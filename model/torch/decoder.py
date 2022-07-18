import torch
from torch import nn
import torch.nn.functional as F

from utils.util_class import MyExceptionToCatch
import model.framework.model_util as mu
import utils.framework.util_function as uf
import config as cfg
import config_dir.util_config as uc


class FeatureDecoder(nn.Module):
    def __init__(self, anchors_per_scale,
                 channel_compos=uc.get_channel_composition(False)):
        super(FeatureDecoder, self).__init__()
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.channel_compos = channel_compos

    def forward(self, features):
        out_features = []
        for i, (key, feature) in enumerate(features.items()):
            slices = uf.slice_feature(feature, self.channel_compos)
            anchors_ratio = torch.tensor(self.anchors_per_scale[i])

            decoded = dict()
            print("yxhw", slices["yxhw"][:, :2, ...].shape)
            box_yx = self.decode_yx(slices["yxhw"][:, :2, ...])
            box_hw = self.decode_hw(slices["yxhw"][:, :2, ...], anchors_ratio)
            decoded["yxhw"] = torch.cat([box_yx, box_hw], dim=1)
            decoded["object"] = torch.sigmoid(slices["object"])
            decoded["category"] = torch.sigmoid(slices["category"])
            decoded["distance"] = torch.exp(slices["distance"])

            if cfg.ModelOutput.IOU_AWARE:
                decoded["ioup"] = torch.sigmoid(slices["ioup"])
                decoded["object"] = self.obj_post_process(torch.sigmoid(slices["object"]), decoded["ioup"])
            else:
                decoded["object"] = torch.sigmoid(slices["object"])
            if cfg.ModelOutput.MINOR_CTGR:
                decoded["sign_ctgr"] = torch.sigmoid(slices["sign_ctgr"])
                decoded["mark_ctgr"] = torch.sigmoid(slices["mark_ctgr"])
            if cfg.ModelOutput.SPEED_LIMIT:
                decoded["sign_speed"] = torch.sigmoid(slices["sign_speed"])
                decoded["mark_speed"] = torch.sigmoid(slices["mark_speed"])
            bbox_pred = [decoded[key] for key in self.channel_compos]
            bbox_pred = torch.cat(bbox_pred, dim=1)
            out_features.append(bbox_pred)

        return out_features

    def decode_yx(self, yx_raw):
        """
        :param yx_raw: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        grid_h, grid_w = yx_raw.shape[-2:]
        """
        Original yolo v3 implementation: yx_dec = tf.sigmoid(yx_raw)
        For yx_dec to be close to 0 or 1, yx_raw should be -+ infinity
        By expanding activation range -0.2 ~ 1.4, yx_dec can be close to 0 or 1 from moderate values of yx_raw 
        """
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = torch.meshgrid(torch.range(0, grid_w - 1), torch.range(0, grid_h - 1))
        # grid: (grid_h, grid_w, 2)
        grid = torch.stack([grid_y, grid_x], dim=-1)
        grid = grid.view(1, 2, 1, grid_h, grid_w)
        grid = grid.to(torch.float32)
        divider = torch.tensor([grid_h, grid_w])
        divider = divider.view(1, 2, 1, 1, 1)
        divider = divider.to(torch.float32)

        yx_box = torch.sigmoid(yx_raw) * 1.4 - 0.2
        # [(batch, grid_h, grid_w, anchor, 2) + (1, grid_h, grid_w, 1, 2)] / (1, 1, 1, 1, 2)
        yx_dec = (yx_box + grid) / divider
        return yx_dec

    def obj_post_process(self, obj, ioup):
        iou_aware_factor = 0.4
        new_obj = torch.pow(obj, (1 - iou_aware_factor)) * torch.pow(ioup, iou_aware_factor)
        return new_obj

    def decode_hw(self, hw_raw, anchors_ratio):
        """
        :param hw_raw: (batch, grid_h, grid_w, anchor, 2)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        num_anc, channel = anchors_ratio.shape  # (3, 2)
        anchors_tf = torch.reshape(anchors_ratio, (1, channel, num_anc, 1, 1))
        # NOTE: exp activation may result in infinity
        # hw_dec = tf.exp(hw_raw) * anchors_tf
        # hw_dec: 0~3 times of anchor, the delayed sigmoid passes through (0, 1)
        # hw_dec = self.const_3 * tf.sigmoid(hw_raw - self.const_log_2) * anchors_tf
        hw_dec = torch.exp(hw_raw) * anchors_tf
        return hw_dec
