import time

from train.framework.loss_pool import *
import train.framework.loss_pool as loss
import utils.framework.util_function as uf
import config as cfg


class IntegratedLoss:
    def __init__(self, loss_weights, valid_category):
        self.loss_names = [key for key in loss_weights.keys()]
        box_loss_weights, lane_loss_weights = self.split_weights(loss_weights)
        self.box_loss = IntegratedBoxLoss(box_loss_weights, valid_category)
        self.lane_loss = IntegratedLaneLoss(lane_loss_weights)

    def split_weights(self, loss_weights):
        box_loss_weights, lane_loss_weights = {}, {}
        for key, value in loss_weights.items():
            if key.startswith("lane"):
                lane_loss_weights[key] = value
            else:
                box_loss_weights[key] = value
        return box_loss_weights, lane_loss_weights

    def __call__(self, features, predictions):
        total_loss = 0
        loss_by_type = {}
        if cfg.ModelOutput.BOX_DET:
            total_loss, loss_by_type = self.box_loss(features, predictions)
        if cfg.ModelOutput.LANE_DET:
            lane_scalar_loss, lane_loss_by_type = self.lane_loss(features, predictions)
            total_loss += lane_scalar_loss
            loss_by_type.update(lane_loss_by_type)
        return total_loss, loss_by_type


class IntegratedBoxLoss:
    def __init__(self, loss_weights, valid_category):
        self.train_minor_ctgr = cfg.ModelOutput.MINOR_CTGR
        self.train_speed_ctgr = cfg.ModelOutput.SPEED_LIMIT
        self.use_ignore_mask = cfg.Train.IGNORE_MASK
        self.loss_weights = loss_weights
        self.num_scale = len(cfg.ModelOutput.FEATURE_SCALES)
        self.sign_idx = cfg.Dataloader.CATEGORY_NAMES["major"].index("Traffic sign")
        self.mark_idx = cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark")
        self.sign_speed_idx = cfg.Dataloader.CATEGORY_NAMES["sign"].index("TS_SPEED_LIMIT")
        self.mark_speed_idx = cfg.Dataloader.CATEGORY_NAMES["mark"].index("RM_SPEED_LIMIT")
        # self.valid_category: binary mask of categories, (1, 1, K)
        self.valid_category = uf.convert_to_tensor(valid_category, "float32")
        self.scaled_loss_objects = self.create_loss_objects(loss_weights)

    def create_loss_objects(self, loss_weights):
        loss_objects = dict()
        for loss_name, values in loss_weights.items():
            loss_objects[loss_name] = eval(values[1])(*values[2:])
        return loss_objects

    def __call__(self, features, predictions):
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_weights}
        for scale in range(self.num_scale):
            auxi = self.prepare_box_auxiliary_data(features["feat_box"], features["inst_box"],
                                                   predictions["feat_box"], scale)
            for loss_name, loss_object in self.scaled_loss_objects.items():
                loss_map_suffix = loss_name + "_map"
                if loss_map_suffix not in loss_by_type:
                    loss_by_type[loss_map_suffix] = []

                scalar_loss, loss_map = loss_object(features["feat_box"], predictions["feat_box"], auxi, scale)
                weight = self.loss_weights[loss_name][0][scale]
                total_loss += scalar_loss * weight
                loss_by_type[loss_name] += scalar_loss * weight
                loss_by_type[loss_map_suffix].append(loss_map)
        return total_loss, loss_by_type

    def prepare_box_auxiliary_data(self, grtr_feat, grtr_boxes, pred_feat, scale):
        auxiliary = dict()
        # As object_count is used as a denominator, it must NOT be 0.
        auxiliary["object_count"] = uf.maximum(uf.reduce_sum(grtr_feat["object"][scale]), 1)
        auxiliary["valid_category"] = self.valid_category
        auxiliary["ignore_mask"] = self.get_ignore_mask(grtr_boxes, pred_feat, scale)
        auxiliary["sign_ctgr"] = self.sign_idx
        auxiliary["mark_ctgr"] = self.mark_idx
        auxiliary["sign_speed"] = self.sign_speed_idx
        auxiliary["mark_speed"] = self.mark_speed_idx
        return auxiliary

    def get_ignore_mask(self, grtr, pred, scale):
        if not self.use_ignore_mask:
            return 1
        iou = uf.compute_iou_general(pred["yxhw"][scale], grtr["yxhw"])
        best_iou = uf.reduce_max(iou, axis=-1)
        ignore_mask = uf.cast(best_iou < 0.65, dtype='float32')
        return ignore_mask


class IntegratedLaneLoss:
    def __init__(self, loss_weights):
        self.loss_weights = loss_weights
        self.loss_objects = self.create_loss_objects(loss_weights)

    def create_loss_objects(self, loss_weights):
        loss_objects = dict()
        for loss_name, values in loss_weights.items():
            loss_objects[loss_name] = eval(values[1])(*values[2:])
        return loss_objects

    def __call__(self, features, predictions):
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_weights}
        auxi = self.prepare_box_auxiliary_data(features["feat_lane"], features["inst_lane"],
                                               predictions["feat_lane"])
        for loss_name, loss_object in self.loss_objects.items():
            # auxi = self.prepare_lane_auxiliary_data(grtr_lane_slices, pred_lane_slices)
            scalar_loss, loss_map = loss_object(features, predictions, auxi, 0)
            total_loss += scalar_loss * self.loss_weights[loss_name][0]
            loss_by_type[loss_name] = scalar_loss * self.loss_weights[loss_name][0]
            loss_by_type[loss_name + "_map"] = [loss_map]  # (batch, HWA)

        return total_loss, loss_by_type

    def prepare_box_auxiliary_data(self, grtr_feat, grtr_inst, pred_feat):
        auxiliary = dict()
        auxiliary["ignore_mask"] = self.get_ignore_mask(grtr_feat)
        return auxiliary

    def get_ignore_mask(self, grtr):
        centerness = grtr["whole"][0][..., -2:-1]
        ignore_mask = uf.avg_pool(centerness)
        mask = uf.cast(ignore_mask > 0, "float32")
        ignore_mask = mask - centerness
        return ignore_mask
