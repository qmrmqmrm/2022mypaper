import numpy as np

import config as cfg
import utils.framework.util_function as uf


class FeatureMapDistributer:
    def __init__(self, ditrib_policy, anchors_per_scale):
        self.ditrib_policy = eval(ditrib_policy)(anchors_per_scale)

    def create(self, bboxes, feat_sizes):
        bbox_map = self.ditrib_policy(bboxes, feat_sizes)
        return bbox_map


class ObjectDistribPolicy:
    def __init__(self, anchors_per_scale):
        self.feat_order = cfg.ModelOutput.FEATURE_SCALES
        self.anchor_ratio = np.concatenate([anchor for anchor in anchors_per_scale])
        self.num_anchor = len(self.anchor_ratio) // len(self.feat_order)


class SinglePositivePolicy(ObjectDistribPolicy):
    def __init__(self, anchors_per_scale):
        super().__init__(anchors_per_scale)

    def __call__(self, bboxes, feat_sizes):
        """
            :param bboxes: bounding boxes in image ratio (0~1) [cy, cx, h, w, obj, major_category, minor_category, depth] (N, 8)
            :param anchors: anchors in image ratio (0~1) (9, 2)
            :param feat_sizes: feature map sizes for 3 feature maps
            :return:
            """

        boxes_hw = bboxes[:, np.newaxis, 2:4]  # (N, 1, 8)
        anchors_hw = self.anchor_ratio[np.newaxis, :, :]  # (1, 9, 2)
        inter_hw = np.minimum(boxes_hw, anchors_hw)  # (N, 9, 2)
        inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]  # (N, 9)
        union_area = boxes_hw[:, :, 0] * boxes_hw[:, :, 1] + anchors_hw[:, :, 0] * anchors_hw[:, :, 1] - inter_area
        iou = inter_area / union_area
        best_anchor_indices = np.argmax(iou, axis=1)

        gt_features = [np.zeros((feat_shape[0], feat_shape[1], self.num_anchor, bboxes.shape[-1]), dtype=np.float32)
                       for feat_shape in feat_sizes]

        # TODO split anchor indices by scales, create each feature map by single operation
        for anchor_index, bbox in zip(best_anchor_indices, bboxes):
            scale_index = anchor_index // self.num_anchor
            anchor_index_in_scale = anchor_index % self.num_anchor
            feat_map = gt_features[scale_index]
            # bbox: [y, x, h, w, category]
            grid_yx = (bbox[:2] * feat_sizes[scale_index]).astype(np.int32)
            assert (grid_yx >= 0).all() and (grid_yx < feat_sizes[scale_index]).all()
            # # bbox: [y, x, h, w, 1, major_category, minor_category, depth]
            feat_map[grid_yx[0], grid_yx[1], anchor_index_in_scale] = bbox
            gt_features[scale_index] = feat_map
        return gt_features


class MultiPositivePolicy(ObjectDistribPolicy):
    def __init__(self, anchors_per_scale):
        super().__init__(anchors_per_scale)
        self.iou_threshold = cfg.FeatureDistribPolicy.IOU_THRESH
        self.image_shape = cfg.Datasets.DATASET_CONFIG.INPUT_RESOLUTION
        self.strides = cfg.ModelOutput.FEATURE_SCALES

# TODO 1. make anchor map 2. anchor and gt bbox match 3. iou threshold
    def __call__(self, bboxes, feat_sizes):

        anchor_map = self.make_anchor_map(feat_sizes)
        gt_features = self.make_bbox_map(bboxes, anchor_map, feat_sizes)
        return gt_features

    def make_anchor_map(self, feat_sizes):
        anchors = []
        for scale, (feat_size, stride) in enumerate(zip(feat_sizes, self.feat_order)):
            ry = (np.arange(0, feat_size[0]) + 0.5) * stride / self.image_shape[0]
            rx = (np.arange(0, feat_size[1]) + 0.5) * stride / self.image_shape[1]
            ry, rx = np.meshgrid(ry, rx)

            grid_map = np.vstack((ry.ravel(), rx.ravel(), np.zeros(feat_size[0] * feat_size[1]), np.zeros(feat_size[0] * feat_size[1]))).transpose()
            anchor_ratio = np.concatenate([np.zeros((3, 2)), self.anchor_ratio[scale * 3:(scale+1) * 3]], axis=1)
            anchor_map = (anchor_ratio.reshape((1, self.num_anchor, 4))
                          + grid_map.reshape((1, grid_map.shape[0], 4)).transpose((1, 0, 2)))
            anchor_map = anchor_map.reshape((self.num_anchor * grid_map.shape[0], 4))
            anchors.append(anchor_map)
        anchors = np.concatenate(anchors, axis=0)
        return anchors

    def make_bbox_map(self, bboxes, anchor_map, feat_sizes):

        bboxes_tlbr = uf.convert_box_format_yxhw_to_tlbr(bboxes)
        anchor_tlbr = uf.convert_box_format_yxhw_to_tlbr(anchor_map)
        bboxes_area = (bboxes_tlbr[:, 2] - bboxes_tlbr[:, 0]) * (bboxes_tlbr[:, 3] - bboxes_tlbr[:, 1])
        anchor_area = (anchor_tlbr[:, 2] - anchor_tlbr[:, 0]) * (anchor_tlbr[:, 3] - anchor_tlbr[:, 1])
        width_height = np.minimum(bboxes_tlbr[:, np.newaxis, 2:4], anchor_tlbr[np.newaxis, :, 2:]) - \
                       np.maximum(bboxes_tlbr[:, np.newaxis, :2], anchor_tlbr[np.newaxis, :, :2])
        width_height = np.clip(width_height, 0, 1)
        intersection = np.prod(width_height, axis=-1)
        iou = np.where(intersection > 0, intersection / (bboxes_area[:, np.newaxis] + anchor_area[np.newaxis, :] - intersection), np.zeros(1))
        max_iou = np.amax(iou, axis=0)
        max_gt_idx = np.argmax(iou, axis=1)
        max_idx = np.argmax(iou, axis=0)
        positive = max_iou > self.iou_threshold[0]
        negative = max_iou < self.iou_threshold[1]

        max_match = np.zeros((anchor_tlbr.shape[0], bboxes_tlbr.shape[-1]))
        max_match[max_gt_idx] = bboxes
        max_match = max_match * (~positive[:, np.newaxis])
        iou_match = bboxes[max_idx, ...] * positive[:, np.newaxis]
        gt_match = iou_match + max_match

        gt_features = []
        start_channel = 0
        for scale in feat_sizes:
            last_channel = start_channel + scale[0] * scale[1] * self.num_anchor
            gt_feat = gt_match[start_channel: last_channel, ...].astype(np.float32)
            start_channel = last_channel
            gt_features.append(gt_feat.reshape(scale[0], scale[1], self.num_anchor, -1))
        return gt_features


class OTAPolicy:
    pass