import numpy as np

import config as cfg
import utils.framework.util_function as uf
import config_dir.util_config as uc
from model.framework.encoder import FeatureLaneEncoder


class FeatureMapDistributer:
    def __init__(self, distrib_policy, imshape, anchors_per_scale, include_lane=cfg.ModelOutput.LANE_DET):
        self.include_lane = include_lane
        self.box_ditrib_policy = eval(distrib_policy)(imshape, anchors_per_scale,
                                                      sum(uc.get_bbox_composition(True).values()))
        if include_lane:
            self.lane_ditrib_policy = LaneFeatureGenerator(imshape, anchors_per_scale,
                                                           sum(uc.get_lane_channel_composition(True).values()))
            self.lane_encoder = FeatureLaneEncoder(uc.get_lane_channel_composition(False))

    def __call__(self, features):
        feat_keys = [key for key in features.keys() if "inst" in key]
        for key in feat_keys:
            features[key] = uf.merge_and_slice_features(features[key], True, key)
        features = uf.convert_tensor_to_numpy(features)
        if cfg.ModelOutput.BOX_DET:
            features["feat_box"] = self.box_ditrib_policy(features["inst_box"])
        if self.include_lane:
            features["feat_lane"] = self.lane_ditrib_policy(features)
            features["feat_lane_logit"] = self.lane_encoder.inverse(features["feat_lane"])
        return features


class ObjectDistribPolicy:
    def __init__(self, imshape, anchors_per_scale):
        self.anchor_ratio = np.concatenate([anchor for anchor in anchors_per_scale])
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.num_anchor = len(self.anchor_ratio) // len(self.feat_scales)
        self.feat_shapes = [np.array(imshape[:2]) // scale for scale in self.feat_scales]

    def slice_and_merge(self, features):
        slice_feat = uf.merge_and_slice_features(features, True, "feat_box")
        merge_features = {key: list() for key in slice_feat.keys()}
        for key, feat in slice_feat.items():
            if key is "whole":
                merge_features[key] = feat
                continue
            for dict_per_feat in feat:
                merge_features[key].append(uf.merge_dim_hwa(dict_per_feat))
        return merge_features


class SinglePositivePolicy(ObjectDistribPolicy):
    def __init__(self, imshape, anchors_per_scale, box_channels):
        super().__init__(imshape, anchors_per_scale)
        self.box_channels = box_channels

    def __call__(self, box_features):
        """
            :param features: {inst_box: {"yxhw": ..., "object": ..., ...}, inst_dc{...}, ...}
            :return:
            """
        bboxes = box_features["yxhw"]
        boxes_hw = bboxes[..., np.newaxis, 2:4]  # (B, N, 1, 2)
        anchors_hw = self.anchor_ratio[np.newaxis, np.newaxis, :, :]  # (1, 1, 9, 2)
        inter_hw = np.minimum(boxes_hw, anchors_hw)  # (B, N, 9, 2)
        inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (B, N, 9)
        union_area = boxes_hw[..., 0] * boxes_hw[..., 1] + anchors_hw[..., 0] * anchors_hw[..., 1] - inter_area
        iou = inter_area / union_area
        best_anchor_indices = np.argmax(iou, axis=-1)
        out_features = [np.zeros((bboxes.shape[0], feat_shape[0], feat_shape[1], self.num_anchor,
                                  box_features["merged"].shape[-1] + 1), dtype=np.float32)
                          for feat_shape in self.feat_shapes]
        batch_features = [np.zeros((bboxes.shape[0], feat_shape[0], feat_shape[1], self.num_anchor, box_features["merged"].shape[-1]),
                                   dtype=np.float32)
                          for feat_shape in self.feat_shapes]
        anchor_map = [np.ones((bboxes.shape[0], feat_shape[0], feat_shape[1], self.num_anchor, 1), dtype=np.float32) * (-1)
                      for feat_shape in self.feat_shapes]
        for batch in range(bboxes.shape[0]):
            for anchor_index, box in zip(best_anchor_indices[batch], box_features["merged"][batch]):
                if np.all(box == 0):
                    break
                scale_index = anchor_index // self.num_anchor
                anchor_index_in_scale = anchor_index % self.num_anchor
                feat_map = batch_features[scale_index]
                anchor_scale_map = anchor_map[scale_index]
                # bbox: [y, x, h, w, category]
                grid_yx = (box[:2] * self.feat_shapes[scale_index]).astype(np.int32)
                assert (grid_yx >= 0).all() and (grid_yx < self.feat_shapes[scale_index]).all()
                # # bbox: [y, x, h, w, 1, major_category, minor_category, depth]
                feat_map[batch, grid_yx[0], grid_yx[1], anchor_index_in_scale] = box
                anchor_scale_map[batch, grid_yx[0], grid_yx[1], anchor_index_in_scale, 0] = anchor_index
                out_features[scale_index] = np.concatenate([feat_map, anchor_scale_map], axis=-1)

        out_features = self.slice_and_merge(out_features)
        return out_features


class FasterRCNNPolicy(ObjectDistribPolicy):
    def __init__(self, imshape, anchors_per_scale, box_channels):
        super().__init__(imshape, anchors_per_scale)
        self.iou_threshold = cfg.FeatureDistribPolicy.IOU_THRESH
        self.image_shape = cfg.Datasets.DATASET_CONFIG.INPUT_RESOLUTION
        self.strides = cfg.ModelOutput.FEATURE_SCALES
        self.box_channels = box_channels

    # TODO 1. make anchor map 2. anchor and gt bbox match 3. iou threshold
    def __call__(self, bboxes, feat_sizes):

        anchor_map = self.make_anchor_map(feat_sizes)
        gt_features = self.make_bbox_map(bboxes, anchor_map, feat_sizes)
        return gt_features

    def make_anchor_map(self, feat_sizes):
        anchors = []
        for scale, (feat_size, stride) in enumerate(zip(feat_sizes, self.feat_scales)):
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


class MultiPositiveGenerator:
    def __init__(self, imshape, center_radius=cfg.FeatureDistribPolicy.CENTER_RADIUS,
                 multi_positive_weight=cfg.FeatureDistribPolicy.MULTI_POSITIVE_WIEGHT):
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.feat_shapes = [np.array(imshape[:2]) // scale for scale in self.feat_scales]
        self.box_size_range_per_scale = self.get_box_size_ranges()
        self.center_radius = center_radius
        self.multi_positive_weight = multi_positive_weight

    def get_box_size_ranges(self):
        size_ranges = []
        for feat_size in self.feat_scales:
            size_ranges.append([feat_size * np.sqrt(2) / 2, feat_size * np.sqrt(2) * 3])
        size_ranges = np.array(size_ranges)
        # no upper bound for large scale
        size_ranges[0, 0] = 0
        size_ranges[-1, 1] = 100000
        return size_ranges

    def __call__(self, inst_feat):
        box = inst_feat["yxhw"]
        grid_yx, belong_to_scale = self.to_grid_over_scales(box)
        single_positive = self.create_featmap_single_positive(inst_feat, grid_yx, belong_to_scale)

        # for key in ["feat2d", "feat3d"]:
        # single_positive = uf.merge_and_slice_features(single_positive, True, "feat_box")
        single_positive = self.slice_and_merge(single_positive)
        multi_positive = self.multi_positive_objectness(box, belong_to_scale, single_positive["object"])
        single_positive["object"] = multi_positive
        return single_positive

    def to_grid_over_scales(self, box2d):
        grid_yx, belong_to_scale = [], []
        for i, s in enumerate(self.feat_scales):
            grid_yx_in_scale, belong_in_scale = self.to_grid(box2d, i)
            grid_yx.append(grid_yx_in_scale)
            belong_to_scale.append(belong_in_scale)
        return grid_yx, belong_to_scale

    def to_grid(self, box2d, scale_index):
        box2d_pixel = uf.convert_box_scale_01_to_pixel(box2d)
        diag_length = np.linalg.norm(box2d_pixel[..., 2:4], axis=-1)
        box_range = self.box_size_range_per_scale[scale_index]
        belong_to_scale = np.logical_and(diag_length > box_range[0], diag_length < box_range[1])
        box_grid_yx = box2d_pixel[..., :2] // np.array(self.feat_scales[scale_index])
        return box_grid_yx, belong_to_scale

    def create_featmap_single_positive(self, inst, grid_yx, belong_to_scale):
        feats_over_scale = []
        for i, s in enumerate(self.feat_scales):
            feats_in_batch = []
            for b in range(inst["merged"].shape[0]):
                feat_map = self.create_featmap(self.feat_shapes[i], inst["merged"][b], grid_yx[i][b],
                                               belong_to_scale[i][b])
                feats_in_batch.append(feat_map)
            feats_over_scale.append(np.stack(feats_in_batch, axis=0))
        return feats_over_scale

    def create_featmap(self, feat_shapes, instances, grid_yx, valid):
        valid_grid_yx = grid_yx[valid].astype(np.int)
        gt_features = np.zeros((feat_shapes[0], feat_shapes[1], 1, instances.shape[-1]), dtype=np.float32)
        gt_features[valid_grid_yx[:, 0], valid_grid_yx[:, 1], ...] = instances[valid][..., np.newaxis, :]
        return gt_features

    def slice_and_merge(self, features):
        slice_feat = uf.merge_and_slice_features(features, True, "feat_box")
        merge_features = {key: list() for key in slice_feat.keys()}
        for key, feat in slice_feat.items():
            if key is "whole":
                merge_features[key] = feat
                continue
            for dict_per_feat in feat:
                merge_features[key].append(uf.merge_dim_hwa(dict_per_feat))
        return merge_features

    def multi_positive_objectness(self, box2d, validity, single_positive_map):
        mp_object_over_scale = []
        for i, s in enumerate(self.feat_scales):
            grid_center_yx = self.make_grid_center(self.feat_shapes[i])
            positive_tlbr_from_box = self.get_positive_range_from_box(box2d, validity[i])
            positive_tlbr_from_radius = self.get_positive_range_from_radius(box2d, self.center_radius, i, validity[i])
            object_from_box = self.get_positive_map_in_boxes(positive_tlbr_from_box, grid_center_yx)
            object_from_radius = self.get_positive_map_in_boxes(positive_tlbr_from_radius, grid_center_yx)
            mp_object = self.merge_positive_maps(single_positive_map[i], object_from_box, object_from_radius)
            mp_object_over_scale.append(mp_object)
        return mp_object_over_scale

    def make_grid_center(self, feat_map_size):
        rx = np.arange(0, feat_map_size[1])
        ry = np.arange(0, feat_map_size[0])
        x_grid, y_grid = np.meshgrid(rx, ry)
        y_grid_center = ((y_grid + 0.5) / feat_map_size[0])
        x_grid_center = ((x_grid + 0.5) / feat_map_size[1])
        # (h, w, 2)
        grid_center_yx = np.stack([y_grid_center, x_grid_center], axis=-1)
        return grid_center_yx

    def get_positive_range_from_box(self, box2d_in_scale, validity):
        half_box = np.concatenate([box2d_in_scale[..., :2], box2d_in_scale[..., 2:4] * 0.5], axis=-1)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(half_box)
        box_tlbr *= validity[..., np.newaxis]
        return box_tlbr

    def get_positive_range_from_radius(self, box2d_in_scale, center_radius, scale_index, validity):
        norm_radius = np.array(center_radius) / self.feat_shapes[scale_index]
        positive_t = box2d_in_scale[..., :1] - norm_radius[0]
        positive_l = box2d_in_scale[..., 1:2] - norm_radius[1]
        positive_b = box2d_in_scale[..., :1] + norm_radius[0]
        positive_r = box2d_in_scale[..., 1:2] + norm_radius[1]
        positive_tlbr = np.concatenate([positive_t, positive_l, positive_b, positive_r], axis=-1)
        positive_tlbr *= validity[..., np.newaxis]
        return positive_tlbr

    def get_positive_map_in_boxes(self, positive_tlbr, grid_center_yx):
        """
        :param positive_tlbr: (B, N, 4)
        :param grid_center_yx: (H, W, 2)
        :return:
        """
        grid_center_yx = grid_center_yx[np.newaxis, ...]  # (1,H,W,2)
        positive_tlbr = positive_tlbr[:, np.newaxis, np.newaxis, ...]  # (B,1,1,N,4)
        # (1,H,W,1) - (B,1,1,N) -> (B,H,W,N)
        delta_t = grid_center_yx[..., 0:1] - positive_tlbr[..., 0]
        delta_l = grid_center_yx[..., 1:2] - positive_tlbr[..., 1]
        delta_b = positive_tlbr[..., 2] - grid_center_yx[..., 0:1]
        delta_r = positive_tlbr[..., 3] - grid_center_yx[..., 1:2]
        # (B, H, W, N, 4)
        tblr_grid_x_box = np.stack([delta_t, delta_l, delta_b, delta_r], axis=-1)
        # (B, H, W, N)
        positive_mask = np.all(tblr_grid_x_box > 0, axis=-1)
        # (B, H, W)
        positive_mask = np.any(positive_mask > 0, axis=-1)
        return positive_mask

    def merge_positive_maps(self, single_positive_map, object_from_box, object_from_radius):
        multi_positive_map = np.logical_and(object_from_box, object_from_radius).astype(int).reshape((single_positive_map.shape[0], -1, 1))
        output_map = single_positive_map + (1 - single_positive_map) * multi_positive_map * self.multi_positive_weight
        return output_map


class MultiPositivePolicy(ObjectDistribPolicy):
    def __init__(self, imshape, anchors_per_scale, box_channels, center_radius=cfg.FeatureDistribPolicy.CENTER_RADIUS,
                 resolution=cfg.Datasets.DATASET_CONFIG.INPUT_RESOLUTION):
        super().__init__(imshape, anchors_per_scale)
        self.center_radius = center_radius
        self.resolution = resolution
        self.generate_feature_maps = MultiPositiveGenerator(imshape)
        self.box_channels = box_channels

    def __call__(self, features):
        """
        :param features["inst2d"]: (B, N, 6), np.array, [yxhw, objectness, category]
        :param features["inst3d"]: (B, N, 9), np.array, [yxhwl, z, theta, objecntess, category]
        :return:
        """
        features = self.generate_feature_maps(features)
        # features = self.slice_and_merge(features)
        return features


class LaneFeatureGenerator(ObjectDistribPolicy):
    def __init__(self, imshape, anchors_per_scale, lane_channels):
        super().__init__(imshape, anchors_per_scale)
        self.lane_num = cfg.Dataloader.MAX_LANE_PER_IMAGE
        self.lane_points = cfg.Dataloader.MAX_POINTS_PER_LANE
        self.lane_feature = cfg.ModelOutput.LANE_FEATURE
        self.lane_channels = lane_channels

    def __call__(self, features):
        """
        :param lanes_point: lanes point in image ratio (0~1) (10(n), 30(p), 2)
        :param lanes: lanes params(five points, centerness, category) in image ratio (0~1) (10(n), 12)
        :param feat_sizes: feature map sizes for 3 feature maps
        :return:
        """
        inst_lane = features["inst_lane"]
        lane_points = features["lanes_point"]
        center_points = inst_lane["lane_fpoints"][..., 4:6]
        lanes = inst_lane["merged"]
        batch = lane_points.shape[0]
        lane_gt_features = np.zeros((batch, self.feat_shapes[self.lane_feature][0], self.feat_shapes[self.lane_feature][1], 1, self.lane_channels),
                                    dtype=np.float32)
        for b in range(batch):
            center_point = center_points[b]
            center_valid_mask = center_point[..., 0] > 0
            center_point = center_point[center_valid_mask]
            for points in lane_points[b, ...]:
                valid_lanes_point_mask = points[..., 0] > 0
                points = points[valid_lanes_point_mask]
                spts = points[0:-1]
                epts = points[1:]
                # (11, N-1, 2)
                heat_points = [((i / 10) * spts + ((10 - i) / 10) * epts) * self.feat_shapes[self.lane_feature] for i in range(11)]
                heat_points = np.array(heat_points, dtype=np.int32).reshape((-1, 2))  # (11*(N-1), 2)
                heat_points = np.unique(heat_points, axis=0)
                lane_gt_features[b, heat_points[:, 0], heat_points[:, 1], 0, 0] = 1

            center_points_pixel = (center_point * self.feat_shapes[self.lane_feature]).astype(np.int)
            lanes_per_batch = lanes[b][center_valid_mask]
            lane_gt_features[b, center_points_pixel[:, 0], center_points_pixel[:, 1], 0, 1:] = lanes_per_batch
        lane_gt_features = self.slice_and_merge([lane_gt_features])
        return lane_gt_features

    def slice_and_merge(self, features):
        slice_feat = uf.merge_and_slice_features(features, True, "feat_lane")
        merge_features = {key: list() for key in slice_feat.keys()}
        for key, feat in slice_feat.items():
            if key is "whole":
                merge_features[key] = feat
                continue
            for dict_per_feat in feat:
                merge_features[key].append(uf.merge_dim_hwa(dict_per_feat))
        return merge_features
