import tensorflow as tf

import utils.framework.util_function as uf
import config as cfg
import config_dir.util_config as uc
import model.framework.model_util as mu


class FeatureDecoder:
    def __init__(self, anchors_per_scale,
                 channel_compos=uc.get_channel_composition(False)):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.num_scale = len(cfg.ModelOutput.FEATURE_SCALES)
        self.const_3 = tf.constant(3, dtype=tf.float32)
        self.const_log_2 = tf.math.log(tf.constant(2, dtype=tf.float32))
        self.channel_compos = channel_compos
        self.sensitive_value = 1.05

    def __call__(self, feature):
        """
        :param feature: raw feature map predicted by model (batch, grid_h, grid_w, anchor, channel)
        :param scale_ind: scale name e.g. 0~2
        :return: decoded feature in the same shape e.g. (yxhw, objectness, category probabilities)
        """
        decoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            anchors_ratio = self.anchors_per_scale[scale_index]
            box_yx = self.decode_yx(feature["yxhw"][scale_index][..., :2])
            box_hw = self.decode_hw(feature["yxhw"][scale_index][..., 2:], anchors_ratio)
            decoded["yxhw"].append(tf.concat([box_yx, box_hw], axis=-1))
            decoded["category"].append(mu.sigmoid_with_margin(feature["category"][scale_index], 0, 0, 1))
            decoded["distance"].append(tf.exp(feature["distance"][scale_index]))
            # TODO check performance improve
            # decoded["distance"] = 1/tf.sigmoid(slices["distance"])
            # TODO check iou-aware
            if cfg.ModelOutput.IOU_AWARE:
                decoded["ioup"].append(mu.sigmoid_with_margin(feature["ioup"][scale_index], 0, 0, 1))
                decoded["object"].append(self.obj_post_process(mu.sigmoid_with_margin(feature["object"], 0), decoded["ioup"]))
            else:
                decoded["object"].append(mu.sigmoid_with_margin(feature["object"][scale_index], 0, 0, 1))
            if cfg.ModelOutput.MINOR_CTGR:
                decoded["sign_ctgr"].append(mu.sigmoid_with_margin(feature["sign_ctgr"][scale_index], 0, 0, 1))
                decoded["mark_ctgr"].append(mu.sigmoid_with_margin(feature["mark_ctgr"][scale_index], 0, 0, 1))
            if cfg.ModelOutput.SPEED_LIMIT:
                decoded["sign_speed"].append(mu.sigmoid_with_margin(feature["sign_speed"][scale_index], 0, 0, 1))
                decoded["mark_speed"].append(mu.sigmoid_with_margin(feature["mark_speed"][scale_index], 0, 0, 1))

            bbox_pred = [decoded[key][scale_index] for key in self.channel_compos]
            decoded["whole"].append(tf.concat(bbox_pred, axis=-1))

            assert decoded["whole"][scale_index].shape == feature["whole"][scale_index].shape
        return decoded

    def obj_post_process(self, obj, ioup):
        iou_aware_factor = 0.4
        new_obj = tf.pow(obj, (1 - iou_aware_factor)) * tf.pow(ioup, iou_aware_factor)
        return new_obj

    def decode_yx(self, yx_raw):
        """
        :param yx_raw: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        grid_h, grid_w = yx_raw.shape[1:3]
        """
        Original yolo v3 implementation: yx_dec = tf.sigmoid(yx_raw)
        For yx_dec to be close to 0 or 1, yx_raw should be -+ infinity
        By expanding activation range -0.2 ~ 1.4, yx_dec can be close to 0 or 1 from moderate values of yx_raw
        """
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        # grid: (grid_h, grid_w, 2)
        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.reshape(grid, (1, grid_h, grid_w, 1, 2))
        grid = tf.cast(grid, tf.float32)
        divider = tf.reshape([grid_h, grid_w], (1, 1, 1, 1, 2))
        divider = tf.cast(divider, tf.float32)

        # yx_box = tf.sigmoid(yx_raw) * 1.4 - 0.2
        yx_box = mu.sigmoid_with_margin(yx_raw, 0.2, 0, 1)
        # [(batch, grid_h, grid_w, anchor, 2) + (1, grid_h, grid_w, 1, 2)] / (1, 1, 1, 1, 2)
        yx_dec = (yx_box + grid) / divider
        return yx_dec

    def decode_hw(self, hw_raw, anchors_ratio):
        """
        :param hw_raw: (batch, grid_h, grid_w, anchor, 2)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        num_anc, channel = anchors_ratio.shape     # (3, 2)
        anchors_tf = tf.reshape(anchors_ratio, (1, 1, 1, num_anc, channel))
        # NOTE: exp activation may result in infinity
        # hw_dec = tf.exp(hw_raw) * anchors_tf
        # hw_dec: 0~3 times of anchor, the delayed sigmoid passes through (0, 1)
        # hw_dec = self.const_3 * tf.sigmoid(hw_raw - self.const_log_2) * anchors_tf
        hw_dec = tf.exp(hw_raw) * anchors_tf
        return hw_dec


class FeatureLaneDecoder:
    def __init__(self, channel_compos=uc.get_lane_channel_composition(False)):
        self.channel_compos = channel_compos

    def __call__(self, feature):
        decoded = {key: [] for key in feature.keys()}

        decoded["laneness"].append(mu.sigmoid_with_margin(feature["laneness"][0], 0, 0, 1))
        decoded["lane_centerness"].append(mu.sigmoid_with_margin(feature["lane_centerness"][0], 0, 0, 1))
        lane_fpoints = self.decode_fpoints(feature["lane_fpoints"][0])
        decoded["lane_fpoints"].append(lane_fpoints)
        # decoded["lane_category"].append(tf.nn.softmax(feature["lane_category"][0]))
        decoded["lane_category"].append(mu.sigmoid_with_margin(feature["lane_category"][0], 0, 0, 1))
        lane_pred = [decoded[key][0] for key in self.channel_compos]
        decoded["whole"].append(tf.concat(lane_pred, axis=-1))

        assert decoded["whole"][0].shape == feature["whole"][0].shape
        return decoded

    def decode_fpoints(self, feature):
        batch, grid_h, grid_w, anchor, channel = feature.shape
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        # grid: (grid_h, grid_w, 2)
        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.reshape(grid, (1, grid_h, grid_w, 1, 1, 2))
        grid = tf.cast(grid, tf.float32)
        divider = tf.reshape([grid_h, grid_w], (1, 1, 1, 1, 1, 2))
        divider = tf.cast(divider, tf.float32) # five_points = five_points.reshape(-1, 5, 2) / np.array([[height, width]])

        lane_fpoints = mu.sigmoid_with_margin(feature, 0, -1, 1)
        lane_fpoints = tf.reshape(lane_fpoints, (batch, grid_h, grid_w, anchor, -1, 2)) + grid/divider
        lane_fpoints = tf.clip_by_value(lane_fpoints, 0, 1)
        lane_fpoints = tf.reshape(lane_fpoints, (batch, grid_h, grid_w, anchor, channel))
        return lane_fpoints

