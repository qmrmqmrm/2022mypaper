import numpy as np
import tensorflow as tf

import config as cfg3d
import config_dir.util_config as uc
import model.framework.model_util as mu
from model.framework.decoder import FeatureLaneDecoder
import utils.framework.util_function as uf


class FeatureLaneEncoder:
    def __init__(self,
                 channel_compos=uc.get_lane_channel_composition(False)):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """

        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.channel_compos = channel_compos

    def inverse(self, feature):
        encoded = {key: [] for key in feature.keys()}
        encoded["lane_fpoints"].append(self.encode_fpoints(feature["lane_fpoints"][0], feature["whole"][0].shape))
        assert encoded["lane_fpoints"][0].shape == feature["lane_fpoints"][0].shape
        return encoded

    def encode_fpoints(self, decode_fpoints, feat_shape):
        """
        :param decode_fpoints: (batch, grid_h * grid_w, 10)
        :return: yx_raw = yx logit (batch, grid_h, grid_w, anchor, 2)
        """

        batch, grid_h, grid_w = feat_shape[:3]
        vaild_mask = tf.cast(decode_fpoints[..., 1] > 0, dtype=tf.float32)[..., tf.newaxis]
        decode_fpoints = tf.reshape(decode_fpoints, (batch, grid_h, grid_w, 1, -1, 2))

        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        # grid: (grid_h, grid_w, 2)
        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.reshape(grid, (1, grid_h, grid_w, 1, 1, 2))
        grid = tf.cast(grid, tf.float32)
        divider = tf.reshape([grid_h, grid_w], (1, 1, 1, 1, 1, 2))
        divider = tf.cast(divider, tf.float32)

        fpoints = decode_fpoints - grid / divider
        fpoints_raw = mu.inv_sigmoid_with_margin(fpoints, 0, -1, 1)
        return tf.reshape(fpoints_raw, (batch, grid_w * grid_h, -1)) * vaild_mask


# TODO : encode decode 확인
if __name__ == '__main__':
    encoder = FeatureLaneEncoder()
    decoder = FeatureLaneDecoder()
    feature = {"whole": [], "laneness": [], "lane_fpoints": [], "lane_centerness": [], "lane_category": []}
    feature["laneness"].append(np.zeros((1, 10, 20, 1, 1), dtype=np.float32))
    feature["lane_centerness"].append(np.zeros((1, 10, 20, 1, 1), dtype=np.float32))
    feature["lane_category"].append(np.zeros((1, 10, 20, 1, 1), dtype=np.float32))
    lane_fpoints = np.arange(0, 1, 0.0005).astype(np.float32)
    lane_fpoints = lane_fpoints.reshape((1, 10, 20, 1, 10))
    test_fpoints = lane_fpoints.reshape((1, 200, 10))
    feature["lane_fpoints"].append(lane_fpoints)
    feature["whole"].append(np.concatenate(
        [feature["laneness"][0], lane_fpoints, feature["lane_centerness"][0], feature["lane_category"][0]], axis=-1))

    check_feature = {key: np.copy(value) for key, value in feature.items()}
    # for key in output_features.keys():
    for slice_key in feature.keys():
        if slice_key != "whole":
            for scale_index in range(len(feature[slice_key])):
                feature[slice_key][scale_index] = uf.merge_dim_hwa(
                    feature[slice_key][scale_index])
    encode = encoder.inverse(feature)


    for slice_key in encode.keys():
        for scale_index in range(len(encode[slice_key])):
            batch, grid_h, grid_w, anchor, featdim = check_feature[slice_key][scale_index].shape
            encode[slice_key][scale_index] = \
                tf.reshape(encode[slice_key][scale_index], (batch, grid_h, grid_w, anchor, featdim))


    output_features = decoder(encode)
    test = output_features["lane_fpoints"][0].numpy().reshape(1, 200, 10)

    print(test)
