import tensorflow as tf
import numpy as np

from utils.util_class import MyExceptionToCatch
from model.framework.drop_block import DropBlock2D
import model.framework.model_util as mu
import utils.framework.util_function as uf


def neck_factory(head, conv_args, training, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels):
    if head == "FPN":
        return FPN(conv_args, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)
    elif head == "PAN":
        return PAN(conv_args, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)
    elif head == "PPFPN":
        return PPFPN(conv_args, training, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)
    elif head == "BiFPN":
        return BiFPN(conv_args, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {head}")


class NeckBase:
    def __init__(self, conv_args, num_anchors_per_scale, out_channels, num_lane_anchors=None, lane_out_channels=None):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **conv_args)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **conv_args)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **conv_args)
        self.conv2d_output = mu.CustomConv2D(kernel_size=1, strides=1, activation=False, scope="output", bn=False)
        self.num_anchors_per_scale = num_anchors_per_scale
        self.num_lane_anchors = num_lane_anchors if num_lane_anchors is not None else None
        self.out_channels = out_channels
        self.lane_out_channels = lane_out_channels if lane_out_channels is not None else None

    def __call__(self, input_features):
        raise NotImplementedError()

    def conv_5x(self, x, channel):
        x = self.conv2d_k1(x, channel)
        x = self.conv2d(x, channel * 2)
        x = self.conv2d_k1(x, channel)
        x = self.conv2d(x, channel * 2)
        x = self.conv2d_k1(x, channel)
        return x

    def coordconv(self, feature, with_r=False):
        batch_size = tf.shape(feature)[0]
        x_dim = tf.shape(feature)[2]
        y_dim = tf.shape(feature)[1]

        xx_indices = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(x_dim), 0), 0), [batch_size, y_dim, 1])
        xx_indices = tf.expand_dims(xx_indices, -1)

        yy_indices = tf.tile(tf.expand_dims(tf.reshape(tf.range(y_dim), (y_dim, 1)), 0), [batch_size, 1, x_dim])
        yy_indices = tf.expand_dims(yy_indices, -1)

        xx_indices = tf.divide(xx_indices, x_dim - 1)
        yy_indices = tf.divide(yy_indices, y_dim - 1)

        xx_indices = tf.cast(tf.subtract(tf.multiply(xx_indices, 2.), 1.), dtype=feature.dtype)
        yy_indices = tf.cast(tf.subtract(tf.multiply(yy_indices, 2.), 1.), dtype=feature.dtype)

        coord_feature = tf.concat([feature, xx_indices, yy_indices], axis=-1)
        if with_r:
            rr = tf.sqrt(tf.add(tf.square(xx_indices - 0.5),
                                tf.square(yy_indices - 0.5)))
            coord_feature = tf.concat([coord_feature, rr], axis=-1)

        return coord_feature


class FPN(NeckBase):
    def __init__(self, model_cfg, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels):
        super().__init__(model_cfg, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)

    def __call__(self, input_features):
        features = list()
        large = input_features["feature_l"]
        medium = input_features["feature_m"]
        small = input_features["feature_s"]
        conv_large = self.conv_5x(large, 512)
        conv_large = self.conv2d(conv_large, 1024)

        conv_medium = self.upsample_concat(large, medium, 256)
        conv_medium = self.conv_5x(conv_medium, 256)
        conv_medium = self.conv2d(conv_medium, 512)

        conv_small = self.upsample_concat(conv_medium, small, 128)
        conv_small = self.conv_5x(conv_small, 128)
        conv_small = self.conv2d(conv_small, 256)
        features.append(conv_small)
        features.append(conv_medium)
        features.append(conv_large)
        return features

    def upsample_concat(self, upper, lower, channel):
        x = self.conv2d_k1(upper, channel)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.concat([x, lower], axis=-1)
        return x


class PPFPN(NeckBase):
    def __init__(self, model_cfg, training, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels):
        super().__init__(model_cfg, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)
        self.training = training
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(5, 5), padding='same', strides=(1, 1))
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), padding='same', strides=(1, 1))
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(13, 13), padding='same', strides=(1, 1))
        self.conv2d_k1v = mu.CustomConv2D(1, 1, padding="valid", **model_cfg)
        self.dropblock = DropBlock2D(keep_prob=0.9, block_size=3)

    def __call__(self, input_features):
        features = list()
        large = input_features["feature_l"]
        medium = input_features["feature_m"]
        small = input_features["feature_s"]

        conv_l = self.coordconv(large)
        conv_l = self.conv2d_k1v(conv_l, 512)
        conv_l = self.conv_block(conv_l, 512, True)
        conv_l = self.sppblock(conv_l)
        conv_l = self.conv_block(conv_l, 512)

        conv_m = self.upsample_concat(conv_l, medium, 512)
        conv_m = self.coordconv(conv_m)
        conv_m = self.conv2d_k1v(conv_m, 256)
        conv_m = self.conv_block(conv_m, 256, True)
        conv_m = self.conv_block(conv_m, 256)

        conv_s = self.upsample_concat(conv_m, small, 256)
        conv_s = self.coordconv(conv_s)
        conv_s = self.conv2d_k1v(conv_s, 128)
        conv_s = self.conv_block(conv_s, 128, True)
        conv_s = self.conv_block(conv_s, 128)
        features.append(conv_s)
        features.append(conv_m)
        features.append(conv_l)
        return features

    def upsample_concat(self, upper, lower, channel):
        x = self.coordconv(upper)
        x = self.conv2d_k1v(x, channel / 2)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.concat([x, lower], axis=-1)
        return x

    def conv_block(self, inputs, channel, is_dropblock=False):
        x = self.conv2d(inputs, channel * 2)
        if is_dropblock:
            x = self.dropblock(x, training=self.training)
        x = self.coordconv(x)
        x = self.conv2d_k1v(x, channel)
        return x

    def sppblock(self, input_tensors):
        pooling_1 = self.pool1(input_tensors)
        pooling_2 = self.pool2(input_tensors)
        pooling_3 = self.pool3(input_tensors)
        output = tf.concat([pooling_3, pooling_2, pooling_1, input_tensors], axis=-1)

        return output

    def output_conv(self, input_tensors, channel):
        x = self.coordconv(input_tensors)
        x = self.conv2d(x, channel * 2)
        x = self.conv2d_output(x, self.num_anchors_per_scale * self.out_channels)
        batch, height, width, channel = x.shape
        output = tf.reshape(x, (batch, height, width, self.num_anchors_per_scale, self.out_channels))
        return output


class PAN(NeckBase):
    def __init__(self, model_cfg, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels):
        super().__init__(model_cfg, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(5, 5), padding='same', strides=(1, 1))
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), padding='same', strides=(1, 1))
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(13, 13), padding='same', strides=(1, 1))

    def __call__(self, input_features):
        conv_result = list()
        if cfg.Architecture.USE_SPP:
            large = self.sppblock(input_features[2])
        else:
            large = input_features[2]
        medium = input_features[1]
        small = input_features[0]

        if cfg.Architecture.COORD_CONV:
            large = self.coordconv(large)
            large = self.conv2d_k1(large, 1024)
            medium = self.coordconv(medium)
            medium = self.conv2d_k1(medium, 512)
            small = self.coordconv(small)
            small = self.conv2d_k1(small, 256)

        conv = self.upsample_concat(large, medium, 256)
        medium_bridge = self.conv_5x(conv, 256)

        conv = self.upsample_concat(medium_bridge, small, 128)
        conv_small = self.conv_5x(conv, 128)
        conv_result.append(conv_small)

        conv_common = self.downsample_concat(conv_small, medium_bridge, 256)
        conv_medium = self.conv_5x(conv_common, 256)
        conv_result.append(conv_medium)

        conv = self.downsample_concat(conv_medium, large, 512)
        conv_large = self.conv_5x(conv, 512)
        conv_result.append(conv_large)
        return conv_result

    def upsample_concat(self, upper, lower, channel):
        """
        :param upper: higher level feature
        :param lower: lower level feature, 2x resolution
        :param channel:
        :return: 2x resolution
        """
        conv_u = self.conv2d_k1(upper, channel)
        conv_u = tf.keras.layers.UpSampling2D()(conv_u)
        conv_l = self.conv2d_k1(lower, channel)
        conv = tf.concat([conv_l, conv_u], axis=-1)
        return conv

    def downsample_concat(self, lower, upper, channel):
        """
        :param lower: lower level feature
        :param upper: higher level feature, 0.5x resolution
        :param channel:
        :return: 0.5x resolution
        """
        conv = self.conv2d_s2(lower, channel)
        conv = tf.concat([conv, upper], axis=-1)
        return conv

    def sppblock(self, input_tensors):
        x = self.conv2d_k1(input_tensors, 512)
        x = self.conv2d(x, 1024)
        x = self.conv2d_k1(x, 512)
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        spp_output = tf.concat([x3, x2, x1, x], axis=-1)
        x = self.conv2d_k1(spp_output, 512)
        x = self.conv2d(x, 1024)
        output = self.conv2d_k1(x, 512)
        return output

    def sppfblock(self, input_tensors):
        x = self.conv2d_k1(input_tensors, 512)
        x = self.conv2d(x, 1024)
        x = self.conv2d_k1(x, 512)
        x1 = self.pool1(x)
        x2 = self.pool1(x1)
        x3 = self.pool1(x2)
        spp_output = tf.concat([x3, x2, x1, x], axis=-1)
        x = self.conv2d_k1(spp_output, 512)
        x = self.conv2d(x, 1024)
        output = self.conv2d_k1(x, 512)
        return output


class BiFPN(NeckBase):
    def __init__(self,  model_cfg, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels):
        super().__init__(model_cfg, num_anchors_per_scale, out_channels, num_lane_anchors, lane_out_channels)
        self.num_channels = cfg.Architecture.Efficientnet.Channels[0]
        self.d_bifpns = cfg.Architecture.Efficientnet.Channels[1]
        self.maxpool2d_p3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2))
        self.spconv2d = mu.CustomSeparableConv2D(kernel_size=3, strides=1,  **model_cfg)

    def __call__(self, input_features):
        conv_result = self.frist_block(input_features)
        for i in range(self.d_bifpns - 1):
            conv_result = self.blocks(conv_result)
        return conv_result

    def frist_block(self, features):
        out_features = list()
        P3_in = features[0]
        P4_in = features[1]
        P5_in = features[2]

        P6_in = self.conv2d_k1(P5_in, self.num_channels)
        P6_in = self.maxpool2d_p3(P6_in)
        P7_in = self.maxpool2d_p3(P6_in)

        P6_td = self.up_sampling(P7_in, P6_in)
        P6_td = self.spconv2d(P6_td, self.num_channels)

        P5_in_1 = self.conv2d_k1(P5_in, self.num_channels)
        P5_td = self.up_sampling(P6_td, P5_in_1)
        P5_td = self.spconv2d(P5_td, self.num_channels)

        P4_in_1 = self.conv2d_k1(P4_in, self.num_channels)
        P4_td = self.up_sampling(P5_td, P4_in_1)
        P4_td = self.spconv2d(P4_td, self.num_channels)

        P3_in = self.conv2d_k1(P3_in, self.num_channels)
        P3_out = self.up_sampling(P4_td, P3_in)
        P3_out = self.spconv2d(P3_out, self.num_channels)
        out_features.append(P3_out)

        P4_in_2 = self.conv2d_k1(P4_in, self.num_channels)
        P4_out = self.down_sampling(P3_out, P4_td, P4_in_2)
        P4_out = self.spconv2d(P4_out, self.num_channels)
        out_features.append(P4_out)

        P5_in_2 = self.conv2d_k1(P5_in, self.num_channels)
        P5_out = self.down_sampling(P4_out, P5_td, P5_in_2)
        P5_out = self.spconv2d(P5_out, self.num_channels)
        out_features.append(P5_out)

        P6_out = self.down_sampling(P5_out, P6_td, P6_in)
        P6_out = self.spconv2d(P6_out, self.num_channels)
        out_features.append(P6_out)

        P6_D = self.maxpool2d_p3(P6_out)
        P7_out = tf.keras.layers.Add()([P7_in, P6_D])
        P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = self.spconv2d(P7_out, self.num_channels)
        out_features.append(P7_out)
        return out_features

    def blocks(self, features):
        out_features = list()
        P3_in = features[0]
        P4_in = features[1]
        P5_in = features[2]
        P6_in = features[3]
        P7_in = features[4]
        P6_td = self.up_sampling(P7_in, P6_in)
        P6_td = self.spconv2d(P6_td, self.num_channels)

        P5_td = self.up_sampling(P6_td, P5_in)
        P5_td = self.spconv2d(P5_td, self.num_channels)

        P4_td = self.up_sampling(P5_td, P4_in)
        P4_td = self.spconv2d(P4_td, self.num_channels)

        P3_out = self.up_sampling(P4_td, P3_in)
        P3_out = self.spconv2d(P3_out, self.num_channels)
        out_features.append(P3_out)

        P4_out = self.down_sampling(P3_out, P4_td, P4_in)
        P4_out = self.spconv2d(P4_out, self.num_channels)
        out_features.append(P4_out)

        P5_out = self.down_sampling(P4_out, P5_td, P5_in)
        P5_out = self.spconv2d(P5_out, self.num_channels)
        out_features.append(P5_out)

        P6_out = self.down_sampling(P5_out, P6_td, P6_in)
        P6_out = self.spconv2d(P6_out, self.num_channels)
        out_features.append(P6_out)

        P6_D = self.maxpool2d_p3(P6_out)
        P7_out = tf.keras.layers.Add()([P7_in, P6_D])
        P7_out = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = self.spconv2d(P7_out, self.num_channels)
        out_features.append(P7_out)
        return out_features

    def up_sampling(self, upper, lower):
        conv_u = tf.keras.layers.UpSampling2D()(upper)
        conv = tf.keras.layers.Add()([lower, conv_u])
        conv = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(conv)
        return conv

    def down_sampling(self, lower, upper, upper_in):
        conv_l = self.maxpool2d_p3(lower)
        conv = tf.keras.layers.Add()([upper_in, upper, conv_l])
        conv = tf.keras.layers.Activation(lambda x: tf.nn.swish(x))(conv)
        return conv


# ==================================================
import config as cfg


def test_feature_decoder():
    print("===== start test_feature_decoder")
    anchors = {"anchor_l": [[10, 20], [30, 40], [50, 60]],
               "anchor_m": [[10, 20], [30, 40], [50, 60]],
               "anchor_s": [[10, 20], [30, 40], [50, 60]],
               }
    imshape = (128, 256, 3)
    decode_feature = FeatureDecoder(cfg.Model, anchors, imshape)
    # feature: (batch, grid_h, grid_w, anchor, channel(yxhw+obj+categories))
    feature = tf.zeros((4, 8, 16, 3, 9))
    decoded = decode_feature(feature, "feature_l")
    # batch=0, grid_y=2, grid_x=2, anchor=0 ([10,20])
    single_pred = decoded[0, 2, 2, 0]
    print("decoded feature in single box prediction", single_pred)
    # objectness, category probabilities: 0 -> 0.5
    assert np.isclose(single_pred[4:].numpy(), 0.5).all()
    # check y, x, h, w
    assert single_pred[0] == (2 + 0.5) / 8.
    assert single_pred[1] == (2 + 0.5) / 16.
    assert single_pred[2] == 10. / 128
    assert single_pred[3] == 20. / 256
    print("!!! test_feature_decoder passed !!!")


if __name__ == "__main__":
    test_feature_decoder()
