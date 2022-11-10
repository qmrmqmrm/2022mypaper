import tensorflow as tf

from model.framework.model_util import PriorProbability
from utils.util_class import MyExceptionToCatch
import config as cfg
import model.framework.model_util as mu
import utils.framework.util_function as uf


def head_factory(output_name, conv_args, training, num_anchors_per_scale, head_composition, num_lane_anchors,
                 lane_out_channels):
    if output_name == "Double":
        return DoubleOutput(conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels)
    elif output_name == "Single":
        return SingleOutput(conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels)
    elif output_name == "Efficient":
        return EfficientOutput(conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels)

    else:
        raise MyExceptionToCatch(f"[head_factory[ invalid output name : {output_name}")


class HeadBase:
    def __init__(self, conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **conv_args)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **conv_args)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **conv_args)
        self.conv2d_k1na = mu.CustomConv2D(kernel_size=1, strides=1, activation=False, bn=False, scope="head")
        self.conv2d_output = mu.CustomConv2D(kernel_size=1, strides=1, activation=False, scope="output", bn=False)
        self.num_anchors_per_scale = num_anchors_per_scale
        self.head_composition = head_composition
        self.num_lane_anchors = num_lane_anchors if num_lane_anchors is not None else None
        self.lane_out_channels = lane_out_channels if lane_out_channels is not None else None

    def make_lane_output(self, x, channel):
        batch_mesh = x.shape[0]
        height_mesh, width_mesh = tf.cast(x.shape[1], dtype=tf.float32), tf.cast(x.shape[2], dtype=tf.float32)
        feat_size = tf.cast([height_mesh // 2, width_mesh // 2], dtype=tf.float32)
        spatial_layer = (uf.get_meshgrid(height_mesh, width_mesh) - feat_size) / feat_size

        spatial_layer = tf.expand_dims(spatial_layer, axis=0)
        x = tf.concat([x, tf.repeat(spatial_layer, batch_mesh, axis=0)], axis=-1)
        x = self.conv2d(x, channel)
        x = self.conv2d_output(x, self.lane_out_channels * self.num_lane_anchors)
        batch, height, width, _ = x.shape
        x_5d = tf.reshape(x, (batch, height, width, self.num_lane_anchors,  self.lane_out_channels))
        return x_5d


class SingleOutput(HeadBase):
    def __init__(self, conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels):
        super().__init__(conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels)
        self.out_channels = sum(head_composition.values())

    def __call__(self, input_features):
        output_features = []
        for feature, channel in zip(input_features, [256, 512, 1024]):
            output_features.append(self.make_output(feature, channel))

        if cfg.ModelOutput.LANE_DET:
            conv_lane = self.make_lane_output(input_features[1], 512)
            output_features.append(conv_lane)
        return output_features

    def make_output(self, x, channel):
        x = self.conv2d(x, channel)
        x = self.conv2d_output(x, self.num_anchors_per_scale * self.out_channels)
        batch, height, width, channel = x.shape
        x_5d = tf.reshape(x, (batch, height, width, self.num_anchors_per_scale, self.out_channels))
        return x_5d


class DoubleOutput(HeadBase):
    def __init__(self, conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels):
        super().__init__(conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels)

    def __call__(self, input_features):
        output_features = list()
        for feature in input_features:
            conv_common = self.conv2d_k1(feature, 256)
            features = []
            for key, channel in self.head_composition.items():
                conv_out = self.conv2d(conv_common, 256)
                conv_out = self.conv2d(conv_out, 256)
                feat = self.conv2d_k1na(conv_out, channel * self.num_anchors_per_scale)
                features.append(feat)
            b, h, w, c = features[0].shape
            output_feature = tf.concat(features, axis=-1)
            output_features.append(tf.reshape(output_feature, (b, h, w, self.num_anchors_per_scale, -1)))

        if cfg.ModelOutput.LANE_DET:
            conv_lane = self.make_lane_output(input_features[1], 512)
            output_features.append(conv_lane)
        return output_features


class EfficientOutput(HeadBase):
    def __init__(self, conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels,
                 ):
        super().__init__(conv_args, num_anchors_per_scale, head_composition, num_lane_anchors, lane_out_channels)
        separable_conv = cfg.Architecture.Efficientnet.Separable
        if separable_conv:
            kernel_initializer = {"depthwise_initializer": tf.keras.initializers.VarianceScaling(),
                                  "pointwise_initializer": tf.keras.initializers.VarianceScaling()}
            conv_args.update(kernel_initializer)
            self.conv2d = mu.CustomSeparableConv2D(kernel_size=3, strides=1, **conv_args)
            conv_args.update({"activation": False})
            self.conv2d_boxout = mu.CustomSeparableConv2D(kernel_size=3, strides=1, bn=False, **conv_args)
            self.conv2d_clsout = mu.CustomSeparableConv2D(kernel_size=3, strides=1, bn=False,
                                                          bias_initializer=PriorProbability(probability=0.01),
                                                          **conv_args)
        else:
            kernel_initializer = {
                'kernel_initializer': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                "activation": False

            }
            conv_args.update(kernel_initializer)
            self.conv2d_boxout = mu.CustomConv2D(kernel_size=3, strides=1, bn=False, **conv_args)
            self.conv2d_clsout = mu.CustomConv2D(kernel_size=3, strides=1, bn=False,
                                                 bias_initializer=PriorProbability(probability=0.01),
                                                 **conv_args)
        self.num_channels = cfg.Architecture.Efficientnet.Channels[0]
        self.d_head = cfg.Architecture.Efficientnet.Channels[2]
        self.separable_conv = separable_conv

    def __call__(self, input_features):
        output_features = list()
        for feature in input_features:
            box_feat = self.boxnet(feature)
            cls_feat = self.clsnet(feature)
            b, h, w, _ = cls_feat.shape
            concat_feature = tf.concat([cls_feat, box_feat], axis=-1)
            output_features.append(tf.reshape(concat_feature, (b, h, w, self.num_anchors_per_scale, -1)))
        if cfg.ModelOutput.LANE_DET:
            conv_lane = self.make_lane_output(input_features[1], 512)
            output_features.append(conv_lane)
        return output_features

    def boxnet(self, feature):
        conv = feature
        for i in range(self.d_head):
            conv = self.conv2d(conv, self.num_channels)
        conv = self.conv2d_boxout(conv, self.num_anchors_per_scale * self.head_composition['reg'])
        return conv

    def clsnet(self, feature):
        conv = feature
        for i in range(self.d_head):
            conv = self.conv2d(conv, self.num_channels)
        conv = self.conv2d_clsout(conv, self.num_anchors_per_scale * self.head_composition['cls'])
        return conv