import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import *

from utils.util_class import MyExceptionToCatch
import model.framework.model_util as mu
from model.framework.drop_block import DropBlock2D
from model.framework.deformable_conv_v2.deformable_layer import DeformableConv2D
import config as cfg


def backbone_factory(backbone, conv_kwargs, training):
    if backbone == "Darknet53":
        return Darknet53(conv_kwargs, training)
    elif backbone == "CSPDarknet53":
        return CSPDarkNet53(conv_kwargs, training)
    elif backbone == "Resnet":
        return ResNet(conv_kwargs, training)
    elif backbone == "Resnet_vd":
        return ResNetVD(conv_kwargs, training)
    elif backbone == "Efficientnet":
        return Efficientnet(conv_kwargs, training)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {backbone}")


class BackboneBase:
    def __init__(self, conv_kwargs):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **conv_kwargs)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **conv_kwargs)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **conv_kwargs)

    def residual(self, x, filters):
        short_cut = x
        conv = self.conv2d_k1(x, filters // 2)
        conv = self.conv2d(conv, filters)
        return short_cut + conv


class Darknet53(BackboneBase):
    def __init__(self, conv_kwargs, training):
        super().__init__(conv_kwargs)
        self.training = training

    def __call__(self, input_tensor):
        """
        conv'n' represents a feature map of which resolution is (input resolution / 2^n)
        e.g. input_tensor.shape[:2] == conv0.shape[:2], conv0.shape[:2]/8 == conv3.shape[:2]
        """
        features = list()
        conv0 = self.conv2d(input_tensor, 32)
        conv1 = self.conv2d_s2(conv0, 64)
        conv1 = self.residual(conv1, 64)

        conv2 = self.conv2d_s2(conv1, 128)
        for i in range(2):
            conv2 = self.residual(conv2, 128)

        conv3 = self.conv2d_s2(conv2, 256)
        for i in range(8):
            conv3 = self.residual(conv3, 256)
        # feature small
        features.append(conv3)

        conv4 = self.conv2d_s2(conv3, 512)
        for i in range(8):
            conv4 = self.residual(conv4, 512)
        # feature medium
        features.append(conv4)

        conv5 = self.conv2d_s2(conv4, 1024)
        for i in range(4):
            conv5 = self.residual(conv5, 1024)
        # feature large
        features.append(conv5)

        return features


class CSPDarkNet53(BackboneBase):
    def __init__(self, conv_kwargs, training):
        super().__init__(conv_kwargs)
        self.training = training

    def __call__(self, input_tensor):
        features = list()
        conv = self.conv2d(input_tensor, 32)
        # conv = self.conv2d(input_tensor["image"], 32)
        conv = self.csp_block(conv, 64, 1, allow_narrow=False, training=self.training)
        conv = self.csp_block(conv, 128, 2, training=self.training)
        route_small = self.csp_block(conv, 256, 8, training=self.training)
        features.append(route_small)
        route_medium = self.csp_block(route_small, 512, 8, training=self.training)
        features.append(route_medium)
        route_large = self.csp_block(route_medium, 1024, 4, training=self.training)
        features.append(route_large)
        return features

    def csp_block(self, inputs, filters, num_blocks, allow_narrow=True, training=True):
        """
        Create a CSPBlock which applies the following scheme to the input (N, H, W, C):
            - the first part (N, H, W, C // 2) goes into a series of residual connection
            - the second part is directly concatenated to the output of the previous operation
        Args:
            inputs (tf.Tensor): 4D (N,H,W,C) input tensor
            filters (int): Number of filters to use
            num_blocks (int): Number of residual blocks to apply
        Returns:
            tf.Tensor: 4D (N,H/2,W/2,filters) output tensor
        """
        half_filters = filters // 2 if allow_narrow else filters
        x = self.conv2d_s2(inputs, filters=filters)
        shortcut = self.conv2d_k1(x, filters=half_filters)
        mainstream = self.conv2d_k1(x, filters=half_filters)
        for i in range(num_blocks):
            mainstream = self.csp_res(mainstream, half_filters, training, allow_narrow)
        mainstream = self.conv2d_k1(mainstream, half_filters)
        x = tf.concat([mainstream, shortcut], axis=-1)
        x = self.conv2d_k1(x, filters)
        return x

    def csp_res(self, inputs, filters, training, first):
        dropblock = DropBlock2D(keep_prob=0.9, block_size=3)
        half_filters = filters if first else filters // 2
        x = self.conv2d_k1(inputs, half_filters)
        x = self.conv2d(x, filters)
        x = dropblock(x, training=training)
        residual_out = inputs + x
        return residual_out


class ResNet(BackboneBase):
    def __init__(self, conv_kwargs, training):
        super().__init__(conv_kwargs)
        self.block_nums = cfg.Architecture.Resnet.LAYER[1]
        self.chennels = cfg.Architecture.Resnet.CHENNELS
        self.conv2d_k7s2 = mu.CustomConv2D(kernel_size=7, strides=2, padding="same", **conv_kwargs)
        self.maxpool2d_p3 = mu.CustomMax2D(pool_size=3, strides=2, padding="same", scope="back")
        self.conv2d_k1s2 = mu.CustomConv2D(kernel_size=1, strides=2, **conv_kwargs)
        self.conv2d_k1s2na = mu.CustomConv2D(kernel_size=1, strides=2, activation=False)
        self.conv2d_k1na = mu.CustomConv2D(kernel_size=1, activation=False)
        self.conv2d_k3s2na = mu.CustomConv2D(kernel_size=3, strides=2, activation=False)
        self.dcn = mu.CustomDeformConv2D(scope="back")
        self.act = layers.ReLU()

    def __call__(self, input_tensor):
        conv = self.stem(input_tensor)
        res_conv = conv
        features = list()
        layer_features = list()

        for layer_num, block_num in enumerate(self.block_nums):
            filters = [self.chennels[layer_num], self.chennels[layer_num + 2]]
            first_layer = True if layer_num == 0 else False
            dcn = True if layer_num == 3 else False
            for num in range(block_num):
                first_conv = True if num == 0 else False
                res_conv = self.bottleneck(res_conv, filters, dcn=dcn, first_conv=first_conv, first_layer=first_layer)
            layer_features.append(res_conv)

        features.append(layer_features[1])
        features.append(layer_features[2])
        features.append(layer_features[3])
        return features

    def stem(self, inputs):
        x = self.conv2d_k7s2(inputs, 64)
        x = self.maxpool2d_p3(x)
        return x

    def bottleneck(self, inputs, filters, dcn=False, first_conv=None, first_layer=False):
        filter1, filter2 = filters

        if not first_layer and first_conv:
            x = self.conv2d_k1s2(inputs, filter1)
            shortcut = self.conv2d_k1s2na(inputs, filter2)

        elif first_layer and first_conv:
            x = self.conv2d_k1(inputs, filter1)
            shortcut = self.conv2d_k1na(inputs, filter2)

        else:
            x = self.conv2d_k1(inputs, filter1)
            shortcut = inputs

        if dcn:
            x = self.dcn(x, 512)
        else:
            x = self.conv2d(x, filter1)
        x = self.conv2d_k1na(x, filter2)

        x = x + shortcut
        act = layers.ReLU()
        x = act(x)
        return x


class ResNetVD(ResNet):
    def __init__(self, conv_kwargs, training):
        super().__init__(conv_kwargs, training)
        self.avg_pool_k2s2 = layers.AvgPool2D(strides=2, padding='same')

    def __call__(self, input_tensor):
        conv = self.stem(input_tensor)
        res_conv = conv
        features = list()
        layer_features = list()

        for layer_num, block_num in enumerate(self.block_nums):
            filters = [self.chennels[layer_num], self.chennels[layer_num + 2]]
            first_layer = True if layer_num == 0 else False
            dcn = True if layer_num == 3 else False
            for num in range(block_num):
                first_conv = True if num == 0 else False
                res_conv = self.bottleneck(res_conv, filters, dcn=dcn, first_conv=first_conv, first_layer=first_layer)
            layer_features.append(res_conv)

        # feature small, medium, large
        features.append(layer_features[1])
        features.append(layer_features[2])
        features.append(layer_features[3])
        return features

    def stem(self, inputs):
        x = self.conv2d_s2(inputs, 64)
        x = self.conv2d(x, 64)
        x = self.conv2d(x, 64)
        x = self.maxpool2d_p3(x)
        return x

    def bottleneck(self, inputs, filters, dcn=False, first_conv=None, first_layer=False):
        filter1, filter2 = filters
        if not first_layer and first_conv:
            x = self.conv2d_k1na(inputs, filter1)
            x = self.conv2d_k3s2na(x, filter1)
            shortcut = self.avg_pool_k2s2(inputs)
            shortcut = self.conv2d_k1na(shortcut, filter2)
        elif first_layer and first_conv:
            x = self.conv2d_k1(inputs, filter1)
            shortcut = self.conv2d_k1na(inputs, filter2)
        else:
            x = self.conv2d_k1(inputs, filter1)
            shortcut = inputs

        if dcn:
            x = self.dcn(x, 512)
        else:
            x = self.conv2d(x, filter1)
        x = self.conv2d_k1na(x, filter2)

        x = x + shortcut
        act = layers.ReLU()
        x = act(x)
        return x


class Efficientnet(BackboneBase):
    def __init__(self, conv_kwargs, training):
        super().__init__(conv_kwargs)
        self.training = training
        self.architecture = eval(cfg.Architecture.Efficientnet.NAME)

    def __call__(self, input_tensor):
        features = list()
        efficient = self.architecture(False, weights=None, input_tensor=input_tensor, )
        feature_s = efficient.get_layer("block4a_expand_conv").input
        features.append(feature_s)
        feature_m = efficient.get_layer("block6a_expand_conv").input
        features.append(feature_m)
        feature_l = efficient.get_layer("top_conv").input
        features.append(feature_l)
        return features
