import torch
from torch import nn
import torch.nn.functional as F

import model.framework.model_util as mu
from utils.util_class import MyExceptionToCatch
import config as cfg


def backbone_factory(backbone, conv_kwargs, training):
    if backbone == "Resnet":
        return ResNet(conv_kwargs, training)

    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {backbone}")


class ResNet(nn.Module):
    def __init__(self, conv_kwargs, training):
        super(ResNet, self).__init__()
        self.stem = mu.CustomConv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            **conv_kwargs)

        # self.bottle = BottleneckBlock()
        self.block_nums = cfg.Architecture.Resnet.LAYER[1]
        self.chennels = cfg.Architecture.Resnet.CHENNELS
        self.training = training
        self.layers0, out_channel0 = self.make_layer(self.block_nums[0], chennels=[64, 256], stride=1,
                                                     conv_kwargs=conv_kwargs)
        self.layers1, out_channel1 = self.make_layer(self.block_nums[1], chennels=[256, 512], stride=2,
                                                     conv_kwargs=conv_kwargs)
        self.layers2, out_channel2 = self.make_layer(self.block_nums[2], chennels=[512, 1024], stride=2,
                                                     conv_kwargs=conv_kwargs)
        self.layers3, out_channel3 = self.make_layer(self.block_nums[3], chennels=[1024, 2048], stride=2,
                                                     conv_kwargs=conv_kwargs)

        self.out_channels = [out_channel1, out_channel2, out_channel3]

    def forward(self, x):
        x = self.stem(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        feat_0 = self.layers0(x)
        feat_1 = self.layers1(feat_0)
        feat_2 = self.layers2(feat_1)
        feat_3 = self.layers3(feat_2)
        feature = {'feature_5': feat_3, 'feature_4': feat_2, 'feature_3': feat_1}

        return feature

    def make_layer(self, block_num, chennels, stride, conv_kwargs):
        layers = list()
        # for i in range(1):
        layers.append(BottleneckBlock(in_channels=chennels[0],
                                      bottleneck_channels=int(chennels[0] / stride),
                                      out_channels=chennels[1], stride=stride,
                                      shortcut=True, **conv_kwargs))
        for num in range(1, block_num):
            layers.append(BottleneckBlock(in_channels=chennels[1],
                                          bottleneck_channels=int(chennels[0] / stride),
                                          out_channels=chennels[1], stride=1,
                                          shortcut=False, **conv_kwargs))
        out_channel = layers[-1].out_channel
        return nn.Sequential(*layers), out_channel


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride, shortcut, **conv_kwargs):
        super(BottleneckBlock, self).__init__()
        self.shortcut = mu.CustomConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            activation=False
        ) if shortcut else None

        self.conv1 = mu.CustomConv2D(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            **conv_kwargs
        )

        self.conv2 = mu.CustomConv2D(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            **conv_kwargs
        )

        self.conv3 = mu.CustomConv2D(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=False
        )
        self.out_channel = self.conv3.out_channels

    def forward(self, x):
        cut = x
        if self.shortcut:
            cut = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + cut
        out = F.relu(out)
        return out
