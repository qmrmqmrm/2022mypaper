import torch
from torch import nn
import torch.nn.functional as F

from utils.util_class import MyExceptionToCatch
import model.framework.model_util as mu
import config as cfg


def neck_factory(head, conv_args, out_channels):
    if head == "FPN":
        return FPN(conv_args, out_channels)
    elif head == "PAN":
        return PAN(conv_args, out_channels)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {head}")


class NeckBase(nn.Module):
    def __init__(self):
        super(NeckBase, self).__init__()
        pass

    def upsample_block(self, in_channels, out_channels, conv_kwargs):
        up_sample = nn.Sequential(
            *[
                mu.CustomConv2D(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                **conv_kwargs),
                nn.Upsample(scale_factor=2),

            ]

        )
        return up_sample

    def conv5x(self, in_channels, out_channels, conv_kwargs):
        conv = nn.Sequential(
            *[
                mu.CustomConv2D(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                **conv_kwargs),
                mu.CustomConv2D(in_channels=out_channels,
                                out_channels=out_channels * 2,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                **conv_kwargs),
                mu.CustomConv2D(in_channels=out_channels * 2,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                **conv_kwargs),
                mu.CustomConv2D(in_channels=out_channels,
                                out_channels=out_channels * 2,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                **conv_kwargs),
                mu.CustomConv2D(in_channels=out_channels * 2,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                **conv_kwargs),
            ]
        )
        return conv, out_channels


class FPN(nn.Module):
    def __init__(self, conv_kwargs, in_channels):
        super(FPN, self).__init__()
        print("FPN", in_channels)

    def forward(self, features):
        feature_3 = features["feature_3"]
        feature_4 = features["feature_4"]
        feature_5 = features["feature_5"]
        return features


class PAN(NeckBase):
    def __init__(self, conv_kwargs, in_channels):
        super(PAN, self).__init__()
        print(in_channels)
        self.large_upsample = self.upsample_block(in_channels[2], 256, conv_kwargs)
        self.medium_conv = mu.CustomConv2D(in_channels=in_channels[1],
                                           out_channels=256,
                                           kernel_size=1,
                                           stride=1,
                                           **conv_kwargs)
        self.med_b5x, _ = self.conv5x(512, 256, conv_kwargs)

        self.medium_upsample = self.upsample_block(256, 128, conv_kwargs)
        self.small_conv = mu.CustomConv2D(in_channels=in_channels[0],
                                          out_channels=128,
                                          kernel_size=1,
                                          stride=1,
                                          **conv_kwargs)

        self.sml_5x, out_channel_s = self.conv5x(256, 128, conv_kwargs)

        self.small_down = mu.CustomConv2D(in_channels=128,
                                          out_channels=256,
                                          kernel_size=1,
                                          stride=2,
                                          padding=0,
                                          **conv_kwargs)

        self.med_5x, out_channel_m = self.conv5x(512, 256, conv_kwargs)

        self.medium_down = mu.CustomConv2D(in_channels=256,
                                           out_channels=512,
                                           kernel_size=1,
                                           stride=2,
                                           padding=0,
                                           **conv_kwargs)

        self.lag_5x, out_channel_l = self.conv5x(2560, 512, conv_kwargs)
        self.out_channels = {"feature_3": out_channel_s, "feature_4":out_channel_m, "feature_5":out_channel_l}
    def forward(self, features):
        print("pan forward")
        feature_3 = features["feature_3"]
        feature_4 = features["feature_4"]
        feature_5 = features["feature_5"]

        large_up = self.large_upsample(feature_5)
        medium_feat = self.medium_conv(feature_4)
        conv_lm_up = torch.concat([medium_feat, large_up], dim=1)
        medium_bridge = self.med_b5x(conv_lm_up)

        medium_up = self.medium_upsample(medium_bridge)
        small_feat = self.small_conv(feature_3)
        conv_ms_up = torch.concat([small_feat, medium_up], dim=1)
        conv_small = self.sml_5x(conv_ms_up)

        small_down = self.small_down(conv_small)
        conv_sm_down = torch.concat([small_down, medium_bridge], dim=1)
        conv_medium = self.med_5x(conv_sm_down)

        medium_down = self.medium_down(conv_medium)
        conv_ml_down = torch.concat([medium_down, feature_5], dim=1)
        conv_large = self.lag_5x(conv_ml_down)

        conv_result = {"feature_3": conv_small, "feature_4": conv_medium, "feature_5": conv_large}
        return conv_result
