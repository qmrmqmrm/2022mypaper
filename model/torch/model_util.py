import torch

from torch import nn
from torch.nn import functional as F

from utils.util_class import MyExceptionToCatch


class CustomConv2D(nn.Module):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, *args, **kwargs):

        """
        in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
        bias=True, padding_mode='zeros', device=None, dtype=None
        """
        super().__init__()
        act_name = kwargs.pop("activation", "relu")
        bn_name = kwargs.pop("bn", True)
        scope = kwargs.pop("scope", None)
        self.conv = nn.Conv2d(*args, **kwargs)
        self.bn = nn.BatchNorm2d(self.conv.out_channels) if bn_name else None
        self.activation = select_activation(act_name)
        self.out_channels = self.conv.out_channels

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = self.conv(x)

        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


def select_activation(activation):
    if activation == "leaky_relu":
        act = nn.LeakyReLU()
    elif activation == "mish":
        act = nn.Mish()
    elif activation == "relu":
        act = nn.ReLU()
    elif activation == "swish":
        act = nn.Hardswish()
    elif activation is False:
        act = None
    else:
        raise MyExceptionToCatch(f"[CustomConv2D] invalid activation name: {activation}")
    return act


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
