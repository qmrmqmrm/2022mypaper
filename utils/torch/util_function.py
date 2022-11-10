import sys
import numpy as np
import torch

import config as cfg
import config_dir.util_config as uc


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def slice_feature(feature, channel_composition):
    """
    :param feature: (batch, grid_h, grid_w, anchors, dims)
    :param channel_composition:
    :return: sliced feature maps
    """
    names, channels = list(channel_composition.keys()), list(channel_composition.values())
    slices = torch.split(feature, channels, dim=1)
    slices = dict(zip(names, slices))  # slices = {'yxhw': (B,H,W,A,4), 'object': (B,H,W,A,1), ...}
    return slices


def print_structure(title, data, key=""):
    if isinstance(data, list):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    elif isinstance(data, dict):
        for subkey, datum in data.items():
            print_structure(title, datum, f"{key}/{subkey}")
    elif isinstance(data, str):
        print(title, key, data)
    elif isinstance(data, tuple):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    elif data is None:
        print(f'{title} : None')
    elif isinstance(data, int):
        print(title, key, data)
    elif type(data) == np.ndarray:
        print(title, key, data.shape, type(data))
    else:
        print(title, key, data.shape)
