import numpy as np
import tensorflow as tf

import config as cfg


def do_nothing(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


mode_decor = None
if cfg.Train.MODE in ["graph", "distribute"]:
    mode_decor = tf.function
else:
    mode_decor = do_nothing


def gt_feat_rename(features):
    new_feat = {"inst": {}, "feat": []}
    for key, val in features.items():
        if "feat_lane" in key:
            new_feat["feat_lane"] = val
        elif "feat" in key:
            new_feat["feat"].extend(val)
        elif "image" in key:
            new_feat[key] = val
        else:
            new_feat["inst"][key] = val
    return new_feat


def create_batch_featmap(features_feat, featmap, key):
    if f"{key}" not in features_feat.keys():
        features_feat[f"{key}"] = []
    for scale, value in enumerate(featmap):
        value = value[np.newaxis, ...]
        if len(features_feat[f"{key}"]) < len(featmap):
            features_feat[f"{key}"].append(value)
        else:
            features_feat[f"{key}"][scale] = np.concatenate([features_feat[f"{key}"][scale], value], axis=0)
    return features_feat[f"{key}"]


def load_weights(model, ckpt_file):
    model.load_weights(ckpt_file)
    return model
