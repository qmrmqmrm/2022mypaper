import numpy as np

import config as cfg

def create_batch_featmap(features_feat, featmap):
    if "feat" not in features_feat.keys():
        features_feat["feat"] = []
    for scale, value in enumerate(featmap):
        value = value[np.newaxis, ...]
        if len(features_feat["feat"]) < len(cfg.ModelOutput.FEATURE_SCALES):
            features_feat["feat"].append(value)
        else:
            features_feat["feat"][scale] = np.concatenate([features_feat["feat"][scale], value], axis=0)
    return features_feat["feat"]


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