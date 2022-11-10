import os
import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from matplotlib import pyplot as plt

import settings
import config_dir.config_generator as cg
import config as cfg
from dataloader.framework.dataset_reader import DatasetReader
import utils.framework.util_function as uf
from train.feature_generator import FeatureMapDistributer


class AnchorParams:
    """
    evaluate performance for each param combination
    -> total_eval_result.csv
    """

    def __init__(self):
        self.dataset_name = cfg.Datasets.TARGET_DATASET
        self.ckpt_path = op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.num_ctgr = len(cfg.Dataloader.CATEGORY_NAMES["lane"])
        self.num_anchors = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE * len(cfg.ModelOutput.FEATURE_SCALES)
        self.num_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["category"]
        self.num_anchors_per_scale = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        self.num_scales = len(cfg.ModelOutput.FEATURE_SCALES)
        self.ctgr_anchor_data = pd.DataFrame()

    def create_eval_file(self):
        dataset, steps, anchors_per_scale, feature_creator = self.load_dataset_model(self.dataset_name)
        self.collect_recall_precision(dataset, steps, self.num_ctgr, anchors_per_scale, feature_creator)

    def load_dataset_model(self, dataset_name):
        batch_size, train_mode, anchors = cfg.Train.BATCH_SIZE, cfg.Train.MODE, cfg.AnchorGeneration.ANCHORS
        tfrd_path = cfg.Paths.DATAPATH
        dataset, steps, imshape, anchors_per_scale \
            = self.get_dataset(tfrd_path, dataset_name, False, batch_size, "val", anchors)
        feature_creator = FeatureMapDistributer(cfg.FeatureDistribPolicy.POLICY_NAME, imshape, anchors_per_scale)
        return dataset, steps, anchors_per_scale, feature_creator

    def get_dataset(self, tfrd_path, dataset_name, shuffle, batch_size, split, anchors):
        tfrpath = op.join(tfrd_path, f"{dataset_name}_{split}")
        reader = DatasetReader(tfrpath, shuffle, batch_size, 1)
        dataset = reader.get_dataset()
        frames = reader.get_total_frames()
        tfr_cfg = reader.get_dataset_config()
        image_shape = tfr_cfg["image"]["shape"]
        # anchor sizes per scale in pixel
        anchors_per_scale = np.array([anchor / np.array([image_shape[:2]]) for anchor in anchors], dtype=np.float32)
        print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, "
              f"frames={frames},\n\tanchors={anchors_per_scale}")
        return dataset, frames // batch_size, image_shape, anchors_per_scale

    def collect_recall_precision(self, dataset, steps, num_ctgr, anchors_per_scale, feature_creator):
        for step, grtr in enumerate(dataset):
            start = timer()
            features = feature_creator(grtr)
            grtr_box_map = self.split_anchor(features["feat_box"], False)

            ctgr_anchor_box_num = self.ctgr_anchor_num_box(grtr_box_map, range(1, self.num_categs),
                                                           range(0, self.num_anchors), step)
            self.ctgr_anchor_data = self.ctgr_anchor_data.append(ctgr_anchor_box_num, ignore_index=True)
            uf.print_progress(f"=== step: {step}/{steps}, took {timer() - start:1.2f}s")
        sum_data = self.ctgr_anchor_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_SUM]
        sum_ctgr_anchor_data = sum_data.groupby(["anchor", "ctgr"], as_index=False).sum()
        sum_ctgr_anchor_data["trpo"] = -1

        sum_anchor_data = sum_data.groupby(["anchor"], as_index=False).sum()
        sum_anchor_data["ctgr"] = -1
        sum_anchor_data["trpo"] = -1

        sum_summary = pd.concat([sum_anchor_data, sum_ctgr_anchor_data], join='outer', ignore_index=True)
        sum_summary.to_csv("anchor_check_.csv",
                          encoding='utf-8', index=False, float_format='%.4f')

    def split_anchor(self, features, is_loss):
        if is_loss:
            new_features = {key[:-4]: [] for key in features.keys() if "map" in key}
        else:
            new_features = {key: [] for key in features.keys() if key is not "whole"}

        for key, feature in features.items():
            if key == "whole" or isinstance(feature, np.float32):
                continue
            for scale in range(self.num_scales):
                if is_loss:
                    batch, hwa = feature[scale].shape
                    new_features[key[:-4]].append(np.reshape(feature[scale], (batch, hwa//self.num_anchors_per_scale, self.num_anchors_per_scale)))
                else:
                    batch, hwa, channel = feature[scale].shape
                    new_features[key].append(np.reshape(feature[scale], (batch, hwa//self.num_anchors_per_scale, self.num_anchors_per_scale, channel)))
        return new_features

    def ctgr_anchor_num_box(self, grtr_map, categories, anchors, step):
        metric_data = []
        for anchor in anchors:
            scale = anchor // 3
            grtr_anchor_mask = self.create_scale_mask(grtr_map, anchor, scale, "anchor_ind")
            for category in categories:
                grtr_catgr_mask = self.create_scale_mask(grtr_map, category, scale, "category")
                grtr_match = self.box_scale_matcher(grtr_map, grtr_catgr_mask * grtr_anchor_mask, scale)
                grtr_num_box = np.sum(grtr_match["object"] > 0)
                metric_data.append(
                    {"step": step, "anchor": anchor, "ctgr": category, "grtr": grtr_num_box, "pred": 0, "trpo": 0})
        return metric_data

    def box_scale_matcher(self, bbox, mask, scale):
        match_bbox = dict()
        for key in bbox.keys():
            match_bbox[key] = bbox[key][scale] * mask
        return match_bbox

    def create_scale_mask(self, data, index, scale, key):
        if key:
            valid_mask = data[key][scale] == index
        else:
            valid_mask = data == index
        return valid_mask


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    eval_param = AnchorParams()
    eval_param.create_eval_file()
