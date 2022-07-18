import numpy as np
import pandas as pd
from timeit import default_timer as timer

import config as cfg
from log.metric import count_true_positives
from log.logger_pool import LogMeanDetailLoss, LogPositiveDetailObj, LogNegativeDetailObj, LogTrueClass, \
    LogFalseClass, LogIouMean, LogBoxYX, LogBoxHW


class ExhaustiveLog:
    def __init__(self):
        self.columns = cfg.Log.LOSS_NAME + cfg.Log.ExhaustiveLog.DETAIL
        self.loggers = self.create_loggers(self.columns)
        self.num_anchors = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE * len(cfg.ModelOutput.FEATURE_SCALES)
        self.num_categs = cfg.ModelOutput.PRED_MAIN_COMPOSITION["category"]
        self.data = pd.DataFrame()
        self.metric_data = pd.DataFrame()
        self.ctgr_anchor_data = pd.DataFrame()
        self.summary = dict()

    def __call__(self, step, grtr, pred, loss, total_loss):
        result = {"step": [step] * self.num_anchors * self.num_categs,
                  "anchor": [i for i in range(self.num_anchors) for k in range(self.num_categs)],
                  "ctgr": list(range(self.num_categs)) * self.num_anchors,
                  }
        grtr_map = self.extract_feature_map(grtr["feat"])
        pred_map = self.extract_feature_map(pred["feat"])
        loss_map = self.extract_feature_map(loss, True)
        for key, log_object in self.loggers.items():
            result[key] = (log_object(grtr_map, pred_map, loss_map))
        metric = self.box_category_match(grtr["inst"], pred["inst"]["bboxes"], range(1, self.num_categs), step)
        self.metric_data = self.metric_data.append(metric, ignore_index=True)
        ctgr_anchor_box_num = self.ctgr_anchor_num_box(grtr_map, pred["inst"]["bboxes"], range(1, self.num_categs), range(0, self.num_anchors), step)
        self.ctgr_anchor_data = self.ctgr_anchor_data.append(ctgr_anchor_box_num, ignore_index=True)
        data = pd.DataFrame(result)
        self.data = self.data.append(data, ignore_index=True)

    def extract_feature_map(self, features, is_loss=False):
        features = self.merge_scale(features, is_loss)
        features = self.split_anchor(features, is_loss)
        return features

    def gather_feature_maps_to_level1_dict(self, features, target_key):
        out_feats = dict()
        for feat_list in features:
            for subkey, feat_map in feat_list.items():
                out_feats[subkey] = feat_map
        return out_feats

    def merge_scale(self, features, is_loss):
        out_features = dict()
        if is_loss:
            slice_keys = list(key for key in features.keys() if "map" in key)  # ['ciou_map', 'object_map', 'category_map']
            for key in slice_keys:
                # scaled_preds = [features[key][scale_name] for scale_name in range(len(features[key]))]
                scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
                out_features[key] = scaled_preds
        else:
            slice_keys = list(features.keys())  # ['yxhw', 'object', 'category']
            for key in slice_keys:
                # list of (batch, HWA in scale, dim)
                # scaled_preds = [features[scale_name][key] for scale_name in range(len(features))]
                scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
                out_features[key] = scaled_preds
        return out_features

    def split_anchor(self, features, is_loss):
        new_features = dict()
        for key, feature in features.items():
            if is_loss:
                batch, hwa = feature.shape
                new_features[key[:-4]] = np.reshape(feature, (batch, hwa // self.num_anchors, self.num_anchors))
            else:
                batch, hwa, channel = feature.shape
                new_features[key] = np.reshape(feature, (batch, hwa // self.num_anchors, self.num_anchors, channel))
        return new_features

    def create_loggers(self, columns):
        loggers = dict()
        if "ciou" in columns:
            loggers["ciou"] = LogMeanDetailLoss("ciou")
        if "iou_aware" in columns:
            loggers["iou_aware"] = LogMeanDetailLoss("iou_aware")
        if "object" in columns:
            loggers["object"] = LogMeanDetailLoss("object")
        if "category" in columns:
            loggers["category"] = LogMeanDetailLoss("category")
        if "sign_ctgr" in columns:
            loggers["sign_ctgr"] = LogMeanDetailLoss("sign_ctgr")
        if "mark_ctgr" in columns:
            loggers["mark_ctgr"] = LogMeanDetailLoss("mark_ctgr")
        if "sign_speed" in columns:
            loggers["sign_speed"] = LogMeanDetailLoss("sign_speed")
        if "mark_speed" in columns:
            loggers["mark_speed"] = LogMeanDetailLoss("mark_speed")
        if "distance" in columns:
            loggers["distance"] = LogMeanDetailLoss("distance")
        if "pos_obj" in columns:
            loggers["pos_obj"] = LogPositiveDetailObj("pos_obj")
        if "neg_obj" in columns:
            loggers["neg_obj"] = LogNegativeDetailObj("neg_obj")
        if "iou_mean" in columns:
            loggers["iou_mean"] = LogIouMean("iou_mean")
        if "box_yx" in columns:
            loggers["box_yx"] = LogBoxYX("box_yx")
        if "box_hw" in columns:
            loggers["box_hw"] = LogBoxHW("box_hw")
        if "true_class" in columns:
            loggers["true_class"] = LogTrueClass("true_class")
        if "false_class" in columns:
            loggers["false_class"] = LogFalseClass("false_class")
        return loggers

    def box_category_match(self, grtr, pred_bbox, categories, step):
        metric_data = [{"step": step, "anchor": -1, "ctgr": 0, "trpo": 0, "grtr": 0, "pred": 0}]
        for category in categories:
            pred_mask = self.create_mask(pred_bbox, category, "category")
            grtr_mask = self.create_mask(grtr["bboxes"], category, "category")
            grtr_match_box = self.box_matcher(grtr["bboxes"], grtr_mask)
            pred_match_box = self.box_matcher(pred_bbox, pred_mask)
            metric = count_true_positives(grtr_match_box, pred_match_box, grtr["dontcare"], self.num_categs)
            metric["anchor"] = -1
            metric["ctgr"] = category
            metric["step"] = step
            metric_data.append(metric)
        return metric_data

    def create_mask(self, data, index, key):
        if key:
            valid_mask = data[key] == index
        else:
            valid_mask = data == index
        return valid_mask

    def box_matcher(self, bbox, mask):
        match_bbox = dict()
        for key in bbox.keys():
            match_bbox[key] = bbox[key] * mask
        return match_bbox

    def ctgr_anchor_num_box(self, grtr_map, pred, categories, anchors, step):
        metric_data = []
        for anchor in anchors:
            grtr_anchor_mask = grtr_map["category"][..., anchor, :]
            pred_anchor_mask = self.create_mask(pred, anchor, "anchor_ind")
            for category in categories:
                pred_ctgr_mask = self.create_mask(pred, category, "category")
                pred_match = self.box_matcher(pred, pred_ctgr_mask * pred_anchor_mask)
                pred_far_mask = pred["distance"] > cfg.Validation.DISTANCE_LIMIT
                pred_valid = {key: (val * (1 - pred_far_mask).astype(np.float32)) for key, val in pred_match.items()}
                pred_num_box = np.sum(pred_valid["object"] > 0)

                grtr_catgr_mask = self.create_mask(grtr_anchor_mask, category, None)
                grtr_num_box = np.sum(grtr_map["object"][..., anchor, :] * grtr_catgr_mask)
                metric_data.append({"step": step, "anchor": anchor, "ctgr": category, "grtr": grtr_num_box, "pred": pred_num_box, "trpo": 0})
        return metric_data

    def finalize(self, start):
        epoch_time = (timer() - start) / 60
        # make summary dataframe
        self.make_summary(epoch_time)
        # write total_data to file

    def make_summary(self, epoch_time):
        mean_summary = self.compute_mean_summary(epoch_time)
        sum_summary = self.compute_sum_summary()
        # if exist sum_summary
        summary = pd.merge(left=mean_summary, right=sum_summary, how="outer", on=["anchor", "ctgr"])
        # self.summary = summary

        # summary["time_m"] = round(epoch_time, 5)

        self.summary = summary

    def compute_mean_summary(self, epoch_time):
        mean_data = self.data[cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN]
        mean_category_data = mean_data.groupby("ctgr", as_index=False).mean()
        mean_category_data["anchor"] = -1

        mean_anchor_data = mean_data.groupby("anchor", as_index=False).mean()
        mean_anchor_data["ctgr"] = -1

        mean_category_anchor_data = mean_data.groupby(["anchor", "ctgr"], as_index=False).mean()

        mean_epoch_data = pd.DataFrame([mean_data.mean(axis=0)])
        mean_epoch_data["anchor"] = -1
        mean_epoch_data["ctgr"] = -1
        mean_epoch_data["time_m"] = epoch_time

        mean_summary = pd.concat([mean_epoch_data, mean_anchor_data, mean_category_data, mean_category_anchor_data],
                                 join='outer', ignore_index=True)
        return mean_summary

    def compute_sum_summary(self):
        sum_data = self.metric_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_SUM]
        sum_category_data = sum_data.groupby("ctgr", as_index=False).sum()
        sum_category_data = pd.DataFrame({"anchor": sum_data["anchor"][:self.num_categs],
                                          "ctgr": sum_data["ctgr"][:self.num_categs],
                                          "recall": sum_category_data["trpo"] / (sum_category_data["grtr"] + 1e-5),
                                          "precision": sum_category_data["trpo"] / (sum_category_data["pred"] + 1e-5),
                                          "grtr": sum_category_data["grtr"], "pred": sum_category_data["pred"],
                                          "trpo": sum_category_data["trpo"]})

        sum_epoch_data = pd.DataFrame([sum_data.sum(axis=0).to_dict()])
        sum_epoch_data = pd.DataFrame({"anchor": -1, "ctgr": -1,
                                       "recall": sum_epoch_data["trpo"] / (sum_epoch_data["grtr"] + 1e-5),
                                       "precision": sum_epoch_data["trpo"] / (sum_epoch_data["pred"] + 1e-5),
                                       "grtr": sum_epoch_data["grtr"], "pred": sum_epoch_data["pred"],
                                       "trpo": sum_epoch_data["trpo"]})

        sum_data = self.ctgr_anchor_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_SUM]
        sum_ctgr_anchor_data = sum_data.groupby(["anchor", "ctgr"], as_index=False).sum()
        sum_ctgr_anchor_data["trpo"] = -1

        sum_summary = pd.concat([sum_epoch_data, sum_category_data, sum_ctgr_anchor_data], join='outer', ignore_index=True)
        return sum_summary

    def get_summary(self):
        return self.summary






