import numpy as np
import pandas as pd
from timeit import default_timer as timer

import config as cfg
from log.metric import count_true_positives
from log.logger_pool import *
import utils.framework.util_function as uf


class ExhaustiveBoxLog:
    def __init__(self, loss_names):
        self.columns = loss_names + cfg.Log.ExhaustiveLog.DETAIL
        box_loss_columns, lane_loss_columns = self.split_columns(self.columns)
        self.box_loggers = self.create_loggers(box_loss_columns)
        self.num_anchors = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE * len(cfg.ModelOutput.FEATURE_SCALES)
        self.num_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["category"]
        if cfg.ModelOutput.MINOR_CTGR:
            self.num_sign_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["sign_ctgr"]
            self.num_mark_categs = cfg.ModelOutput.PRED_FMAP_COMPOSITION["mark_ctgr"]
            self.sign_summary = dict()
            self.mark_summary = dict()
        self.num_anchors_per_scale = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        self.num_scales = len(cfg.ModelOutput.FEATURE_SCALES)
        self.data = pd.DataFrame()
        self.metric_data = pd.DataFrame()
        self.sign_metric_data = pd.DataFrame()
        self.mark_metric_data = pd.DataFrame()
        self.ctgr_anchor_data = pd.DataFrame()
        self.summary = dict()

    def split_columns(self, columns):
        box_loss_columns, lane_loss_columns = [], []
        for column in columns:
            if "lane" in column:
                lane_loss_columns.append(column)
            else:
                box_loss_columns.append(column)
        return box_loss_columns, lane_loss_columns

    def __call__(self, step, grtr, pred, loss, total_loss):
        result = {"step": [step] * self.num_anchors * self.num_categs,
                  "anchor": [i for i in range(self.num_anchors) for k in range(self.num_categs)],
                  "ctgr": list(range(self.num_categs)) * self.num_anchors,
                  }
        grtr_box_map = self.extract_feature_map(grtr["feat_box"])
        pred_box_map = self.extract_feature_map(pred["feat_box"])
        box_loss = {key: value for key, value in loss.items() if not "lane" in key}
        loss_map = self.extract_feature_map(box_loss, True)
        for key, log_object in self.box_loggers.items():
            result[key] = log_object(grtr_box_map, pred_box_map, loss_map)
        data = pd.DataFrame(result)
        self.data = self.data.append(data, ignore_index=True)

        metric, sign_metric, mark_metric = self.box_category_match(grtr, pred["inst_box"], range(1, self.num_categs), step)
        self.metric_data = self.metric_data.append(metric, ignore_index=True)
        self.sign_metric_data = self.sign_metric_data.append(sign_metric, ignore_index=True)
        self.mark_metric_data = self.mark_metric_data.append(mark_metric, ignore_index=True)

        ctgr_anchor_box_num = self.ctgr_anchor_num_box(grtr_box_map, pred["inst_box"], range(1, self.num_categs),
                                                       range(0, self.num_anchors), step)
        self.ctgr_anchor_data = self.ctgr_anchor_data.append(ctgr_anchor_box_num, ignore_index=True)

    def extract_feature_map(self, features, is_loss=False):
        # features = uf.merge_scale(features, is_loss)
        features = self.split_anchor(features, is_loss)
        return features

    def merge_scale(self, feature, is_loss=False):
        features = uf.merge_scale(feature, is_loss)
        return features

    def gather_feature_maps_to_level1_dict(self, features, target_key):
        out_feats = dict()
        for feat_list in features:
            for subkey, feat_map in feat_list.items():
                out_feats[subkey] = feat_map
        return out_feats

    def split_anchor(self, features, is_loss):
        if is_loss:
            new_features = {key[:-4]: [] for key in features.keys() if "map" in key}
        else:
            new_features = {key: [] for key in features.keys() if key != "whole"}

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

    def create_loggers(self, columns):
        # TODO use for and eval
        loggers = dict()
        if "iou" in columns:
            loggers["iou"] = LogMeanDetailLoss("iou")
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
        sign_metric_data = []
        mask_metric_data = []

        for category in categories:
            pred_mask = self.create_mask(pred_bbox, category, "category")
            grtr_mask = self.create_mask(grtr["inst_box"], category, "category")
            grtr_match_box = self.box_matcher(grtr["inst_box"], grtr_mask)
            pred_match_box = self.box_matcher(pred_bbox, pred_mask)
            metric = count_true_positives(grtr_match_box, pred_match_box, grtr["inst_dc"], self.num_categs)

            metric["anchor"] = -1
            metric["ctgr"] = category
            metric["step"] = step
            metric_data.append(metric)
            if cfg.ModelOutput.MINOR_CTGR:
                if category == cfg.Dataloader.CATEGORY_NAMES["major"].index("Traffic sign") or category == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark"):
                    minor_metric = count_true_positives(grtr_match_box, pred_match_box, grtr["inst_dc"], self.num_categs,
                                                        per_class=True, num_sign_ctgr=self.num_sign_categs,
                                                        num_mark_ctgr=self.num_mark_categs)
                    sign_metric = {key: val for key, val in minor_metric.items() if "sign" in key}
                    mark_metric = {key: val for key, val in minor_metric.items() if "mark" in key}
                    sign_metric_data.append(sign_metric)
                    mask_metric_data.append(mark_metric)
        return metric_data, sign_metric_data, mask_metric_data

    def create_mask(self, data, index, key):
        if key:
            valid_mask = data[key] == index
        else:
            valid_mask = data == index
        return valid_mask

    def create_scale_mask(self, data, index, scale, key):
        if key:
            valid_mask = data[key][scale] == index
        else:
            valid_mask = data == index
        return valid_mask

    def box_matcher(self, bbox, mask):
        match_bbox = dict()
        for key in bbox.keys():
            match_bbox[key] = bbox[key] * mask
        return match_bbox

    def box_scale_matcher(self, bbox, mask, scale):
        match_bbox = dict()
        for key in bbox.keys():
            match_bbox[key] = bbox[key][scale] * mask
        return match_bbox

    def ctgr_anchor_num_box(self, grtr_map, pred, categories, anchors, step):
        metric_data = []
        for anchor in anchors:
            scale = anchor // 3
            grtr_anchor_mask = self.create_scale_mask(grtr_map, anchor, scale, "anchor_ind")
            pred_anchor_mask = self.create_mask(pred, anchor, "anchor_ind")
            for category in categories:
                pred_ctgr_mask = self.create_mask(pred, category, "category")
                pred_match = self.box_matcher(pred, pred_ctgr_mask * pred_anchor_mask)
                pred_far_mask = pred["distance"] > cfg.Validation.DISTANCE_LIMIT
                pred_valid = {key: (val * (1 - pred_far_mask).astype(np.float32)) for key, val in pred_match.items()}
                pred_num_box = np.sum(pred_valid["object"] > 0)

                grtr_catgr_mask = self.create_scale_mask(grtr_map, category, scale, "category")
                grtr_match = self.box_scale_matcher(grtr_map, grtr_catgr_mask * grtr_anchor_mask, scale)
                grtr_num_box = np.sum(grtr_match["object"] > 0)
                metric_data.append({"step": step, "anchor": anchor, "ctgr": category, "grtr": grtr_num_box, "pred": pred_num_box, "trpo": 0})
        return metric_data

    def finalize(self, start):
        epoch_time = (timer() - start) / 60
        # make summary dataframe
        self.make_summary(epoch_time)
        # write total_data to file

    def make_summary(self, epoch_time):
        mean_summary = self.compute_mean_summary(epoch_time)
        sum_summary, sign_summary, mark_summary = self.compute_sum_summary()
        # if exist sum_summary
        summary = pd.merge(left=mean_summary, right=sum_summary, how="outer", on=["anchor", "ctgr"])
        # self.summary = summary

        # summary["time_m"] = round(epoch_time, 5)

        self.summary = summary
        self.mark_summary = mark_summary
        self.sign_summary = sign_summary

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

        sum_anchor_data = sum_data.groupby(["anchor"], as_index=False).sum()
        sum_anchor_data["ctgr"] = -1
        sum_anchor_data["trpo"] = -1

        sum_summary = pd.concat([sum_epoch_data, sum_anchor_data, sum_category_data, sum_ctgr_anchor_data], join='outer', ignore_index=True)
        if cfg.ModelOutput.MINOR_CTGR:
            sign_data = self.sign_metric_data
            sign_trpo = np.sum(np.stack(sign_data["sign_trpo"].tolist(), axis=0), axis=0)
            sign_grtr = np.sum(np.stack(sign_data["sign_grtr"].tolist(), axis=0), axis=0)
            sign_pred = np.sum(np.stack(sign_data["sign_pred"].tolist(), axis=0), axis=0)
            sign_summary = pd.DataFrame({"recall": sign_trpo / (sign_grtr + 1e-5),
                                         "precision": sign_trpo / (sign_pred + 1e-5),
                                         "grtr": sign_grtr, "pred": sign_pred,
                                         "trpo": sign_trpo})
            mark_data = self.mark_metric_data
            mark_trpo = np.sum(np.stack(mark_data["mark_trpo"].tolist(), axis=0), axis=0)
            mark_grtr = np.sum(np.stack(mark_data["mark_grtr"].tolist(), axis=0), axis=0)
            mark_pred = np.sum(np.stack(mark_data["mark_pred"].tolist(), axis=0), axis=0)
            mark_summary = pd.DataFrame({"recall": mark_trpo / (mark_grtr + 1e-5),
                                         "precision": mark_trpo / (mark_pred + 1e-5),
                                         "grtr": mark_grtr, "pred": mark_pred,
                                         "trpo": mark_trpo})
            return sum_summary, sign_summary, mark_summary
        return sum_summary, None, None

    def get_summary(self):
        return self.summary

    def get_minor_summary(self):
        return self.sign_summary, self.mark_summary



def summarize_data():
    raw_data = pd.read_csv("/home/falcon/kim_workspace/ckpt/scaled_weight_inverse+lane_bgd/exhaust_log/exhaust_box_val_total.csv")
    log_data = raw_data.copy()
    log_data = log_data.fillna(0)
    print(log_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN].head())
    print(log_data[["grtr"]].head())
    for key in cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN[2:]:
        log_data[key] = log_data[key] * log_data["grtr"]
    print(log_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN].head())

    ctgr_data = log_data.groupby("ctgr", as_index=False).sum()
    ctgr_data["anchor"] = -1
    anch_data = log_data.groupby("anchor", as_index=False).sum()
    anch_data["ctgr"] = -1
    single_summary = pd.DataFrame(np.sum(log_data.values, axis=0, keepdims=True), columns=list(log_data))
    single_summary[["anchor"]] = -1
    single_summary[["ctgr"]] = -1
    print(single_summary)

    summary_data = pd.concat([single_summary, ctgr_data, anch_data], axis=0)
    print(summary_data)
    for key in cfg.Log.ExhaustiveLog.COLUMNS_TO_MEAN[2:]:
        summary_data[key] = summary_data[key] / (summary_data["grtr"] + 1e-5)
    print(summary_data)

    total_data = pd.concat([summary_data, raw_data])
    print(total_data)
    total_data.to_csv("/home/falcon/kim_workspace/ckpt/scaled_weight_inverse+lane_bgd/exhaust_log/exhaust_box_val_new.csv",
                      encoding='utf-8', index=False, float_format='%.4f')




if __name__ == "__main__":
    summarize_data()





