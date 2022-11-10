import numpy as np
import pandas as pd
from timeit import default_timer as timer

import config as cfg
from log.metric import count_true_positives_lane
from log.logger_pool import *
import utils.framework.util_function as uf


class ExhaustiveLaneLog:
    def __init__(self, loss_names):
        self.columns = loss_names + cfg.Log.ExhaustiveLog.LANE_DETAIL
        box_loss_columns, lane_loss_columns = self.split_columns(self.columns)
        self.lane_loggers = self.create_loggers(lane_loss_columns)
        self.num_anchors = 1
        self.num_categs = cfg.ModelOutput.PRED_FMAP_LANE_COMPOSITION["lane_category"]
        self.data = pd.DataFrame()
        self.metric_data = pd.DataFrame()
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
        result = {"step": [step] * self.num_categs,
                  "ctgr": list(range(self.num_categs)),
                  }
        grtr_lane_map = self.extract_feature_map(grtr["feat_lane"])
        pred_lane_map = self.extract_feature_map(pred["feat_lane"])
        lane_loss = {key: value for key, value in loss.items() if "lane" in key}
        loss_map = self.extract_feature_map(lane_loss, True)
        for key, log_object in self.lane_loggers.items():
            result[key] = log_object(grtr_lane_map, pred_lane_map, loss_map)
        metric = self.lane_category_match(grtr, pred["inst_lane"], range(self.num_categs), step)
        self.metric_data = self.metric_data.append(metric, ignore_index=True)
        # self.ctgr_anchor_data = self.ctgr_anchor_data.append(ctgr_anchor_box_num, ignore_index=True)
        data = pd.DataFrame(result)
        self.data = self.data.append(data, ignore_index=True)

    def extract_feature_map(self, features, is_loss=False):
        features = uf.merge_scale(features, is_loss)
        features = self.split_anchor(features, is_loss)
        return features

    def gather_feature_maps_to_level1_dict(self, features, target_key):
        out_feats = dict()
        for feat_list in features:
            for subkey, feat_map in feat_list.items():
                out_feats[subkey] = feat_map
        return out_feats

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
        # TODO use for and eval
        loggers = dict()
        if "lane_true_class" in columns:
            loggers["lane_true_class"] = LogTrueLaneClass("lane_true_class")
        if "lane_false_class" in columns:
            loggers["lane_false_class"] = LogFalseLaneClass("lane_false_class")

        if "laneness" in columns:
            loggers["laneness"] = LogMeanDetailLaneLoss("laneness")
        if "lane_fpoints" in columns:
            loggers["lane_fpoints"] = LogMeanDetailLaneLoss("lane_fpoints")

        if "lane_centerness" in columns:
            loggers["lane_centerness"] = LogMeanDetailLaneLoss("lane_centerness")
        if "lane_category" in columns:
            loggers["lane_category"] = LogMeanDetailLaneLoss("lane_category")

        if "pos_lane" in columns:
            loggers["pos_lane"] = LogPositiveDetaiLaneness("pos_lane")
        if "neg_lane" in columns:
            loggers["neg_lane"] = LogNegativeDetailLaneness("neg_lane")

        if "pos_lanecenter" in columns:
            loggers["pos_lanecenter"] = LogPositiveDetaiCenterness("pos_lanecenter")
        if "neg_lanecenter" in columns:
            loggers["neg_lanecenter"] = LogNegativeDetailCenterness("neg_lanecenter")


        return loggers

    def lane_category_match(self, grtr, pred_bbox, categories, step):
        metric_data = []
        img_shape = np.array(grtr["image"].shape[1:3])
        for category in categories:
            pred_mask = self.create_mask(pred_bbox, category, "lane_category")
            grtr_mask = self.create_mask(grtr["inst_lane"], category, "lane_category")
            grtr_match_box = self.box_matcher(grtr["inst_lane"], grtr_mask)
            pred_match_box = self.box_matcher(pred_bbox, pred_mask)
            metric = count_true_positives_lane(grtr_match_box, pred_match_box, self.num_categs, img_shape, is_train=False)
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

    def finalize(self, start):
        epoch_time = (timer() - start) / 60
        # make summary dataframe
        self.make_summary(epoch_time)
        # write total_data to file

    def make_summary(self, epoch_time):
        mean_summary = self.compute_mean_summary(epoch_time)
        sum_summary = self.compute_sum_summary()
        # if exist sum_summary
        summary = pd.merge(left=mean_summary, right=sum_summary, how="outer", on=["ctgr"])
        # self.summary = summary

        # summary["time_m"] = round(epoch_time, 5)

        self.summary = summary

    def compute_mean_summary(self, epoch_time):
        mean_data = self.data[cfg.Log.ExhaustiveLog.COLUMNS_TO_LANE_MEAN]
        mean_category_data = mean_data.groupby("ctgr", as_index=False).mean()
        
        mean_data["ctgr"] = -1
        mean_epoch_data = pd.DataFrame([mean_data.mean(axis=0)])
        mean_epoch_data["time_m"] = epoch_time

        mean_summary = pd.concat([mean_epoch_data, mean_category_data], join='outer', ignore_index=True)
        return mean_summary

    def compute_sum_summary(self):
        sum_data = self.metric_data[cfg.Log.ExhaustiveLog.COLUMNS_TO_LANE_SUM]
        sum_category_data = sum_data.groupby("ctgr", as_index=False).sum()
        sum_category_data = pd.DataFrame({"ctgr": sum_data["ctgr"][:self.num_categs],
                                          "recall_lane": sum_category_data["trpo_lane"] / (sum_category_data["grtr_lane"] + 1e-5),
                                          "precision_lane": sum_category_data["trpo_lane"] / (sum_category_data["pred_lane"] + 1e-5),
                                          "grtr_lane": sum_category_data["grtr_lane"], "pred_lane": sum_category_data["pred_lane"],
                                          "trpo_lane": sum_category_data["trpo_lane"]})

        sum_epoch_data = pd.DataFrame([sum_data.sum(axis=0).to_dict()])
        sum_epoch_data = pd.DataFrame({"ctgr": -1,
                                       "recall_lane": sum_epoch_data["trpo_lane"] / (sum_epoch_data["grtr_lane"] + 1e-5),
                                       "precision_lane": sum_epoch_data["trpo_lane"] / (sum_epoch_data["pred_lane"] + 1e-5),
                                       "grtr_lane": sum_epoch_data["grtr_lane"], "pred_lane": sum_epoch_data["pred_lane"],
                                       "trpo_lane": sum_epoch_data["trpo_lane"]})

        sum_summary = pd.concat([sum_epoch_data, sum_category_data], join='outer', ignore_index=True)
        return sum_summary

    def get_summary(self):
        return self.summary






