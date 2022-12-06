import pandas as pd
from timeit import default_timer as timer

import config as cfg
from log.metric import *
from log.logger_pool import LogMeanLoss, LogPositiveObj, LogNegativeObj, LogPositiveLaneness, LogPositiveCenterness, \
    LogNegativeLaneness, LogNegativeCenterness
import utils.framework.util_function as uf


class HistoryLog:
    def __init__(self, loss_names, is_train):
        self.is_train = is_train
        self.columns = loss_names + cfg.Log.HistoryLog.SUMMARY
        self.loggers = self.create_loggers(self.columns)
        self.data = pd.DataFrame()
        self.summary = dict()

    def __call__(self, step, grtr, pred, loss, total_loss):
        # result = {"step": step}
        result = dict()
        for key, log_object in self.loggers.items():
            result[key] = log_object(grtr, pred, loss)
        result.update({"total_loss": total_loss.numpy()})
        if cfg.ModelOutput.BOX_DET:
            num_ctgr = pred["feat_box"]["category"][0].shape[-1]
            metric = count_true_positives(grtr["inst_box"], pred["inst_box"], grtr["inst_dc"], num_ctgr)
            result.update(metric)

            result["dist_diff"] = self.distance_diff(grtr, pred)
        # TODO: add metric lane
        if cfg.ModelOutput.LANE_DET:
            num_lane_ctgr = pred["feat_lane"]["lane_category"][0].shape[-1]
            img_shape = np.array(grtr["image"].shape[1:3])
            metric_lane = count_true_positives_lane(grtr["inst_lane"], pred["inst_lane"], num_lane_ctgr, img_shape,
                                                    is_train=self.is_train)
            result.update(metric_lane)
        self.data = self.data.append(result, ignore_index=True)

    def create_loggers(self, columns):
        loggers = dict()
        if "iou" in columns:
            loggers["iou"] = LogMeanLoss("iou")
        if "iou_aware" in columns:
            loggers["iou_aware"] = LogMeanLoss("iou_aware")
        if "object" in columns:
            loggers["object"] = LogMeanLoss("object")
        if "category" in columns:
            loggers["category"] = LogMeanLoss("category")
        if "sign_ctgr" in columns:
            loggers["sign_ctgr"] = LogMeanLoss("sign_ctgr")
        if "mark_ctgr" in columns:
            loggers["mark_ctgr"] = LogMeanLoss("mark_ctgr")
        if "sign_speed" in columns:
            loggers["sign_speed"] = LogMeanLoss("sign_speed")
        if "mark_speed" in columns:
            loggers["mark_speed"] = LogMeanLoss("mark_speed")
        if "distance" in columns:
            loggers["distance"] = LogMeanLoss("distance")
        if "laneness" in columns:
            loggers["laneness"] = LogMeanLoss("laneness")
        if "lane_fpoints" in columns:
            loggers["lane_fpoints"] = LogMeanLoss("lane_fpoints")
        if "lane_centerness" in columns:
            loggers["lane_centerness"] = LogMeanLoss("lane_centerness")
        if "lane_category" in columns:
            loggers["lane_category"] = LogMeanLoss("lane_category")
        if "pos_obj" in columns:
            loggers["pos_obj"] = LogPositiveObj("pos_obj")
        if "neg_obj" in columns:
            loggers["neg_obj"] = LogNegativeObj("neg_obj")
        if "pos_lane" in columns:
            loggers["pos_lane"] = LogPositiveLaneness("pos_lane")
        if "neg_lane" in columns:
            loggers["neg_lane"] = LogNegativeLaneness("neg_lane")
        if "pos_center" in columns:
            loggers["pos_center"] = LogPositiveCenterness("pos_center")
        if "neg_center" in columns:
            loggers["neg_center"] = LogNegativeCenterness("neg_center")
        return loggers

    def distance_diff(self, grtr, pred):
        grtr_bbox_augmented = self.exapand_grtr_bbox(grtr, pred)
        iou_thresh = cfg.Validation.TP_IOU_THRESH
        pred_valid, pred_far = split_pred_far(pred["inst_box"])
        grtr_far, grtr_valid = split_grtr_far(pred_far, grtr_bbox_augmented, iou_thresh)
        batch, M, _ = pred_valid["category"].shape
        iou = uf.compute_iou_general(grtr_valid["yxhw"], pred_valid["yxhw"]).numpy()  # (batch, N, M)

        best_iou = np.max(iou, axis=-1)  # (batch, N)

        best_idx = np.argmax(iou, axis=-1)  # (batch, N)
        if len(iou_thresh) > 1:
            iou_thresh = get_iou_thresh_per_class(grtr_valid["category"], iou_thresh)
        iou_match = best_iou > iou_thresh  # (batch, N)

        pred_ctgr_aligned = numpy_gather(pred_valid["category"], best_idx, 1)  # (batch, N, 8)
        pred_dist_aligned = numpy_gather(pred_valid["distance"], best_idx, 1)  # (batch, N, 8)
        grtr_dist_mask = grtr_valid["distance"][..., 0] > 0
        ctgr_match = grtr_valid["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
        grtr_dist_tp_mask = np.expand_dims(iou_match * ctgr_match * grtr_dist_mask, axis=-1)
        grtr_valid_dist = grtr_valid["distance"] * grtr_dist_tp_mask
        pred_valid_dist = pred_dist_aligned * grtr_dist_tp_mask[..., 0]
        grtr_dist = grtr_valid_dist[grtr_dist_tp_mask > 0]
        pred_dist = pred_valid_dist[grtr_dist_tp_mask[..., 0] > 0]
        distance_diff = np.abs(grtr_dist - pred_dist) / (grtr_dist + 1e-5)
        return np.mean(distance_diff)

    def exapand_grtr_bbox(self, grtr, pred):
        grtr_boxes = uf.merge_scale(grtr["feat_box"])
        pred_boxes = uf.merge_scale(pred["feat_box"])

        best_probs = np.max(pred_boxes["category"], axis=-1, keepdims=True)
        grtr_boxes["pred_ctgr_prob"] = best_probs
        grtr_boxes["pred_object"] = pred_boxes["object"]
        grtr_boxes["pred_score"] = best_probs * pred_boxes["object"]

        batch, _, _ = grtr["inst_box"]["yxhw"].shape
        numbox = cfg.Validation.MAX_BOX
        objectness = grtr_boxes["object"]
        for key in grtr_boxes:
            features = []
            for frame_idx in range(batch):
                valid_mask = (objectness[frame_idx, :, 0] == 1).astype(np.bool)
                feature = grtr_boxes[key]
                feature = feature[frame_idx, valid_mask, :]
                feature = np.pad(feature, [(0, numbox - feature.shape[0]), (0, 0)])
                features.append(feature)

            features = np.stack(features, axis=0)
            grtr_boxes[key] = features

        return grtr_boxes

    def finalize(self, start):
        epoch_time = (timer() - start) / 60
        self.make_summary(epoch_time)
        # make summary dataframe

    def make_summary(self, epoch_time):
        mean_result = self.data.mean(axis=0).to_dict()
        sum_result_ = self.data.sum(axis=0).to_dict()
        sum_result = {}
        metric_keys = []
        if cfg.ModelOutput.BOX_DET:
            sum_result.update({"recall": sum_result_["trpo"] / (sum_result_["grtr"] + 1e-5),
                          "precision": sum_result_["trpo"] / (sum_result_["pred"] + 1e-5),
                          })

            metric_keys.extend(["trpo", "grtr", "pred"])

        if cfg.ModelOutput.LANE_DET:
            sum_result.update({"recall_lane": sum_result_["trpo_lane"] / (sum_result_["grtr_lane"] + 1e-5),
                               "precision_lane": sum_result_["trpo_lane"] / (sum_result_["pred_lane"] + 1e-5)
                               })
            sum_result.update({"f1": (2 * sum_result["recall_lane"] * sum_result["precision_lane"]) / (
                        sum_result["recall_lane"] + sum_result["precision_lane"])
                          })
            metric_keys.extend(["trpo_lane", "grtr_lane", "pred_lane"])
        summary = {key: val for key, val in mean_result.items() if key not in metric_keys}
        summary.update(sum_result)
        summary["time_m"] = round(epoch_time, 5)
        print("epoch summary:", summary)
        self.summary = summary

    def get_summary(self):
        return self.summary
