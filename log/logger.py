import time

import numpy as np
import os.path as op
import pandas as pd
import os

from log.box_exhaustive_log import ExhaustiveBoxLog
from log.lane_exhaustive_log import ExhaustiveLaneLog
from log.history_log import HistoryLog
from log.visual_log import VisualLog
import utils.framework.util_function as uf
import model.framework.model_util as mu
import config as cfg
import config_dir.util_config as uc
from log.save_pred import SavePred


class Logger:
    def __init__(self, visual_log, exhaustive_log, loss_names, ckpt_path, epoch, is_train, val_only):
        self.history_logger = HistoryLog(loss_names, is_train)
        self.exhaustive_log = exhaustive_log
        self.save_pred = SavePred(op.join(ckpt_path, "result"))
        self.exhaustive_box_logger = ExhaustiveBoxLog(loss_names) if exhaustive_log and cfg.ModelOutput.BOX_DET else None
        self.exhaustive_lane_logger = ExhaustiveLaneLog(loss_names) if exhaustive_log and cfg.ModelOutput.LANE_DET else None
        self.visual_logger = VisualLog(ckpt_path, epoch,  val_only) if visual_log else None
        self.history_filename = op.join(ckpt_path, "history.csv")
        self.num_channel = cfg.ModelOutput.NUM_BOX_MAIN_CHANNELS
        self.exhaust_path = op.join(ckpt_path, "exhaust_log")
        if not op.isdir(self.exhaust_path):
            os.makedirs(self.exhaust_path, exist_ok=True)
        self.nms_box = mu.NonMaximumSuppressionBox()
        self.nms_lane = mu.NonMaximumSuppressionLane()
        self.is_train = is_train
        self.epoch = epoch
        self.ckpt_path = ckpt_path
        self.val_only = val_only

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        self.check_nan(grtr, "grtr")
        self.check_nan(pred, "pred")
        self.check_nan(loss_by_type, "loss")
        if cfg.ModelOutput.BOX_DET:
            nms_boxes = self.nms_box(pred["feat_box"])
            inst_box = uf.slice_feature(nms_boxes, uc.get_bbox_composition(False))
            inst_box = self.nms_box(inst_box, merged=True, is_inst=True)
            pred["inst_box"] = uf.slice_feature(inst_box, uc.get_bbox_composition(False))

        if cfg.ModelOutput.LANE_DET:
            lane_hw = pred["feat_lane"]["whole"][0].shape[1:3]
            nms_lanes = self.nms_lane(pred["feat_lane"], lane_hw)
            pred["inst_lane"] = uf.slice_feature(nms_lanes, uc.get_lane_composition(False))

        for key, feature_slices in grtr.items():
            grtr[key] = uf.convert_tensor_to_numpy(feature_slices)
        for key, feature_slices in pred.items():
            pred[key] = uf.convert_tensor_to_numpy(feature_slices)
        loss_by_type = uf.convert_tensor_to_numpy(loss_by_type)

        if step == 0 and self.epoch == 0:
            structures = {"grtr": grtr, "pred": pred, "loss": loss_by_type}
            self.save_model_structure(structures)
        self.history_logger(step, grtr, pred, loss_by_type, total_loss)

        if self.exhaustive_log:
            if cfg.ModelOutput.BOX_DET:
                self.exhaustive_box_logger(step, grtr, pred, loss_by_type, total_loss)
            if cfg.ModelOutput.LANE_DET:
                self.exhaustive_lane_logger(step, grtr, pred, loss_by_type, total_loss)
        if self.visual_logger:
            self.visual_logger(step, grtr, pred)
        if self.val_only:
            self.save_pred(step, grtr, pred)

    def check_nan(self, features, feat_name):
        valid_result = True
        if isinstance(features, dict):
            for name, value in features.items():
                self.check_nan(value, f"{feat_name}_{name}")
        elif isinstance(features, list):
            for idx, tensor in enumerate(features):
                self.check_nan(tensor, f"{feat_name}_{idx}")
        elif isinstance(features, str):
            pass
        else:
            if features.ndim == 0 and (np.isnan(features) or np.isinf(features) or features > 100000000):
                print(f"nan loss: {feat_name}, {features}")
                valid_result = False
            elif not np.isfinite(features.numpy()).all():
                print(f"nan {feat_name}:", np.quantile(features.numpy(), np.linspace(0, 1, self.num_channel)))
                valid_result = False
        assert valid_result

    def finalize(self, start):
        self.history_logger.finalize(start)
        if self.exhaustive_log:
            if cfg.ModelOutput.BOX_DET:
                self.save_exhaustive_log(start, self.exhaustive_box_logger, "box")
            if cfg.ModelOutput.LANE_DET:
                self.save_exhaustive_log(start, self.exhaustive_lane_logger, "lane")
        if self.val_only:
            self.save_val_log()
        else:
            self.save_log()

    def save_log(self):
        logger_summary = self.history_logger.get_summary()
        if self.is_train:
            train_summary = {"epoch": self.epoch}
            train_summary.update({"!" + key: val for key, val in logger_summary.items()})
            train_summary.update({"|": "|"})
            if op.isfile(self.history_filename):
                history_summary = pd.read_csv(self.history_filename, encoding='utf-8',
                                              converters={'epoch': lambda c: int(c)})
                history_summary = history_summary.append(train_summary, ignore_index=True)
            else:
                history_summary = pd.DataFrame([train_summary])
        else:
            history_summary = pd.read_csv(self.history_filename, encoding='utf-8',
                                          converters={'epoch': lambda c: int(c)})
            for key, val in logger_summary.items():
                history_summary.loc[self.epoch, "`" + key] = val

        history_summary["epoch"] = history_summary["epoch"].astype(int)
        print("=== history\n", history_summary)
        history_summary.to_csv(self.history_filename, encoding='utf-8', index=False, float_format='%.4f')

    def save_exhaustive_log(self, start, exhaustive_logger, name):
        exhaustive_logger.finalize(start)
        exhaust_summary = exhaustive_logger.get_summary()
        if self.val_only:
            exhaust_filename = self.exhaust_path + f"/exhaust_{name}_val.csv"
            if cfg.ModelOutput.MINOR_CTGR and (name != "lane"):
                sign_summary, mark_summary = exhaustive_logger.get_minor_summary()
                sign_filename = self.exhaust_path + f"/sign_val.csv"
                mark_filename = self.exhaust_path + f"/mark_val.csv"
                sign_summary = pd.DataFrame(sign_summary)
                mark_summary = pd.DataFrame(mark_summary)
                sign_summary.to_csv(sign_filename, encoding='utf-8', index=False, float_format='%.4f')
                mark_summary.to_csv(mark_filename, encoding='utf-8', index=False, float_format='%.4f')
        else:
            exhaust_filename = self.exhaust_path + f"/{name}_epoch{self.epoch:02d}.csv"
        exhaust = pd.DataFrame(exhaust_summary)
        exhaust.to_csv(exhaust_filename, encoding='utf-8', index=False, float_format='%.4f')

    def save_val_log(self):
        logger_summary = self.history_logger.get_summary()
        val_filename = self.history_filename[:-4] + "_val.csv"
        epoch_summary = {"epoch": self.epoch}
        epoch_summary.update({"`" + key: val for key, val in logger_summary.items()})
        history_summary = pd.DataFrame([epoch_summary])
        history_summary["epoch"] = history_summary["epoch"].astype(int)
        print("=== validation history\n", history_summary)
        history_summary.to_csv(val_filename, encoding='utf-8', index=False, float_format='%.4f')

    def save_model_structure(self, structures):
        structure_file = op.join(self.ckpt_path, "structure.md")
        f = open(structure_file, "w")
        for key, structure in structures.items():
            f.write(f"- {key}\n")
            space_count = 1
            self.analyze_structure(structure, f, space_count)
        f.close()

    def analyze_structure(self, data, f, space_count, key=""):
        space = "    " * space_count
        if isinstance(data, list):
            for i, datum in enumerate(data):
                if isinstance(datum, dict):
                    # space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    # space_count -= 1
                elif type(datum) == np.ndarray:
                    f.write(f"{space}- {key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {datum}\n")
                    space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    space_count -= 1
        elif isinstance(data, dict):
            for sub_key, datum in data.items():
                if type(datum) == np.ndarray:
                    f.write(f"{space}- {sub_key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {sub_key}\n")

                space_count += 1
                self.analyze_structure(datum, f, space_count, sub_key)
                space_count -= 1
