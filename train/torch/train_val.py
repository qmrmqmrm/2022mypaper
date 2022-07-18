import numpy as np
import torch
from timeit import default_timer as timer
import os.path as op

import config as cfg
import utils.framework.util_function as uf
import train.framework.train_util as tu
from log.logger import Logger


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps, feature_creator, anchors_per_scale,
                 ckpt_path):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.is_train = True
        self.device = cfg.Hardware.DEVICE
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.anchor_ratio = np.concatenate([anchor for anchor in anchors_per_scale])
        self.feature_creator = feature_creator

    def run_epoch(self, dataset, epoch,  visual_log=False, exhaustive_log=False, val_only=False):
        #  dataset, scheduler, epoch=0, visual_log=False, exhaustive_log=False, val_only=False
        self.mode_set()
        logger = Logger(visual_log, exhaustive_log, self.ckpt_path, epoch, self.is_train, val_only)
        epoch_start = timer()
        train_loader_iter = iter(dataset)
        steps = len(train_loader_iter)
        for step in range(steps):
            start = timer()
            features = self.to_device(next(train_loader_iter))
            prediction, total_loss, loss_by_type, new_features = self.run_batch(features)
            logger.log_batch_result(step, new_features, prediction, total_loss, loss_by_type)

            # logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"training {step}/{self.epoch_steps} steps, "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, ")

            if step >= self.epoch_steps:
                break

        logger.finalize(epoch_start)

    def to_device(self, features):
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(device=self.device)
            if isinstance(features[key], list):
                data = list()
                for feature in features[key]:
                    if isinstance(feature, torch.Tensor):
                        feature = feature.to(device=self.device)
                    data.append(feature)
                features[key] = data
        return features

    def run_step(self, features):
        raise NotImplementedError()

    def mode_set(self):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps, feature_creator, anchors_per_scale,
                 ckpt_path):
        super().__init__(model, loss_object, optimizer, epoch_steps, feature_creator, anchors_per_scale,
                         ckpt_path)
        self.split = "train"

    def run_batch(self, features):
        features["feat"] = []
        for i in range(features["image"].shape[0]):
            feat_sizes = [np.array(features["image"][i].shape[:2]) // scale for scale in self.feat_scales]
            featmap = self.feature_creator.create(features["bboxes"][i].numpy(), feat_sizes)
            features["feat"] = tu.create_batch_featmap(features, featmap)
        features = tu.gt_feat_rename(features)

        return self.run_step(features)

    def run_step(self, features):
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return prediction, total_loss, loss_by_type, features

    def mode_set(self):
        self.model.train()
        self.model.set_gt_use(True)


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, feature_creator, anchors_per_scale, ckpt_path):
        super().__init__(model, loss_object, None, epoch_steps, feature_creator, anchors_per_scale,
                         ckpt_path)

    def run_batch(self, features):
        for i in range(features["image"].shape[0]):
            feat_sizes = [np.array(features["image"][i].shape[:2]) // scale for scale in self.feat_scales]
            featmap = self.feature_creator.create(features["bboxes"][i].numpy(), feat_sizes)
            features["feat"] = tu.create_batch_featmap(features, featmap)
        features = tu.gt_feat_rename(features)
        return self.run_step(features)

    def run_step(self, features):
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction, False)
        return prediction, total_loss, loss_by_type, features

    def mode_set(self):
        self.model.train()
        self.model.set_gt_use(False)


