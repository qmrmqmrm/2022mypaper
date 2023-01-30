import time
import cv2
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

import config as cfg
import utils.framework.util_function as uf
from train.framework.train_util import mode_decor
import train.framework.train_util as tu
from log.logger import Logger

from log.visual_log import VisualLog
import dataloader.data_util as du


class TrainValBase:
    def __init__(self, model, loss_object, augmenter, optimizer, epoch_steps, feature_creator, ckpt_path):
        self.model = model
        self.loss_object = loss_object
        self.augmenter = augmenter
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.is_train = True
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.feature_creator = feature_creator
        self.batch_size = cfg.Train.BATCH_SIZE

    def run_epoch(self, dataset, scheduler, epoch=0, visual_log=False, exhaustive_log=False, val_only=False):
        logger = Logger(visual_log, exhaustive_log, self.loss_object.loss_names, self.ckpt_path, epoch, self.is_train, val_only)
        epoch_start = timer()
        for step, features in enumerate(dataset):
            start = timer()
            if self.is_train:
                self.optimizer.lr = scheduler(step)
            prediction, total_loss, loss_by_type, new_features = self.run_batch(features)
            logger.log_batch_result(step, new_features, prediction, total_loss, loss_by_type)
            print_loss = f"training {step}/{self.epoch_steps} steps, " + f"time={timer() - start:.3f}, " + f"total={total_loss:.3f}, "
            for key in loss_by_type.keys():
                if "map" not in key:
                    print_loss = print_loss + f"{key}={loss_by_type[key]:.3f}, "
            uf.print_progress(print_loss)
            if step >= self.epoch_steps:
                break
            # if step >= 10:
            #     break

        # print("")
        logger.finalize(epoch_start)
        if self.is_train and cfg.Scheduler.LOG:
            scheduler.save_log()

    def run_batch(self, features):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, augmenter, optimizer, epoch_steps, feature_creator, strategy, ckpt_path):
        super().__init__(model, loss_object, augmenter, optimizer, epoch_steps, feature_creator, ckpt_path)

    def run_batch(self, features):
        if self.augmenter:
            features = self.augmenter(features)
        features = self.feature_creator(features)

        return self.run_step(features)

    @mode_decor
    def run_step(self, features):
        with tf.GradientTape() as tape:
            prediction = self.model(features["image"], training=True)
            total_loss, loss_by_type = self.loss_object(features, prediction)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type, features


class ModelDistribTrainer(ModelTrainer):
    def __init__(self, model, loss_object, augmenter, optimizer, epoch_steps, feature_creator, strategy, ckpt_path):
        super().__init__(model, loss_object, augmenter, optimizer, epoch_steps, feature_creator, strategy, ckpt_path)
        self.strategy = strategy

    def run_epoch(self, dataset, scheduler, epoch=0, visual_log=False, exhaustive_log=False, val_only=False):
        logger = Logger(visual_log, exhaustive_log, self.loss_object.loss_names, self.ckpt_path, epoch, self.is_train, val_only)
        epoch_start = timer()
        for step, features in enumerate(dataset):
            start = timer()

            if self.is_train:
                self.optimizer.lr = scheduler(step)

            if self.augmenter:
                features = self.augmenter(features)
            features = self.feature_creator(features)

            replica_dataset = self.create_dataset(features)
            feat_keys = [key for key in features.keys() if key.startswith("feat")]
            for rep_step, replica_feat in enumerate(replica_dataset):
                for key in feat_keys:
                    replica_feat[key] = self.dict_to_list(replica_feat[key])
                prediction, total_loss, loss_by_type, new_features = self.run_batch(replica_feat)
                logger.log_batch_result(step, new_features, prediction, total_loss, loss_by_type)
                uf.print_progress(f"training {step}/{rep_step}/{self.epoch_steps} steps, "
                                  f"time={timer() - start:.3f}, "
                                  f"total={total_loss:.3f}, "
                                  f"box={loss_by_type['iou']:.3f}, "
                                  f"object={loss_by_type['object']:.3f}, "
                                  f"distance={loss_by_type['distance']:.3f}, "
                                  f"category={loss_by_type['category']:.3f}")

            if step >= self.epoch_steps:
                break
            # if step > 10:
            #     break

        logger.finalize(epoch_start)
        if self.is_train and cfg.Scheduler.LOG:
            scheduler.save_log()

    @mode_decor
    def run_batch(self, features):
        with self.strategy.scope():
            prediction, total_loss, loss_by_type, new_features = self.strategy.run(self.run_step, args=(features,))

            total_loss = self.reduce_replica(total_loss)
            scalar_loss = {key: val for key, val in loss_by_type.items() if "map" not in key}
            tensor_loss = {key: val for key, val in loss_by_type.items() if "map" in key}
            scalar_loss = self.reduce_replica(scalar_loss)
            tensor_loss = self.concat_replica(tensor_loss)
            loss_by_type.update(scalar_loss)
            loss_by_type.update(tensor_loss)

            new_pred = self.concat_replica(prediction)
            new_gt = self.concat_replica(new_features)

        return new_pred, total_loss, loss_by_type, new_gt

    def reduce_replica(self, value):
        if isinstance(value, dict):
            for name in value:
                value[name] = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, value[name], axis=None)
        else:
            value = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
        return value

    def concat_replica(self, value):
        if isinstance(value, dict):
            new_feat = dict()
            for name in value:
                new_feat[name] = self.concat_replica(value[name])
        elif isinstance(value, list):
            new_feat = list()
            for scale, feat in enumerate(value):
                new_feat.append(tf.concat(self.strategy.experimental_local_results(feat), 0))
        else:
            new_feat = tf.concat(self.strategy.experimental_local_results(value), 0)
        return new_feat

    def create_dataset(self, dataset):
        dataset = self.list_to_dict(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.batch(self.batch_size)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        dataset = dataset.with_options(options)
        replica_dataset = self.strategy.experimental_distribute_dataset(dataset)
        return replica_dataset

    def list_to_dict(self, feature):
        if isinstance(feature, dict):
            for key, value in feature.items():
                feature[key] = self.list_to_dict(value)
        elif isinstance(feature, list):
            return {f"list{i}": feat for i, feat in enumerate(feature)}
        return feature

    def dict_to_list(self, feature):
        if isinstance(feature, dict):
            for key, value in feature.items():
                if key.startswith("list"):
                    return [feature.pop(key) for key in list(feature.keys())]
                else:
                    feature[key] = self.dict_to_list(value)
        return feature


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, feature_creator, ckpt_path):
        super().__init__(model, loss_object, None, None, epoch_steps, feature_creator, ckpt_path)
        self.is_train = False

    def run_batch(self, features):
        features = self.feature_creator(features)
        return self.run_step(features)

    @mode_decor
    def run_step(self, features):
        prediction = self.model(features["image"])
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type, features

