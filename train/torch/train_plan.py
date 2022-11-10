import os
import os.path as op
import torch
from torch import nn
import torch.nn.functional as F

from utils.util_class import MyExceptionToCatch
from dataloader.framework.dataset_reader import DatasetReader
import model.framework.model_util as mu
from model.framework.model_factory import ModelFactory



import train.framework.train_util as tu
import config as cfg
import config_dir.util_config as uc


def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save):
    batch_size, data_batch_size, train_mode, anchors = cfg.Train.BATCH_SIZE, cfg.Train.DATA_BATCH_SIZE, cfg.Train.MODE, \
                                                       cfg.AnchorGeneration.ANCHORS
    datapath, ckpt_path = cfg.Paths.DATAPATH, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    valid_category = uc.get_valid_category_mask(dataset_name)
    start_epoch = read_previous_epoch(ckpt_path)
    train_batch_size = cfg.Train.GLOBAL_BATCH if train_mode == "distribute" else data_batch_size
    val_batch_size = data_batch_size if train_mode == "distribute" else batch_size
    if end_epoch <= start_epoch:
        print(f"!! end_epoch {end_epoch} <= start_epoch {start_epoch}, no need to train")
        return

    dataset_train, train_steps, imshape, anchors_per_scale = \
        get_dataset(datapath, dataset_name, True, train_batch_size, "train", anchors)
    dataset_val, val_steps, _, _ = get_dataset(datapath, dataset_name, False, val_batch_size, "val", anchors)

    model, loss_object, augmenter, optimizer, lrs, feature_creator = create_training_parts(batch_size, imshape,
                                                                                           anchors_per_scale, ckpt_path,
                                                                                           learning_rate, loss_weights,
                                                                                           valid_category)
    keras.utils.plot_model(model, to_file='resnet50.png', show_shapes=True, )
    trainer_class = tv.ModelDistribTrainer if train_mode == "distribute" else tv.ModelTrainer
    trainer = trainer_class(model, loss_object, augmenter, optimizer, lrs, train_steps, feature_creator,
                            anchors_per_scale, stretagy, ckpt_path)
    validater = tv.ModelValidater(model, loss_object, val_steps, feature_creator, anchors_per_scale, ckpt_path)

    for epoch in range(start_epoch, end_epoch):
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        detail_log = (epoch in cfg.Train.DETAIL_LOG_EPOCHS)
        trainer.run_epoch(dataset_train, epoch)
        validater.run_epoch(dataset_val, epoch, detail_log, detail_log)
        save_model_ckpt(ckpt_path, model)
    if model_save:
        save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def get_dataset(datapath, dataset_name, shuffle, batch_size, split, anchors):
    data_split_path = op.join(datapath, f"{dataset_name}_{split}")
    reader = DatasetReader(data_split_path, shuffle, batch_size, 1)
    dataset = reader.get_dataset()
    frames = reader.get_total_frames()
    dataset_cfg = reader.get_dataset_config()  # TODO
    image_shape = dataset_cfg["image"]["shape"]
    # anchor sizes per scale in pixel
    anchors_per_scale = np.array([anchor / np.array([image_shape[:2]]) for anchor in anchors], dtype=np.float32)
    print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, "
          f"frames={frames},\n\tanchors={anchors_per_scale}")
    return dataset, frames // batch_size, image_shape, anchors_per_scale


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model = tu.load_weights(model, ckpt_file)  # TODO
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


def read_previous_epoch(ckpt_path):
    filename = op.join(ckpt_path, 'history.csv')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            print("[read_previous_epoch] EMPTY history:", history)
            return 0

        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        print(f"[read_previous_epoch] NO history in {filename}")
        return 0
