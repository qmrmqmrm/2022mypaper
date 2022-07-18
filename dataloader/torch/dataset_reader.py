import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataloader.readers.kitti_reader import KittiReader, KittiBevReader
import dataloader.framework.data_util as du
from dataloader.example_maker import ExampleMaker
import config_dir.config_generator as cg
import config as cfg


class DatasetAdapter(Dataset, ExampleMaker):
    def __init__(self, data_reader, dataset_cfg, split,
                 feat_scales=cfg.ModelOutput.FEATURE_SCALES,
                 anchors_pixel=cfg.Dataloader.ANCHORS_PIXEL,
                 category_names=cfg.Dataloader.CATEGORY_NAMES,
                 max_bbox=cfg.Dataloader.MAX_BBOX_PER_IMAGE,
                 max_lane=cfg.Dataloader.MAX_LANE_PER_IMAGE,
                 max_dontcare=cfg.Dataloader.MAX_DONT_PER_IMAGE

                 ):
        Dataset.__init__(self)
        ExampleMaker.__init__(self, data_reader, dataset_cfg, split, feat_scales, anchors_pixel,
                              category_names, max_bbox, max_lane, max_dontcare)

    def __len__(self):
        return self.data_reader.num_frames()

    def __getitem__(self, index):
        return self.get_example(index)


class DatasetReader:
    def __init__(self, ds_name, data_path, split, shuffle=False, batch_size=cfg.Train.BATCH_SIZE, epochs=1):
        self.data_reader = self.reader_factory(ds_name, data_path, split)
        self.data_path = data_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs

    def get_total_frames(self):
        return len(self.data_reader)

    def get_dataset(self):
        data_loader = DataLoader(dataset=self.data_reader, shuffle=self.shuffle, batch_size=self.batch_size,
                                 drop_last=True, num_workers=4)
        return data_loader

    def reader_factory(self, ds_name, data_path, split):
        if ds_name == "kitti":
            data_reader = KittiReader(data_path, split, cfg.Datasets.Kitti)
        elif ds_name == "kitti_bev":
            data_reader = KittiBevReader(data_path, split, cfg.Datasets.Kitti)
        else:
            data_reader = None
        dataset_cfg = cg.set_dataset_and_get_config(ds_name)
        data_reader = DatasetAdapter(data_reader, dataset_cfg, split)
        return data_reader


def test_read_dataset():
    path = "/media/dolphin/intHDD/kitti_detection/data_object_image_2/training/image_2"
    reader = DatasetReader("kitti", path, "train", False, 2, 1)
    dataset = reader.get_dataset()
    frames = reader.get_total_frames()
    dataset_cfg = cfg.Datasets.Kitti
    print(len(dataset))
    for i, features in enumerate(dataset):
        print('features', type(features))
        # for key, val in features.items():
        image = features["image"].detach().numpy().astype(np.uint8)[0]
        boxes2d = features["bbox2d"].detach().numpy()[0]
        print(image.shape)
        print(boxes2d.shape)
        # boxes_3d = uf.convert_box_format_yxhw_to_tlbr(boxes_3d[:, :4])
        boxed_image = du.draw_boxes(image, boxes2d, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("KITTI", boxed_image)
        key = cv2.waitKey()
        # image_i = image.copy()
        # box_image_3d = du.draw_boxes(image_i, boxes_3d, category_names=None)
        # cv2.imshow('img', box_image)
        # cv2.imshow('img_2', box_image_3d)
        # cv2.waitKey()


if __name__ == '__main__':
    test_read_dataset()
