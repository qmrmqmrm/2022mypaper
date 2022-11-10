import os.path as op
import os
import numpy as np
from glob import glob
import cv2
import json

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
import dataloader.data_util as du
import utils.util_class as uc


class UplusDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)
        self.split = "val" if self.split == "test" else self.split

    def list_drive_paths(self):
        dirlist = glob(op.join(self.datapath, self.split, '*'))
        dirlist = [directory for directory in dirlist if op.isdir(op.join(directory, "image"))]
        return dirlist

    def get_drive_name(self, drive_index):
        drive_path = self.drive_paths[drive_index]
        print("drive_path", drive_path)
        drive_name = op.basename(drive_path)
        return drive_name


class UplusReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)
        self.split = split

    """
    Public methods used outside this class
    """

    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "image", "*.jpg"))
        frame_names.sort()
        print("[UplusReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_image(self, index):
        # print("\nframe name : ", self.frame_names[index])
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index, raw_hw_shape=None):
        """
        :return: bounding boxes in 'yxhw' format
        """
        image_file = self.frame_names[index]
        label_file = image_file.replace("image", "label").replace(".jpg", ".csv")
        bboxes = []
        categories = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[1:]):
                bbox, category = self.extract_box(line, raw_hw_shape)
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)

        if not bboxes:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        bboxes = np.array(bboxes)
        return bboxes, categories

    def extract_box(self, line, raw_hw_shape):
        raw_label = line.strip("\n").split(",")
        category_name, x1, y1, x2, y2, depth = raw_label
        depth = float(depth)
        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        # print("category : ", category_name, self.dataset_cfg.CATEGORIES_TO_USE)
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]
        y1 = round(float(y1) * raw_hw_shape[0])
        x1 = round(float(x1) * raw_hw_shape[1])
        y2 = round(float(y2) * raw_hw_shape[0])
        x2 = round(float(x2) * raw_hw_shape[1])
        bbox = np.array([(y1 + y2) / 2, (x1 + x2) / 2, y2 - y1, x2 - x1, 1, depth], dtype=np.float32)
        return bbox, category_name

    def get_raw_lane_pts(self, index, raw_hw_shape):
        return None, None



# ==================================================
import config as cfg
import shutil


def move_labels():
    dataset_cfg = cfg.Datasets.Uplus
    drive_mngr = UplusDriveManager(dataset_cfg.PATH, "val")
    drive_paths = drive_mngr.get_drive_paths()

    for drive_path in drive_paths:
        print("drive path:", drive_path)
        reader = UplusReader(drive_path, "val", dataset_cfg)
        label_name = reader.frame_names[0].replace("image", "label").replace(".jpg", ".csv")
        label_dir_path = os.path.dirname(label_name)
        os.makedirs(label_dir_path, exist_ok=True)
        lane_name = reader.frame_names[0].replace("image", "lane").replace(".jpg", ".csv")
        lane_dir_path = os.path.dirname(lane_name)
        os.makedirs(lane_dir_path, exist_ok=True)

        for frame_name in reader.frame_names:
            # move labels
            val_label_name = frame_name.replace("image", "label").replace(".jpg", ".csv")
            train_label_name = val_label_name.replace("/val/", "/train/")
            print(f"move label: {train_label_name[-40:]} --> {val_label_name[-40:]}")
            shutil.move(train_label_name, val_label_name)
            # move lanes
            val_lane_name = frame_name.replace("image", "lane").replace(".jpg", ".json")
            train_lane_name = val_lane_name.replace("/val/", "/train/")
            print(f"move lane : {train_lane_name[-40:]} --> {val_lane_name[-40:]}")
            shutil.move(train_lane_name, val_lane_name)


def test_uplus_reader():
    print("===== start test_uplus_reader")
    dataset_cfg = cfg.Datasets.Uplus
    drive_mngr = UplusDriveManager(dataset_cfg.PATH, "val")
    drive_paths = drive_mngr.get_drive_paths()
    reader = UplusReader(drive_paths[0], "train", dataset_cfg)
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes = reader.get_bboxes(i, image.shape)
        print(f"frame {i}, bboxes:\n", bboxes)
        boxed_image = du.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("uplus", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_uplus_reader passed")


if __name__ == "__main__":
    # test_uplus_reader()
    move_labels()