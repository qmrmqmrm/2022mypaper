import os.path as op
import os
import numpy as np
from glob import glob
import cv2

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
import dataloader.framework.data_util as tu
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
        return cv2.imread(self.frame_names[index])

    def get_2d_box(self, index, raw_hw_shape=None):
        """
        :return: bounding boxes in 'yxhw' format
        """
        image_file = self.frame_names[index]
        label_file = image_file.replace("image", "label").replace(".jpg", ".txt")

        bboxes = []
        categories = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            split_line = lines.index("---\n")
            bbox_lines = lines[:split_line]
            for line in bbox_lines:
                bbox, category = self.extract_box(line)
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)

        if not bboxes:
            raise uc.MyExceptionToCatch("[get_2d_box] empty boxes")
        bboxes = np.array(bboxes)
        return bboxes, categories

    def get_depth_map(self, lidar_file):
        depth = np.load(lidar_file)
        depth = depth["depth"]
        return depth

    def extract_box(self, line):
        raw_label = line.strip("\n").split(",")
        category_name, y1, x1, h, w, depth = raw_label
        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]
        y = int(h) / 2 + int(y1)
        x = int(w) / 2 + int(x1)
        h = int(h)
        w = int(w)
        bbox = np.array([y, x, h, w, 1, depth], dtype=np.float32)
        return bbox, category_name

    def extract_depth(self, data):
        value = data[np.where(data > 0)]
        if value.size == 0:
            return 0
        return np.quantile(value, 0.2)

    def get_raw_lane_pts(self, index, raw_hw_shape):
        image_file = self.frame_names[index]
        label_file = image_file.replace("image", "label").replace(".jpg", ".txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()
            split_line = lines.index("---\n")
            lane_lines = lines[split_line + 1:]
            lane_types = []
            lane_points = []
            if len(lane_lines) != 0:
                for lane in lane_lines:
                    if (lane == "[\n") or (lane == "]") or (lane == "]\n"):
                        continue
                    lane_type, lane_point = self.extract_lane(lane)
                    if lane_type is not None:
                        lane_types.append(lane_type)
                        lane_points.append(lane_point)
            else:
                return None, None
        return lane_points, lane_types

    def extract_lane(self, lane):
        lane = eval(lane)
        if isinstance(lane, tuple):
            lane = lane[0]
        if lane[0] not in self.dataset_cfg.LANE_TYPES:
            return None, None
        lane_type = self.dataset_cfg.LANE_REMAP[lane[0]]
        lane_xy = np.array(lane[1:], dtype=np.float32)
        # NOTE: labeler saves coords as (x, y) form, change form into (y, x)
        lane_point = lane_xy[:, [1, 0]]
        return lane_type, lane_point


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
        bboxes = reader.get_2d_box(i, image.shape)
        print(f"frame {i}, bboxes:\n", bboxes)
        boxed_image = tu.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("uplus", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_uplus_reader passed")


if __name__ == "__main__":
    # test_uplus_reader()
    move_labels()
