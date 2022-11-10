import os.path as op
import numpy as np
import zipfile
from glob import glob
from PIL import Image
from PIL import ImageColor
import json
import cv2

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
import utils.util_class as uc
import dataloader.data_util as du


class CityDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        dirlist = glob(op.join(self.datapath, 'leftImg8bit_trainvaltest', 'leftImg8bit', self.split, '*'))
        dirlist = [directory for directory in dirlist if op.isdir(op.join(directory))]
        return dirlist

    def get_drive_name(self, drive_index):
        drive_path = self.drive_paths[drive_index]
        print("drive_path", drive_path)
        drive_name = op.basename(drive_path)
        return drive_name


class CityReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "*.png"))
        print(frame_names)
        frame_names.sort()
        print("[CityReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index, raw_hw_shape=None):
        """
        :return: bounding boxes in 'yxhw' format
        """
        image_file = self.frame_names[index]
        label_file = image_file.replace("leftImg8bit_trainvaltest", "gtFine_trainvaltest")\
            .replace("/leftImg8bit", "/gtFine").replace("_leftImg8bit.png", "_gtFine_polygons.json")
        disparity_file = image_file.replace("leftImg8bit_trainvaltest", "disparity_trainvaltest")\
            .replace("/leftImg8bit", "/disparity").replace("_leftImg8bit", "_disparity")
        camera_file = image_file.replace("leftImg8bit_trainvaltest", "camera_trainvaltest")\
            .replace("/leftImg8bit", "/camera").replace("_leftImg8bit.png", "_camera.json")
        depth_map = self.get_depth(disparity_file, camera_file)
        bboxes = []
        categories = []
        with open(label_file, 'r') as f:
            label_data = json.load(f)
            for obj in label_data["objects"]:
                bbox, category = self.extract_box(obj, depth_map)
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)

        if not bboxes:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        bboxes = np.array(bboxes)
        return bboxes, categories

    def get_depth(self, disparity_file, camera_file):
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)
            disparity = Image.open(disparity_file)

            disparity = np.array(disparity, np.uint16).astype(np.float32)
            disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256.    # why? question
            raw_shape = disparity.shape

            baseline = camera_data["extrinsic"]["baseline"]
            fx = camera_data["intrinsic"]["fx"]

            depth = np.zeros(disparity.shape, dtype=np.float32)
            depth[disparity > 0] = (fx * baseline) / disparity[disparity > 0]
        return depth

    def extract_box(self, objects, depth_map):
        polygon = np.array(objects["polygon"])
        category_name = objects["label"]
        if category_name not in self.dataset_cfg.CATEGORY_REMAP:
            return None, None
        category_index = self.dataset_cfg.CATEGORIES_TO_USE.index(category_name)
        category_renamed = self.dataset_cfg.CATEGORY_REMAP[category_name]

        x = polygon[:, 0]
        y = polygon[:, 1]
        x1 = np.min(x)
        x2 = np.max(x)
        y1 = np.min(y)
        y2 = np.max(y)
        depth = depth_map[y1:y2, x1:x2]
        dist_value = depth[np.where(depth > 0)]
        if dist_value.size == 0:
            dist_value = 0
        else:
            dist_value = np.quantile(dist_value, self.dataset_cfg.DIST_QUANTILE)

        bbox = np.array([(y1+y2)/2, (x1+x2)/2, y2-y1, x2-x1, category_index, dist_value], dtype=np.float32)
        return bbox, category_renamed

# =====================
import config as cfg
import shutil

def test_city_reader():
    print("===== start test_city_reader")
    dataset_cfg = cfg.Datasets.City
    drive_mngr = CityDriveManager(dataset_cfg.PATH, "val")
    drive_path = drive_mngr.get_drive_paths()
    reader = CityReader(drive_path[0], "val", dataset_cfg)
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes = reader.get_bboxes(i, image.shape)
        boxed_image = du.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("City", boxed_image)
        cv2.waitKey()
        print(f"frame {i}, bboxes:\n", bboxes)


if __name__ == "__main__":
    test_city_reader()
