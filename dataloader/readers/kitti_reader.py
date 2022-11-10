import os.path as op
import numpy as np
from glob import glob
import cv2

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
import utils.util_class as uc
import dataloader.data_util as du
from dataloader.data_util import point_cloud_to_depth_map


class KittiDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        kitti_split = "train"    # if self.split == "train" else "testing"
        return [op.join(self.datapath, "data_object_image_2", kitti_split)]

    def get_drive_name(self, drive_index):
        return f"drive{drive_index:02d}"


class KittiReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)

    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "*.png"))
        frame_names.sort()
        if split == "train":
            frame_names = frame_names[:-500]
        else:
            frame_names = frame_names[-500:]
        print("[KittiReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index, raw_hw_shape=None):
        """
        :return: bounding boxes in 'yxhw' format
        """
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")
        velo_file = image_file.replace("image_2", "velodyne").replace(".png", ".bin")
        cali_file = image_file.replace("image_2", "calib").replace(".png", ".txt")
        calib_data = self.load_calib_data(cali_file)
        instrinsic = calib_data["P0"].copy()[:, :3]
        velo_data = self.load_velo_scan(velo_file)
        point_cloud = self.get_point_cloud(velo_data, calib_data["Tr_velo_to_cam"])
        depth_map = point_cloud_to_depth_map(point_cloud, instrinsic, raw_hw_shape)

        bboxes = []
        categories = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                bbox, category = self.extract_box(line, depth_map)
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)
        if not bboxes:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        bboxes = np.array(bboxes)
        return bboxes, categories

    def extract_box(self, line, depth_map):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]
        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]
        y1 = round(float(raw_label[5]))
        x1 = round(float(raw_label[4]))
        y2 = round(float(raw_label[7]))
        x2 = round(float(raw_label[6]))
        depth = depth_map[y1:y2, x1:x2]
        dist_value = depth[np.where(depth > 0)]
        if dist_value.size == 0:
            dist_value = 0
        else:
            dist_value = np.quantile(dist_value, self.dataset_cfg.DIST_QUANTILE)
        bbox = np.array([(y1+y2)/2, (x1+x2)/2, y2-y1, x2-x1, 1, dist_value], dtype=np.int32)
        return bbox, category_name

    def load_calib_data(self, file):
        calib_dict = {}
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                new_line = []
                line = line.split(" ")
                if len(line) == 1:
                    pass
                else:
                    line[0] = line[0].rstrip(":")
                    line[-1] = line[-1].rstrip("\n")
                    for a in line[1:]:
                        new_line.append(float(a))
                    calib_dict[line[0]] = new_line
        calib_dict["Tr_velo_to_cam"] = np.reshape(np.array(calib_dict["Tr_velo_to_cam"]), (3, 4))
        calib_dict["P0"] = np.reshape(np.array(calib_dict["P0"]), (3, 4))
        return calib_dict

    def load_velo_scan(self, file):
        """Load and parse a velodyne binary file."""
        scan = np.fromfile(file, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def get_point_cloud(self, velo_data, T2cam):
        velo_data[:, 3] = 1
        velo_in_camera = np.dot(T2cam, velo_data.T)
        velo_in_camera = velo_in_camera[:3].T
        # remove all velodyne points behind image plane
        velo_in_camera = velo_in_camera[velo_in_camera[:, 2] > 0]
        return velo_in_camera

# ==================================================
import config as cfg


def test_kitti_reader():
    print("===== start test_kitti_reader")
    dataset_cfg = cfg.Datasets.Kitti
    drive_mngr = KittiDriveManager(dataset_cfg.PATH, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = KittiReader(drive_paths[0], "train", dataset_cfg)
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes = reader.get_bboxes(i)
        print(f"frame {i}, bboxes:\n", bboxes)
        boxed_image = du.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("kitti", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_kitti_reader passed")


if __name__ == "__main__":
    test_kitti_reader()
