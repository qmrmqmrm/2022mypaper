import os.path as op
import numpy as np
from glob import glob
import re
import cv2

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
import utils.util_class as uc
import dataloader.data_util as du
from dataloader.data_util import point_cloud_to_depth_map


class CULaneDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        return [self.datapath]

    def get_drive_name(self, drive_index):
        return f"drive{drive_index:02d}"

    def push_list(self, testset_file):
        test_list = []
        with open(testset_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[1:]
                target_file = op.join(self.datapath, line).strip('\n')
                test_list.append(target_file)
        return test_list


class CULaneReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)

    def init_drive(self, drive_path, split):
        testset_file = op.join(drive_path, "list", f'new_{split}.txt')
        frame_names = self.push_list(drive_path, testset_file)
        frame_names.sort()
        print("[CULaneReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def push_list(self, drive_path, testset_file):
        test_list = []
        with open(testset_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # line = line[1:]
                target_file = "/".join(line.strip('\n').split('/')[-3])
                target_file = op.join(drive_path, target_file)
                test_list.append(target_file)
        return test_list

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_raw_lane_pts(self, index, raw_hw_shape):
        image_file = self.frame_names[index]
        label_file = image_file.replace("image", "label").replace(".jpg", ".lines.txt")
        with open(label_file, 'r') as f:
            lane_lines = f.readlines()
            lanes_point = []
            lanes_label = []
            if len(lane_lines) != 0:
                for lane in lane_lines:
                    lanes_point.append(self.extract_lane(lane))
                if len(lanes_point) < 4:
                    a = 1
                lanes_label = self.get_lane_label(lanes_point)

            else:
                return lanes_point, lanes_label
        return lanes_point, lanes_label

    def extract_lane(self, lane):
        lane_points = lane.strip(" \n").split(" ")
        # lane_points = [float(i) for i in lane_points]
        lane_xy = np.array(lane_points, dtype=np.float32).reshape(-1, 2)
        # NOTE: labeler saves coords as (x, y) form, change form into (y, x)
        lane_point = lane_xy[:, [1, 0]]
        return lane_point

    def get_lane_label(self, lanes_points):
        sprit_index = len(lanes_points)
        find_k_flag = False
        if len(lanes_points) > 2:
            for index in range(len(lanes_points) - 1):
                grad_1 = self.get_grad(lanes_points[index])
                grad_2 = self.get_grad(lanes_points[index + 1])
                if grad_1 <= 0 and grad_2 > 0:
                    sprit_index = index + 1
                    find_k_flag = True
                    break
                sprit_index = index
        if not find_k_flag:
            grad_1 = self.get_grad(lanes_points[0])
            if grad_1 > 0:
                sprit_index = 2
        lanes_label = []
        for index in range(len(lanes_points)):
            idx = index + sprit_index - 2
            if idx < 0 or idx > len(lanes_points):
                continue
            else:
                lanes_label.append(index+1)

        return lanes_label

    def get_grad(self, lpoints):
        grad = lpoints[1] - lpoints[0]
        return grad[1] / grad[0]


# ==================================================
import config as cfg


def test_culane_reader():
    print("===== start test_culane_reader")
    dataset_cfg = cfg.Datasets.Culane
    drive_mngr = CULaneDriveManager(dataset_cfg.PATH, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = CULaneReader(drive_paths[0], "train", dataset_cfg)
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes = reader.get_bboxes(i)
        print(f"frame {i}, bboxes:\n", bboxes)
        boxed_image = du.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("culane", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_culane_reader passed")


if __name__ == "__main__":
    test_culane_reader()
