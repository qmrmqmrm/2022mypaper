import os
import numpy as np
import os.path as op
import cv2

import config as cfg
import utils.tflow.util_function as uf


class SavePred:
    def __init__(self, result_path):
        self.result_path = result_path
        if not op.isdir(result_path):
            os.makedirs(result_path, exist_ok=True)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["lane"])}
        self.crop_tlbr = cfg.Datasets.DATASET_CONFIG.CROP_TLBR
        self.y_axis = np.arange(600, 200, -10)
        self.image_files = self.init_drive(cfg.Datasets.DATASET_CONFIG.PATH, "test")
        self.image_files.sort()

    def init_drive(self, drive_path, split):
        testset_file = op.join(drive_path, "list", f'{split}.txt')
        frame_names = self.push_list(drive_path, testset_file)
        frame_names.sort()
        print("[CULaneReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def push_list(self, drive_path, testset_file):
        test_list = []
        with open(testset_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[1:]
                target_file = op.join(drive_path, line).strip('\n')
                test_list.append(target_file)
        return test_list

    def __call__(self, step, grtr, pred):
        batch, _, __ = pred["inst_lane"]["lane_fpoints"].shape
        for i in range(batch):

            image_file = self.image_files[step * batch + i]
            org_image = cv2.imread(image_file)
            im_shape = grtr["image"].shape
            image_file = image_file.split('/')[-3:]
            file_dir = op.join(self.result_path, image_file[0], image_file[1])
            if not op.isdir(file_dir):
                os.makedirs(file_dir)
            filename = op.join(file_dir, image_file[2].replace("jpg", "lines.txt"))
            lane = self.extract_valid_data(pred["inst_lane"], i, "lane_centerness")
            # lane = self.lane_restore(lane)
            # bbox_3d = self.extract_valid_data(pred["inst3d"], i, "category")
            # tlbr_bboxes = uf.convert_box_format_yxhw_to_tlbr(bbox_2d["yxhw"])
            file = open(os.path.join(filename), 'w')
            text_to_write = ''
            for n in range(lane["lane_fpoints"].shape[0]):
                fpoints = lane["lane_fpoints"][n].reshape(-1, 2) * np.array(im_shape[:2]) + \
                          np.array([self.crop_tlbr[0], self.crop_tlbr[1]])

                xys = list()
                for index in range(len(fpoints) - 1):
                    alpha = (fpoints[index + 1, 0] - fpoints[index, 0]) / (fpoints[index + 1, 1] - fpoints[index, 1]+1e-10)
                    beta = fpoints[index, 0] - alpha * fpoints[index, 1]
                    mask = (self.y_axis < fpoints[index, 0]) * (self.y_axis > fpoints[index + 1, 0])
                    y = self.y_axis[mask]
                    x = (y - beta) / alpha
                    xy = np.stack([x, y], axis=-1)
                    xys.append(xy)
                xys = np.concatenate(xys, axis=0).flatten()
                xys_str = ""
                for entry in xys:
                    xys_str += f"{entry} "
                text_to_write = text_to_write + xys_str + "\n"

            file.write(text_to_write)
            file.close()


    def extract_valid_data(self, inst_data, i, mask_key):
        """
        remove zero padding from bboxes
        """
        valid_data = {}
        valid_mask = (inst_data[mask_key][i] > 0).flatten()
        for key, data in inst_data.items():
            valid_data[key] = data[i][valid_mask]
        return valid_data
    #
    # def lane_restore(self, lane, origin_shape, crop_shape, ):
    #     fpoint = lane["lane_fpoints"]
