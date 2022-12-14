import numpy as np
import os
import cv2

import dataloader.data_util as tu

import dataloader.preprocess as pr
import config as cfg


class ExampleMaker:
    def __init__(self, data_reader, dataset_cfg, split, tfr_drive_path,
                 feat_scales=cfg.ModelOutput.FEATURE_SCALES,
                 category_names=cfg.Dataloader.CATEGORY_NAMES,
                 max_lane=cfg.Dataloader.MAX_LANE_PER_IMAGE,
                 max_lpoints=cfg.Dataloader.MAX_POINTS_PER_LANE):
        self.data_reader = data_reader
        self.feat_scales = feat_scales
        self.category_names = category_names
        self.include_lane = dataset_cfg.INCLUDE_LANE
        self.tfr_drive_path = tfr_drive_path
        self.save_image = True
        self.image_mean = 0
        self.preprocess_example = pr.ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                                       dataset_cfg=dataset_cfg,
                                                       max_lane=max_lane,
                                                       max_lpoints=max_lpoints,
                                                       lane_min_pix=cfg.Dataloader.LANE_MIN_PIX[split],
                                                       category_names=category_names
                                                       )

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        # example["image_file"] = self.data_reader.frame_names[index]
        raw_hw_shape = example["image"].shape[:2]
        # bboxes, categories = self.data_reader.get_bboxes(index, raw_hw_shape)
        # example["inst_box"], example["inst_dc"] = self.merge_box_and_category(bboxes, categories)
        example["lanes_point"], example["lanes_type"] = self.data_reader.get_raw_lane_pts(index, raw_hw_shape)
        example = self.preprocess_example(example)
        # if index % 10 == 5:
        self.show_example(example, index)
        return example

    def merge_lane_and_category(self, lanes, categories):
        reamapped_categories = []
        for index, category_str in enumerate(categories):
            if category_str in self.category_names["lane"]:
                major_index = self.category_names["lane"].index(category_str)
            elif category_str in self.category_names["dont_lane"]:
                major_index = -1
            else:
                major_index = -2
            reamapped_categories.append(major_index)
        return lanes, reamapped_categories

    def show_example(self, example, index):
        image_dir = os.path.join(self.tfr_drive_path, "test_image")
        os.makedirs(image_dir, exist_ok=True)

        image = example["image"]
        # if np.sum(example['inst_lane'][:, 10]) >= 2:
        #
        #     if self.del_night(example):
        #         self.save_txt(self.data_reader.frame_names[index])
        # else:
        #     pass

        image = tu.draw_lanes(image, example["lanes_point"], example["inst_lane"], self.category_names)
        cv2.imshow("image with feature bboxes", image)
        cv2.waitKey(1)
        if self.save_image:
            image_file = os.path.join(self.tfr_drive_path, "test_image", str(index) + ".png")
            cv2.imwrite(image_file, image)

    def save_txt(self, data):
        save_txt_file_name = os.path.join(self.tfr_drive_path, f'new_train.txt')

        if not os.path.exists(save_txt_file_name):

            with open(save_txt_file_name, "w") as f:
                f.write(data)
        else:
            with open(save_txt_file_name, "r") as f:
                line_list = f.readlines()

                line_list.append("\n" + data)
            with open(save_txt_file_name, "w") as f:
                for line in line_list:
                    f.write(line)

    def del_night(self, example):
        image = example["image"]
        image_mean = np.mean(image[:, :, :])
        if 50 < image_mean < 120:
            if image_mean > self.image_mean:
                self.image_mean = image_mean
            return True
        elif 120 < image_mean < 140:
            if image_mean > self.image_mean:
                self.image_mean = image_mean
            cv2.imshow("image with feature bboxes", image)
            cv2.waitKey(10)
        else:
            if image_mean > self.image_mean:
                self.image_mean = image_mean
            cv2.imshow("image with feature bboxes", image)
            key = cv2.waitKey()
            if key == ord('s'):
                return True

        return False