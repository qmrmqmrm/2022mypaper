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
        self.save_image = False
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
        os.makedirs(image_dir,exist_ok=True)

        image = example["image"]

        image = tu.draw_lanes(image, example["lanes_point"], example["inst_lane"], self.category_names)
        cv2.imshow("image with feature bboxes", image)
        cv2.waitKey(1)
        if self.save_image:
            image_file = os.path.join(self.tfr_drive_path, "test_image", str(index) + ".png")
            cv2.imwrite(image_file, image)
