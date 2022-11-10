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
                 max_bbox=cfg.Dataloader.MAX_BBOX_PER_IMAGE,
                 max_lane=cfg.Dataloader.MAX_LANE_PER_IMAGE,
                 max_lpoints=cfg.Dataloader.MAX_POINTS_PER_LANE,
                 max_dontcare=cfg.Dataloader.MAX_DONT_PER_IMAGE):
        self.data_reader = data_reader
        self.feat_scales = feat_scales
        self.category_names = category_names
        self.include_lane = dataset_cfg.INCLUDE_LANE
        self.tfr_drive_path = tfr_drive_path
        self.max_bbox = max_bbox
        self.save_image = False
        self.preprocess_example = pr.ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                                       dataset_cfg=dataset_cfg,
                                                       max_bbox=max_bbox,
                                                       max_lane=max_lane,
                                                       max_lpoints=max_lpoints,
                                                       max_dontcare=max_dontcare,
                                                       min_pix=cfg.Dataloader.MIN_PIX[split],
                                                       lane_min_pix=cfg.Dataloader.LANE_MIN_PIX[split],
                                                       category_names=category_names
                                                       )

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        raw_hw_shape = example["image"].shape[:2]
        bboxes, categories = self.data_reader.get_bboxes(index, raw_hw_shape)
        example["inst_box"], example["inst_dc"] = self.merge_box_and_category(bboxes, categories)
        if self.include_lane:
            lanes_point, lane_categories = self.data_reader.get_raw_lane_pts(index, raw_hw_shape)
            example["lanes_point"], example["lanes_type"] = self.merge_lane_and_category(lanes_point, lane_categories)

        example = self.preprocess_example(example)
        if index % 10 == 5:
            self.show_example(example, index)
        return example

    def extract_bbox(self, example):
        scales = [key for key in example if "feature" in key]
        # merge pred features over scales
        total_features = []
        for scale_name in scales:
            height, width, anchor, channel = example[scale_name].shape
            merged_features = np.reshape(example[scale_name], (height * width * anchor, channel))
            total_features.append(merged_features)

        total_features = np.concatenate(total_features, axis=0)  # (batch, N, dim)
        total_features = total_features[total_features[..., 4] > 0]
        if total_features.size != 0:
            num_box = total_features.shape[0]
            pad_num = np.maximum(self.max_bbox - num_box, 0)
            zero_pad = np.zeros((pad_num, total_features.shape[-1]), dtype=np.float32)
            example["bboxes"] = np.concatenate([total_features[:self.max_bbox, :], zero_pad])
        return example

    def merge_box_and_category(self, bboxes, categories):
        reamapped_categories = []
        for category_str in categories:
            if category_str in self.category_names["major"]:
                major_index = self.category_names["major"].index(category_str)
                minor_index = -1
                speed_index = -1
            elif category_str in self.category_names["sign"]:
                major_index = self.category_names["major"].index("Traffic sign")
                minor_index = self.category_names["sign"].index(category_str)
                speed_index = -1
            elif category_str in self.category_names["mark"]:
                major_index = self.category_names["major"].index("Road mark")
                minor_index = self.category_names["mark"].index(category_str)
                speed_index = -1
            elif category_str in self.category_names["sign_speed"]:
                major_index = self.category_names["major"].index("Traffic sign")
                minor_index = self.category_names["sign"].index("TS_SPEED_LIMIT")
                speed_index = self.category_names["sign_speed"].index(category_str)
            elif category_str in self.category_names["mark_speed"]:
                major_index = self.category_names["major"].index("Road mark")
                minor_index = self.category_names["mark"].index("RM_SPEED_LIMIT")
                speed_index = self.category_names["mark_speed"].index(category_str)
            elif category_str in self.category_names["dont"]:
                major_index = -1
                minor_index = -1
                speed_index = -1
            else:
                major_index = -2
                minor_index = -2
                speed_index = -2
            reamapped_categories.append((major_index, minor_index, speed_index))
        reamapped_categories = np.array(reamapped_categories)
        # bbox: yxhw, obj, major_ctgr, minor_ctgr, speed_index, depth (9)
        bboxes = np.concatenate([bboxes[..., :-1], reamapped_categories, bboxes[..., -1:]], axis=-1)
        dontcare = bboxes[bboxes[..., 5] == -1]
        bboxes = bboxes[bboxes[..., 5] >= 0]
        return bboxes, dontcare

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
        image = tu.draw_boxes(image, example["inst_box"], self.category_names, frame_name=self.data_reader.frame_names[index])

        if self.include_lane:
            image = tu.draw_lanes(image, example["lanes_point"], example["inst_lane"], self.category_names)
        cv2.imshow("image with feature bboxes", image)
        cv2.waitKey(100)
        if self.save_image:
            image_file = os.path.join(self.tfr_drive_path, "test_image", str(index) + ".png")
            cv2.imwrite(image_file, image)
