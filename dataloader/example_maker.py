import numpy as np
import cv2


import dataloader.framework.data_util as tu

import dataloader.preprocess as pr
import config as cfg


class ExampleMaker:
    def __init__(self, data_reader, dataset_cfg, split,
                 feat_scales=cfg.ModelOutput.FEATURE_SCALES,
                 anchors_pixel=cfg.Dataloader.ANCHORS_PIXEL,
                 category_names=cfg.Dataloader.CATEGORY_NAMES,
                 max_bbox=cfg.Dataloader.MAX_BBOX_PER_IMAGE,
                 max_lane=cfg.Dataloader.MAX_LANE_PER_IMAGE,
                 max_dontcare=cfg.Dataloader.MAX_DONT_PER_IMAGE):
        self.data_reader = data_reader
        self.feat_scales = feat_scales
        self.category_names = category_names
        self.anchors_ratio = anchors_pixel / np.array([dataset_cfg.INPUT_RESOLUTION])
        self.anchors_lane = cfg.Dataloader.ANCHORS_LANE
        self.lane_detect_rows = cfg.Dataloader.LANE_DETECT_ROWS
        self.include_lane = dataset_cfg.INCLUDE_LANE
        self.max_bbox = max_bbox
        self.preprocess_example = pr.ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                                       dataset_cfg=dataset_cfg,
                                                       max_bbox=max_bbox,
                                                       max_lane=max_lane,
                                                       max_dontcare=max_dontcare,
                                                       min_pix=cfg.Dataloader.MIN_PIX[split],
                                                       category_names=category_names
                                                       )

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        raw_hw_shape = example["image"].shape[:2]
        box2d, categories = self.data_reader.get_2d_box(index, raw_hw_shape)
        example["bbox2d"], example["dontcare"] = self.merge_box_and_category(box2d, categories)
        if self.include_lane:
            example["lane_points"], example["lane_types"] = self.data_reader.get_raw_lane_pts(index, raw_hw_shape)

        example = self.preprocess_example(example)

        if self.include_lane:
            del example["lane_points"]
        if index % 100 == 10:
            self.show_example(example)
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

    def show_example(self, example):
        image = tu.draw_boxes(example["image"], example["bboxes"], self.category_names)
        if self.include_lane:
            image = tu.draw_lanes(image, example["lanes"], self.category_names)

        cv2.imshow("image with feature bboxes", image)
        cv2.waitKey(100)

