import numpy as np
import cv2

import utils.util_class as uc
import utils.framework.util_function as uf


class PreprocessBase:
    def __call__(self, example):
        """
        :param example: source example
        :return: preprocessed example
        """
        raise NotImplementedError()


class ExamplePreprocess(PreprocessBase):
    def __init__(self, target_hw, dataset_cfg, max_bbox, max_lane, max_dontcare, min_pix, category_names):
        self.preprocess = [ExampleCropper(target_hw, dataset_cfg.INCLUDE_LANE, dataset_cfg.CROP_TLBR),
                           ExampleResizer(target_hw, dataset_cfg.INCLUDE_LANE),   # box in pixel scale
                           ExampleMinPixel(min_pix),
                           ExampleBoxScaler(),                                    # box in (0~1) scale
                           ExampleZeroPadBbox(max_bbox),
                           ExampleZeroPadDontCare(max_dontcare),
                           ]
        if dataset_cfg.INCLUDE_LANE:
            self.preprocess.append(ExampleLaneScaler())                           # lane in (0~1) scale
            self.preprocess.append(ExampleLaneParams(category_names))
            self.preprocess.append(ExampleZeroPadLane(max_lane))

    def __call__(self, example):
        for process in self.preprocess:
            example = process(example)
        return example


class ExampleCropper(PreprocessBase):
    """
    crop image for aspect ratio to be the same with that of target
    adjust boxes to be consistent with the cropped image
    """
    def __init__(self, target_hw, include_lane, crop_offset=None):
        # target image aspect ratio: width / height
        self.target_hw_ratio = target_hw[1] / target_hw[0]
        # crop offset: dy1, dx1, dy2, dx2 (top, left, bottom, right)
        self.crop_offset = [0, 0, 0, 0] if crop_offset is None else crop_offset
        self.include_lane = include_lane

    def __call__(self, example: dict):
        source_hw = example["image"].shape[:2]
        crop_tlbr = self.find_crop_range(source_hw)
        example["image"] = self.crop_image(example["image"], crop_tlbr)
        cropped_hw = example["image"].shape[:2]
        example["bbox2d"] = self.crop_bboxes(example["bbox2d"], crop_tlbr, cropped_hw)
        if self.include_lane:
            example["lane_points"] = self.crop_lane_points(example["lane_points"], crop_tlbr)
        return example

    def find_crop_range(self, src_hw):                      # example:
        src_hw = np.array(src_hw, dtype=np.float32)         # [220, 540]
        offset = np.array(self.crop_offset, dtype=np.int32) # [10, 20, 10, 20]
        src_crop_hw = src_hw - (offset[:2] + offset[2:])    # [200, 500]
        src_hw_ratio = src_crop_hw[1] / src_crop_hw[0]      # 2.5
        dst_hw_ratio = self.target_hw_ratio                 # 2
        if dst_hw_ratio < src_hw_ratio:                     # crop x-axis, dst_hw=[200, 400]
            dst_hw = np.array([src_hw[0], src_hw[0] * dst_hw_ratio], dtype=np.int32)
        else:
            dst_hw = np.array([src_hw[1] / dst_hw_ratio, src_hw[1]], dtype=np.int32)
        # crop with fixed center, ([200, 500]-[200, 400])/2 = [0, 50]
        crop_yx = ((src_crop_hw - dst_hw) // 2).astype(np.int32)
        # crop top left bottom right, [10, 20, 10, 20] + [0, 50, 0, 50] = [10, 70, 10, 70]
        crop_tlbr = offset + np.concatenate([crop_yx, crop_yx], axis=0)
        return crop_tlbr

    def crop_image(self, image, crop_tlbr):
        if crop_tlbr[0] > 0:
            image = image[crop_tlbr[0]:]
        if crop_tlbr[2] > 0:
            image = image[:-crop_tlbr[2]]
        if crop_tlbr[1] > 0:
            image = image[:, crop_tlbr[1]:]
        if crop_tlbr[3] > 0:
            image = image[:, :-crop_tlbr[3]]
        return image

    def crop_bboxes(self, bboxes, crop_tlbr, cropped_hw):
        # move image origin
        bboxes[:, 0] = bboxes[:, 0] - crop_tlbr[0]
        bboxes[:, 1] = bboxes[:, 1] - crop_tlbr[1]
        # filter boxes with centers outside image
        inside = (bboxes[:, 0] >= 0) & (bboxes[:, 0] < cropped_hw[0]) & \
                 (bboxes[:, 1] >= 0) & (bboxes[:, 1] < cropped_hw[1])
        bboxes = bboxes[inside]
        if bboxes.size == 0:
            raise uc.MyExceptionToCatch("[get_2d_box] empty boxes")
        # clip into image range
        bboxes = uf.convert_box_format_yxhw_to_tlbr(bboxes)
        bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], cropped_hw[0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], cropped_hw[1])
        bboxes = uf.convert_box_format_tlbr_to_yxhw(bboxes)
        return bboxes

    def crop_lane_points(self, lane_points, crop_tlbr):
        new_lanes = []
        crop_yx = np.array([crop_tlbr[:2]], dtype=np.float32)
        if lane_points is None:
            return None
        for lane_point in lane_points:
            lane_point = lane_point - crop_yx
            new_lanes.append(lane_point)
        return new_lanes


class ExampleResizer(PreprocessBase):
    def __init__(self, target_hw, include_lane):
        self.target_hw = np.array(target_hw, dtype=np.float32)
        self.include_lane = include_lane

    def __call__(self, example):
        source_hw = np.array(example["image"].shape[:2], dtype=np.float32)
        resize_ratio = self.target_hw[0] / source_hw[0]
        assert np.isclose(self.target_hw[0] / source_hw[0], self.target_hw[1] / source_hw[1], atol=0.001)
        # resize image
        image = cv2.resize(example["image"],
                           (self.target_hw[1].astype(np.int32), self.target_hw[0].astype(np.int32)))  # (256, 832)
        bboxes = example["bbox2d"].astype(np.float32)
        # rescale yxhw
        bboxes[:, :4] *= resize_ratio
        example["image"] = image
        example["bbox2d"] = bboxes
        if self.include_lane:
            if example["lane_points"] is not None:
                example["lane_points"] = [lane * resize_ratio for lane in example["lane_points"]]
        return example


class ExampleBoxScaler(PreprocessBase):
    """
    scale bounding boxes into (0~1)
    """
    def __call__(self, example):
        height, width = example["image"].shape[:2]
        bboxes = example["bbox2d"].astype(np.float32)
        bboxes[:, :4] /= np.array([height, width, height, width])
        example["bbox2d"] = bboxes

        dc_boxes = example["dontcare"].astype(np.float32)
        dc_boxes[:, :4] /= np.array([height, width, height, width])
        example["dontcare"] = dc_boxes
        return example


class ExampleLaneScaler(PreprocessBase):
    """
    scale bounding boxes into (0~1)
    """
    def __call__(self, example):
        height, width = example["image"].shape[:2]
        if example["lane_points"] is None:
            return example
        lane_points = example["lane_points"]
        for lane_point in lane_points:
            lane_point /= np.array([[height, width]])
        example["lane_points"] = lane_points
        return example


class ExampleCategoryRemapper(PreprocessBase):
    INVALID_CATEGORY = -1

    def __init__(self, src_categories, src_renamer, dst_categories):
        self.category_remap = self.make_category_remap(src_categories, src_renamer, dst_categories)

    def make_category_remap(self, src_categories, src_renamer, dst_categories):
        # replace src_categories by src_renamer
        renamed_categories = [src_renamer[categ] if categ in src_renamer else categ for categ in src_categories]
        remap = dict()
        for si, categ in enumerate(renamed_categories):
            if categ in dst_categories:
                # category index mapping between renamed_categories and dst_categories
                remap[si] = dst_categories.index(categ)
            else:
                remap[si] = self.INVALID_CATEGORY
        print("[make_category_remap] remap=", remap)
        return remap

    def __call__(self, example):
        old_categs = example["bboxes"][:, 4]
        new_categs = old_categs.copy()
        # replace category indices by category_remap
        for key, val in self.category_remap.items():
            new_categs[old_categs == key] = val
        example["bbox2d"][:, 4] = new_categs
        # filter out invalid category
        example["bbox2d"] = example["bbox2d"][new_categs != self.INVALID_CATEGORY, :]
        return example


class ExampleZeroPadBbox(PreprocessBase):
    def __init__(self, max_bbox):
        self.max_bbox = max_bbox

    def __call__(self, example):
        bboxes = example["bbox2d"]
        if bboxes.shape[0] < self.max_bbox:
            new_bboxes = np.zeros((self.max_bbox, bboxes.shape[1]), dtype=np.float32)
            new_bboxes[:bboxes.shape[0]] = bboxes
            example["bbox2d"] = new_bboxes
        return example


class ExampleZeroPadLane(PreprocessBase):
    def __init__(self, max_lane):
        self._max_lane = max_lane

    def __call__(self, example):
        new_lanes = np.zeros((self._max_lane, 4), dtype=np.float32)
        if example["lanes"] is None:
            example["lanes"] = new_lanes
            return example
        else:
            lanes = example["lanes"][:self._max_lane, :]
            new_lanes[:lanes.shape[0], :] = lanes
            example["lanes"] = new_lanes
            return example


class ExampleZeroPadDontCare(PreprocessBase):
    def __init__(self, max_dontcare):
        self.max_dontcare = max_dontcare

    def __call__(self, example):
        dontcare = example["dontcare"]
        if dontcare.shape[0] < self.max_dontcare:
            new_dontcare = np.zeros((self.max_dontcare, dontcare.shape[1]), dtype=np.float32)
            new_dontcare[:dontcare.shape[0]] = dontcare
            example["dontcare"] = new_dontcare
        return example


class ExampleMinPixel(PreprocessBase):
    def __init__(self, min_pixels):
        self.min_pixels = list(min_pixels.values())

    def __call__(self, example):
        bboxes = example["bbox2d"]
        area = bboxes[:, 2] * bboxes[:, 3]
        min_pixels = np.array([self.min_pixels[int(category)] for category in bboxes[:, 5]])
        dontcare = bboxes[area < min_pixels]
        example["bbox2d"] = bboxes[area >= min_pixels]
        example["dontcare"] = np.concatenate([example["dontcare"], dontcare], axis=0)
        return example


class ExampleLaneParams(PreprocessBase):
    def __init__(self, category_names):
        self.category_names = category_names

    def __call__(self, example):
        lane_points = example["lane_points"]
        lane_types = example["lane_types"]
        if lane_points is None:
            example["lanes"] = None
            del example["lane_types"]
            return example
        else:
            lane_params = []
            for lane_point, lane_type in zip(lane_points, lane_types):
                try:
                    slope, intercept_x = self.polyline(lane_point)
                except np.linalg.LinAlgError as e:
                    print("\n[LinAlgError]:", lane_point)
                    continue
                lane_params.append([slope, intercept_x, 1, self.category_names["lane"].index(lane_type)])
                # start_points = lane_point[0].tolist()
                # end_points = lane_point[-1].tolist()
                # lane_params.append([slope, intercept_x, start_points[0], start_points[1], end_points[0], end_points[1], 1, self.category_names["lane"].index(lane_type)])
            if not lane_params:
                example["lanes"] = None
            else:
                example["lanes"] = np.array(lane_params, dtype=np.float32)
            del example["lane_types"]
            return example

    def polyline(self, lane_point):
        x = lane_point[:, 1]  # x = ay + b
        A = np.concatenate([lane_point[:, 0:1], np.ones((lane_point.shape[0], 1), dtype=np.float32)], axis=-1)
        An = A.T @ A
        bn = A.T @ x
        z = np.linalg.solve(An, bn)  # Az = x
        return z[0], z[1]
