import numpy as np
import cv2

import utils.util_class as uc
import utils.framework.util_function as uf
from utils.util_class import MyExceptionToCatch


class PreprocessBase:
    def __call__(self, example):
        """
        :param example: source example
        :return: preprocessed example
        """
        raise NotImplementedError()


class ExamplePreprocess(PreprocessBase):
    def __init__(self, target_hw, dataset_cfg, max_bbox, max_lane, max_lpoints, max_dontcare, min_pix, lane_min_pix, category_names):
        self.preprocess = [ExampleCropper(target_hw, dataset_cfg.INCLUDE_LANE, dataset_cfg.CROP_TLBR),
                           ExampleResizer(target_hw, dataset_cfg.INCLUDE_LANE),   # box in pixel scale
                           ExampleMinPixel(min_pix),
                           ExampleBoxScaler(),                                    # box in (0~1) scale
                           ExampleZeroPadBbox(max_bbox),
                           ExampleZeroPadDontCare(max_dontcare),
                           ]
        if dataset_cfg.INCLUDE_LANE:
            self.preprocess.extend([ExampleLaneParams(category_names, max_lane, max_lpoints, lane_min_pix),
                                    ExampleLaneScaler(),
                                    ExampleZeroPadLane(max_lane, max_lpoints)]
                                   )

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
        example["inst_box"] = self.crop_bboxes(example["inst_box"], crop_tlbr)
        if self.include_lane:
            example = ExampleLaneCropper(crop_tlbr)(example)
        return example

    def find_crop_range(self, src_hw):  # example:
        src_hw = np.array(src_hw, dtype=np.float32)  # [220, 540]
        offset = np.array(self.crop_offset, dtype=np.int32)  # [10, 20, 10, 20]
        src_crop_hw = src_hw - (offset[:2] + offset[2:])  # [200, 500]
        src_hw_ratio = src_crop_hw[1] / src_crop_hw[0]  # 2.5
        dst_hw_ratio = self.target_hw_ratio  # 2
        if dst_hw_ratio < src_hw_ratio:  # crop x-axis, dst_hw=[200, 400]
            dst_hw = np.array([src_crop_hw[0], src_crop_hw[0] * dst_hw_ratio], dtype=np.int32)
        else:
            dst_hw = np.array([src_crop_hw[1] / dst_hw_ratio, src_crop_hw[1]], dtype=np.int32)
        # crop with fixed center, ([200, 500]-[200, 400])/2 = [0, 50]
        addi_crop_yx = ((src_crop_hw - dst_hw) // 2).astype(np.int32)
        # crop top left bottom right, [10, 20, 10, 20] + [0, 50, 0, 50] = [10, 70, 10, 70]
        crop_tlbr = offset + np.concatenate([addi_crop_yx, addi_crop_yx], axis=0)
        # cropped image range, [10, 70, [220, 540]-[10, 70]] = [10, 70, 210, 470]
        crop_tlbr = np.concatenate([crop_tlbr[:2], src_hw - crop_tlbr[2:]])
        return crop_tlbr

    def crop_image(self, image, crop_tlbr):

        image = image[int(crop_tlbr[0]):int(crop_tlbr[2]), int(crop_tlbr[1]):int(crop_tlbr[3]), :]
        return image

    def crop_bboxes(self, bboxes, crop_tlbr):
        crop_hw = crop_tlbr[2:] - crop_tlbr[:2]
        # move image origin
        bboxes[:, :2] = bboxes[:, :2] - crop_tlbr[:2]
        # filter boxes with centers outside image
        inside = (bboxes[:, 0] >= 0) & (bboxes[:, 0] < crop_hw[0]) & \
                 (bboxes[:, 1] >= 0) & (bboxes[:, 1] < crop_hw[1])
        bboxes = bboxes[inside]
        if bboxes.size == 0:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        # clip into image range
        bboxes = uf.convert_box_format_yxhw_to_tlbr(bboxes)
        bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], crop_hw[0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], crop_hw[1])
        bboxes = uf.convert_box_format_tlbr_to_yxhw(bboxes)
        return bboxes


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
        bboxes = example["inst_box"].astype(np.float32)
        # rescale yxhw
        bboxes[:, :4] *= resize_ratio
        example["image"] = image
        example["inst_box"] = bboxes
        if self.include_lane and example["lanes_point"]:
            example["lanes_point"] = [lane * resize_ratio for lane in example["lanes_point"]]
        return example


class ExampleBoxScaler(PreprocessBase):
    """
    scale bounding boxes into (0~1)
    """

    def __call__(self, example):
        height, width = example["image"].shape[:2]
        bboxes = example["inst_box"].astype(np.float32)
        bboxes[:, :4] /= np.array([height, width, height, width])
        example["inst_box"] = bboxes

        dc_boxes = example["inst_dc"].astype(np.float32)
        dc_boxes[:, :4] /= np.array([height, width, height, width])
        example["inst_dc"] = dc_boxes
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
        old_categs = example["inst_box"][:, 4]
        new_categs = old_categs.copy()
        # replace category indices by category_remap
        for key, val in self.category_remap.items():
            new_categs[old_categs == key] = val
        example["inst_box"][:, 4] = new_categs
        # filter out invalid category
        example["inst_box"] = example["inst_box"][new_categs != self.INVALID_CATEGORY, :]
        return example


class ExampleZeroPadBbox(PreprocessBase):
    def __init__(self, max_bbox):
        self.max_bbox = max_bbox

    def __call__(self, example):
        bboxes = example["inst_box"]
        if bboxes.shape[0] < self.max_bbox:
            new_bboxes = np.zeros((self.max_bbox, bboxes.shape[1]), dtype=np.float32)
            new_bboxes[:bboxes.shape[0]] = bboxes
            example["inst_box"] = new_bboxes
        return example


class ExampleZeroPadDontCare(PreprocessBase):
    def __init__(self, max_dontcare):
        self.max_dontcare = max_dontcare

    def __call__(self, example):
        dontcare = example["inst_dc"]
        if dontcare.shape[0] < self.max_dontcare:
            new_dontcare = np.zeros((self.max_dontcare, dontcare.shape[1]), dtype=np.float32)
            new_dontcare[:dontcare.shape[0]] = dontcare
            example["inst_dc"] = new_dontcare
        return example


class ExampleMinPixel(PreprocessBase):
    def __init__(self, min_pixels):
        self.min_pixels = list(min_pixels.values())

    def __call__(self, example):
        bboxes = example["inst_box"]
        diag = np.sqrt(bboxes[:, 2] * bboxes[:, 2] + bboxes[:, 3] * bboxes[:, 3])
        min_pixels = np.array([self.min_pixels[int(category)] for category in bboxes[:, 5]])
        dontcare = bboxes[diag < min_pixels]
        example["inst_box"] = bboxes[diag >= min_pixels]
        example["inst_dc"] = np.concatenate([example["inst_dc"], dontcare], axis=0)
        return example


class ExampleLaneParams(PreprocessBase):
    def __init__(self, category_names, max_lane, max_lpoints, lane_min_pix):
        self.category_names = category_names
        self._max_lpoints = max_lpoints
        self.max_lane = max_lane
        self.lane_points_num = 0
        self.lane_min_pix = list(lane_min_pix.values())

    def __call__(self, example):
        lanes_point = example["lanes_point"]
        lanes_type = example["lanes_type"]
        example["inst_lane"] = None

        if lanes_point:
            new_lanes_point = []
            five_points = []
            categorys = []
            lanes_point_dc = []
            fpoints_dc = []
            cate_dc = []
            for i, lane_points in enumerate(lanes_point):
                points, lane_length = self.get_five_points(lane_points)
                if lane_length > self.lane_min_pix[lanes_type[i]]:
                    new_lanes_point.append(lanes_point[i])
                    five_points.append(points)
                    categorys.append(lanes_type[i])
                else:
                    lanes_point_dc.append(lanes_point[i])
                    fpoints_dc.append(points)
                    cate_dc.append(-1)
            five_points = np.array(five_points)  # (N,5,2)
            five_points = five_points.reshape(-1, 10)  # (N,10)

            fpoints_dc = np.array(fpoints_dc)  # (N,5,2)
            fpoints_dc = fpoints_dc.reshape(-1, 10)  # (N,10)

            lanes_type = np.array(categorys).reshape(-1, 1)
            lanes_type_dc = np.array(cate_dc).reshape(-1, 1)
            # lanes = [y0,x0,y1,x1, ..., y4,x4, 1(centerness), category] (N, 12)
            lanes = np.concatenate([five_points, np.ones((five_points.shape[0], 1)), lanes_type], axis=-1)
            lanes_dc = np.concatenate([fpoints_dc, np.ones((fpoints_dc.shape[0], 1)), lanes_type_dc], axis=-1)
            example["lanes_point"] = new_lanes_point
            example["lanes_point_dc"] = lanes_point_dc
            example["inst_lane"] = lanes
            example["inst_ldc"] = lanes_dc

        del example["lanes_type"]
        return example

    def get_five_points(self, lane_points):
        lane_length = self.get_points_length(lane_points)
        center_point, center_ind = self.get_center_points(lane_points, lane_length / 2, 0, len(lane_points))
        lane_points = np.insert(lane_points, center_ind, center_point, axis=0)
        second_point, second_ind = self.get_center_points(lane_points, lane_length / 4, 0, len(lane_points))
        fourth_point, fourth_ind = self.get_center_points(lane_points, lane_length / 4, center_ind, len(lane_points))
        five_points = np.stack([lane_points[0], second_point, center_point, fourth_point, lane_points[-1]], axis=0)
        return five_points, lane_length

    def get_points_length(self, lane_points):
        points_length = 0
        for i in range(len(lane_points) - 1):
            points_length += np.abs(np.linalg.norm(lane_points[i + 1] - lane_points[i]))
        return points_length

    def get_center_points(self, lane_points, center_length, start_ind, end_ind):
        points_length = 0
        for i in range(start_ind, end_ind - 1):
            length = np.abs(np.linalg.norm(lane_points[i + 1] - lane_points[i]))
            if points_length + length > center_length:
                vector = lane_points[i + 1] - lane_points[i]
                vector = (center_length - points_length) * (vector / np.linalg.norm(vector))
                return lane_points[i] + vector, start_ind + i + 1
            points_length += length


class ExampleLaneCropper(PreprocessBase):
    def __init__(self, crop_tlbr):
        self.crop_tlbr = crop_tlbr

    def __call__(self, example):
        # list( (N,2) )
        if not example["lanes_point"]:
            return example

        crop_lanes_point = []
        crop_lanes_type = []
        for lane_points, lane_type in zip(example["lanes_point"], example["lanes_type"]):
            try:
                lane_points, crop_from_start, crop_from_end, change_from_inside = self.remove_outside_image(lane_points)
                if crop_from_start:
                    lane_points[0] = self.crop_point_into_image(lane_points[0], lane_points[1])
                if crop_from_end:
                    lane_points[-1] = self.crop_point_into_image(lane_points[-1], lane_points[-2])
                if change_from_inside:
                    for index in change_from_inside:
                        lane_points[index] = self.crop_point_into_image(lane_points[index], lane_points[index + 1])
                self.check_points_inside_crop_range(lane_points)
                # translate by [-top, -left]
                lane_points -= self.crop_tlbr[:2]
                crop_lanes_point.append(lane_points)
                crop_lanes_type.append(lane_type)
            except MyExceptionToCatch as me:
                print("[Exception]:", me)

        example["lanes_point"] = crop_lanes_point
        example["lanes_type"] = crop_lanes_type
        return example

    def remove_outside_image(self, lane_points):
        inside = (lane_points[:, 0] > self.crop_tlbr[0]) & \
                 (lane_points[:, 0] < self.crop_tlbr[2]) & \
                 (lane_points[:, 1] > self.crop_tlbr[1]) & \
                 (lane_points[:, 1] < self.crop_tlbr[3])
        if not inside.any():
            raise MyExceptionToCatch("all lane points are out of crop range")
        # example
        # inside = [F F T F T F F]
        # ~inside[:-1] & inside[1:] = [T T F T F T] & [F T F T F F] = [F T F T F F]
        change_from_start = np.where(~inside[:-1] & inside[1:])[0]  # [1, 3]
        if change_from_start.size > 0:
            crop_index = change_from_start[0]
            lane_points = lane_points[crop_index:]  # lane_points[1:]
        change_from_start = bool(change_from_start.size > 0)

        inside = (lane_points[:, 0] > self.crop_tlbr[0]) & \
                 (lane_points[:, 0] < self.crop_tlbr[2]) & \
                 (lane_points[:, 1] > self.crop_tlbr[1]) & \
                 (lane_points[:, 1] < self.crop_tlbr[3])

        # inside = [F T F T F F]
        # inside[:-1] & ~inside[1:] = [F T F T F] & [F T F T T] = [F T F T F]
        change_from_end = np.where(inside[:-1] & ~inside[1:])[0]  # [1, 3]
        if change_from_end.size > 0:
            crop_index = change_from_end[-1] + 2  # [5]
            lane_points = lane_points[:crop_index]  # lane_points[:5]
        change_from_end = bool(change_from_end.size > 0)


        inside = (lane_points[:, 0] > self.crop_tlbr[0]) & \
                 (lane_points[:, 0] < self.crop_tlbr[2]) & \
                 (lane_points[:, 1] > self.crop_tlbr[1]) & \
                 (lane_points[:, 1] < self.crop_tlbr[3])
        # inside = [F T F T F]
        # inside[:-1] & ~inside[1:] = [F T F T F] & [F T F T T] = [F T F T F]
        change_from_inside = np.where(~inside[:-1] & inside[1:])[0]
        if change_from_inside.size > 0:
            change_from_inside = change_from_inside[1:]  # [5]


        return lane_points, change_from_start, change_from_end, change_from_inside

    def crop_point_into_image(self, crop_pt, inside_pt):
        if crop_pt[0] < self.crop_tlbr[0]:
            out_ratio = (self.crop_tlbr[0] - crop_pt[0]) / (inside_pt[0] - crop_pt[0])
            crop_pt = out_ratio * inside_pt + (1 - out_ratio) * crop_pt + 0.1
        if crop_pt[1] < self.crop_tlbr[1]:
            out_ratio = (self.crop_tlbr[1] - crop_pt[1]) / (inside_pt[1] - crop_pt[1])
            crop_pt = out_ratio * inside_pt + (1 - out_ratio) * crop_pt + 0.1
        if crop_pt[0] >= self.crop_tlbr[2]:
            out_ratio = (crop_pt[0] - self.crop_tlbr[2] + 1) / (crop_pt[0] - inside_pt[0])
            crop_pt = out_ratio * inside_pt + (1 - out_ratio) * crop_pt
        if crop_pt[1] >= self.crop_tlbr[3]:
            out_ratio = (crop_pt[1] - self.crop_tlbr[3] + 1) / (crop_pt[1] - inside_pt[1])
            crop_pt = out_ratio * inside_pt + (1 - out_ratio) * crop_pt
        return crop_pt

    def check_points_inside_crop_range(self, crop_lane_points):
        assert (crop_lane_points[:, 0] >= self.crop_tlbr[0]).all(), f"{crop_lane_points}, {self.crop_tlbr}"
        assert (crop_lane_points[:, 0] < self.crop_tlbr[2]).all(), f"{crop_lane_points}, {self.crop_tlbr}"
        assert (crop_lane_points[:, 1] >= self.crop_tlbr[1]).all(),f"{crop_lane_points}, {self.crop_tlbr}"
        assert (crop_lane_points[:, 1] < self.crop_tlbr[3]).all(),f"{crop_lane_points}, {self.crop_tlbr}"


class ExampleLaneScaler(PreprocessBase):
    """
    scale bounding boxes into (0~1)
    """

    def __call__(self, example):
        height, width = example["image"].shape[:2]
        if not example["lanes_point"]:
            return example

        lanes_point = example["lanes_point"]
        for lane in lanes_point:
            lane /= np.array([[height, width]])
        example["lanes_point"] = lanes_point

        lanes_point_dc = example["lanes_point_dc"]
        for lane in lanes_point_dc:
            lane /= np.array([[height, width]])
        example["lanes_point_dc"] = lanes_point_dc

        five_points = example["inst_lane"][:, :10]
        five_points_remainder = example["inst_lane"][:, 10:]
        five_points = five_points.reshape(-1, 5, 2) / np.array([[height, width]])
        example["inst_lane"] = np.concatenate([five_points.reshape(-1, 10), five_points_remainder], axis=1)

        fpoints_dc = example["inst_ldc"][:, :10]
        fpoints_dc_remainder = example["inst_ldc"][:, 10:]
        fpoints_dc = fpoints_dc.reshape(-1, 5, 2) / np.array([[height, width]])
        example["inst_ldc"] = np.concatenate([fpoints_dc.reshape(-1, 10), fpoints_dc_remainder], axis=1)
        return example


class ExampleZeroPadLane(PreprocessBase):
    def __init__(self, max_lane, max_lpoints):
        self._max_lane = max_lane
        self._max_lpoints = max_lpoints

    def __call__(self, example):
        new_lanes_point = np.zeros((self._max_lane, self._max_lpoints, 2), dtype=np.float32)
        new_lanes_point_dc = np.zeros((self._max_lane, self._max_lpoints, 2), dtype=np.float32)
        new_lanes = np.zeros((self._max_lane, 12), dtype=np.float32)

        new_lanes_dc = np.zeros((self._max_lane, 12), dtype=np.float32)

        if example["lanes_point"]:
            example["lanes_point"] = example["lanes_point"][:self._max_lane][:self._max_lpoints]
            for i, lane_points in enumerate(example["lanes_point"]):
                new_lanes_point[i, :lane_points.shape[0], :] = lane_points

            example["lanes_point_dc"] = example["lanes_point_dc"][:self._max_lane][:self._max_lpoints]
            for i, lane_points in enumerate(example["lanes_point_dc"]):
                new_lanes_point_dc[i, :lane_points.shape[0], :] = lane_points

            example["inst_lane"] = example["inst_lane"][:self._max_lane]
            new_lanes[:example["inst_lane"].shape[0], :] = example["inst_lane"]

            example["inst_ldc"] = example["inst_ldc"][:self._max_lane]
            new_lanes_dc[:example["inst_ldc"].shape[0], :] = example["inst_ldc"]


        example["lanes_point"] = new_lanes_point
        example["lanes_point_dc"] = new_lanes_point_dc
        example["inst_lane"] = new_lanes
        example["inst_ldc"] = new_lanes_dc
        return example


def test_lane_cropper():
    print("========= Test ExampleLaneCropper")
    example = {"lanes_point": [
        np.array([[0, 0], [0.1, 0.1], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5], [0.8, 0.8], [0.9, 0.9]]) * 100]}
    crop_tlbr = np.array([0.2, 0.2, 0.7, 0.7]) * 100
    cropper = ExampleLaneCropper(crop_tlbr)
    new_example = cropper(example)
    print("new example", new_example)
    lanes_point = new_example["lanes_point"][0]
    assert (lanes_point[0] == np.array([20, 20])).all()
    assert (lanes_point[-1] == np.array([69, 69])).all()
    print("========= ExampleLaneCropper passed!!")


if __name__ == "__main__":
    test_lane_cropper()


