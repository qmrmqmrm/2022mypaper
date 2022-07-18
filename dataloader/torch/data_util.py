import numpy as np
import cv2
import torch
import pandas as pd

import utils.framework.util_function as uf


def draw_boxes(image, bboxes, category_names, locations=None, box_format="yxhw"):
    """
    :param grid_feats:
    :param image: (height, width, 3), np.uint8
    :param bboxes: (N, 6), np.float32 (0~1) or np.int32 (pixel scale)
    :param category_names: list of category names
    :param box_format: "yxhw": [y, x, h, w, category] or "2pt": [y1, x1, y2, x2, category]
    """
    image = image.copy()
    bboxes = bboxes.copy()
    if np.max(bboxes[:, :4]) <= 1:
        height, width = image.shape[:2]
        bboxes[:, :4] *= np.array([[height, width, height, width]], np.float32)
    if box_format == "yxhw":
        bboxes = uf.convert_box_format_yxhw_to_tlbr(bboxes)
    bboxes = bboxes[bboxes[:, 2] > 0, :]

    for i, bbox in enumerate(bboxes):
        pt1, pt2 = (bbox[1].astype(np.int32), bbox[0].astype(np.int32)), (bbox[3].astype(np.int32), bbox[2].astype(np.int32))

        # minor_category = category_names["minor"][minor_cat_index]
        image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), thickness=2)

    if locations is not None:
        for i, location in enumerate(locations):
            y, x = location
            image = cv2.circle(image, (int(x), int(y)), 4, (255, 0, 255), -1)
    return image
