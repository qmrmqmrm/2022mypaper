import numpy as np

import config_dir.config_generator as cg
import config as cfg


def get_channel_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_FMAP_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_FMAP_COMPOSITION


def get_bbox_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_INST_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_INST_COMPOSITION


def get_lane_channel_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_FMAP_LANE_COMPOSITION

    else:
        return cfg.ModelOutput.PRED_FMAP_LANE_COMPOSITION


def get_lane_composition(is_gt: bool):
    if is_gt:
        return cfg.ModelOutput.GRTR_INST_LANE_COMPOSITION
    else:
        return cfg.ModelOutput.PRED_INST_LANE_COMPOSITION


def get_img_shape(code="HW", dataset="kitti", scale_div=1):
    dataset_cfg = cfg.Datasets.TARGET_DATASET
    imsize = dataset_cfg.INPUT_RESOLUTION
    code = code.upper()
    if code == "H":
        return imsize[0] // scale_div
    elif code == "W":
        return imsize[1] // scale_div
    elif code == "HW":
        return imsize[0] // scale_div, imsize[1] // scale_div
    elif code == "WH":
        return imsize[1] // scale_div, imsize[0] // scale_div
    elif code == "HWC":
        return imsize[0] // scale_div, imsize[1] // scale_div, 3
    elif code == "BHWC":
        return cfg.Train.BATCH_SIZE, imsize[0] // scale_div, imsize[1] // scale_div, 3
    else:
        assert 0, f"Invalid code: {code}"


def get_valid_category_mask(dataset="kitti"):
    """
    :param dataset: dataset name
    :return: binary mask e.g. when
        Dataloader.MAJOR_CATE = ["Person", "Car", "Van", "Bicycle"] and
        Dataset.CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck"]
        Dataset.CATEGORY_REMAP = {"Pedestrian": "Person"}
        this function returns [1 1 1 0] because ["Person", "Car", "Van"] are included in dataset categories
        but "Bicycle" is not
    """
    dataset_cfg = cg.set_dataset_and_get_config(dataset)
    renamed_categories = [dataset_cfg.CATEGORY_REMAP[categ] if categ in dataset_cfg.CATEGORY_REMAP else categ
                          for categ in dataset_cfg.CATEGORIES_TO_USE]
    if dataset == "uplus":
        for i, categ in enumerate(cfg.Dataloader.CATEGORY_NAMES["major"]):
            if categ not in renamed_categories:
                renamed_categories.insert(i, categ)

    mask = np.zeros((len(cfg.Dataloader.CATEGORY_NAMES["major"]),), dtype=np.int32)
    for categ in renamed_categories:
        if categ in cfg.Dataloader.CATEGORY_NAMES["major"]:
            index = cfg.Dataloader.CATEGORY_NAMES["major"].index(categ)
            if index < len(cfg.Dataloader.CATEGORY_NAMES["major"]):
                mask[index] = 1
    return mask
