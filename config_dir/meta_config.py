import os.path as op
import numpy as np

import config_dir.parameter_pool as params


class Paths:
    RESULT_ROOT = "/home/dolphin/kim_workspace"
    # DATAPATH = op.join(RESULT_ROOT, "tfrecord")
    DATAPATH = "/media/dolphin/intHDD/kitti_detection/data_object_image_2/training/image_2"
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")
    CONFIG_FILENAME = '/home/dolphin/kim_workspace/RILabDetector/config.py'
    META_CFG_FILENAME = '/home/dolphin/kim_workspace/RILabDetector/config_dir/meta_config.py'


class Datasets:
    # specific dataset configs MUST have the same items
    class Kitti:
        NAME = "kitti"
        PATH = "/home/rilab-01/workspace/detlec/kitti"
        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck", "Cyclist", "Misc", "Tram", "DontCare",
                             "Person_sitting"]
        CATEGORY_REMAP = {"Pedestrian": "Person", "Person_sitting": "Person", "Cyclist": "Bicycle"}
        INPUT_RESOLUTION = (256, 832)
        # (4,13) * 64
        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = [True, False][1]
        # crop [top, left, bottom, right] or [y1 x1 y2 x2]

    class Kittibev:
        NAME = "kitti_bev"
        PATH = "/home/rilab-01/workspace/detlec/kitti"
        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck", "Cyclist", "Misc", "Tram", "DontCare",
                             "Person_sitting"]
        CATEGORY_REMAP = {"Pedestrian": "Person", "Person_sitting": "Person", "Cyclist": "Bicycle"}
        INPUT_RESOLUTION = (256, 832)
        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = [True, False][1]
        CELL_SIZE = 0.05
        GRID_SHAPE = 500
        TBEV_POSE = np.arange(0, np.pi / 2, np.pi / 12)

    class Uplus:
        NAME = "uplus"
        PATH = "/home/eagle/mun_workspace/"
        CATEGORIES_TO_USE = params.Uplus2Params.CATEGORIES_TO_USE
        CATEGORY_REMAP = params.Uplus2Params.CATEGORY_REMAP
        LANE_TYPES = params.Uplus2Params.LANE_TYPES
        LANE_REMAP = params.Uplus2Params.LANE_REMAP
        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [300, 0, 0, 0]
        # crop [top, left, bottom, right] or [y1 x1 y2 x2]
        INCLUDE_LANE = [True, False][0]

    # TARGET_DATASET = "Uplus"
    DATASET_CONFIG = None
    TARGET_DATASET = "kittibev"


class Dataloader:
    DATASETS_FOR_TFRECORD = {
        # "kitti": ("train", "val"),
        "uplus": ("train", "val"),
    }
    MAX_BBOX_PER_IMAGE = 50
    MAX_DONT_PER_IMAGE = 50
    MAX_LANE_PER_IMAGE = 10

    CATEGORY_NAMES = params.TfrParams.CATEGORY_NAMES
    SHARD_SIZE = 2000
    ANCHORS_PIXEL = None
    ANCHORS_LANE = None
    LANE_DETECT_ROWS = [20, 24, 29]
    MIN_PIX = params.TfrParams.MIN_PIX


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    LANE_DET = False
    MINOR_CTGR = True
    SPEED_LIMIT = True
    FEAT_RAW = False
    IOU_AWARE = False
    OUTPUT3D = False

    NUM_ANCHORS_PER_SCALE = 3
    # MAIN -> FMAP, NMS -> INST
    GRTR_MAIN_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1}
    PRED_MAIN_COMPOSITION = params.TrainParams.get_pred_composition(MINOR_CTGR, SPEED_LIMIT, IOU_AWARE)
    PRED_HEAD_COMPOSITION = params.TrainParams.get_pred_composition(MINOR_CTGR, SPEED_LIMIT, IOU_AWARE, True)

    GRTR_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1}
    PRED_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1,
                            "distance": 1, "ctgr_prob": 1, "score": 1, "anchor_ind": 1}

    NUM_MAIN_CHANNELS = sum(PRED_MAIN_COMPOSITION.values())

    NUM_LANE_ANCHORS_PER_SCALE = 5
    GRTR_LANE_COMPOSITION = {"angle": 1, "intercept_x": 1, "object": 1, "category": 1}
    PRED_LANE_COMPOSITION = {"angle": 1, "intercept_x": 1, "object": 1,
                             "category": len(params.TfrParams.CATEGORY_NAMES["lane"])}
    NUM_LANE_CHANNELS = sum(PRED_LANE_COMPOSITION.values()) * NUM_LANE_ANCHORS_PER_SCALE

    # OUTPUT3D
    VP_BINS = 16


class Optimizer:
    WEIGHT_DECAY = 0.0001
    WEIGHT_DECAY_BIAS = 0.0001
    WEIGHT_DECAY_NORM = 0.0
    MOMENTUM = 0.9


class Architecture:
    BACKBONE = ["Resnet", "Resnet_vd", "CSPDarknet53", "Efficientnet"][0]
    NECK = ["FPN", "PAN", "BiFPN"][1]
    HEAD = ["Single", "Double", "Efficient"][0]

    BACKBONE_CONV_ARGS = {"activation": ["mish", "relu", "leaky_relu"][0], "scope": "back"}
    NECK_CONV_ARGS = {"activation": "leaky_relu", "scope": "neck"}
    # NECK_CONV_ARGS = {"activation": False, "scope": "neck", "bn": False}
    HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}
    # HEAD_CONV_ARGS = {"activation": "swish", "scope": "head"}
    USE_SPP = [True, False][1]
    COORD_CONV = [True, False][1]

    class Resnet:
        LAYER = {50: ('BottleneckBlock', (3, 4, 6, 3)),
                 101: ('BottleneckBlock', (3, 4, 23, 3)),
                 152: ('BottleneckBlock', (3, 8, 36, 3))
                 }[50]
        CHENNELS = [64, 128, 256, 512, 1024, 2048]

    class Efficientnet:
        NAME = "EfficientNetB2"
        Channels = {"EfficientNetB0": (64, 3, 3), "EfficientNetB1": (88, 4, 3),
                    "EfficientNetB2": (112, 5, 3), "EfficientNetB3": (160, 6, 4),
                    "EfficientNetB4": (224, 7, 4), "EfficientNetB5": (288, 7, 4),
                    "EfficientNetB6": (384, 8, 5)}[NAME]
        Separable = [True, False][1]


class Train:
    CKPT_NAME = "test_dataset"
    MODE = ["eager", "graph", "distribute"][1]
    DEVICE = ['cuda', 'cpu'][0]
    DATA_BATCH_SIZE = 2
    BATCH_SIZE = DATA_BATCH_SIZE * 2
    GLOBAL_BATCH = BATCH_SIZE
    TRAINING_PLAN = params.TrainingPlan.KITTI_SIMPLE
    DETAIL_LOG_EPOCHS = list(range(0, 100, 10))
    IGNORE_MASK = True
    # AUGMENT_PROBS = {"Flip": 0.2}
    AUGMENT_PROBS = {"ColorJitter": 0.5, "CropResize": 1.0, "Blur": 0.2}
    # LOG_KEYS: select options in ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
    LOG_KEYS = ["distance"]
    USE_EMA = [True, False][1]
    EMA_DECAY = 0.9998


class Scheduler:
    MIN_LR = 1e-10
    CYCLE_STEPS = 10000
    WARMUP_EPOCH = 0
    LOG = [True, False][0]


class FeatureDistribPolicy:
    POLICY_NAME = ["SinglePositivePolicy", "MultiPositivePolicy"][0]
    IOU_THRESH = [0.5, 0.3]


class AnchorGeneration:
    ANCHOR_STYLE = "YoloAnchor"
    ANCHORS = None
    MUL_SCALES = [scale / 8 for scale in ModelOutput.FEATURE_SCALES]

    class YoloAnchor:
        BASE_ANCHOR = [80., 120.]
        ASPECT_RATIO = [0.2, 1., 2.]
        SCALES = [1]

    class RetinaNetAnchor:
        BASE_ANCHOR = [20, 20]
        ASPECT_RATIO = [0.5, 1, 2]
        SCALES = [2 ** x for x in [0, 1 / 3, 2 / 3]]


class NmsInfer:
    MAX_OUT = [0, 19, 15, 8, 5, 6, 5, 5, 9, 5, 5, 12, 5]
    IOU_THRESH = [0, 0.3, 0.3, 0.34, 0.26, 0.36, 0.1, 0.1, 0.34, 0.3, 0.1, 0.1, 0.1]
    SCORE_THRESH = [1, 0.28, 0.08, 0.1, 0.08, 0.08, 0.1, 0.38, 0.16, 0.18, 0.22, 0.22, 0.3]


class NmsOptim:
    IOU_CANDIDATES = np.arange(0.1, 0.4, 0.02)
    SCORE_CANDIDATES = np.arange(0.02, 0.4, 0.02)
    MAX_OUT_CANDIDATES = np.arange(5, 20, 1)


class Validation:
    TP_IOU_THRESH = [1, 0.4, 0.5, 0.5, 0.5, 0.4, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5]
    DISTANCE_LIMIT = 25
    VAL_EPOCH = "latest"
    MAP_TP_IOU_THRESH = [0.5]


class Log:
    LOSS_NAME = {ModelOutput.MINOR_CTGR: ["ciou", "object", "category", "sign_ctgr", "mark_ctgr", "distance"],
                 ModelOutput.SPEED_LIMIT: ["ciou", "object", "category", "sign_ctgr", "mark_ctgr", "sign_speed",
                                           "mark_speed", "distance"]}.get(True,
                                                                          ["ciou", "object", "category", "distance"])

    class HistoryLog:
        SUMMARY = ["pos_obj", "neg_obj"]

    class ExhaustiveLog:
        DETAIL = ["pos_obj", "neg_obj", "iou_mean", "iou_aware", "box_yx", "box_hw", "true_class", "false_class"] \
            if ModelOutput.IOU_AWARE else ["pos_obj", "neg_obj", "iou_mean", "box_yx", "box_hw", "true_class",
                                           "false_class"]
        COLUMNS_TO_MEAN = {ModelOutput.MINOR_CTGR: ["anchor", "ctgr", "ciou", "object", "category", "distance",
                                                    "pos_obj", "neg_obj", "iou_mean", "box_hw", "box_yx", "true_class",
                                                    "false_class" "sign_ctgr", "mark_ctgr"],
                           ModelOutput.SPEED_LIMIT: ["anchor", "ctgr", "ciou", "object", "category", "distance",
                                                     "pos_obj", "neg_obj", "iou_mean", "box_hw", "box_yx", "true_class",
                                                     "false_class", "sign_ctgr", "mark_ctgr", "sign_speed", "mark_speed",
                                                     ]}.get(True, ["anchor", "ctgr", "ciou", "object",
                                                                             "category", "distance", "pos_obj",
                                                                             "neg_obj", "iou_mean", "box_hw", "box_yx",
                                                                             "true_class", "false_class"])
        COLUMNS_TO_SUM = ["anchor", "ctgr", "trpo", "grtr", "pred"]
