import os.path as op
import config_dir.parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "/home/dolphin/kim_workspace"
    DATAPATH = "/media/dolphin/intHDD/culane/tfrecord"
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")
    CONFIG_FILENAME = '/home/dolphin/kim_workspace/2022mypaper/config.py'
    META_CFG_FILENAME = '/home/dolphin/kim_workspace/2022mypaper/config_dir/meta_config.py'


class Datasets:
    # specific dataset configs MUST have the same items
    class Kitti:
        NAME = "kitti"
        PATH = "/home/cheetah-01/IntHDD/datasets/kitti"
        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck", "Cyclist"]
        CATEGORY_REMAP = {"Pedestrian": "Person", "Cyclist": "Bicycle"}
        INPUT_RESOLUTION = (256, 832)
        INCLUDE_LANE = [True, False][1]
        # (4,13) * 64
        CROP_TLBR = [0, 0, 0, 0]
        # crop [top, left, bottom, right] or [y1 x1 y2 x2]

    class Culane:
        NAME = "culane"
        PATH = "/media/dolphin/intHDD/culane"
        CATEGORIES_TO_USE = ["Lane1, Lane2, Lane3, Lane4"]
        CATEGORY_REMAP = {}
        # INPUT_RESOLUTION = (590, 1640)
        INPUT_RESOLUTION = (576, 1600)
        INCLUDE_LANE = [True, False][1]
        # (4,13) * 64
        CROP_TLBR = [14, 20, 0, 20]
        # crop [top, left, bottom, right] or [y1 x1 y2 x2]

    class Uplus:
        NAME = "uplus"
        PATH = "/home/cheetah/kim_workspace/uplus22"
        CATEGORIES_TO_USE = params.Uplus2Params.CATEGORIES_TO_USE
        CATEGORY_REMAP = params.Uplus2Params.CATEGORY_REMAP
        LANE_TYPES = params.Uplus2Params.LANE_TYPES
        LANE_REMAP = params.Uplus2Params.LANE_REMAP
        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [450, 180, 25, 180]
        # CROP_TLBR = [300, 0, 0, 0]
        # crop [top, left, bottom, right] or [y1 x1 y2 x2]
        INCLUDE_LANE = [True, False][0]

    class Uplus21:
        NAME = "uplus21"
        PATH = "/home/dolphin/kim_workspace/uplus21"
        CATEGORIES_TO_USE = params.UplusParams.CATEGORIES_TO_USE
        CATEGORY_REMAP = params.UplusParams.CATEGORY_REMAP
        LANE_TYPES = params.UplusParams.LANE_TYPES
        LANE_REMAP = params.UplusParams.LANE_REMAP
        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [0, 0, 0, 0]
        # crop [top, left, bottom, right] or [y1 x1 y2 x2]
        INCLUDE_LANE = [True, False][0]

    class City:
        NAME = "city"
        PATH = "/media/cheetah/IntHDD/datasets/city"
        CATEGORIES_TO_USE = ['person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
        CATEGORY_REMAP = {"person": "Person", "rider": "Person", "car": "Car", "bus": "Bus",
                          "truck": "Truck", "motorcycle": "Motorcycle", "bicycle": "Bicycle"}
        INPUT_RESOLUTION = (1024, 2048)
        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = [True, False][1]
        DIST_QUANTILE = 0.2

    class A2d2:
        NAME = "a2d2"
        PATH = "/media/cheetah/IntHDD/datasets/a2d2/camera_lidar_semantic_bboxes.zip"
        CATEGORIES_TO_USE = ["Car 1", "Car 2", "Car 3", "Car 4",
                             "Bicycle 1", "Bicycle 2", "Bicycle 3", "Bicycle 4",
                             "Pedestrian 1", "Pedestrian 2", "Pedestrian 3",
                             "Truck 1", "Truck 2", "Truck 3",
                             "Small vehicles 1", "Small vehicles 2", "Small vehicles 3"]
        CATEGORY_REMAP = {"Car 1": "Car", "Car 2": "Car", "Car 3": "Car", "Car 4": "Car",
                          "Bicycle 1": "Bicycle", "Bicycle 2": "Bicycle", "Bicycle 3": "Bicycle",
                          "Bicycle 4": "Bicycle",
                          "Pedestrian 1": "Person", "Pedestrian 2": "Person", "Pedestrian 3": "Person",
                          "Truck 1": "Truck", "Truck 2": "Truck", "Truck 3": "Truck",
                          "Small vehicles 1": "Motorcycle", "Small vehicles 2": "Motorcycle",
                          "Small vehicles 3": "Motorcycle"}
        INPUT_RESOLUTION = (512, 1280)  # (4,13) * 64
        CROP_TLBR = [0, 0, 0, 0]        # crop [top, left, bottom, right] or [y1 x1 y2 x2]
        MAX_LANE_PARAM = 50
        CATEGORY_NAMES = ["Bgd", "Pedestrian", "Car"]
        SHARD_SIZE = 2000
        INCLUDE_LANE = [True, False][1]
        SEGMAP_SCALE = 4
        PIXEL_LIMIT = 50
        DIST_QUANTILE = 0.2

    DATASET_CONFIG = None
    TARGET_DATASET = "culane"
    MAX_FRAMES = 100000


class Dataloader:
    DATASETS_FOR_TFRECORD = {
        # "kitti": ("train", "val"),
        # "uplus": ( "val"),
        "culane": ("train","val"),
        # "city": ("train", "val"),
        # "a2d2": ("train", "val"),
    }
    MAX_BBOX_PER_IMAGE = 100
    MAX_DONT_PER_IMAGE = 100
    MAX_LANE_PER_IMAGE = 4
    MAX_POINTS_PER_LANE = 50
    NUM_LANE_POINT = params.TfrParams.NUM_LANE_POINT

    CATEGORY_NAMES = params.TfrParams.CATEGORY_NAMES
    SHARD_SIZE = 2000
    MIN_PIX = params.TfrParams.MIN_PIX
    LANE_MIN_PIX = params.TfrParams.LANE_MIN_PIX


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    LANE_DET = True
    # CATEGORY_LEVEL = 1~3    # TODO
    MINOR_CTGR = False
    SPEED_LIMIT = False
    FEAT_RAW = False
    IOU_AWARE = False
    BOX_DET = False

    NUM_BOX_ANCHORS_PER_SCALE = 3
    GRTR_FMAP_BOX_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1,
                             "anchor_ind": 1}
    PRED_FMAP_BOX_COMPOSITION = params.TrainParams.get_pred_composition(MINOR_CTGR, SPEED_LIMIT, IOU_AWARE)
    HEAD_BOX_COMPOSITION = params.TrainParams.get_pred_composition(MINOR_CTGR, SPEED_LIMIT, IOU_AWARE, True)

    GRTR_INST_BOX_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1}
    PRED_INST_BOX_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1,
                             "distance": 1, "ctgr_prob": 1, "score": 1, "anchor_ind": 1}

    NUM_BOX_MAIN_CHANNELS = sum(PRED_FMAP_BOX_COMPOSITION.values())

    NUM_LANE_ANCHORS_PER_SCALE = 1
    GRTR_FMAP_LANE_COMPOSITION = {"laneness": 1,  "lane_fpoints": params.TfrParams.NUM_LANE_POINT *2, "lane_centerness": 1, "lane_category": 1}
    PRED_FMAP_LANE_COMPOSITION =  params.TrainParams.get_pred_lane_composition(False)
    HEAD_LANE_COMPOSITION = params.TrainParams.get_pred_lane_composition(True)

    GRTR_INST_LANE_COMPOSITION = {"lane_fpoints": params.TfrParams.NUM_LANE_POINT *2, "lane_centerness": 1, "lane_category": 1}
    PRED_INST_LANE_COMPOSITION = {"lane_fpoints": params.TfrParams.NUM_LANE_POINT *2, "lane_centerness": 1, "lane_category": 1}
    NUM_LANE_MAIN_CHANNELS = sum(PRED_FMAP_LANE_COMPOSITION.values())


class Architecture:
    BACKBONE = ["Resnet", "Resnet_vd", "CSPDarknet53", "Efficientnet"][3]
    NECK = ["FPN", "PAN", "BiFPN"][1]
    HEAD = ["Single", "Double", "Efficient"][0]

    BACKBONE_CONV_ARGS = {"activation": "mish", "scope": "back"}
    NECK_CONV_ARGS = {"activation": "leaky_relu", "scope": "neck"}
    HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}
    USE_SPP = True
    COORD_CONV = True
    SIGMOID_DELTA = 0.2

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
    CKPT_NAME = "culane_v5_efficient"
    MODE = ["eager", "graph", "distribute"][1]
    AUGMENT_PROBS = None
    # AUGMENT_PROBS = {"ColorJitter": 0.5, "CropResize": 1.0, "Blur": 0.2}
    DATA_BATCH_SIZE = 2
    BATCH_SIZE = DATA_BATCH_SIZE * 2 if AUGMENT_PROBS else DATA_BATCH_SIZE
    DATSET_SIZE = DATA_BATCH_SIZE * 20
    TRAINING_PLAN = params.TrainingPlan.UPLUS_PLAN
    DETAIL_LOG_EPOCHS = list(range(5, 100, 5))
    IGNORE_MASK = True
    # AUGMENT_PROBS = {"Flip": 0.2}

    # LOG_KEYS: select options in ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
    LOG_KEYS = ["distance"]
    USE_EMA = True
    EMA_DECAY = 0.9998
    INTRINSIC = np.zeros([3, 4])


class Scheduler:
    MIN_LR = 1e-10
    CYCLE_STEPS = 10000
    WARMUP_EPOCH = 0
    LOG = [True, False][0]


class FeatureDistribPolicy:
    POLICY_NAME = ["SinglePositivePolicy", "FasterRCNNPolicy", "MultiPositivePolicy"][0]
    IOU_THRESH = [0.5, 0.3]
    CENTER_RADIUS = 2.5
    # [Small Max, Medium Max]
    BOX_SIZE_STANDARD = np.array([128, 256])
    MULTI_POSITIVE_WIEGHT = 0.8


class AnchorGeneration:
    # ANCHOR_STYLE : Manual
    ANCHOR_STYLE = "YoloAnchor"
    # ANCHORS = np.array([[[30, 29], [91, 40], [56, 101]], [[43, 447], [193, 105], [104, 207]], [[60, 818], [247, 340], [72, 1207]]])
    ANCHORS = None
    # MUL_SCALES = [scale / 8 for scale in ModelOutput.FEATURE_SCALES]
    MUL_SCALES = [1.0, 2.5, 4.0]

    class YoloAnchor:
        BASE_ANCHOR = [70., 110.]
        ASPECT_RATIO = [0.14, 1., 2.5]
        SCALES = [1]

    class RetinaNetAnchor:
        BASE_ANCHOR = [20, 20]
        ASPECT_RATIO = [0.5, 1, 2]
        SCALES = [2 ** x for x in [0, 1 / 3, 2 / 3]]

    class YoloxAnchor:
        BASE_ANCHOR = [8, 8]
        ASPECT_RATIO = [1]
        SCALES = [1]


class NmsInfer:
    MAX_OUT = [0, 10, 5, 15, 7, 9, 5, 7, 5, 8, 9, 5, 6, 11, 7, 5]
    IOU_THRESH = [0, 0.28, 0.1, 0.34, 0.1, 0.3, 0.38, 0.32, 0.28, 0.4, 0.1, 0.26, 0.4, 0.26, 0.38, 0.1]
    SCORE_THRESH = [1, 0.3, 0.36, 0.34, 0.2, 0.14, 0.28, 0.16, 0.24, 0.24, 0.26, 0.3, 0.16, 0.3, 0.18, 0.38]

    LANE_MAX_OUT = [0, 1, 1, 1, 1]
    LANE_OVERLAP_THRESH = [0, 0.65, 0.4, 0.45, 0.1]
    LANE_SCORE_THRESH = [1, 0.08, 0.12, 0.08, 0.04]


class NmsOptim:
    IOU_CANDIDATES = np.arange(0.1, 0.4, 0.02)
    SCORE_CANDIDATES = np.arange(0.02, 0.4, 0.02)
    MAX_OUT_CANDIDATES = np.arange(5, 20, 1)

    LANE_IOU_CANDIDATES = np.arange(0.1, .9, 0.05)
    LANE_SCORE_CANDIDATES = np.arange(0.02, 0.4, 0.02)
    LANE_MAX_OUT_CANDIDATES = np.arange(1, 3, 1)


class Validation:
    # 사람 차 트럭 버스 오토바이 신호등 표지판 노면표시 자전거 콘 규제봉 방지턱 포트홀
    TP_IOU_THRESH = [1, 0.4, 0.5, 0.5, 0.5, 0.4, 0.2, 0.3, 0.3, 0.4, 0.5, 0.2, 0.3, 0.3]
    DISTANCE_LIMIT = 25
    VAL_EPOCH = "latest"
    MAP_TP_IOU_THRESH = [0.5]
    MAX_BOX = 200
    LANE_TP_IOU_THRESH = [0.5]


class Log:
    VISUAL_HEATMAP = True

    class HistoryLog:
        SUMMARY = {"base": ["pos_obj", "neg_obj"],
                   "lane": ["pos_lane", "neg_lane", "pos_center", "neg_center"]}

    class ExhaustiveLog:
        LANE_DETAIL = {"lane": ["pos_lane", "neg_lane", "pos_lanecenter", "neg_lanecenter", "lane_true_class",
                                "lane_false_class"]}
        COLUMNS_TO_LANE_MEAN = {"lane": ["ctgr", "laneness", "lane_fpoints", "lane_category", "lane_centerness",
                                         "pos_lane", "neg_lane", "pos_lanecenter", "neg_lanecenter",
                                         "lane_true_class", "lane_false_class"]}
        COLUMNS_TO_LANE_SUM = ["ctgr", "trpo_lane", "grtr_lane", "pred_lane"]
