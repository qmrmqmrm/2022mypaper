import numpy as np


class Paths:
    RESULT_ROOT = "/home/dolphin/kim_workspace"
    DATAPATH = "/media/dolphin/intHDD/kitti_detection/data_object_image_2/training/image_2"
    CHECK_POINT = "/home/dolphin/kim_workspace/ckpt"
    CONFIG_FILENAME = "/home/dolphin/kim_workspace/RILabDetector/config.py"
    META_CFG_FILENAME = "/home/dolphin/kim_workspace/RILabDetector/config_dir/meta_config.py"


class Datasets:

    class Kitti:
        NAME = "kitti"
        PATH = "/home/rilab-01/workspace/detlec/kitti"
        CATEGORIES_TO_USE = ['Pedestrian', 'Car', 'Van', 'Truck', 'Cyclist', 'Misc', 'Tram', 'DontCare', 'Person_sitting']
        CATEGORY_REMAP = {"Pedestrian": "Person", "Person_sitting": "Person", "Cyclist": "Bicycle", 
                          
                          }
        INPUT_RESOLUTION = (256, 832)
        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = False

    class Kittibev:
        NAME = "kitti_bev"
        PATH = "/home/rilab-01/workspace/detlec/kitti"
        CATEGORIES_TO_USE = ['Pedestrian', 'Car', 'Van', 'Truck', 'Cyclist', 'Misc', 'Tram', 'DontCare', 'Person_sitting']
        CATEGORY_REMAP = {"Pedestrian": "Person", "Person_sitting": "Person", "Cyclist": "Bicycle", 
                          
                          }
        INPUT_RESOLUTION = (256, 832)
        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = False
        CELL_SIZE = 0.05
        GRID_SHAPE = 500
        TBEV_POSE = np.array([0.0, 0.2618, 0.5236, 0.7854, 1.0472, 1.309])

    class Uplus:
        NAME = "uplus"
        PATH = "/home/eagle/mun_workspace/"
        CATEGORIES_TO_USE = ['사람', '승용차', '트럭', '버스', '이륜차', '신호등', '자전거', '삼각콘', '차선규제봉', '과속방지턱', 'TS이륜차금지', 'TS우회전금지', 'TS좌회전금지', 'TS유턴금지', 'TS주정차금지', 'TS자전거전용', 'TS유턴', 'TS어린이보호', 'TS횡단보도', 'TS좌회전', 'TS속도표시판', 'TS속도표시판_30', 'TS속도표시판_50', 'TS속도표시판_80', 'RM우회전금지', 'RM좌회전금지', 'RM직진금지', 'RM우회전', 'RM좌회전', 'RM직진', 'RM유턴', 'RM횡단예고', 'RM정지선', 'RM횡단보도', 'RM속도제한', 'RM속도제한_30', 'RM속도제한_50', 'RM속도제한_80', "don't care"]
        CATEGORY_REMAP = {"사람": "Person", "승용차": "Car", "트럭": "Truck", 
                          "버스": "Bus", "이륜차": "Motorcycle", "신호등": "Traffic light", 
                          "자전거": "Bicycle", "삼각콘": "Cone", "차선규제봉": "Lane_stick", 
                          "과속방지턱": "Bump", "don't care": "Don't Care", "TS이륜차금지": "TS_NO_TW", 
                          "TS우회전금지": "TS_NO_RIGHT", "TS좌회전금지": "TS_NO_LEFT", "TS유턴금지": "TS_NO_TURN", 
                          "TS주정차금지": "TS_NO_STOP", "TS자전거전용": "TS_Only_Bic", "TS유턴": "TS_U_TURN", 
                          "TS어린이보호": "TS_CHILDREN", "TS횡단보도": "TS_CROSSWK", "TS좌회전": "TS_GO_LEFT", 
                          "TS속도표시판": "TS_SPEED_LIMIT_ETC", "TS속도표시판_30": "TS_SPEED_LIMIT_30", "TS속도표시판_50": "TS_SPEED_LIMIT_50", 
                          "TS속도표시판_80": "TS_SPEED_LIMIT_80", "RM우회전금지": "RM_NO_RIGHT", "RM좌회전금지": "RM_NO_LEFT", 
                          "RM직진금지": "RM_NO_STR", "RM우회전": "RM_GO_RIGHT", "RM좌회전": "RM_GO_LEFT", 
                          "RM직진": "RM_GO_STR", "RM유턴": "RM_U_TURN", "RM횡단예고": "RM_ANN_CWK", 
                          "RM정지선": "RM_STOP", "RM횡단보도": "RM_CROSSWK", "RM속도제한": "RM_SPEED_LIMIT_ETC", 
                          "RM속도제한_30": "RM_SPEED_LIMIT_30", "RM속도제한_50": "RM_SPEED_LIMIT_50", "RM속도제한_80": "RM_SPEED_LIMIT_80", 
                          
                          }
        LANE_TYPES = ['차선1', '차선2', '차선3', '차선4', 'RM정지선']
        LANE_REMAP = {"차선1": "Lane", "차선2": "Lane", "차선3": "Lane", 
                      "차선4": "Lane", "RM정지선": "Stop_Line", 
                      }
        INPUT_RESOLUTION = (512, 1280)
        CROP_TLBR = [300, 0, 0, 0]
        INCLUDE_LANE = True
    DATASET_CONFIG = Kittibev
    TARGET_DATASET = "kittibev"


class Dataloader:
    DATASETS_FOR_TFRECORD = {"uplus": ('train', 'val'), 
                             }
    MAX_BBOX_PER_IMAGE = 50
    MAX_DONT_PER_IMAGE = 50
    MAX_LANE_PER_IMAGE = 10
    CATEGORY_NAMES = {"major": ['Bgd', 'Person', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Traffic light', 'Traffic sign', 'Road mark', 'Bicycle', 'Cone', 'Lane_stick', 'Bump'], "sign": ['TS_NO_TW', 'TS_NO_RIGHT', 'TS_NO_LEFT', 'TS_NO_TURN', 'TS_NO_STOP', 'TS_Only_Bic', 'TS_U_TURN', 'TS_CHILDREN', 'TS_CROSSWK', 'TS_GO_LEFT', 'TS_SPEED_LIMIT'], "mark": ['RM_NO_RIGHT', 'RM_NO_LEFT', 'RM_NO_STR', 'RM_GO_RIGHT', 'RM_GO_LEFT', 'RM_GO_STR', 'RM_U_TURN', 'RM_ANN_CWK', 'RM_STOP', 'RM_CROSSWK', 'RM_SPEED_LIMIT'], 
                      "sign_speed": ['TS_SPEED_LIMIT_30', 'TS_SPEED_LIMIT_50', 'TS_SPEED_LIMIT_80', 'TS_SPEED_LIMIT_ETC'], "mark_speed": ['RM_SPEED_LIMIT_30', 'RM_SPEED_LIMIT_50', 'RM_SPEED_LIMIT_80', 'RM_SPEED_LIMIT_ETC'], "dont": ["Don't Care"], 
                      "lane": ['Lane', 'Stop_Line'], 
                      }
    SHARD_SIZE = 2000
    ANCHORS_PIXEL = np.array([[42.0, 51.0], [121.0, 52.0], [79.0, 52.0], [51.0, 323.0], [251.0, 112.0], [166.0, 231.0], [85.0, 692.0], [92.0, 1079.0], [282.0, 396.0]])
    ANCHORS_LANE = None
    LANE_DETECT_ROWS = [20, 24, 29]
    MIN_PIX = {"train": {'Bgd': 0, 'Person': 0, 'Car': 0, 'Truck': 0, 'Bus': 0, 'Motorcycle': 0, 'Traffic light': 0, 'Traffic sign': 0, 'Road mark': 0, 'Bicycle': 0, 'Cone': 0, 'Lane_stick': 0, 'Bump': 0}, "val": {'Bgd': 0, 'Person': 0, 'Car': 0, 'Truck': 0, 'Bus': 0, 'Motorcycle': 0, 'Traffic light': 0, 'Traffic sign': 0, 'Road mark': 0, 'Bicycle': 0, 'Cone': 0, 'Lane_stick': 0, 'Bump': 0}, 
               }


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    LANE_DET = False
    MINOR_CTGR = True
    SPEED_LIMIT = True
    FEAT_RAW = False
    IOU_AWARE = False
    OUTPUT3D = False
    NUM_ANCHORS_PER_SCALE = 3
    GRTR_MAIN_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, 
                             "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, 
                             
                             }
    PRED_MAIN_COMPOSITION = {"category": 13, "sign_ctgr": 11, "mark_ctgr": 11, 
                             "sign_speed": 4, "mark_speed": 4, "yxhw": 4, 
                             "object": 1, "distance": 1, 
                             }
    PRED_HEAD_COMPOSITION = {"cls": 43, "reg": 6, 
                             }
    GRTR_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, 
                            "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, 
                            
                            }
    PRED_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, 
                            "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, 
                            "ctgr_prob": 1, "score": 1, "anchor_ind": 1, 
                            
                            }
    NUM_MAIN_CHANNELS = 49
    NUM_LANE_ANCHORS_PER_SCALE = 5
    GRTR_LANE_COMPOSITION = {"angle": 1, "intercept_x": 1, "object": 1, 
                             "category": 1, 
                             }
    PRED_LANE_COMPOSITION = {"angle": 1, "intercept_x": 1, "object": 1, 
                             "category": 2, 
                             }
    NUM_LANE_CHANNELS = 25
    VP_BINS = 16


class Optimizer:
    WEIGHT_DECAY = 0.0001
    WEIGHT_DECAY_BIAS = 0.0001
    WEIGHT_DECAY_NORM = 0.0
    MOMENTUM = 0.9


class Architecture:
    BACKBONE = "Resnet"
    NECK = "PAN"
    HEAD = "Single"
    BACKBONE_CONV_ARGS = {"activation": "mish", "scope": "back", 
                          }
    NECK_CONV_ARGS = {"activation": "leaky_relu", "scope": "neck", 
                      }
    HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head", 
                      }
    USE_SPP = False
    COORD_CONV = False

    class Resnet:
        LAYER = ('BottleneckBlock', (3, 4, 6, 3))
        CHENNELS = [64, 128, 256, 512, 1024, 2048]

    class Efficientnet:
        NAME = "EfficientNetB2"
        Channels = (112, 5, 3)
        Separable = False


class Train:
    CKPT_NAME = "test_dataset"
    MODE = "graph"
    DEVICE = "cuda"
    DATA_BATCH_SIZE = 2
    BATCH_SIZE = 4
    GLOBAL_BATCH = 4
    TRAINING_PLAN = [
                     ('kitti', 10, 0.0001, {'ciou': ([1.0, 1.0, 1.0], 'CiouLoss'), 'object': ([1.0, 1.0, 1.0], 'BoxObjectnessLoss', 1, 1), 'category': ([1.0, 1.0, 1.0], 'MajorCategoryLoss')}, True),
                     ('kitti', 50, 1e-05, {'ciou': ([1.0, 1.0, 1.0], 'CiouLoss'), 'object': ([1.0, 1.0, 1.0], 'BoxObjectnessLoss', 1, 1), 'category': ([1.0, 1.0, 1.0], 'MajorCategoryLoss')}, True),
                     ]
    DETAIL_LOG_EPOCHS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    IGNORE_MASK = True
    AUGMENT_PROBS = {"ColorJitter": 0.5, "CropResize": 1.0, "Blur": 0.2, 
                     
                     }
    LOG_KEYS = ['distance']
    USE_EMA = False
    EMA_DECAY = 0.9998


class Scheduler:
    MIN_LR = 1e-10
    CYCLE_STEPS = 10000
    WARMUP_EPOCH = 0
    LOG = True


class FeatureDistribPolicy:
    POLICY_NAME = "SinglePositivePolicy"
    IOU_THRESH = [0.5, 0.3]


class AnchorGeneration:
    ANCHOR_STYLE = "YoloAnchor"
    ANCHORS = np.array([[[35, 268], [80, 120], [113, 84]], [[71, 536], [160, 240], [226, 169]], [[143, 1073], [320, 480], [452, 339]]])
    MUL_SCALES = [1.0, 2.0, 4.0]

    class YoloAnchor:
        BASE_ANCHOR = [80.0, 120.0]
        ASPECT_RATIO = [0.2, 1.0, 2.0]
        SCALES = [1]

    class RetinaNetAnchor:
        BASE_ANCHOR = [20, 20]
        ASPECT_RATIO = [0.5, 1, 2]
        SCALES = [1, 1.2599210498948732, 1.5874010519681994]


class NmsInfer:
    MAX_OUT = [0, 19, 15, 8, 5, 6, 5, 5, 9, 5, 5, 12, 5]
    IOU_THRESH = [0, 0.3, 0.3, 0.34, 0.26, 0.36, 0.1, 0.1, 0.34, 0.3, 0.1, 0.1, 0.1]
    SCORE_THRESH = [1, 0.28, 0.08, 0.1, 0.08, 0.08, 0.1, 0.38, 0.16, 0.18, 0.22, 0.22, 0.3]


class NmsOptim:
    IOU_CANDIDATES = np.array([0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4])
    SCORE_CANDIDATES = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38])
    MAX_OUT_CANDIDATES = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])


class Validation:
    TP_IOU_THRESH = [1, 0.4, 0.5, 0.5, 0.5, 0.4, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5]
    DISTANCE_LIMIT = 25
    VAL_EPOCH = "latest"
    MAP_TP_IOU_THRESH = [0.5]


class Log:
    LOSS_NAME = ['ciou', 'object', 'category', 'sign_ctgr', 'mark_ctgr', 'sign_speed', 'mark_speed', 'distance']

    class HistoryLog:
        SUMMARY = ['pos_obj', 'neg_obj']

    class ExhaustiveLog:
        DETAIL = ['pos_obj', 'neg_obj', 'iou_mean', 'box_yx', 'box_hw', 'true_class', 'false_class']
        COLUMNS_TO_MEAN = ['anchor', 'ctgr', 'ciou', 'object', 'category', 'distance', 'pos_obj', 'neg_obj', 'iou_mean', 'box_hw', 'box_yx', 'true_class', 'false_class', 'sign_ctgr', 'mark_ctgr', 'sign_speed', 'mark_speed']
        COLUMNS_TO_SUM = ['anchor', 'ctgr', 'trpo', 'grtr', 'pred']
