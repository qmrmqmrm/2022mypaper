import numpy as np


class LossComb:
    STANDARD = {"ciou": ([1., 1., 1.], "CiouLoss"), "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                "category": ([1., 1., 1.], "MajorCategoryLoss")}
    UPLUS_BASIC = {"ciou": ([1., 1., 1.], "CiouLoss"), "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                   "category": ([1, 1, 1.], "MajorCategoryLoss"), "distance": ([1., 1., 1.], "DistanceLoss")}
    UPLUS_SCALE_WEIGHT = {"ciou": ([3., 3., 3.], "CiouLoss"), "object": ([4., 2., 2.], "BoxObjectnessLoss", 1, 3),
                          "category": ([4, 2, 2.], "MajorCategoryLoss"), "distance": ([1., 1., 1.], "DistanceLoss")}
    UPLUS_MINOR_WEIGHT = {"ciou": ([3., 3., 3.], "CiouLoss"), "object": ([4., 2., 2.], "BoxObjectnessLoss", 1, 3),
                          "category": ([4., 2., 2.], "MajorCategoryLoss"),
                          "sign_ctgr": ([4., 2., 2.], "MinorCategoryLoss", "sign_ctgr"),
                          "mark_ctgr": ([4., 2., 2.], "MinorCategoryLoss", "mark_ctgr"),
                          "distance": ([1., 1., 1.], "DistanceLoss")}
    UPLUS_SPEED_WEIGHT = {"ciou": ([3., 3., 3.], "CiouLoss"), "object": ([4., 2., 2.], "BoxObjectnessLoss", 1, 3),
                          "category": ([4., 2., 2.], "MajorCategoryLoss"),
                          "sign_ctgr": ([4., 2., 2.], "MinorCategoryLoss", "sign_ctgr"),
                          "mark_ctgr": ([4., 2., 2.], "MinorCategoryLoss", "mark_ctgr"),
                          "sign_speed": ([4, 2, 2.], "MinorSpeedCategoryLoss", "sign_ctgr", "sign_speed"),
                          "mark_speed": ([4, 2, 2.], "MinorSpeedCategoryLoss", "mark_ctgr", "mark_speed"),
                          "distance": ([1., 1., 1.], "DistanceLoss")}


class Anchor:
    """
    anchor order MUST be compatible with Config.ModelOutput.FEATURE_ORDER
    in the current setting, the smallest anchor comes first
    """
    COCO_YOLOv3 = np.array([[13, 10], [30, 16], [23, 33],
                            [61, 30], [45, 62], [119, 59],
                            [90, 116], [198, 156], [326, 373]], dtype=np.float32)

    # UPLUS_YOLOv4 = np.array([[34, 39], [60, 90], [109, 50],
    #                       [103, 144], [151, 224], [238, 164],
    #                       [211, 312], [259, 411], [449, 427]], dtype=np.float32)

    UPLUS_YOLOv4 = np.array([[42, 51], [121, 52], [79, 52],
                             [51, 323], [251, 112], [166, 231],
                             [85, 692], [92, 1079], [282, 396]], dtype=np.float32)

    KITTI = np.array([[42, 51], [121, 52], [79, 52],
                             [51, 323], [251, 112], [166, 231],
                             [85, 692], [92, 1079], [282, 396]], dtype=np.float32)

    UPLUS_LANE = np.tan(np.array([-70, -40, 0, 40, 70], dtype=np.float32) / 180. * np.pi)
    COCO_RESOLUTION = (416, 416)
    UPLUS_RESOLUTION = (512, 1280)
    KITTI_RESOLUTION = (256, 832)


class TrainingPlan:
    KITTI_SIMPLE = [
        ("kitti", 10, 0.0001, LossComb.STANDARD, True),
        ("kitti", 50, 0.00001, LossComb.STANDARD, True)
    ]

    UPLUS_SIMPLE = [
        ("uplus", 25, 0.001, LossComb.UPLUS_SCALE_WEIGHT, True),
        ("uplus", 15, 0.0001, LossComb.UPLUS_SCALE_WEIGHT, True),
        ("uplus", 15, 0.0001, LossComb.UPLUS_SCALE_WEIGHT, True),
        ("uplus", 30, 0.00001, LossComb.UPLUS_SCALE_WEIGHT, True),
        ("uplus", 10, 0.000001, LossComb.UPLUS_SCALE_WEIGHT, True),
        ("uplus", 10, 0.0000001, LossComb.UPLUS_SCALE_WEIGHT, True),
    ]

    UPLUS_MINOR = [
        ("uplus", 25, 0.0001, LossComb.UPLUS_MINOR_WEIGHT, True),
        ("uplus", 25, 0.0001, LossComb.UPLUS_MINOR_WEIGHT, True),
        ("uplus", 15, 0.00001, LossComb.UPLUS_MINOR_WEIGHT, True),
        ("uplus", 30, 0.000001, LossComb.UPLUS_MINOR_WEIGHT, True),
        ("uplus", 10, 0.0000001, LossComb.UPLUS_MINOR_WEIGHT, True),
        ("uplus", 10, 0.00000001, LossComb.UPLUS_MINOR_WEIGHT, True),
    ]

    UPLUS_SPEED = [
        ("uplus", 25, 0.0001, LossComb.UPLUS_SPEED_WEIGHT, True),
        ("uplus", 25, 0.0001, LossComb.UPLUS_SPEED_WEIGHT, True),
        ("uplus", 10, 0.00001, LossComb.UPLUS_SPEED_WEIGHT, True),
        ("uplus", 10, 0.000001, LossComb.UPLUS_SPEED_WEIGHT, True),
        ("uplus", 10, 0.0000001, LossComb.UPLUS_SPEED_WEIGHT, True),
        ("uplus", 10, 0.00000001, LossComb.UPLUS_SPEED_WEIGHT, True),
    ]


class UplusParams:
    CATEGORIES_TO_USE = ["사람", "차", "차량/트럭", "차량/버스", "차량/오토바이", "신호등",
                         "차량/자전거", "삼각콘", "표지판/이륜 통행 금지", "표지판/유턴 금지", "표지판/주정차 금지",
                         "표지판/서행", "표지판/우로 굽은 도로", "표지판/어린이 보호", "표지판/횡단보도", "노면표시/직진",
                         "노면표시/횡단보도 예고", "노면표시/직진금지", "노면표시/정지선", "노면표시/횡단보도", "don't care"]
    CATEGORY_REMAP = {"사람": "Person", "차량/자전거": "Bicycle", "차": "Car", "차량/트럭": "Truck",
                      "차량/버스": "Bus", "신호등": "Traffic light", "삼각콘": "Cone", "don't care": "Don't Care",
                      "차량/오토바이": "Motorcycle",

                      "표지판/유턴 금지": "NO_TURN",
                      "표지판/이륜 통행 금지": "NO_TW",
                      "표지판/주정차 금지": "NO_STOP", "표지판/서행": "SLOW",
                      "표지판/우로 굽은 도로": "RIGHT_CURVE", "표지판/어린이 보호": "CHILDREN",
                      "표지판/횡단보도": "CROSSWK",

                      "노면표시/직진": "GO_STR",
                      "노면표시/횡단보도 예고": "CW_NOTI", "노면표시/직진금지": "NO_STR",
                      "노면표시/정지선": "STOP_L",
                      "노면표시/횡단보도": "CW_MARK",
                      }
    LANE_TYPES = ["차선/황색 단선 실선", "차선/백색 단선 실선", "차선/황색 단선 점선", "차선/백색 단선 점선",
                  "차선/황색 겹선 실선"]
    LANE_REMAP = {"차선/황색 단선 실선": "YSL", "차선/백색 단선 실선": "WSL", "차선/황색 단선 점선": "YSDL",
                  "차선/백색 단선 점선": "WSDL", "차선/황색 겹선 실선": "YDL"}


class Uplus2Params:
    CATEGORIES_TO_USE = ["사람", "승용차", "트럭", "버스", "이륜차", "신호등", "자전거", "삼각콘", "차선규제봉", "과속방지턱",

                         "TS이륜차금지", "TS우회전금지", "TS좌회전금지", "TS유턴금지", "TS주정차금지", "TS자전거전용", "TS유턴",
                         "TS어린이보호", "TS횡단보도", "TS좌회전",
                         "TS속도표시판", "TS속도표시판_30", "TS속도표시판_50", "TS속도표시판_80",

                         "RM우회전금지", "RM좌회전금지", "RM직진금지", "RM우회전", "RM좌회전", "RM직진", "RM유턴", "RM횡단예고",
                         "RM정지선", "RM횡단보도",
                         "RM속도제한", "RM속도제한_30", "RM속도제한_50", "RM속도제한_80",
                         "don't care"]
    CATEGORY_REMAP = {"사람": "Person", "승용차": "Car", "트럭": "Truck", "버스": "Bus", "이륜차": "Motorcycle",
                      "신호등": "Traffic light", "자전거": "Bicycle", "삼각콘": "Cone", "차선규제봉": "Lane_stick", "과속방지턱": "Bump",
                      "don't care": "Don't Care",

                      "TS이륜차금지": "TS_NO_TW", "TS우회전금지": "TS_NO_RIGHT", "TS좌회전금지": "TS_NO_LEFT",
                      "TS유턴금지": "TS_NO_TURN", "TS주정차금지": "TS_NO_STOP", "TS자전거전용": "TS_Only_Bic",
                      "TS유턴": "TS_U_TURN", "TS어린이보호": "TS_CHILDREN", "TS횡단보도": "TS_CROSSWK",
                      "TS좌회전": "TS_GO_LEFT",
                      "TS속도표시판": "TS_SPEED_LIMIT_ETC",
                      # "TS속도표시판_30": "TS_SPEED_LIMIT", "TS속도표시판_50": "TS_SPEED_LIMIT", "TS속도표시판_80": "TS_SPEED_LIMIT",
                      "TS속도표시판_30": "TS_SPEED_LIMIT_30", "TS속도표시판_50": "TS_SPEED_LIMIT_50", "TS속도표시판_80": "TS_SPEED_LIMIT_80",

                      "RM우회전금지": "RM_NO_RIGHT", "RM좌회전금지": "RM_NO_LEFT", "RM직진금지": "RM_NO_STR",
                      "RM우회전": "RM_GO_RIGHT", "RM좌회전": "RM_GO_LEFT", "RM직진": "RM_GO_STR", "RM유턴": "RM_U_TURN",
                      "RM횡단예고": "RM_ANN_CWK", "RM정지선": "RM_STOP", "RM횡단보도": "RM_CROSSWK",
                      "RM속도제한": "RM_SPEED_LIMIT_ETC",
                      # "RM속도제한_30": "RM_SPEED_LIMIT", "RM속도제한_50": "RM_SPEED_LIMIT", "RM속도제한_80": "RM_SPEED_LIMIT"
                      "RM속도제한_30": "RM_SPEED_LIMIT_30", "RM속도제한_50": "RM_SPEED_LIMIT_50", "RM속도제한_80": "RM_SPEED_LIMIT_80"

                      }
    LANE_TYPES = ["차선1", "차선2", "차선3", "차선4", "RM정지선"]
    # LANE_REMAP = {"차선1": "Lane1", "차선2": "Lane2", "차선3": "Lane3", "차선4": "Lane4", "RM정지선": "Stop_Line"}
    LANE_REMAP = {"차선1": "Lane", "차선2": "Lane", "차선3": "Lane", "차선4": "Lane", "RM정지선": "Stop_Line"}


class TfrParams:
    MIN_PIX = {'train': {"Bgd": 0, "Person": 0, "Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0,
                         "Traffic light": 0, "Traffic sign": 0, "Road mark": 0, "Bicycle": 0, "Cone": 0, "Lane_stick": 0,
                         "Bump": 0
                         },
               'val': {"Bgd": 0, "Person": 0, "Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0,
                       "Traffic light": 0, "Traffic sign": 0, "Road mark": 0, "Bicycle": 0, "Cone": 0, "Lane_stick": 0,
                       "Bump": 0
                       }
               }

    CATEGORY_NAMES = {"major": ["Bgd", "Person", "Car", "Truck", "Bus", "Motorcycle", "Traffic light",
                                "Traffic sign", "Road mark", "Bicycle", "Cone", "Lane_stick", "Bump"],
                      "sign": ["TS_NO_TW", "TS_NO_RIGHT", "TS_NO_LEFT", "TS_NO_TURN", "TS_NO_STOP",
                               "TS_Only_Bic", "TS_U_TURN", "TS_CHILDREN", "TS_CROSSWK", "TS_GO_LEFT",
                               "TS_SPEED_LIMIT"],
                      "mark": ["RM_NO_RIGHT", "RM_NO_LEFT", "RM_NO_STR", "RM_GO_RIGHT", "RM_GO_LEFT", "RM_GO_STR",
                               "RM_U_TURN", "RM_ANN_CWK", "RM_STOP", "RM_CROSSWK", "RM_SPEED_LIMIT"
                              ],
                      "sign_speed": ["TS_SPEED_LIMIT_30", "TS_SPEED_LIMIT_50", "TS_SPEED_LIMIT_80", "TS_SPEED_LIMIT_ETC"],
                      "mark_speed": ["RM_SPEED_LIMIT_30", "RM_SPEED_LIMIT_50", "RM_SPEED_LIMIT_80", "RM_SPEED_LIMIT_ETC"],
                      "dont": ["Don't Care"],
                      "lane": ["Lane", "Stop_Line"]
                      }


class TrainParams:
    @classmethod
    def get_pred_composition(cls, minor_ctgr, speed_limit, iou_aware, categorized=False):
        cls_composition = {"category": len(TfrParams.CATEGORY_NAMES["major"])}
        if minor_ctgr:
            cls_composition["sign_ctgr"] = len(TfrParams.CATEGORY_NAMES["sign"])
            cls_composition["mark_ctgr"] = len(TfrParams.CATEGORY_NAMES["mark"])
        if speed_limit:
            cls_composition["sign_speed"] = len(TfrParams.CATEGORY_NAMES["sign_speed"])
            cls_composition["mark_speed"] = len(TfrParams.CATEGORY_NAMES["mark_speed"])

        reg_composition = {"yxhw": 4, "object": 1, "distance": 1}
        if iou_aware:
            reg_composition["ioup"] = 1
        composition = {"cls": cls_composition, "reg": reg_composition}

        if categorized:
            out_composition = {name: sum(list(subdic.values())) for name, subdic in composition.items()}
        else:
            out_composition = dict()
            for names, subdic in composition.items():
                out_composition.update(subdic)

        return out_composition


assert list(TfrParams.MIN_PIX["train"].keys()) == TfrParams.CATEGORY_NAMES["major"]
assert list(TfrParams.MIN_PIX["val"].keys()) == TfrParams.CATEGORY_NAMES["major"]
