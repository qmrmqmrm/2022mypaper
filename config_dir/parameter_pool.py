import numpy as np


class LossComb:
    STANDARD = {"iou": ([1., 1., 1.], "IouLoss"), "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                "category": ([1., 1., 1.], "MajorCategoryLoss")}
    UPLUS_BASIC = {"iou": ([1., 1., 1.], "CiouLoss"), "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                   "category": ([1, 1, 1.], "MajorCategoryLoss"), "distance": ([1., 1., 1.], "DistanceLoss")}
    UPLUS_SCALE_WEIGHT = {"iou": ([1., 1., 1.], "CiouLoss"), "object": ([2., 2., 2.], "BoxObjectnessLoss", 3., 1.),
                          "category": ([2, 2, 3.], "MajorCategoryLoss"), "distance": ([1., 1., 1.], "DistanceLoss")}
    UPLUS_MINOR_WEIGHT = {"sign_ctgr": ([2., 1., 1.], "MinorCategoryLoss", "sign_ctgr"),
                          "mark_ctgr": ([2., 1., 1.], "MinorCategoryLoss", "mark_ctgr"),
                          }
    UPLUS_SPEED_WEIGHT = {"sign_speed": ([2, 1, 1.], "MinorSpeedCategoryLoss", "sign_ctgr", "sign_speed"),
                          "mark_speed": ([2, 1, 1.], "MinorSpeedCategoryLoss", "mark_ctgr", "mark_speed"),
                          }

    UPLUS_LANE_WEIGHT = {"laneness": (3., "LanenessLoss", 1, 1),
                         "lane_fpoints": (10000., "FpointLoss"),
                         "lane_centerness": (5., "CenternessLoss", 3, 1),
                         "lane_category": (10., "LaneCategLoss")}

    FULL_COMBINATION = {"basic": UPLUS_SCALE_WEIGHT, "minor": UPLUS_MINOR_WEIGHT,
                        "speed": UPLUS_SPEED_WEIGHT, "lane": UPLUS_LANE_WEIGHT}


class TrainingPlan:
    UPLUS_PLAN = [
        # ("uplus", 1, 0.0000001, LossComb.FULL_COMBINATION, True),
        ("uplus", 40, 0.0001,  LossComb.FULL_COMBINATION, True),
        ("uplus", 40, 0.00001,  LossComb.FULL_COMBINATION, True),
        ("uplus", 40, 0.000001, LossComb.FULL_COMBINATION, True),
    ]


class UplusParams:
    CATEGORIES_TO_USE = ["사람", "차", "차량/트럭", "차량/버스", "차량/오토바이", "신호등",
                         "차량/자전거", "삼각콘", "표지판/이륜 통행 금지", "표지판/유턴 금지", "표지판/주정차 금지",
                         "표지판/어린이 보호", "표지판/횡단보도", "노면표시/직진",
                         "노면표시/횡단보도 예고", "노면표시/직진금지", "노면표시/횡단보도", "don't care"]
    CATEGORY_REMAP = {"사람": "Pedestrian", "차량/자전거": "Bicycle", "차": "Car", "차량/트럭": "Truck",
                      "차량/버스": "Bus", "신호등": "Traffic light", "삼각콘": "Cone", "don't care": "Don't Care",
                      "차량/오토바이": "Motorcycle",

                      "표지판/유턴 금지": "TS_NO_TURN",
                      "표지판/이륜 통행 금지": "TS_NO_TW",
                      "표지판/주정차 금지": "TS_NO_STOP",
                      "표지판/어린이 보호": "TS_CHILDREN",
                      "표지판/횡단보도": "TS_CROSSWK",

                      "노면표시/직진": "RM_GO_STR",
                      "노면표시/횡단보도 예고": "RM_ANN_CWK", "노면표시/직진금지": "RM_NO_STR",
                      "노면표시/횡단보도": "RM_CROSSWK",
                      }
    LANE_TYPES = ["차선/황색 단선 실선", "차선/백색 단선 실선", "차선/황색 단선 점선", "차선/백색 단선 점선",
                  "차선/황색 겹선 실선"]
    LANE_REMAP = {"차선/황색 단선 실선": "YSL", "차선/백색 단선 실선": "WSL", "차선/황색 단선 점선": "YSDL",
                  "차선/백색 단선 점선": "WSDL", "차선/황색 겹선 실선": "YDL"}


class Uplus2Params:
    CATEGORIES_TO_USE = ["보행자", "승용차", "트럭", "버스", "이륜차", "신호등", "자전거", "삼각콘",
                         "차선규제봉", "과속방지턱", "포트홀",

                         "TS이륜차금지", "TS우회전금지", "TS좌회전금지", "TS유턴금지", "TS주정차금지", "TS자전거전용", "TS유턴",
                         "TS어린이보호", "TS횡단보도", "TS좌회전",
                         "TS속도제한_기타", "TS속도제한_30", "TS속도제한_50", "TS속도제한_80",

                         "RM우회전금지", "RM좌회전금지", "RM직진금지", "RM우회전", "RM좌회전", "RM직진", "RM유턴", "RM횡단예고",
                         "RM횡단보도",
                         "RM속도제한_기타", "RM속도제한_30", "RM속도제한_50", "RM속도제한_80",
                         "don't care"]
    CATEGORY_REMAP = {"보행자": "Pedestrian",  "승용차": "Car", "트럭": "Truck", "버스": "Bus",
                      "이륜차": "Motorcycle", "신호등": "Traffic light", "자전거": "Bicycle", "삼각콘": "Cone",
                      "차선규제봉": "Lane_stick", "과속방지턱": "Bump", "포트홀": "Pothole",
                      "don't care": "Don't Care", "lane don't care": "Lane Don't Care",

                      "TS이륜차금지": "TS_NO_TW", "TS우회전금지": "TS_NO_RIGHT", "TS좌회전금지": "TS_NO_LEFT",
                      "TS유턴금지": "TS_NO_TURN", "TS주정차금지": "TS_NO_STOP", "TS자전거전용": "TS_Only_Bic",
                      "TS유턴": "TS_U_TURN", "TS어린이보호": "TS_CHILDREN", "TS횡단보도": "TS_CROSSWK",
                      "TS좌회전": "TS_GO_LEFT",
                      "TS속도제한_기타": "TS_SPEED_LIMIT_ETC", "TS속도제한_30": "TS_SPEED_LIMIT_30",
                      "TS속도제한_50": "TS_SPEED_LIMIT_50", "TS속도제한_80": "TS_SPEED_LIMIT_80",

                      "RM우회전금지": "RM_NO_RIGHT", "RM좌회전금지": "RM_NO_LEFT", "RM직진금지": "RM_NO_STR",
                      "RM우회전": "RM_GO_RIGHT", "RM좌회전": "RM_GO_LEFT", "RM직진": "RM_GO_STR", "RM유턴": "RM_U_TURN",
                      "RM횡단예고": "RM_ANN_CWK", "RM횡단보도": "RM_CROSSWK",
                      "RM속도제한_기타": "RM_SPEED_LIMIT_ETC", "RM속도제한_30": "RM_SPEED_LIMIT_30",
                      "RM속도제한_50": "RM_SPEED_LIMIT_50", "RM속도제한_80": "RM_SPEED_LIMIT_80"
                      }
    LANE_TYPES = ["차선1", "차선2", "차선3", "차선4", "RM정지선"]
    LANE_REMAP = {"차선1": "Lane", "차선2": "Lane", "차선3": "Lane", "차선4": "Lane", "RM정지선": "Stop_Line"}


class TfrParams:
    # # [300 0 0 0] crop 픽셀 50% Road makr 30
    # MIN_PIX = {
    #     'train': {"Bgd": 0, "Pedestrian": 68, "Rider": 54, "Car": 87, "Truck": 98, "Bus": 150, "Motorcycle": 38,
    #               "Traffic light": 41, "Pedestrian light": 30, "Traffic sign": 26, "Road mark": 20, "Bicycle": 38,
    #               "Cone": 34, "Lane_stick": 30, "Bump": 75, "Pothole": 0
    #               },
    #     'val': {"Bgd": 0, "Pedestrian": 76, "Rider": 60, "Car": 98, "Truck": 110, "Bus": 168, "Motorcycle": 42,
    #             "Traffic light": 46, "Pedestrian light": 34, "Traffic sign": 29, "Road mark": 20, "Bicycle": 42,
    #             "Cone": 38, "Lane_stick": 34, "Bump": 85, "Pothole": 0
    #             },
    # }

    # # [450, 180, 25, 180] crop 픽셀 80% Road makr 30
    # MIN_PIX = {
    #     'train': {"Bgd": 0, "Pedestrian": 86, "Rider": 48, "Car": 110, "Truck": 124, "Bus": 230, "Motorcycle": 48,
    #               "Traffic light": 52, "Pedestrian light": 38, "Traffic sign": 33, "Road mark": 30, "Bicycle": 48,
    #               "Cone": 43, "Lane_stick": 38, "Bump": 96, "Pothole": 0
    #               },
    #     'val': {"Bgd": 0, "Pedestrian": 97, "Rider": 54, "Car": 124, "Truck": 140, "Bus": 259, "Motorcycle": 54,
    #             "Traffic light": 59, "Pedestrian light": 43, "Traffic sign": 37, "Road mark": 30, "Bicycle": 54,
    #             "Cone": 48, "Lane_stick": 43, "Bump": 108, "Pothole": 0
    #             },
    # }
    MIN_PIX = {
        'train': {"Bgd": 0, "Pedestrian": 86, "Car": 110, "Truck": 124, "Bus": 230, "Motorcycle": 48,
                  "Traffic light": 52, "Traffic sign": 33, "Road mark": 30, "Bicycle": 48,
                  "Cone": 43, "Lane_stick": 38, "Bump": 96, "Pothole": 0
                  },
        'val': {"Bgd": 0, "Pedestrian": 97, "Car": 124, "Truck": 140, "Bus": 259, "Motorcycle": 54,
                "Traffic light": 59, "Traffic sign": 37, "Road mark": 30, "Bicycle": 54,
                "Cone": 48, "Lane_stick": 43, "Bump": 108, "Pothole": 0
                },
    }

    LANE_MIN_PIX = {'train': {"Bgd": 0, "Lane": 55, "Stop_Line": 55},
                    'val': {"Bgd": 0, "Lane": 50, "Stop_Line": 50},
                    }

    CATEGORY_NAMES = {"major": ["Bgd", "Pedestrian", "Car", "Truck", "Bus", "Motorcycle", "Traffic light",
                                "Traffic sign", "Road mark", "Bicycle", "Cone", "Lane_stick",
                                "Bump", "Pothole"],
                      "sign": ["TS_NO_TW", "TS_NO_RIGHT", "TS_NO_LEFT", "TS_NO_TURN", "TS_NO_STOP",
                               "TS_Only_Bic", "TS_U_TURN", "TS_CHILDREN", "TS_CROSSWK", "TS_GO_LEFT",
                               "TS_SPEED_LIMIT"],
                      "mark": ["RM_NO_RIGHT", "RM_NO_LEFT", "RM_NO_STR", "RM_GO_RIGHT", "RM_GO_LEFT", "RM_GO_STR",
                               "RM_U_TURN", "RM_ANN_CWK", "RM_CROSSWK", "RM_SPEED_LIMIT"
                              ],
                      "sign_speed": ["TS_SPEED_LIMIT_30", "TS_SPEED_LIMIT_50", "TS_SPEED_LIMIT_80", "TS_SPEED_LIMIT_ETC"],
                      "mark_speed": ["RM_SPEED_LIMIT_30", "RM_SPEED_LIMIT_50", "RM_SPEED_LIMIT_80", "RM_SPEED_LIMIT_ETC"],
                      "dont": ["Don't Care"],
                      "lane": ["Bgd", "Lane", "Stop_Line"],
                      "dont_lane": ["Lane Don't Care"],
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
        composition = {"reg": reg_composition, "cls": cls_composition}

        if categorized:
            out_composition = {name: sum(list(subdic.values())) for name, subdic in composition.items()}
        else:
            out_composition = dict()
            for names, subdic in composition.items():
                out_composition.update(subdic)

        return out_composition


assert list(TfrParams.MIN_PIX["train"].keys()) == TfrParams.CATEGORY_NAMES["major"]
assert list(TfrParams.MIN_PIX["val"].keys()) == TfrParams.CATEGORY_NAMES["major"]
