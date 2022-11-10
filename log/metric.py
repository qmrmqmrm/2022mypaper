import numpy as np
import utils.framework.util_function as uf
import config as cfg


def count_true_positives(grtr, pred, grtr_dontcare, num_ctgr, iou_thresh=cfg.Validation.TP_IOU_THRESH, per_class=False,
                         num_sign_ctgr=None, num_mark_ctgr=None):
    """
    :param grtr: slices of features["bboxes"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param grtr_dontcare: slices of features["dontcare"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :param num_minor_ctgr
    :return:
    """
    splits = split_true_false(grtr, pred, grtr_dontcare, iou_thresh)
    # ========== use split instead grtr, pred
    grtr_valid_tp = splits["grtr_tp"]["yxhw"][..., 2:3] > 0
    grtr_valid_fn = splits["grtr_fn"]["yxhw"][..., 2:3] > 0
    pred_valid_tp = splits["pred_tp"]["yxhw"][..., 2:3] > 0
    pred_valid_fp = splits["pred_fp"]["yxhw"][..., 2:3] > 0
    if cfg.ModelOutput.MINOR_CTGR:
        grtr_valid_sign_tp = np.where(((grtr_valid_tp > 0) * (splits["grtr_tp"]["category"][..., :1] == 8)),
                                      splits["grtr_tp"]["minor_ctgr"][..., :1], -1) >= 0
        grtr_valid_mark_tp = np.where(((grtr_valid_tp > 0) * (splits["grtr_tp"]["category"][..., :1] == 9)),
                                      splits["grtr_tp"]["minor_ctgr"][..., :1], -1) >= 0

        grtr_valid_sign_fn = np.where(((grtr_valid_fn > 0) * (splits["grtr_fn"]["category"][..., :1] == 8)),
                                      splits["grtr_fn"]["minor_ctgr"][..., :1], -1) >= 0
        grtr_valid_mark_fn = np.where(((grtr_valid_fn > 0) * (splits["grtr_fn"]["category"][..., :1] == 9)),
                                      splits["grtr_fn"]["minor_ctgr"][..., :1], -1) >= 0

        pred_valid_sign_tp = np.where(((pred_valid_tp > 0) * (splits["pred_tp"]["category"][..., :1] == 8)),
                                      splits["pred_tp"]["minor_ctgr"][..., :1], -1) >= 0
        pred_valid_mark_tp = np.where(((pred_valid_tp > 0) * (splits["pred_tp"]["category"][..., :1] == 9)),
                                      splits["pred_tp"]["minor_ctgr"][..., :1], -1) >= 0

        pred_valid_sign_fp = np.where(((pred_valid_fp > 0) * (splits["pred_fp"]["category"][..., :1] == 9)),
                                      splits["pred_fp"]["minor_ctgr"][..., :1], -1) >= 0
        pred_valid_mark_fp = np.where(((pred_valid_fp > 0) * (splits["pred_fp"]["category"][..., :1] == 10)),
                                      splits["pred_fp"]["minor_ctgr"][..., :1], -1) >= 0

    if per_class:
        grtr_tp_count = count_per_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        grtr_fn_count = count_per_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        pred_tp_count = count_per_class(splits["pred_tp"], pred_valid_tp, num_ctgr)
        pred_fp_count = count_per_class(splits["pred_fp"], pred_valid_fp, num_ctgr)
        if cfg.ModelOutput.MINOR_CTGR:
            grtr_sign_tp_count = count_per_minor_class(splits["grtr_tp"], grtr_valid_sign_tp, num_sign_ctgr)
            grtr_sign_fn_count = count_per_minor_class(splits["grtr_fn"], grtr_valid_sign_fn, num_sign_ctgr)
            pred_sign_tp_count = count_per_minor_class(splits["pred_tp"], pred_valid_sign_tp, num_sign_ctgr)
            pred_sign_fp_count = count_per_minor_class(splits["pred_fp"], pred_valid_sign_fp, num_sign_ctgr)

            grtr_mark_tp_count = count_per_minor_class(splits["grtr_tp"], grtr_valid_mark_tp, num_mark_ctgr)
            grtr_mark_fn_count = count_per_minor_class(splits["grtr_fn"], grtr_valid_mark_fn, num_mark_ctgr)
            pred_mark_tp_count = count_per_minor_class(splits["pred_tp"], pred_valid_mark_tp, num_mark_ctgr)
            pred_mark_fp_count = count_per_minor_class(splits["pred_fp"], pred_valid_mark_fp, num_mark_ctgr)
            return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                    "pred": (pred_tp_count + pred_fp_count), "sign_trpo": pred_sign_tp_count,
                    "sign_grtr": (grtr_sign_tp_count + grtr_sign_fn_count),
                    "sign_pred": (pred_sign_tp_count + pred_sign_fp_count),
                    "mark_trpo": pred_mark_tp_count,
                    "mark_grtr": (grtr_mark_tp_count + grtr_mark_fn_count),
                    "mark_pred": (pred_mark_tp_count + pred_mark_fp_count)
                    }
        return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                "pred": (pred_tp_count + pred_fp_count)}
    else:
        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


def split_true_false(grtr, pred, grtr_dc, iou_thresh):
    pred_valid, pred_far = split_pred_far(pred)
    grtr_far, grtr_valid = split_grtr_far(pred_far, grtr, iou_thresh)
    splits = split_tp_fp_fn(pred_valid, grtr_valid, iou_thresh)
    fp_pred, dc_pred = split_dontcare_pred(splits["pred_fp"], grtr_dc)
    splits["pred_fp"] = fp_pred
    splits["pred_dc"] = dc_pred
    splits["grtr_dc"] = grtr_dc
    splits["pred_far"] = pred_far
    splits["grtr_far"] = grtr_far
    return splits


def split_pred_far(pred):
    pred_far_mask = pred["distance"] > cfg.Validation.DISTANCE_LIMIT
    pred_valid = {key: (val * (1 - pred_far_mask).astype(np.float32)) for key, val in pred.items()}
    pred_far = {key: (val * pred_far_mask).astype(np.float32) for key, val in pred.items()}
    return pred_valid, pred_far


def split_grtr_far(pred_far, grtr, iou_thresh):
    iou_far = uf.compute_iou_general(grtr["yxhw"], pred_far["yxhw"]).numpy()
    best_iou_far = np.max(iou_far, axis=-1)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh)
    iou_match = np.expand_dims(best_iou_far > iou_thresh, axis=-1)
    grtr_valid = {key: (val * (1 - iou_match)).astype(np.float32) for key, val in grtr.items()}
    grtr_far = {key: (val * iou_match).astype(np.float32) for key, val in grtr.items()}
    grtr_far["iou"] = best_iou_far
    return grtr_far, grtr_valid


def split_tp_fp_fn(pred, grtr, iou_thresh):
    batch, M, _ = pred["category"].shape
    valid_mask = grtr["object"]
    iou = uf.compute_iou_general(grtr["yxhw"], pred["yxhw"]).numpy()  # (batch, N, M)

    # ctgr_mask = grtr["category"][:, :, np.newaxis, 0] == pred["category"][:, np.newaxis, :, 0]
    # iou_ctgr = iou * ctgr_mask
    best_iou = np.max(iou, axis=-1)  # (batch, N)

    best_idx = np.argmax(iou, axis=-1)  # (batch, N)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh)
    iou_match = best_iou > iou_thresh  # (batch, N)

    pred_ctgr_aligned = numpy_gather(pred["category"], best_idx, 1)  # (batch, N)

    ctgr_match = grtr["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)

    if cfg.ModelOutput.MINOR_CTGR:
        minor_mask = (grtr["category"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["major"].index("Traffic sign")) | \
                     (grtr["category"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["major"].index("Road mark"))
        pred_minor_ctgr_aligned = numpy_gather(pred["minor_ctgr"], best_idx, 1)
        minor_ctgr_match = grtr["minor_ctgr"][..., 0] == pred_minor_ctgr_aligned
        grtr_tp_mask = np.expand_dims((1 - minor_mask) * grtr_tp_mask[..., 0]
                                      + minor_mask * minor_ctgr_match, axis=-1).astype(bool)

        if cfg.ModelOutput.SPEED_LIMIT:
            speed_mask = (grtr["minor_ctgr"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["sign"].index("TS_SPEED_LIMIT")) | \
                         (grtr["minor_ctgr"][..., 0] == cfg.Dataloader.CATEGORY_NAMES["mark"].index("RM_SPEED_LIMIT"))
            pred_speed_ctgr_aligned = numpy_gather(pred["speed_ctgr"], best_idx, 1)
            speed_ctgr_match = grtr["speed_ctgr"][..., 0] == pred_speed_ctgr_aligned
            grtr_tp_mask = np.expand_dims((1 - speed_mask) * grtr_tp_mask[..., 0]
                                          + speed_mask * speed_ctgr_match, axis=-1).astype(bool)

    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def split_dontcare_pred(pred_fp, grtr_dc):
    B, M, _ = pred_fp["category"].shape
    iou_dc = uf.compute_iou_general(grtr_dc["yxhw"], pred_fp["yxhw"])
    best_iou_dc = np.max(iou_dc, axis=-1)  # (batch, D)
    grtr_dc["iou"] = best_iou_dc
    dc_match = np.expand_dims(best_iou_dc > 0.5, axis=-1)  # (batch, D)
    best_idx_dc = np.argmax(iou_dc, axis=-1)
    pred_dc_mask = indices_to_binary_mask(best_idx_dc, dc_match, M)  # (batch, M, 1)
    dc_pred = {key: val * pred_dc_mask for key, val in pred_fp.items()}
    fp_pred = {key: val * (1 - pred_dc_mask) for key, val in pred_fp.items()}
    return fp_pred, dc_pred


def indices_to_binary_mask(best_idx, valid_mask, depth):
    best_idx_onehot = one_hot(best_idx, depth) * valid_mask
    binary_mask = np.expand_dims(np.max(best_idx_onehot, axis=1), axis=-1)  # (batch, M, 1)
    return binary_mask.astype(np.float32)


def get_iou_thresh_per_class(grtr_ctgr, tp_iou_thresh):
    ctgr_idx = grtr_ctgr.astype(np.int32)
    tp_iou_thresh = np.asarray(tp_iou_thresh, np.float32)
    iou_thresh = numpy_gather(tp_iou_thresh, ctgr_idx)
    return iou_thresh[..., 0]


def count_per_class(boxes, mask, num_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    boxes_ctgr = boxes["category"][..., 0].astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_ctgr) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count


def count_per_minor_class(boxes, mask, num_minor_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """

    boxes_ctgr = boxes["minor_ctgr"][..., 0].astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_minor_ctgr) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count


def count_per_lane_class(lanes, mask, num_ctgr):
    """
    :param lanes: slices of object info {'lanes': (batch, N, 10), 'lane_category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    lanes_ctgr = lanes["lane_category"][..., 0].astype(np.int32)  # (batch, N')
    lanes_onehot = one_hot(lanes_ctgr, num_ctgr) * mask
    lanes_count = np.sum(lanes_onehot, axis=(0, 1))
    return lanes_count


def one_hot(grtr_category, category_shape):
    one_hot_data = np.eye(category_shape)[grtr_category.astype(np.int32)]
    return one_hot_data


def numpy_gather(params, index, dim=0):
    if dim is 1:
        batch_list = []
        for i in range(params.shape[0]):
            batch_param = params[i]
            batch_index = index[i]
            batch_gather = np.take(batch_param, batch_index)
            batch_list.append(batch_gather)
        gathar_param = np.stack(batch_list)
    else:
        gathar_param = np.take(params, index)
    return gathar_param


def count_true_positives_lane(grtr, pred, num_ctgr, img_shape, iou_thresh=cfg.Validation.LANE_TP_IOU_THRESH,
                              per_class=False, is_train=True):
    splits = split_lane_true_false(grtr, pred, iou_thresh, img_shape, is_train)

    grtr_valid_tp = splits["grtr_tp"]["lane_fpoints"][..., 1:2] > 0
    grtr_valid_fn = splits["grtr_fn"]["lane_fpoints"][..., 1:2] > 0
    pred_valid_tp = splits["pred_tp"]["lane_fpoints"][..., 1:2] > 0
    pred_valid_fp = splits["pred_fp"]["lane_fpoints"][..., 1:2] > 0
    if per_class:
        grtr_tp_count = count_per_lane_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        grtr_fn_count = count_per_lane_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        pred_tp_count = count_per_lane_class(splits["pred_tp"], pred_valid_tp, num_ctgr)
        pred_fp_count = count_per_lane_class(splits["pred_fp"], pred_valid_fp, num_ctgr)
        return {"trpo_lane": pred_tp_count, "grtr_lane": (grtr_tp_count + grtr_fn_count),
                "pred_lane": (pred_tp_count + pred_fp_count)}
    else:
        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo_lane": trpo_count, "grtr_lane": grtr_count, "pred_lane": pred_count}


def split_lane_true_false(grtr, pred, iou_thresh, img_shape, is_train):
    splits = split_lane_tp_fp_fn(pred, grtr, iou_thresh, img_shape, is_train)
    return splits


def split_lane_tp_fp_fn(pred, grtr, iou_thresh, img_shape, is_train):
    batch, M, _ = pred["lane_category"].shape
    valid_mask = grtr["lane_centerness"]

    if is_train:
        iou = uf.compute_lane_iou(grtr["lane_fpoints"], pred["lane_fpoints"])  # (batch, N, M)
    else:
        iou = uf.compute_lane_iou_with_cv2(grtr["lane_fpoints"], pred["lane_fpoints"], img_shape)  # (batch, N, M)
    best_iou = np.max(iou, axis=-1)  # (batch, N)
    best_idx = np.argmax(iou, axis=-1)  # (batch, N)

    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["lane_category"], iou_thresh)
    iou_match = best_iou > iou_thresh  # (batch, N)

    pred_ctgr_aligned = numpy_gather(pred["lane_category"], best_idx, 1)  # (batch, N, 8)

    ctgr_match = grtr["lane_category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)

    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]

    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}
    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


