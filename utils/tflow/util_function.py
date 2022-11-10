import sys
import numpy as np
import cv2
import tensorflow as tf

import config as cfg
import config_dir.util_config as uc


def set_gpu_configs():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def to_float_image(im_tensor):
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.float32)


def to_uint8_image(im_tensor):
    im_tensor = tf.clip_by_value(im_tensor, -1, 1)
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.uint8)


def convert_box_format_tlbr_to_yxhw(boxes_tlbr):
    """
    :param boxes_tlbr: type=tf.Tensor or np.array, shape=(numbox, dim) or (batch, numbox, dim)
    :return:
    """
    boxes_yx = (boxes_tlbr[..., 0:2] + boxes_tlbr[..., 2:4]) / 2  # center y,x
    boxes_hw = boxes_tlbr[..., 2:4] - boxes_tlbr[..., 0:2]  # y2,x2 = y1,x1 + h,w
    output = [boxes_yx, boxes_hw]
    output = concat_box_output(output, boxes_tlbr)
    return output


def convert_box_format_yxhw_to_tlbr(boxes_yxhw):
    """
    :param boxes_yxhw: type=tf.Tensor or np.array, shape=(numbox, dim) or (batch, numbox, dim)
    :return:
    """
    boxes_tl = boxes_yxhw[..., 0:2] - (boxes_yxhw[..., 2:4] / 2)  # y1,x1 = cy,cx + h/2,w/2
    boxes_br = boxes_tl + boxes_yxhw[..., 2:4]  # y2,x2 = y1,x1 + h,w
    output = [boxes_tl, boxes_br]
    output = concat_box_output(output, boxes_yxhw)
    return output


def concat_box_output(output, boxes):
    num, dim = boxes.shape[-2:]
    # if there is more than bounding box, append it  e.g. category, distance
    if dim > 4:
        auxi_data = boxes[..., 4:]
        output.append(auxi_data)

    if tf.is_tensor(boxes):
        output = tf.concat(output, axis=-1)
        output = tf.cast(output, boxes.dtype)
    else:
        output = np.concatenate(output, axis=-1)
        output = output.astype(boxes.dtype)
    return output


def convert_box_scale_01_to_pixel(boxes_norm):
    """
    :boxes_norm: yxhw format boxes scaled into 0~1
    :return:
    """
    img_res = cfg.Datasets.DATASET_CONFIG.INPUT_RESOLUTION
    img_res = [*img_res, *img_res]
    output = [boxes_norm[..., :4] * img_res]
    output = concat_box_output(output, boxes_norm)
    return output


def merge_and_slice_features(featin, is_gt: bool, feat_type: str):
    featout = {}
    if feat_type == "inst_box":
        composition = uc.get_bbox_composition(is_gt)
        featout["merged"] = featin
        featout.update(slice_feature(featin, composition))

    if feat_type == "inst_dc":
        composition = uc.get_bbox_composition(is_gt)
        featout["merged"] = featin
        featout.update(slice_feature(featin, composition))

    if feat_type == "inst_lane":
        composition = uc.get_lane_composition(is_gt)
        featout["merged"] = featin
        featout.update(slice_feature(featin, composition))

    if feat_type.startswith("feat_box"):
        composition = uc.get_channel_composition(is_gt)
        newfeat = []
        featout["whole"] = featin
        for scale_data in featin:
            newfeat.append(slice_feature(scale_data, composition))
        newfeat = scale_align_featmap(newfeat)
        featout.update(newfeat)

    if feat_type.startswith("feat_lane"):
        composition = uc.get_lane_channel_composition(is_gt)
        newfeat = []
        featout["whole"] = featin
        for scale_data in featin:
            newfeat.append(slice_feature(scale_data, composition))
        newfeat = scale_align_featmap(newfeat)
        featout.update(newfeat)

    return featout


def slice_feature(feature, channel_composition):
    """
    :param feature: (batch, grid_h, grid_w, anchors, dims)
    :param channel_composition:
    :return: sliced feature maps
    """
    names, channels = list(channel_composition.keys()), list(channel_composition.values())
    slices = tf.split(feature, channels, axis=-1)
    slices = dict(zip(names, slices))  # slices = {'yxhw': (B,H,W,A,4), 'object': (B,H,W,A,1), ...}
    return slices


def merge_dim_hwa(feature_map):
    """
    :param feature_map: (batch, grid_h, grid_w, anchor, 5+K)
    :return: (batch, grid_h * grid_w * anchor, 5+K)
    """
    batch, grid_h, grid_w, anchor, featdim = feature_map.shape
    merged_feat = tf.reshape(feature_map, (batch, grid_h * grid_w * anchor, featdim))
    return merged_feat


def scale_align_featmap(features):
    slice_keys = list(features[0].keys())
    aline_feat = dict()
    for key in slice_keys:
        aline_feat[key] = [features[scale_name][key] for scale_name in range(len(features))]
    return aline_feat


def compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: GT bounding boxes in yxhw format (batch, HWA, D(>4))
    :param pred_yxhw: predicted bounding boxes aligned with GT in yxhw format (batch, HWA, D(>4))
    :return: iou (batch, HWA)
    """
    if grtr_tlbr is None:
        grtr_tlbr = convert_box_format_yxhw_to_tlbr(grtr_yxhw)
    if pred_tlbr is None:
        pred_tlbr = convert_box_format_yxhw_to_tlbr(pred_yxhw)
    inter_tl = tf.maximum(grtr_tlbr[..., :2], pred_tlbr[..., :2])
    inter_br = tf.minimum(grtr_tlbr[..., 2:4], pred_tlbr[..., 2:4])
    inter_hw = inter_br - inter_tl
    positive_mask = tf.cast(inter_hw > 0, dtype=tf.float32)
    inter_hw = inter_hw * positive_mask
    inter_area = inter_hw[..., 0] * inter_hw[..., 1]
    pred_area = pred_yxhw[..., 2] * pred_yxhw[..., 3]
    grtr_area = grtr_yxhw[..., 2] * grtr_yxhw[..., 3]
    iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)
    return iou


def compute_iou_general(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: GT bounding boxes in yxhw format (batch, N1, D1(>4))
    :param pred_yxhw: predicted bounding box in yxhw format (batch, N2, D2(>4))
    :return: iou (batch, N1, N2)
    """
    grtr_yxhw = tf.expand_dims(grtr_yxhw, axis=-2)  # (batch, N1, 1, D1)
    pred_yxhw = tf.expand_dims(pred_yxhw, axis=-3)  # (batch, 1, N2, D2)

    if grtr_tlbr is None:
        grtr_tlbr = convert_box_format_yxhw_to_tlbr(grtr_yxhw)  # (batch, N1, 1, D1)
    if pred_tlbr is None:
        pred_tlbr = convert_box_format_yxhw_to_tlbr(pred_yxhw)  # (batch, 1, N2, D2)

    inter_tl = tf.maximum(grtr_tlbr[..., :2], pred_tlbr[..., :2])  # (batch, N1, N2, 2)
    inter_br = tf.minimum(grtr_tlbr[..., 2:4], pred_tlbr[..., 2:4])  # (batch, N1, N2, 2)
    inter_hw = inter_br - inter_tl  # (batch, N1, N2, 2)
    inter_hw = tf.maximum(inter_hw, 0)
    inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (batch, N1, N2)

    pred_area = pred_yxhw[..., 2] * pred_yxhw[..., 3]  # (batch, 1, N2)
    grtr_area = grtr_yxhw[..., 2] * grtr_yxhw[..., 3]  # (batch, N1, 1)
    iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)  # (batch, N1, N2)
    return iou


def compute_lane_iou(grtr, pred):
    grtr = tf.convert_to_tensor(grtr)
    pred = tf.convert_to_tensor(pred)
    overlap = []
    grtr_lane_num = grtr.shape[1]
    pred_lane_num = pred.shape[1]
    for batch_grtr, batch_pred in zip(grtr, pred):
        # batch_grtr_np = batch_grtr.numpy()
        # batch_pred_np = batch_pred.numpy()
        grtr_ind = np.where(batch_grtr[..., 1] > 0)[0]
        pred_ind = np.where(batch_pred[..., 1] > 0)[0]
        batch_grtr = batch_grtr[batch_grtr[..., 1] > 0]
        batch_pred = batch_pred[batch_pred[..., 1] > 0]

        gt_line_params, gt_line_spts, gt_line_epts = compute_line_segment(batch_grtr, "grtr")
        pred_line_params, pred_line_spts, pred_line_epts = compute_line_segment(batch_pred, "pred")
        batch_overlap = compute_overlap(gt_line_params, gt_line_spts, gt_line_epts,
                                        pred_line_params, pred_line_spts, pred_line_epts)

        batch_overlap = overlap_zero_padding(batch_overlap, grtr_lane_num, pred_lane_num, grtr_ind, pred_ind)
        overlap.append(batch_overlap)
    overlap = tf.stack(overlap, axis=0)
    return overlap.numpy()


def overlap_zero_padding(overlap, grtr_num, pred_num, grtr_ind, pred_ind):
    zero_pad = np.zeros((grtr_num, pred_num))
    n, m = overlap.shape
    zero_pad[grtr_ind[...,np.newaxis], pred_ind[np.newaxis,...]] = overlap.numpy()
    return tf.convert_to_tensor(zero_pad)


def compute_lane_iou_with_cv2(grtr, pred, image_shape):
    """
    grtr : ndarray (4,10,10)
    pred : ndarray (4,10,10)
    """
    batch, n, points = grtr.shape
    batch, m, points = pred.shape
    h, w = image_shape
    image_shape = np.array(image_shape)
    batch_overlap = np.zeros((batch, n, m), dtype=np.float32)
    for batch_idx in range(batch):
        batch_grtr = grtr[batch_idx]
        batch_pred = pred[batch_idx]
        grtr_ind = np.where(batch_grtr[..., 0] > 0)
        pred_ind = np.where(batch_pred[..., 0] > 0)
        for grtr_idx in grtr_ind[0]:
            grtr_zero_image = np.zeros((h, w, 3))
            grtr_lane = batch_grtr[grtr_idx].reshape(-1, 2) * image_shape
            grtr_image = draw_line(grtr_lane, grtr_zero_image, (0, 0, 255))
            gt_pixel = np.sum(grtr_image[..., 2] == 255)

            for pred_idx in pred_ind[0]:
                pred_zero_image = np.zeros((h, w, 3))
                pred_lane = batch_pred[pred_idx].reshape(-1, 2) * image_shape
                pred_image = draw_line(pred_lane, pred_zero_image, (255, 0, 0))
                pred_pixel = np.sum(pred_image[..., 0] == 255)
                lane_image = (pred_image + grtr_image)
                # cv2.imshow("nms", lane_image)
                # cv2.waitKey(100)
                overlap_pixel = np.sum((lane_image[..., 0] == 255) & (lane_image[..., 2] == 255))
                overlap = overlap_pixel / (gt_pixel + pred_pixel - overlap_pixel)
                batch_overlap[batch_idx][grtr_idx][pred_idx] = overlap
    return batch_overlap


def compute_line_segment(five_points, name):
    """
    param five_points : (lane_num,10)

    """
    lane_num, c = five_points.shape
    fpoints = tf.reshape(five_points, (lane_num, -1, 2))  # (b,h,w, 5, 2)
    fpoints_t = tf.transpose(fpoints, perm=[0, 2, 1])
    ptp = tf.matmul(fpoints_t, fpoints)
    det = tf.linalg.det(ptp)[..., tf.newaxis, tf.newaxis]
    inv = (1 / det) * tf.concat(
        [tf.concat([tf.convert_to_tensor(ptp[..., 1:2, 1:2]), tf.convert_to_tensor(-ptp[..., 0:1, 1:2])], axis=-1),
         tf.concat([tf.convert_to_tensor(-ptp[..., 1:2, 0:1]), tf.convert_to_tensor(ptp[..., 0:1, 0:1])], axis=-1)],
        axis=-2)
    ptpinv = tf.matmul(inv, fpoints_t)
    n, c1, c2 = ptpinv.shape
    y = tf.ones((n, c2, 1), dtype=tf.float32)
    x = tf.matmul(ptpinv, y)
    spts = fpoints[..., 0:1, :]
    epts = fpoints[..., -1:, :]
    line_params = tf.transpose(x, perm=[0, 2, 1])

    line_spts = spts - tf.matmul((tf.matmul(spts, x) - 1) / tf.linalg.norm(x), line_params)
    line_epts = epts - tf.matmul((tf.matmul(epts, x) - 1) / tf.linalg.norm(x), line_params)
    return tf.reshape(line_params, [-1, 2]), tf.reshape(line_spts, [-1, 2]), tf.reshape(line_epts, [-1, 2])


def compute_overlap(gt_line_params, gt_spts, gt_epts, pred_line_params, pred_spts, pred_epts):
    gt_lane_num, c = gt_spts.shape
    gt_norm_x = tf.linalg.norm(gt_line_params, axis=-1, keepdims=True)
    pred_norm_x = tf.linalg.norm(pred_line_params, axis=-1, keepdims=True)
    gt_ones_tenser = tf.ones((gt_lane_num, 1))
    gt_spts = tf.concat([gt_spts, gt_ones_tenser], axis=-1)
    gt_epts = tf.concat([gt_epts, gt_ones_tenser], axis=-1)
    gt_x = tf.concat([gt_line_params, -gt_ones_tenser], axis=-1)
    pred_lane_num, c = pred_spts.shape
    pred_ones_tenser = tf.ones((pred_lane_num, 1))
    pred_spts = tf.concat([pred_spts, pred_ones_tenser], axis=-1)
    pred_epts = tf.concat([pred_epts, pred_ones_tenser], axis=-1)
    pred_x = tf.concat([pred_line_params, -pred_ones_tenser], axis=-1)

    dist_gt_spts = tf.squeeze(1 / gt_norm_x) * tf.abs(tf.matmul(pred_x, tf.transpose(gt_spts)))
    dist_gt_epts = tf.squeeze(1 / gt_norm_x) * tf.abs(tf.matmul(pred_x, tf.transpose(gt_epts)))

    dist_pred_spts = tf.abs(tf.matmul(gt_x, tf.transpose(pred_spts))) * tf.squeeze(1 / pred_norm_x)
    dist_pred_epts = tf.abs(tf.matmul(gt_x, tf.transpose(pred_epts))) * tf.squeeze(1 / pred_norm_x)

    dist_gt = (dist_gt_spts + dist_gt_epts) / 2
    dist_pred = (dist_pred_spts + dist_pred_epts) / 2
    overlap = 1 - tf.minimum(tf.transpose(dist_gt), dist_pred)
    return overlap


def get_meshgrid(height, width):
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    meshgrid = tf.stack([grid_y, grid_x], axis=-1)
    return meshgrid


def draw_line(lines, image, color):
    for i in range(lines.shape[0] - 1):
        cv2.line(image,
                 (int(lines[i, 1]), int(lines[i, 0])),
                 (int(lines[i + 1, 1]), int(lines[i + 1, 0])),
                 color, int(50 * 0.4)
                 )
    return image


def convert_tensor_to_numpy(feature):
    if isinstance(feature, dict):
        dict_feat = dict()
        for key, value in feature.items():
            dict_feat[key] = convert_tensor_to_numpy(value)
        return dict_feat
    elif isinstance(feature, list):
        list_feat = []
        for value in feature:
            list_feat.append(convert_tensor_to_numpy(value))
        return list_feat
    else:
        return feature.numpy()


def convert_to_tensor(value, dtype):
    return tf.convert_to_tensor(value=value, dtype=dtype)


def reduce_sum(value, axis=None):
    return tf.reduce_sum(input_tensor=value, axis=axis)


def reduce_max(value, axis=None):
    return tf.reduce_max(input_tensor=value, axis=axis)


def reduce_min(value, axis=None):
    return tf.reduce_min(input_tensor=value, axis=axis)


def maximum(x, y):
    return tf.maximum(x=x, y=y)


def cast(value, dtype):
    return tf.cast(x=value, dtype=dtype)


def print_structure(title, data, key=""):
    if isinstance(data, list):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    elif isinstance(data, dict):
        for subkey, datum in data.items():
            print_structure(title, datum, f"{key}/{subkey}")
    elif isinstance(data, str):
        print(title, key, data)
    elif isinstance(data, tuple):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    elif data is None:
        print(f'{title} : None')
    elif isinstance(data, int):
        print(title, key, data)
    elif type(data) == np.ndarray:
        print(title, key, data.shape, type(data))
    else:
        print(title, key, data.shape, type(data))


# [B, H, W, A, C]
def merge_scale(features, is_loss=False):
    stacked_feat = {}
    if is_loss:
        slice_keys = list(key for key in features.keys() if "map" in key)  # ['ciou_map', 'object_map', 'category_map']
    else:
        slice_keys = list(features.keys())  # ['yxhw', 'object', 'category']
    for key in slice_keys:
        if key is not "whole":
            # list of (batch, HWA in scale, dim)
            scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
            stacked_feat[key] = scaled_preds
    return stacked_feat


def avg_pool(feat):
    b, h, w, a, c = feat.shape
    feat = tf.reshape(feat, (b, h, w, -1))
    feat = tf.keras.layers.AveragePooling2D((3, 3), 1, "same")(feat)
    return tf.reshape(feat, (b, h, w, a, c))
