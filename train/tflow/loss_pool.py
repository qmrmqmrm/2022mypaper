import tensorflow as tf
import numpy as np

import utils.framework.util_function as uf


class LossBase:
    def __call__(self, grtr, pred, auxi, scale):
        dummy_large = tf.reduce_mean(tf.square(pred["feature_l"]))
        dummy_medium = tf.reduce_mean(tf.square(pred["feature_m"]))
        dummy_small = tf.reduce_mean(tf.square(pred["feature_s"]))
        dummy_total = dummy_large + dummy_medium + dummy_small
        return dummy_total, {"dummy_large": dummy_large, "dummy_medium": dummy_medium, "dummy_small": dummy_small}


class CiouLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: complete-iou loss (batch, HWA)
        """
        # object_mask: (batch, HWA, 1)
        object_mask = grtr["object"][scale]
        ciou_loss = self.compute_ciou(grtr["yxhw"][scale], pred["yxhw"][scale]) * object_mask[..., 0]
        # sum over object-containing grid cells
        scalar_loss = tf.reduce_sum(ciou_loss)
        return scalar_loss, ciou_loss

    def compute_ciou(self, grtr_yxhw, pred_yxhw):
        """
        :param grtr_yxhw: (batch, HWA, 4)
        :param pred_yxhw: (batch, HWA, 4)
        :return: ciou loss (batch, HWA)
        """
        grtr_tlbr = uf.convert_box_format_yxhw_to_tlbr(grtr_yxhw)
        pred_tlbr = uf.convert_box_format_yxhw_to_tlbr(pred_yxhw)
        # iou: (batch, HWA)
        iou = uf.compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr, pred_tlbr)
        cbox_tl = tf.minimum(grtr_tlbr[..., :2], pred_tlbr[..., :2])
        cbox_br = tf.maximum(grtr_tlbr[..., 2:], pred_tlbr[..., 2:])
        cbox_hw = cbox_br - cbox_tl
        c = tf.reduce_sum(cbox_hw * cbox_hw, axis=-1)
        center_diff = grtr_yxhw[..., :2] - pred_yxhw[..., :2]
        u = tf.reduce_sum(center_diff * center_diff, axis=-1)
        # NOTE: divide_no_nan results in nan gradient
        # d = tf.math.divide_no_nan(u, c)
        d = u / (c + 1.0e-5)

        # grtr_hw_ratio = tf.math.divide_no_nan(grtr_yxhw[..., 2], grtr_yxhw[..., 3])
        # pred_hw_ratio = tf.math.divide_no_nan(pred_yxhw[..., 2], pred_yxhw[..., 3])
        grtr_hw_ratio = grtr_yxhw[..., 3] / (grtr_yxhw[..., 2] + 1.0e-5)
        pred_hw_ratio = pred_yxhw[..., 3] / (pred_yxhw[..., 2] + 1.0e-5)
        coeff = tf.convert_to_tensor(4.0 / (np.pi * np.pi), dtype=tf.float32)
        v = coeff * tf.pow((tf.atan(grtr_hw_ratio) - tf.atan(pred_hw_ratio)), 2)
        alpha = v / (1 - iou + v)
        penalty = d + alpha * v
        loss = 1 - iou + penalty
        return loss


class IouLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: complete-iou loss (batch, HWA)
        """
        # object_mask: (batch, HWA, 1)
        object_mask = grtr["object"][scale]
        iou_loss = self.compute_iou(grtr["yxhw"][scale], pred["yxhw"][scale]) * object_mask[..., 0]
        # sum over object-containing grid cells
        scalar_loss = tf.reduce_sum(iou_loss)
        return scalar_loss, iou_loss

    def compute_iou(self, grtr_yxhw, pred_yxhw):
        grtr_tlbr = uf.convert_box_format_yxhw_to_tlbr(grtr_yxhw)
        pred_tlbr = uf.convert_box_format_yxhw_to_tlbr(pred_yxhw)
        iou = uf.compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr, pred_tlbr)
        iou_loss = 1 - iou
        return iou_loss


class L1smooth(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        object_mask = grtr["object"][scale]
        # grtr_tlbr = uf.convert_box_format_yxhw_to_tlbr(grtr["yxhw"][scale]) * object_mask
        # pred_tlbr = uf.convert_box_format_yxhw_to_tlbr(pred["yxhw"][scale]) * object_mask

        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)  # (batch, HWA)
        l1_loss = huber_loss(grtr["yxhw"][scale], pred["yxhw"][scale]) * object_mask[..., 0]
        scalar_loss = tf.reduce_sum(l1_loss)
        return scalar_loss, l1_loss


class IouAware(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        """
                :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
                :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
                :param auxi: auxiliary data
                :return: complete-iou loss (batch, HWA)
                """
        # object_mask: (batch, HWA, 1)
        pred_ioup = pred["ioup"]
        object_mask = grtr["object"]
        iou = uf.compute_iou_aligned(grtr["yxhw"], pred["yxhw"])[..., 0]
        iou_aware = tf.keras.losses.binary_crossentropy(iou, pred_ioup, label_smoothing=0.04) * object_mask[..., 0]
        # sum over object-containing grid cells
        scalar_loss = tf.reduce_sum(iou_aware)
        return scalar_loss, iou_aware


class BoxObjectnessLoss(LossBase):
    def __init__(self, pos_weight, neg_weight):
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: objectness loss (batch, HWA)
        """
        grtr_obj = grtr["object"][scale]
        pred_obj = pred["object"][scale]
        ignore_mask = auxi["ignore_mask"]
        conf_focal = tf.pow(grtr_obj - pred_obj, 2)
        obj_loss = tf.keras.losses.binary_crossentropy(grtr_obj, pred_obj, label_smoothing=0.04) * conf_focal[..., 0]
        obj_positive = obj_loss * grtr_obj[..., 0] * self.pos_weight
        obj_negative = obj_loss * (1 - grtr_obj[..., 0]) * ignore_mask * self.neg_weight
        scalar_loss = tf.reduce_sum(obj_positive) + tf.reduce_sum(obj_negative)
        return scalar_loss, obj_loss


class CategoryLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        pass

    def compute_category_loss(self, grtr, pred, mask):
        """
        :param grtr: (batch, HWA, 1)
        :param pred: (batch, HWA, K)
        :param mask: (batch, HWA, K or 1)
        :return: (batch, HWA)
        """
        num_cate = pred.shape[-1]
        grtr_label = tf.cast(grtr, dtype=tf.int32)
        grtr_onehot = tf.one_hot(grtr_label[..., 0], depth=num_cate, axis=-1)[..., tf.newaxis]
        # category_loss: (batch, HWA, K)
        category_loss = tf.losses.binary_crossentropy(grtr_onehot, pred[..., tf.newaxis], label_smoothing=0.04)
        loss_map = category_loss * mask
        loss_map = tf.reduce_sum(loss_map, axis=-1)
        loss = tf.reduce_sum(loss_map)
        return loss, loss_map


class MajorCategoryLoss(CategoryLoss):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        object_mask, valid_category = grtr["object"][scale], auxi["valid_category"]
        scalar_loss, category_loss = self.compute_category_loss(grtr["category"][scale], pred["category"][scale],
                                                                object_mask * valid_category)
        return scalar_loss, category_loss


class MinorCategoryLoss(CategoryLoss):
    def __init__(self, minor_ctgr):
        self.minor_ctgr = minor_ctgr

    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        object_mask = grtr["object"][scale]
        minor_ctgr_index = auxi[self.minor_ctgr]
        loss_mask = object_mask * tf.cast(grtr["category"][scale] == minor_ctgr_index, tf.float32)
        scalar_loss, category_loss = self.compute_category_loss(grtr["minor_ctgr"][scale], pred[self.minor_ctgr][scale],
                                                                loss_mask)
        return scalar_loss, category_loss


class MinorSpeedCategoryLoss(CategoryLoss):
    def __init__(self, minor_ctgr, speed_ctgr):
        self.minor_ctgr = minor_ctgr
        self.speed_ctgr = speed_ctgr

    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        object_mask = grtr["object"][scale]
        speed_ctgr_index = auxi[self.speed_ctgr]
        loss_mask = object_mask * tf.cast(grtr["minor_ctgr"][scale] == speed_ctgr_index, tf.float32)
        scalar_loss, category_loss = self.compute_category_loss(grtr["speed_ctgr"][scale], pred[self.speed_ctgr][scale],
                                                                loss_mask)
        return scalar_loss, category_loss


class DistanceLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: distance loss (batch, HWA)
        """
        object_mask = grtr["object"][scale]
        grtr_dist = grtr["distance"][scale] * object_mask      # (batch, HWA, 1)

        valid_dist_mask = tf.cast(grtr_dist > 0, dtype=tf.float32)
        pred_dist = pred["distance"][scale] * object_mask    # (batch, HWA, 1)

        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)    # (batch, HWA)
        dist_loss = huber_loss(grtr_dist, pred_dist) * valid_dist_mask[..., 0]
        scalar_loss = tf.reduce_sum(dist_loss)
        return scalar_loss, dist_loss


class LaneParamLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT lane feature map slices of some scale, {'angle': (batch, HWA, 1), 'intercept_x', ..., 'object': ..., 'category', ...}
        :param pred: pred. lane feature map slices of some scale, {'angle': (batch, HWA, 1), 'intercept_x', ..., 'object': ..., 'category', ...}
        :param auxi: auxiliary data
        :return: parameter loss (batch, HWA)
        """
        object_mask = grtr["object"]
        grtr_angle, grtr_intercept_x = grtr["angle"], grtr["intercept_x"]
        pred_angle, pred_intercept_x = pred["angle"], pred["intercept_x"]

        lane_angle_loss = (tf.abs(tf.sin(grtr_angle) - tf.sin(pred_angle))) * object_mask
        lane_inter_loss = (tf.abs(grtr_intercept_x - pred_intercept_x)) * object_mask
        lane_angle_loss_scalar = tf.reduce_sum(lane_angle_loss)
        lane_inter_loss_scalar = tf.reduce_sum(lane_inter_loss)
        scalar_loss = lane_angle_loss_scalar + lane_inter_loss_scalar
        param_loss = (lane_angle_loss + lane_inter_loss)[..., 0]
        return scalar_loss, param_loss


class LaneObjectnessLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: objectness loss (batch, HWA)
        """
        grtr_obj = grtr["object"][scale]
        pred_obj = pred["object"][scale]
        conf_focal = tf.pow(grtr_obj - pred_obj, 2)
        obj_loss = tf.keras.losses.binary_crossentropy(grtr_obj, pred_obj, label_smoothing=0.04) * conf_focal[..., 0]
        scalar_loss = tf.reduce_sum(obj_loss)
        return scalar_loss, obj_loss


class LaneCategLoss(CategoryLoss):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT lane feature map slices of some scale, {'angle': (batch, HWA, 1), 'intercept_x', ..., 'object': ..., 'category', ...}
        :param pred: pred. lane feature map slices of some scale, {'angle': (batch, HWA, 1), 'intercept_x', ..., 'object': ..., 'category', ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA)
        """
        object_mask = grtr["object"][scale]
        grtr_cate = grtr["category"][scale]    # (batch, HWA, 1)
        pred_cate = pred["category"][scale]    # (batch, HWA, K)
        scalar_loss, lane_ctgr_loss = self.compute_category_loss(grtr_cate, pred_cate, object_mask)
        return scalar_loss, lane_ctgr_loss
