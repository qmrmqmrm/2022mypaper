import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import math

import utils.framework.util_function as uf
from model.framework.deformable_conv_v2.deformable_layer import DeformableConv2D
from utils.util_class import MyExceptionToCatch
import config as cfg


class CustomConv2D:
    CALL_COUNT = -1

    def __init__(self, kernel_size=3, strides=1, padding="same", activation="leaky_relu", scope=None, bn=True,
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.random_normal_initializer(stddev=0.001)):
        # save arguments for Conv2D layer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope
        self.bn = bn
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

    def __call__(self, x, filters, name=None):
        CustomConv2D.CALL_COUNT += 1
        index = CustomConv2D.CALL_COUNT
        name = f"conv{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name

        x = layers.Conv2D(filters, self.kernel_size, self.strides, self.padding,
                          use_bias=not self.bn,
                          kernel_regularizer=tf.keras.regularizers.l2(0.001),
                          kernel_initializer=self.kernel_initializer,
                          bias_initializer=self.bias_initializer, name=name
                          )(x)

        if self.bn:
            x = layers.BatchNormalization()(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "swish":
            x = tf.nn.swish(x)
        elif self.activation is False:
            x = x
        else:
            raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {self.activation}")

        return x


class CustomSeparableConv2D:
    CALL_COUNT = -1

    def __init__(self, kernel_size=3, strides=1, padding="same", activation="leaky_relu", scope=None, bn=True,
                 **kwargs):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope
        self.bn = bn
        self.kwargs = kwargs

    def __call__(self, x, filters, name=None):
        CustomSeparableConv2D.CALL_COUNT += 1
        index = CustomSeparableConv2D.CALL_COUNT
        name = f"spconv{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name
        x = layers.SeparableConv2D(filters, self.kernel_size, self.strides, self.padding,
                                   use_bias=not self.bn, name=name,
                                   **self.kwargs
                                   )(x)

        if self.bn:
            x = layers.BatchNormalization()(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "swish":
            x = tf.nn.swish(x)
        elif self.activation is False:
            x = x
        else:
            raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {self.activation}")

        return x


class CustomDeformConv2D:
    CALL_COUNT = -1

    def __init__(self, scope=None):
        # save arguments for Conv2D layer
        self.scope = scope

    def __call__(self, x, filters, name=None):
        CustomDeformConv2D.CALL_COUNT += 1
        index = CustomDeformConv2D.CALL_COUNT
        name = f"dcn{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name

        x = DeformableConv2D(filters, name=name)(x)
        return x


class CustomMax2D:
    CALL_COUNT = -1

    def __init__(self, pool_size=3, strides=1, padding="same", scope=None):
        # save arguments for Conv2D layer
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.scope = scope

    def __call__(self, x, name=None):
        CustomMax2D.CALL_COUNT += 1
        index = CustomMax2D.CALL_COUNT
        name = f"maxpool{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name

        x = layers.MaxPooling2D(self.pool_size, self.strides, self.padding, name=name)(x)
        return x

class NonMaximumSuppressionBox:
    def __init__(self, max_out=cfg.NmsInfer.MAX_OUT,
                 iou_thresh=cfg.NmsInfer.IOU_THRESH,
                 score_thresh=cfg.NmsInfer.SCORE_THRESH,
                 category_names=cfg.Dataloader.CATEGORY_NAMES["major"],
                 sign_ctgr=cfg.Dataloader.CATEGORY_NAMES["sign"],
                 mark_ctgr=cfg.Dataloader.CATEGORY_NAMES["mark"],
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.category_names = category_names
        self.sign_ctgr = sign_ctgr
        self.mark_ctgr = mark_ctgr

    def __call__(self, pred, max_out=None, iou_thresh=None, score_thresh=None, merged=False, is_inst=False):
        self.max_out = max_out if max_out is not None else self.max_out
        self.iou_thresh = iou_thresh if iou_thresh is not None else self.iou_thresh
        self.score_thresh = score_thresh if score_thresh is not None else self.score_thresh

        if is_inst:
            nms_res = self.inst_nms(pred)
        else:
            nms_res = self.pure_nms(pred, merged)
        return nms_res

    # @tf.function
    def pure_nms(self, pred, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        if merged is False:
            pred = self.append_anchor_inds(pred)
            pred = self.merged_scale(pred)

        boxes = uf.convert_box_format_yxhw_to_tlbr(pred["yxhw"])  # (batch, N, 4)
        categories = tf.argmax(pred["category"], axis=-1)  # (batch, N)
        if cfg.ModelOutput.MINOR_CTGR:
            sign_ctgr = tf.cast(categories == self.category_names.index("Traffic sign"), dtype=tf.float32) * \
                        tf.cast(tf.argmax(pred["sign_ctgr"], axis=-1), dtype=tf.float32)
            mark_ctgr = tf.cast(categories == self.category_names.index("Road mark"), dtype=tf.float32) * \
                        tf.cast(tf.argmax(pred["mark_ctgr"], axis=-1), dtype=tf.float32)
            minor_ctgr = sign_ctgr + mark_ctgr
            if cfg.ModelOutput.SPEED_LIMIT:
                sign_speed = tf.cast(sign_ctgr == self.sign_ctgr.index("TS_SPEED_LIMIT"), dtype=tf.float32) * \
                             tf.cast(tf.argmax(pred["sign_speed"], axis=-1), dtype=tf.float32)
                mark_speed = tf.cast(mark_ctgr == self.mark_ctgr.index("RM_SPEED_LIMIT"), dtype=tf.float32) * \
                             tf.cast(tf.argmax(pred["mark_speed"], axis=-1), dtype=tf.float32)
                speed_ctgr = sign_speed + mark_speed
            else:
                speed_ctgr = tf.zeros_like(categories, dtype=tf.float32)
        else:
            minor_ctgr = tf.zeros_like(categories, dtype=tf.float32)
            speed_ctgr = tf.zeros_like(categories, dtype=tf.float32)

        best_probs = tf.reduce_max(pred["category"], axis=-1)  # (batch, N)
        objectness = pred["object"][..., 0]  # (batch, N)
        scores = objectness * best_probs  # (batch, N)
        batch, numbox, numctgr = pred["category"].shape

        distances = pred["distance"][..., 0]
        anchor_inds = pred["anchor_ind"][..., 0]

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(1, numctgr):
            ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)  # (batch, N)
            ctgr_boxes = boxes * ctgr_mask[..., tf.newaxis]  # (batch, N, 4)

            ctgr_scores = scores * ctgr_mask  # (batch, N)
            for frame_idx in range(batch):
                selected_indices = tf.image.non_max_suppression(
                    boxes=ctgr_boxes[frame_idx],
                    scores=ctgr_scores[frame_idx],
                    max_output_size=self.max_out[ctgr_idx],
                    iou_threshold=self.iou_thresh[ctgr_idx],
                    score_threshold=self.score_thresh[ctgr_idx],
                )
                # zero padding that works in tf.function
                numsel = tf.shape(selected_indices)[0]
                zero = tf.ones((self.max_out[ctgr_idx] - numsel), dtype=tf.int32) * -1
                selected_indices = tf.concat([selected_indices, zero], axis=0)
                batch_indices[frame_idx].append(selected_indices)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
        batch_indices = tf.stack(batch_indices, axis=0)  # (batch, K*max_output)
        valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
        batch_indices = tf.maximum(batch_indices, 0)

        # list of (batch, N) -> (batch, N, 4)
        categories = tf.cast(categories, dtype=tf.float32)
        # "bbox": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, "best_probs": 1, "score": 1, "anchor_inds": 1
        result = tf.stack([objectness, categories, minor_ctgr, speed_ctgr, distances,
                           best_probs, scores, anchor_inds], axis=-1)

        result = tf.concat([pred["yxhw"], result], axis=-1)  # (batch, N, 10)
        result = tf.gather(result, batch_indices, batch_dims=1)  # (batch, K*max_output, 10)
        result = result * valid_mask[..., tf.newaxis]  # (batch, K*max_output, 10)
        return result

    def append_anchor_inds(self, pred):
        pred["anchor_ind"] = []
        num_anchor = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        for scale in range(len(cfg.ModelOutput.FEATURE_SCALES)):
            for key in pred:
                if key != "whole":
                    fmap_shape = pred[key][scale].shape[:-1]
                    break
            fmap_shape = (*fmap_shape, 1)

            ones_map = tf.ones(fmap_shape, dtype=tf.float32)
            anchor_list = range(scale * num_anchor, (scale + 1) * num_anchor)
            pred["anchor_ind"].append(self.anchor_indices(ones_map, anchor_list))
        return pred

    def inst_nms(self, pred):
        boxes = uf.convert_box_format_yxhw_to_tlbr(pred["yxhw"])  # (batch, N, 4)
        batch, numbox, numctgr = pred["category"].shape

        batch_indices = [[] for i in range(batch)]
        for frame_idx in range(batch):
            selected_indices = tf.image.non_max_suppression(
                boxes=boxes[frame_idx],
                scores=pred["score"][frame_idx][..., 0],
                max_output_size=numbox,
                iou_threshold=0.8,
            )
            # zero padding that works in tf.function
            numsel = tf.shape(selected_indices)[0]
            zero = tf.ones((numbox - numsel), dtype=tf.int32) * -1
            selected_indices = tf.concat([selected_indices, zero], axis=0)
            batch_indices[frame_idx].append(selected_indices)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
        batch_indices = tf.stack(batch_indices, axis=0)  # (batch, K*max_output)
        valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
        batch_indices = tf.maximum(batch_indices, 0)

        # list of (batch, N) -> (batch, N, 4)
        # "bbox": 4, "object": 1, "category": 1, "minor_ctgr": 1, "speed_ctgr": 1, "distance": 1, "best_probs": 1, "score": 1, "anchor_inds": 1
        result = tf.stack([pred["object"][..., 0], pred["category"][..., 0], pred["minor_ctgr"][..., 0],
                           pred["speed_ctgr"][..., 0], pred["distance"][..., 0], pred["ctgr_prob"][..., 0],
                           pred["score"][..., 0], pred["anchor_ind"][..., 0]], axis=-1)

        result = tf.concat([pred["yxhw"], result], axis=-1)  # (batch, N, 10)
        result = tf.gather(result, batch_indices, batch_dims=1)  # (batch, K*max_output, 10)
        result = result * valid_mask[..., tf.newaxis]  # (batch, K*max_output, 10)
        return result

    def merged_scale(self, pred):
        slice_keys = list(pred.keys())  # ['yxhw', 'object', 'category']
        merged_pred = {}
        # merge pred features over scales
        for key in slice_keys:
            if key != "whole":
                merged_pred[key] = tf.concat(pred[key], axis=1)  # (batch, N, dim)
        return merged_pred

    def anchor_indices(self, ones_map, anchor_list):
        batch, hwa, _ = ones_map.shape
        num_anchor = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        anchor_index = tf.cast(anchor_list, dtype=tf.float32)[..., tf.newaxis]
        split_anchor_shape = tf.reshape(ones_map, (batch, hwa // num_anchor, num_anchor, 1))

        split_anchor_map = split_anchor_shape * anchor_index
        merge_anchor_map = tf.reshape(split_anchor_map, (batch, hwa, 1))

        return merge_anchor_map

    def compete_diff_categories(self, nms_res, foo_ctgr, bar_ctgr, iou_thresh, score_thresh):
        """
        :param nms_res: (batch, numbox, 10)
        :return:
        """
        batch, numbox = nms_res.shape[:2]
        boxes = nms_res[..., :4]
        category = nms_res[..., 5:6]
        score = nms_res[..., -1:]

        foo_ctgr = self.category_names.index(foo_ctgr)
        bar_ctgr = self.category_names.index(bar_ctgr)
        boxes_tlbr = uf.convert_box_format_yxhw_to_tlbr(boxes)
        batch_survive_mask = []
        for frame_idx in range(batch):
            foo_mask = tf.cast(category[frame_idx] == foo_ctgr, dtype=tf.float32)
            bar_mask = tf.cast(category[frame_idx] == bar_ctgr, dtype=tf.float32)
            target_mask = foo_mask + bar_mask
            target_score_mask = foo_mask + (bar_mask * 0.9)
            target_boxes = boxes_tlbr[frame_idx] * target_mask
            target_score = score[frame_idx] * target_score_mask

            selected_indices = tf.image.non_max_suppression(
                boxes=target_boxes,
                scores=target_score[:, 0],
                max_output_size=20,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh,
            )
            if tf.size(selected_indices) != 0:
                selected_onehot = tf.one_hot(selected_indices, depth=numbox, axis=-1)  # (20, numbox)
                survive_mask = 1 - target_mask + tf.reduce_max(selected_onehot, axis=0)[..., tf.newaxis]  # (numbox,)
                batch_survive_mask.append(survive_mask)

        if len(batch_survive_mask) == 0:
            return nms_res

        batch_survive_mask = tf.stack(batch_survive_mask, axis=0)  # (batch, numbox)
        nms_res = nms_res * batch_survive_mask
        return nms_res


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)

        return result


class NonMaximumSuppressionLane:
    def __init__(self, max_out=cfg.NmsInfer.LANE_MAX_OUT,
                 overlap_thresh=cfg.NmsInfer.LANE_OVERLAP_THRESH,
                 score_thresh=cfg.NmsInfer.LANE_SCORE_THRESH,
                 ):
        self.max_out = max_out
        self.overlap_thresh = overlap_thresh
        self.score_thresh = score_thresh
        self.imshape = cfg.Datasets.DATASET_CONFIG.INPUT_RESOLUTION

    def __call__(self, pred, feat_shape, max_out=None, overlap_thresh=None, score_thresh=None, merged=False, flat_indices=None):
        self.max_out = max_out if max_out is not None else self.max_out
        self.overlap_thresh = overlap_thresh if overlap_thresh is not None else self.overlap_thresh
        self.score_thresh = score_thresh if score_thresh is not None else self.score_thresh

        nms_res = self.nms_lane(pred, feat_shape, flat_indices)
        return nms_res

    def nms_lane(self, pred, feat_shape, flat_indices=None):
        """
        pred : dict("laneness : list((4,2560,1))
                    "five_points : list((4,2560,10))
                    "lane_centerness" : list((4,2560,1))
                    "lane_category" : list((4,2560,3))
                    )
        """
        """
        after is list
        pred : dict("laneness : (4,2560,1)
                    "five_points : (4,2560,10)
                    "lane_centerness" : (4,2560,1)
                    "lane_category" : (4,2560,3)
                    )
        """
        pred = self.strip_list(pred)
        five_points = pred["lane_fpoints"]  # (b, hw, 10)
        # TODO
        pixel_fpoints = self.scale_upto_pixel(five_points, feat_shape)
        line_params, line_spts, line_epts = self.compute_line_segment(pixel_fpoints)
        overlap = self.compute_overlap(line_params, line_spts, line_epts, flat_indices)
        batch_indices, valid_mask = self.nms_lane_overlap(pred, overlap)
        lane_ctgr = tf.argmax(pred["lane_category"], axis=-1)
        centerness = pred["lane_centerness"][..., 0]
        lane_ctgr = tf.cast(lane_ctgr, dtype=tf.float32)
        result = tf.stack([centerness, lane_ctgr], axis=-1)
        result = tf.concat([five_points, result], axis=-1)
        result = tf.gather(result, batch_indices, batch_dims=1)  # (batch, max_output, 12)

        result = result * valid_mask[..., tf.newaxis]  # (batch, max_output, 12)
        return result

    def strip_list(self, pred):
        pred_ = {}
        for key, value in pred.items():
            if isinstance(value, list):
                pred_[key] = value[0]
            else:
                pred_ = pred
        return pred_

    def scale_upto_pixel(self, five_points, feat_shape):
        b, hw, c = five_points.shape
        fpoints = tf.reshape(five_points, (b, feat_shape[0], feat_shape[1], -1, 2))
        return fpoints * self.imshape

    def compute_line_segment(self, fpoints):
        # b, hw, c = five_points.shape
        # fpoints = tf.reshape(five_points, (b, feat_shape[0], feat_shape[1], -1, 2))     # (b,h,w,5,2)
        fpoints_t = tf.transpose(fpoints, perm=[0, 1, 2, 4, 3])                         # (b,h,w,2,5)
        ptp = tf.matmul(fpoints_t, fpoints)                     # (b,h,w,2,2)
        det = tf.linalg.det(ptp)[..., tf.newaxis, tf.newaxis]   # (b,h,w,1,1)
        # EPS = 1e-6
        # singular_mask = tf.cast(tf.abs(det) < EPS, tf.float32)
        # ptp = ptp + singular_mask * tf.eye(2)                   # (b,h,w,2,2)
        # inv = tf.linalg.inv(ptp) * (1 - singular_mask)          # (b,h,w,2,2)
        inv = (1 / (det + 1e-6)) * tf.concat(
            [tf.concat([tf.convert_to_tensor(ptp[..., 1:2, 1:2]), tf.convert_to_tensor(-ptp[..., 0:1, 1:2])], axis=-1),
             tf.concat([tf.convert_to_tensor(-ptp[..., 1:2, 0:1]), tf.convert_to_tensor(ptp[..., 0:1, 0:1])], axis=-1)],
            axis=-2)
        ptpinv = tf.matmul(inv, fpoints_t)                      # (b,h,w,2,2) * (b,h,w,2,5) = (b,h,w,2,5)
        b, h, w, c1, c2 = ptpinv.shape
        y = tf.ones((b, h, w, c2, 1), dtype=tf.float32)         # (b,h,w,5,1)
        param = tf.matmul(ptpinv, y)                            # (b, h, w, 2, 5)  * (b, h ,w, 5, 1) = (b,h,w,2,1)
        # param: line parameter (a,b) in ax+by=1
        param = tf.squeeze(param)                               # (b,h,w,2)
        spts = fpoints[..., 0, :]                               # (b,h,w,2)
        epts = fpoints[..., 1, :]                               # (b,h,w,2)
        numerator = tf.reduce_sum(spts * param, axis=-1, keepdims=True) - 1     # (b,h,w,1)
        denominator = tf.linalg.norm(param, axis=-1, keepdims=True)             # (b,h,w,1)
        # (b,h,w,2) = (b,h,w,2) - (b,h,w,1)/(b,h,w,1) * (b,h,w,2)
        line_spts = spts - numerator / denominator * param
        numerator = tf.reduce_sum(epts * param, axis=-1, keepdims=True) - 1     # (b,h,w,1)
        line_epts = epts - numerator / denominator * param
        return param, line_spts, line_epts                      # (b,h,w,2)

    def compute_overlap(self, line_params, spts, epts, flat_indices=None):
        b, h, w, c = spts.shape
        line_params = tf.reshape(line_params, (b, h*w, 1, c))  # (b,hw,1,2)
        spts = tf.reshape(spts, (b, 1, h * w, c))              # (b,1,hw,2)
        epts = tf.reshape(epts, (b, 1, h * w, c))              # (b,1,hw,2)
        param_norm = tf.linalg.norm(line_params, axis=-1)      # (b,hw,1)
        # dist = |ax+by-1| / sqrt(a^2+b^2), shape=(b,h,w)
        # (b,hw,hw) / (b,hw,1) = (b,hw,hw)
        dist_spts = tf.abs(tf.reduce_sum(line_params * spts, axis=-1) - 1) / param_norm
        dist_epts = tf.abs(tf.reduce_sum(line_params * epts, axis=-1) - 1) / param_norm
        dist = (dist_spts + dist_epts) / 2
        dist = tf.minimum(dist, tf.transpose(dist, perm=[0, 2, 1]))
        overlap = tf.maximum((50 - dist) / 50, 0)
        return overlap

    def nms_lane_overlap(self, pred, overlap):
        centerness = pred["lane_centerness"][..., 0]
        best_probs = tf.reduce_max(pred["lane_category"], axis=-1)
        categories = tf.argmax(pred["lane_category"], axis=-1)
        scores = centerness * best_probs
        batch, numbox, numctgr = pred["lane_category"].shape
        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(1, numctgr):
            ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)  # (batch, N)
            ctgr_overlap = overlap * ctgr_mask[..., tf.newaxis]  # (batch, N, M)

            ctgr_scores = scores * ctgr_mask  # (batch, N)
            for frame_idx in range(batch):
                selected_indices = tf.image.non_max_suppression_overlaps(ctgr_overlap[frame_idx],
                                                                         ctgr_scores[frame_idx],
                                                                         max_output_size=self.max_out[ctgr_idx],
                                                                         overlap_threshold=self.overlap_thresh[ctgr_idx],
                                                                         score_threshold=self.score_thresh[ctgr_idx])
                numsel = tf.shape(selected_indices)[0]
                padding = tf.ones((self.max_out[ctgr_idx] - numsel), dtype=tf.int32) * -1
                selected_indices = tf.concat([selected_indices, padding], axis=0)
                batch_indices[frame_idx].append(selected_indices)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
        batch_indices = tf.stack(batch_indices, axis=0)  # (batch, K*max_output)
        valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
        batch_indices = tf.maximum(batch_indices, 0)
        return batch_indices, valid_mask


def sigmoid_with_margin(x, delta=cfg.Architecture.SIGMOID_DELTA, low=0, high=1):
    y = tf.sigmoid(x)
    z = (high - low + 2 * delta) * y + low - delta
    # (1 + 1 + 0 ) * 1 - 1 - 0= 1
    # (1 + 1 + 0 ) * 0 - 1 - 0 = -1
    return z


def inv_sigmoid_with_margin(z, delta=cfg.Architecture.SIGMOID_DELTA, low=0, high=1, eps=1e-7):
    y = (z - low + delta) / (high - low + 2*delta)
    assert tf.reduce_all(tf.logical_and(z > low - delta, z < high + delta))
    assert tf.reduce_all(tf.logical_and(y >= 0, y <= 1))
    x = tf.clip_by_value(y, eps, 1 - eps)
    x = x / (1 - x)
    x = tf.math.log(x)
    return x
