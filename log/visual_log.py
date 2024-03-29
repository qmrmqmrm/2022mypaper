import os
import os.path as op
import numpy as np
import cv2

import config as cfg
from log.metric import split_true_false, split_lane_true_false
import utils.framework.util_function as uf


class VisualLog:
    def __init__(self, ckpt_path, epoch):
        self.grtr_log_keys = cfg.Train.LOG_KEYS
        self.pred_log_keys = cfg.Train.LOG_KEYS
        self.vlog_path = op.join(ckpt_path, "vlog", f"ep{epoch:02d}")
        self.visual_heatmap_path = op.join(ckpt_path, "heatmap", f"ep{epoch:02d}") if cfg.Log.VISUAL_HEATMAP else None
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        if not op.isdir(self.visual_heatmap_path):
            os.makedirs(self.visual_heatmap_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["major"])}
        self.sign_ctgr = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["sign"])}
        self.mark_ctgr = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["mark"])}
        self.sign_speed_ctgr = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["sign_speed"])}
        self.mark_speed_ctgr = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["mark_speed"])}
        self.lane_ctgr = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["lane"])}

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        """
        grtr_bbox_augmented = self.exapand_grtr_bbox(grtr, pred)
        splits = split_true_false(grtr_bbox_augmented, pred["inst_box"], grtr["inst_dc"],
                                  cfg.Validation.TP_IOU_THRESH)

        if cfg.ModelOutput.LANE_DET:
            splits_lane = split_lane_true_false(grtr["inst_lane"], pred["inst_lane"],
                                                cfg.Validation.LANE_TP_IOU_THRESH, grtr['image'].shape[1:3],
                                                is_train=False)

        batch = splits["grtr_tp"]["yxhw"].shape[0]

        for i in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_grtr = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"], i, self.grtr_log_keys, (0, 255, 0))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"], i, self.grtr_log_keys, (0, 0, 255))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_dc"], i, self.grtr_log_keys, (100, 100, 100))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_far"], i, self.grtr_log_keys, (255, 100, 100))

            image_pred = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"], i, self.pred_log_keys, (0, 255, 0))
            image_pred = self.draw_boxes(image_pred, splits["pred_fp"], i, self.pred_log_keys, (0, 0, 255))
            image_pred = self.draw_boxes(image_pred, splits["pred_dc"], i, self.pred_log_keys, (100, 100, 100))
            image_pred = self.draw_boxes(image_pred, splits["pred_far"], i, self.pred_log_keys, (255, 100, 100))

            if cfg.ModelOutput.LANE_DET:
                image_grtr = self.draw_lanes(image_grtr, splits_lane['grtr_tp'], i, (0, 255, 0))
                image_grtr = self.draw_lanes(image_grtr, splits_lane['grtr_fn'], i, (0, 0, 255))
                image_pred = self.draw_lanes(image_pred, splits_lane["pred_tp"], i, (0, 255, 0))
                image_pred = self.draw_lanes(image_pred, splits_lane["pred_fp"], i, (0, 0, 255))

            if self.visual_heatmap_path:
                image_zero = uf.to_uint8_image(np.zeros((512, 1280, 3))).numpy()
                self.draw_box_heatmap(grtr, pred, image_zero, i, step, batch)
                if cfg.ModelOutput.LANE_DET:
                    self.draw_lane_heatmap(grtr, pred, image_zero, i, step, batch)

            vlog_image = np.concatenate([image_pred, image_grtr], axis=0)
            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(10)
            filename = op.join(self.vlog_path, f"{step * batch + i:05d}.jpg")
            cv2.imwrite(filename, vlog_image)

    def exapand_grtr_bbox(self, grtr, pred):
        grtr_boxes = uf.merge_scale(grtr["feat_box"])
        pred_boxes = uf.merge_scale(pred["feat_box"])

        best_probs = np.max(pred_boxes["category"], axis=-1, keepdims=True)
        grtr_boxes["pred_ctgr_prob"] = best_probs
        grtr_boxes["pred_object"] = pred_boxes["object"]
        grtr_boxes["pred_score"] = best_probs * pred_boxes["object"]

        batch, _, _ = grtr["inst_box"]["yxhw"].shape
        numbox = cfg.Validation.MAX_BOX
        objectness = grtr_boxes["object"]
        for key in grtr_boxes:
            features = []
            for frame_idx in range(batch):
                valid_mask = (objectness[frame_idx, :, 0] == 1).astype(np.bool)
                feature = grtr_boxes[key]
                feature = feature[frame_idx, valid_mask, :]
                feature = np.pad(feature, [(0, numbox - feature.shape[0]), (0, 0)])
                features.append(feature)

            features = np.stack(features, axis=0)
            grtr_boxes[key] = features

        return grtr_boxes

    def merge_scale_hwa(self, features):
        stacked_feat = {}
        slice_keys = list(features.keys())  # ['yxhw', 'object', 'category']
        for key in slice_keys:
            # list of (batch, HWA in scale, dim)
            # scaled_preds = [features[scale_name][key] for scale_name in range(len(features))]
            scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
            stacked_feat[key] = scaled_preds
        return stacked_feat

    def draw_boxes(self, image, bboxes, frame_idx, log_keys, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param log_keys
        :param color: box color
        :return: box drawn image
        """
        height, width = image.shape[:2]
        box_yxhw = bboxes["yxhw"][frame_idx]  # (N, 4)
        category = bboxes["category"][frame_idx]  # (N, 1)
        minor_ctgr = bboxes["minor_ctgr"][frame_idx]
        speed_ctgr = bboxes["speed_ctgr"][frame_idx]
        valid_mask = box_yxhw[:, 2] > 0  # (N,) h>0

        box_yxhw = box_yxhw[valid_mask, :] * np.array([[height, width, height, width]], dtype=np.float32)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(box_yxhw)  # (N', 4)
        category = category[valid_mask, 0].astype(np.int32)  # (N',)
        minor_ctgr = minor_ctgr[valid_mask, 0].astype(np.int32)
        speed_ctgr = speed_ctgr[valid_mask, 0].astype(np.int32)

        valid_boxes = {}
        for key in log_keys:
            scale = 1 if key == "distance" else 100
            feature = (bboxes[key][frame_idx] * scale)
            feature = feature.astype(np.int32) if key != "distance" else feature
            valid_boxes[key] = feature[valid_mask, 0]

        for i in range(box_yxhw.shape[0]):
            y1, x1, y2, x2 = box_tlbr[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            if cfg.ModelOutput.MINOR_CTGR:
                if annotation == "Traffic sign":
                    annotation = f"{self.sign_ctgr[minor_ctgr[i]]}"
                    if cfg.ModelOutput.SPEED_LIMIT:
                        if annotation == "TS_SPEED_LIMIT":
                            annotation = f"{self.sign_speed_ctgr[speed_ctgr[i]]}"
                elif annotation == "Road mark":
                    annotation = f"{self.mark_ctgr[minor_ctgr[i]]}"
                    if cfg.ModelOutput.SPEED_LIMIT:
                        if annotation == "RM_SPEED_LIMIT":
                            annotation = f"{self.mark_speed_ctgr[speed_ctgr[i]]}"

            for key in log_keys:
                annotation += f",{valid_boxes[key][i]:02d}" if key != "distance" else f",{valid_boxes[key][i]:.2f}"

            cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def draw_lanes(self, image, lanes, frame_idx, color):
        image = image.copy()
        height, width = image.shape[:2]
        fpoints = lanes["lane_fpoints"][frame_idx]  # (N, 10)
        category = lanes["lane_category"][frame_idx]

        valid_mask = fpoints[:, 4] > 0
        fpoints = fpoints[valid_mask, :]
        category = category[valid_mask, 0]

        for n in range(fpoints.shape[0]):
            point = (fpoints[n].reshape(-1, 2) * np.array([height, width])).astype(np.int32)
            annotation = f"{self.lane_ctgr[category[n]]}"
            for i in range(point.shape[0]):
                cv2.circle(image, (point[i, 1], point[i, 0]), 1, color, 6)
            cv2.putText(image, annotation, (point[2, 1], point[2, 0]), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def draw_lane_heatmap(self, grtr, pred, image_zero, i, step, batch):
        feat_shape = pred["feat_lane"]["whole"][0].shape[1:4]
        image_laneness = self.draw_object(image_zero,
                                          grtr["feat_lane"]["laneness"][0],
                                          pred["feat_lane"]["laneness"][0],
                                          i, feat_shape)
        image_center = self.draw_object(image_zero,
                                        grtr["feat_lane"]["lane_centerness"][0],
                                        pred["feat_lane"]["lane_centerness"][0],
                                        i, feat_shape)
        lane_heatmap = np.concatenate([image_laneness, image_center], axis=1)
        filename = op.join(self.visual_heatmap_path, f"{step * batch + i:05d}_lane.jpg")
        cv2.imwrite(filename, lane_heatmap)

    def draw_box_heatmap(self, grtr, pred, image_zero, i, step, batch):
        box_heatmap = list()
        for scale in range(len(cfg.ModelOutput.FEATURE_SCALES)):
            feat_shape = pred["feat_box"]["whole"][scale].shape[1:4]
            v_objectness = self.draw_object(image_zero,
                                            grtr["feat_box"]["object"][scale],
                                            pred["feat_box"]["object"][scale],
                                            i, feat_shape)
            box_heatmap.append(v_objectness)
        box_heatmap = np.concatenate(box_heatmap, axis=1)
        filename = op.join(self.visual_heatmap_path, f"{step * batch + i:05d}_box.jpg")
        cv2.imwrite(filename, box_heatmap)

    def draw_object(self, bev_img, gt_object_feature, pred_objectness_feat, batch_idx, feat_shape):
        # object
        gt_obj_imgs = []
        pred_obj_imgs = []

        org_img = bev_img.copy()
        gt_object_per_image = self.convert_img(gt_object_feature[batch_idx], feat_shape, org_img)
        pred_object_per_image = self.convert_img(pred_objectness_feat[batch_idx], feat_shape, org_img)
        gt_obj_imgs.append(gt_object_per_image)
        pred_obj_imgs.append(pred_object_per_image)
        gt_obj_img = np.concatenate(gt_obj_imgs, axis=1)
        pred_obj_img = np.concatenate(pred_obj_imgs, axis=1)
        obj_img = np.concatenate([pred_obj_img, gt_obj_img], axis=0)
        return obj_img

    def convert_img(self, feature, feat_shape, org_img):
        feature_image = feature.reshape(feat_shape) * 255
        if feature_image.shape[-1] == 1:
            feature_image = cv2.cvtColor(feature_image, cv2.COLOR_GRAY2BGR)
        feature_image = cv2.resize(feature_image, (1280, 512), interpolation=cv2.INTER_NEAREST)
        feature_image = org_img + feature_image
        feature_image[-1, :] = [255, 255, 255]
        feature_image[:, -1] = [255, 255, 255]
        return feature_image
