import albumentations as A
import tensorflow as tf
import numpy as np

import config as cfg
import utils.framework.util_function as uf


def augmentation_factory(augment_probs=None):
    if augment_probs:
        augmenters = []
        for key, prob in augment_probs.items():
            if key == "ColorJitter":
                augmenters.append(A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0, p=prob))
            elif key == "Flip":
                augmenters.append(A.HorizontalFlip(p=prob))
            elif key == "CropResize":
                augmenters.append(A.OneOf([A.RandomSizedCrop((256, 512), 512, 1280, w2h_ratio=2.5, p=prob),
                                  A.RandomScale((-0.5, -0.5), p=prob)], p=1.0))
            elif key == "Blur":
                augmenters.append(A.Blur(p=prob))
        augmenters.append(A.PadIfNeeded(512, 1280, border_mode=0, value=0, always_apply=True))
        aug_func = A.Compose(augmenters, bbox_params=A.BboxParams(format="yolo", min_visibility=0.5,
                                                                  label_fields=["remainder", "dont_remainder"]),
                             additional_targets={"dontbbox": "dontbbox"})
        # "remainder" mean object, class, minor_class, distance. except bbox coord
        total_augment = TotalAugment(aug_func)
    else:
        total_augment = None
    return total_augment


class TotalAugment:
    def __init__(self, augment_objects=None):
        self.augment_objects = augment_objects
        self.max_bbox = cfg.Dataloader.MAX_BBOX_PER_IMAGE

    def __call__(self, features):
        total_image = []
        total_bboxes = []
        total_dontcare = []
        total_lane = []
        total_lpoint = []
        batch_size = features["image"].shape[0]
        for i in range(batch_size):
            image = features["image"][i].numpy()
            inst_box = features["inst_box"][i]
            inst_dc = features["inst_dc"][i]
            lanes_point = features["lanes_point"][i]
            inst_lane = features["inst_lane"][i]
            raw_data = self.preprocess(image, inst_box, inst_dc, lanes_point, inst_lane)
            aug_data = self.transformation(raw_data)
            aug_data = self.post_process(aug_data)
            aug_lpoint = tf.zeros_like(lanes_point)[np.newaxis, ...]
            aug_inst_lane = tf.zeros_like(inst_lane)[np.newaxis, ...]

            total_image.extend([image[np.newaxis, ...], aug_data["image"]])
            total_bboxes.extend([inst_box[np.newaxis, ...], aug_data["total_inst_box"]])
            total_dontcare.extend([inst_dc[np.newaxis, ...], aug_data["total_inst_dc"]])
            total_lane.extend([inst_lane[np.newaxis, ...], aug_inst_lane])
            total_lpoint.extend([lanes_point[np.newaxis, ...], aug_lpoint])
        features["image"] = tf.convert_to_tensor(np.concatenate(total_image, axis=0), dtype=tf.float32)
        features["inst_box"] = tf.convert_to_tensor(np.concatenate(total_bboxes, axis=0), dtype=tf.float32)
        features["inst_dc"] = tf.convert_to_tensor(np.concatenate(total_dontcare, axis=0), dtype=tf.float32)
        features["lanes_point"] = tf.convert_to_tensor(np.concatenate(total_lpoint, axis=0), dtype=tf.float32)
        features["inst_lane"] = tf.convert_to_tensor(np.concatenate(total_lane, axis=0), dtype=tf.float32)
        return features

    def preprocess(self, image, inst_box, inst_dc, lanes_points, inst_lane):
        data = {"image": image}
        yxhw = uf.convert_tensor_to_numpy(inst_box[:, :4])
        remainder = uf.convert_tensor_to_numpy(inst_box[:, 4:])
        dontbbox = uf.convert_tensor_to_numpy(inst_dc[:, :4])
        dont_remainder = uf.convert_tensor_to_numpy(inst_dc[:, 4:])
        valid_mask = yxhw[:, 2] > 0
        yxhw = yxhw[valid_mask, :]
        dontbbox = dontbbox[valid_mask, :]

        data["remainder"] = remainder[valid_mask, :]
        data["dont_remainder"] = dont_remainder[valid_mask, :]
        data["xywh"] = self.convert_coord(yxhw)
        data["dontbbox"] = self.convert_coord(dontbbox)
        return data

    def convert_coord(self, coord):
        if len(coord) < 1:
            convert_coord = np.array([[0, 0, 0, 0]], dtype=np.float32)
        else:
            convert_coord = np.array([coord[:, 1], coord[:, 0], coord[:, 3], coord[:, 2]], dtype=np.float32).T
        return convert_coord

    def transformation(self, data):
        if not np.all(data["xywh"]):
            aug_data = dict()
            aug_data["image"] = data["image"]
            aug_data["bboxes"] = data["xywh"]
            aug_data["remainder"] = data["remainder"]
            aug_data["dontbbox"] = data["dontbbox"]
            aug_data["dont_remainder"] = data["dont_remainder"]
        else:
            augment_data = self.augment_objects(image=data["image"], bboxes=data["xywh"], remainder=data["remainder"],
                                                dontbbox=data["dontbbox"], dont_remainder=data["dont_remainder"])
            aug_data = {key: np.array(val) if isinstance(val, list) else val for key, val in augment_data.items()}
            aug_data["bboxes"] = self.convert_coord(aug_data["bboxes"])
            aug_data["dontbbox"] = self.convert_coord(aug_data["dontbbox"])
        if len(aug_data["remainder"]) == 0:
            # TODO fix hard code
            aug_data["remainder"] = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
            aug_data["dont_remainder"] = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
        return aug_data

    def convert_list_to_numpy(self, data):
        convert_data = np.asarray(data, dtype=np.float32)
        return convert_data

    def post_process(self, aug_data):
        aug_data["image"] = tf.convert_to_tensor(aug_data["image"], dtype=tf.float32)[np.newaxis, ...]
        if aug_data["bboxes"].shape[0] < self.max_bbox and len(aug_data["bboxes"]) != 0:
            bboxes = aug_data["bboxes"]
            remainder = aug_data["remainder"]
            dontbbox = aug_data["dontbbox"]
            dont_remain = aug_data["dont_remainder"]
            aug_data["bboxes"] = np.zeros((self.max_bbox, aug_data["bboxes"].shape[1]), dtype=np.float32)
            aug_data["remainder"] = np.zeros((self.max_bbox, aug_data["remainder"].shape[1]), dtype=np.float32)
            aug_data["bboxes"][:bboxes.shape[0]] = bboxes
            aug_data["remainder"][:remainder.shape[0]] = remainder
            # distance aug value is 0
            aug_data["remainder"][:, -1:] = 0

            aug_data["dontbbox"] = np.zeros((self.max_bbox, aug_data["dontbbox"].shape[1]), dtype=np.float32)
            aug_data["dont_remainder"] = np.zeros((self.max_bbox, aug_data["dont_remainder"].shape[1]),
                                                  dtype=np.float32)
            aug_data["dontbbox"][:dontbbox.shape[0]] = dontbbox
            aug_data["dont_remainder"][:dont_remain.shape[0]] = dont_remain

        elif len(aug_data["bboxes"]) == 0:
            # TODO fix hard code
            aug_data["bboxes"] = np.zeros_like((self.max_bbox, 5), dtype=np.float32)
            aug_data["remainder"] = np.zeros_like((self.max_bbox, 5), dtype=np.float32)

            aug_data["dontbbox"] = np.zeros_like((self.max_bbox, 5), dtype=np.float32)
            aug_data["dont_remainder"] = np.zeros_like((self.max_bbox, 5), dtype=np.float32)
        else:
            aug_data["bboxes"] = aug_data["bboxes"]
            aug_data["remainder"] = aug_data["remainder"]

            aug_data["dontbbox"] = aug_data["dontbbox"]
            aug_data["dont_remainder"] = aug_data["dont_remainder"]

        bbox_total_labels = np.concatenate([aug_data["bboxes"], aug_data["remainder"]], axis=-1)
        dont_total_labels = np.concatenate([aug_data["dontbbox"], aug_data["dont_remainder"]], axis=-1)
        aug_data["total_inst_box"] = tf.convert_to_tensor(bbox_total_labels, dtype=tf.float32)[np.newaxis, ...]
        aug_data["total_inst_dc"] = tf.convert_to_tensor(dont_total_labels, dtype=tf.float32)[np.newaxis, ...]
        return aug_data

