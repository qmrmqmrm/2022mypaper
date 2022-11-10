import os.path as op
import numpy as np
import zipfile
from PIL import Image
from PIL import ImageColor
import json
import cv2

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
from dataloader.data_util import depth_map_to_point_cloud, point_cloud_to_depth_map


class A2D2DriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        return [self.datapath]

    def get_drive_name(self, drive_index):
        return self.split


class A2D2Reader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)
    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path, split):
        zip_file = drive_path
        self.zipinfile = self.load_zipfile(zip_file)
        frame_names = self.zipinfile.namelist()
        frame_names = [name for name in frame_names if '/camera/' in name if name.endswith(".png")]
        frame_names.sort()
        configfile = op.join(op.dirname(zip_file), "cams_lidars.json")
        print("[A2D2Reader] sensor config_dir file:", configfile)
        self.sensor_config = SensorConfig(configfile)
        self.class_color = self.read_class_color()
        if split == "train":
            frame_names = frame_names[:-500]
        else:
            frame_names = frame_names[-500:]
        print("[A2D2Reader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def load_zipfile(self, zip_file):
        return zipfile.ZipFile(zip_file, 'r')

    def read_class_color(self):
        color_class = self.zipinfile.open(op.join("camera_lidar_semantic_bboxes", "class_list.json"))
        class_color = json.load(color_class)
        # convert hex code to [r, g, b] list
        class_color = {class_name: list(ImageColor.getcolor(color_code, "RGB"))
                       for color_code, class_name in class_color.items()
                       if class_name in self.dataset_cfg.CATEGORIES_TO_USE}
        # rgb code to bgr code
        class_color = {class_name: [color_code[2], color_code[1], color_code[0]]
                       for class_name, color_code in class_color.items()}
        return class_color

    def get_image(self, index):
        image_bytes = self.zipinfile.open(self.frame_names[index])
        image = Image.open(image_bytes)
        image = np.array(image, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def get_bboxes(self, index, raw_hw_shape=None):
        """
        :return: bounding boxes in 'yxhw' format
        """
        scale = self.dataset_cfg.SEGMAP_SCALE
        seg_image = self.get_seg_img(index, scale)
        depth_map = self._read_depth_map(index, scale)
        bboxes, categories = self.extract_box(seg_image, depth_map)
        return bboxes, categories

    def get_seg_img(self, index, scale):
        image_file = self.frame_names[index]
        segmentic_name = image_file.replace("/camera/", "/label/").replace("_camera_", "_label_")
        seg_image_bytes = self.zipinfile.open(segmentic_name)
        seg_image = Image.open(seg_image_bytes)
        seg_image = np.array(seg_image, np.uint8)
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
        resize_seg_img = self.seg_rescale(seg_image, scale)
        return resize_seg_img

    def seg_rescale(self, seg_image, scale):
        crop = cv2.resize(seg_image, (seg_image.shape[1] // scale, seg_image.shape[0] // scale),
                          interpolation=cv2.INTER_NEAREST)
        return crop

    def extract_box(self, segmap, depth_map):
        H, W = segmap.shape[:2]
        mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
        black = (0, 0, 0)
        bboxes = []
        categories = []
        for v in range(0, H, 2):
            for u in range(0, W, 2):
                value = segmap[v, u]
                if (value == 0).all():
                    continue
                for dict_name, dict_val in self.class_color.items():
                    if (value[0] == dict_val[0]) and (value[1] == dict_val[1]) and (value[2] == dict_val[2]):
                        npixels, segmap, mask, ltwh = cv2.floodFill(segmap, mask, (u, v), black,
                                                                    flags=cv2.FLOODFILL_FIXED_RANGE | 8)
                        if npixels < self.dataset_cfg.PIXEL_LIMIT:
                            continue
                        bbox_tlbr = (ltwh[1], ltwh[0], ltwh[1] + ltwh[3], ltwh[0] + ltwh[2])
                        bbox_yxhw = self.convert_to_bbox_format(bbox_tlbr)
                        if dict_name not in self.dataset_cfg.CATEGORIES_TO_USE:
                            dict_name = None
                        # category_index = self.dataset_cfg.CATEGORIES_TO_USE.index(dict_name)
                        if dict_name in self.dataset_cfg.CATEGORY_REMAP:
                            dict_name = self.dataset_cfg.CATEGORY_REMAP[dict_name]
                        category_index = dict_name
                        depth_patch = depth_map[int(bbox_tlbr[0]):int(bbox_tlbr[2]),
                                      int(bbox_tlbr[1]):int(bbox_tlbr[3])]
                        dist_value = self.get_depth_value(depth_patch)
                        categories.append(category_index)
                        bbox_yxhw += [1, dist_value]
                        bboxes.append(bbox_yxhw)
        # return np.asarray(bboxes, dtype=np.float32), np.asarray(categories, dtype=np.float32)
        if len(categories) < 1:
            category_index = 0
            categories.append(category_index)
            bbox_yxhw = [0, 0, 0, 0, 0, 0]
            bboxes.append(bbox_yxhw)
        return np.asarray(bboxes, dtype=np.float32), categories

    def convert_to_bbox_format(self, tlbr):
        scale = self.dataset_cfg.SEGMAP_SCALE
        height = tlbr[2] - tlbr[0]
        width = tlbr[3] - tlbr[1]
        center_y = tlbr[0] + (height / 2)
        center_x = tlbr[1] + (width / 2)
        bbox = [center_y * scale, center_x * scale, height * scale, width * scale]
        # bbox = [center_y, center_x, height, width]
        return bbox

    def get_depth_value(self, depth_patch):
        dist_value = depth_patch[np.where(depth_patch > 0)]
        if dist_value.size == 0:
            return 0
        else:
            return np.quantile(dist_value, self.dataset_cfg.DIST_QUANTILE)

    def _read_depth_map(self, index, scale):
        image_name = self.frame_names[index]
        npz_name = image_name.replace("/camera/", "/lidar/").replace("_camera_", "_lidar_").replace(".png", ".npz")
        npzfile = self.zipinfile.open(npz_name)
        npzfile = np.load(npzfile)
        lidar_depth = npzfile["depth"]
        lidar_row = (npzfile["row"] + 0.5).astype(np.int32)
        lidar_col = (npzfile["col"] + 0.5).astype(np.int32)
        camera_key = "front_center"
        imsize_hw = self.sensor_config.get_resolution_hw(camera_key)

        assert (lidar_row >= 0).all() and (lidar_row < imsize_hw[0]).all(), \
            f"wrong index: {lidar_row[lidar_row >= 0]}, {lidar_row[lidar_row < imsize_hw[0]]}"
        assert (lidar_col >= 0).all() and (lidar_col < imsize_hw[1]).all(), \
            f"wrong index: {lidar_col[lidar_col >= 0]}, {lidar_col[lidar_col < imsize_hw[1]]}"

        depth_map = np.zeros(imsize_hw, dtype=np.float32)
        depth_map[lidar_row, lidar_col] = lidar_depth

        srcshape_hw = self.sensor_config.get_resolution_hw("front_center")
        rszshape_hw = (srcshape_hw[0] // scale, srcshape_hw[1] // scale)
        # depth_map = self.rescale_shape(depth_map, True)
        intrinsic = self.sensor_config.get_cam_matrix("front_center")
        point_cloud = depth_map_to_point_cloud(depth_map, intrinsic)
        scaled_intrinsic = intrinsic.copy()
        scaled_intrinsic[:2] /= scale
        depth_map = point_cloud_to_depth_map(point_cloud, scaled_intrinsic, rszshape_hw)
        return depth_map


class SensorConfig:
    def __init__(self, cfgfile):
        if cfgfile:
            with open(cfgfile, "r") as fr:
                self.sensor_config = json.load(fr)
        self.undist_remap = dict()

    def get_resolution_hw(self, cam_key):
        resolution = self.sensor_config["cameras"][cam_key]["Resolution"]
        resolution = np.asarray([resolution[1], resolution[0]], dtype=np.int32)
        return resolution

    def get_cam_matrix(self, cam_key):
        intrinsic = np.asarray(self.sensor_config["cameras"][cam_key]["CamMatrix"], dtype=np.float32)
        return intrinsic

# ==================================================

