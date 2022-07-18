import os.path as op
import numpy as np
from glob import glob
import cv2
import open3d as o3d

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
# import dataloader.framework.data_util as tu
import utils.util_class as uc


class KittiDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        kitti_split = "training"  # if self.split == "train" else "testing"
        return [op.join(self.datapath, kitti_split, "image_2")]

    def get_drive_name(self, drive_index):
        return f"drive{drive_index:02d}"


class KittiReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg=None):
        super().__init__(drive_path, split, dataset_cfg)

    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "*.png"))
        frame_names.sort()
        if split == 'train':
            frame_names = frame_names[:-500]
        else:
            frame_names = frame_names[-500:]
        print("[KittiReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_calibration(self, index):
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "calib").replace(".png", ".txt")
        return SensorConfig(label_file)

    def get_2d_box(self, index, raw_hw_shape=None):
        """
        :return: bounding boxes in 'yxhw' format
        """
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")

        bboxes = []
        categories = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                bbox, category = self.extract_2d_box(line)
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)

        if not bboxes:
            raise uc.MyExceptionToCatch("[get_2d_box] empty boxes")
        bboxes = np.array(bboxes)
        return bboxes, categories

    def extract_2d_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]

        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]
        y1 = round(float(raw_label[5]))
        x1 = round(float(raw_label[4]))
        y2 = round(float(raw_label[7]))
        x2 = round(float(raw_label[6]))
        bbox = np.array([(y1 + y2) / 2, (x1 + x2) / 2, y2 - y1, x2 - x1, 1, 0], dtype=np.int32)
        return bbox, category_name

    def get_3d_box(self, index):
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")
        bboxes = []
        categories = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                bbox3d, category = self.extract_3d_box(line)
                if bbox3d is not None:
                    bboxes.append(bbox3d)
                    categories.append(category)

        if not bboxes:
            raise uc.MyExceptionToCatch("[get_2d_box] empty boxes")
        bboxes = np.array(bboxes)
        return bboxes, categories

    def extract_3d_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]

        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]

        dimensions_1 = round(float(raw_label[8]))
        dimensions_2 = round(float(raw_label[9]))
        dimensions_3 = round(float(raw_label[10]))
        location_1 = round(float(raw_label[11]))
        location_2 = round(float(raw_label[12]))
        location_3 = round(float(raw_label[13]))
        rotation_y = round(float(raw_label[14]))

        bbox3d = np.array([dimensions_1, dimensions_2, dimensions_3, location_1, location_2, location_3, rotation_y], dtype=np.int32)
        return bbox3d, category_name

    def get_point_cloud(self, index):
        """
        :param index: image index in self.frame_names
        :return:
        """
        image_file = self.frame_names[index]
        lidar_file = image_file.replace("image_2", "velodyne").replace(".png", ".bin")
        point_clouds = np.fromfile(lidar_file, dtype=np.float32)
        point_clouds = point_clouds.reshape((-1, 4))
        point_clouds[:, :3] = self.get_calibration(index).project_velo_to_rect(point_clouds[:, :3])
        return point_clouds

    def get_depth_map(self, index):
        """
        :param index: image index in self.frame_names
        :return:
        """
        image_file = self.frame_names[index]
        lidar_file = image_file.replace("image_2", "velodyne").replace(".png", ".bin")
        point_clouds = np.fromfile(lidar_file, dtype=np.float32)
        point_clouds = point_clouds.reshape((-1, 4))
        depth = self.get_calibration(index).project_velo_to_image(point_clouds[:, :3])
        return depth


class KittiBevReader(KittiReader):
    def __init__(self,  drive_path, split, dataset_cfg=None, cell_size=0.05, grid_shape=500, tbev_pose=0):
        super(KittiBevReader, self).__init__(drive_path, split, dataset_cfg)
        self.cell_size = cell_size
        self.grid_shape = grid_shape
        self.limited_meter = cell_size * grid_shape
        self.tbev_pose = tbev_pose

    def get_bev_box(self, index):
        point_cloud = self.get_point_cloud(index)
        plane_model = self.get_ground_plane(point_cloud[:, :3])
        bbox3d, category_name = self.get_3d_box(index)
        calib = self.get_calibration(index)
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(calib.R0), np.transpose( bbox3d[3:6])))
        he = np.array([0, 0, 1.73 / 2]).reshape([1, 3])
        centroid = np.dot(pts_3d_ref, np.transpose(calib.C2V)) + he

        corners = get_box3d_corner(bbox3d[0:3])  # wlh
        R = create_rotation_matrix([bbox3d[-1], 0, 0])
        corners = np.dot(corners, R) + centroid
        corners = np.concatenate([corners, centroid], axis=0)
        rotated_corners, normal_theta = self.get_rotation_and_normal_vector(corners, self.tbev_pose)
        height = self.cal_height(centroid, plane_model) * 2
        value_mask = (rotated_corners[:, 0] > 0) & (rotated_corners[:, 1] < self.limited_meter / 2) & (
                rotated_corners[:, 1] > -self.limited_meter / 2)
        rotated_corners = rotated_corners[value_mask, :]

        pixels = self.pixel_coordinates(rotated_corners[:, :2], self.tbev_pose)

        imshape = [self.grid_shape, self.grid_shape, 3]
        valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < imshape[1] - 1) & (pixels[:, 1] >= 0) & (
                pixels[:, 1] < imshape[0] - 1)
        pixels = pixels[valid_mask, :]

        ann = {'centroid': centroid.tolist(), 'dimension': bbox3d[0:3].tolist(),
                       'height': height.tolist(), 'corners': rotated_corners.tolist(),
                       'p_corners': pixels.tolist(), 'category': category_name,
                       'rotation_y': bbox3d[-1]}

    def get_bev_image(self, index, tbev_pose=0):
        point_cloud = self.get_point_cloud(index)
        plane_model = self.get_ground_plane(point_cloud[:, :3])
        image_param = self.get_image_param(point_cloud, plane_model, tbev_pose)
        flpixels = self.pixel_coordinates(image_param[:, :2], tbev_pose)
        result_depthmap = self.interpolation(flpixels, image_param, 3, 6)
        normal_bev = normalization(result_depthmap)
        result_bev = (normal_bev * 255).astype(np.uint8)
        return result_bev

    def get_ground_plane(self, points):
        points_vaild = (points[:, 2] < -1.0)  # & (points[:, 2] < -1.5)
        rote_pcd = o3d.geometry.PointCloud()
        rote_pcd.points = o3d.utility.Vector3dVector(points[points_vaild, :])
        plane_model, inliers = rote_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=200)
        plane_model = np.array(plane_model)
        return plane_model

    def get_image_param(self, value, plane_model, tbev_pose):
        points = value[:, :3]
        rotated_points, normal_theta = self.get_rotation_and_normal_vector(points, tbev_pose)
        height = self.cal_height(points, plane_model)
        value_mask = (rotated_points[:, 0] > 0) & (rotated_points[:, 1] < self.limited_meter / 2) & (
                rotated_points[:, 1] > -self.limited_meter / 2)
        crop_points = rotated_points[value_mask, :]
        normal_theta = normal_theta[value_mask, :]
        height = height[value_mask, :]
        reflence = value[value_mask, 3:4]
        # image_param shape : (N,6) :[rotated_points(x,y,z), points_z, reflence, normal_theta]
        image_param = np.concatenate([crop_points, height, reflence, normal_theta], axis=1)
        return image_param

    def get_rotation_and_normal_vector(self, points, tbev_pose):
        """

        :param points: velodyne_points
        :param tbev_pose: axis-y rotation angle
        :return:
        rotated_points : transformation points with tbev_pose
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20)
        pcd.estimate_normals(search_param)
        points_normals = np.asarray(pcd.normals)[:, 0:3:2]
        normal_theta = np.arctan2(points_normals[:, 1:], points_normals[:, 0:1])
        normal_theta = normal_theta % (2 * np.pi)
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, tbev_pose, 0)), center=(0, 0, 0))
        rotated_points = np.asarray(pcd.points)[:, :3]

        return rotated_points, normal_theta

    def cal_height(self, points, plane_model):
        height = np.abs(plane_model[0:1] * points[:, 0:1] + plane_model[1:2] * points[:, 1:2] +
                        plane_model[2:3] * points[:, 2:3] + plane_model[3:4]) / np.sum(np.power(plane_model, 2))
        return height

    def pixel_coordinates(self, rotated_xy, tbev_pose):
        """

        :param rotated_xy: rotated_points(x,y)
        :param tbev_pose: rotation_y(default : 0)
        :return:
        """
        image_x = (self.grid_shape / 2) - (rotated_xy[:, 1:2] / self.cell_size)
        image_y = (self.grid_shape) - (rotated_xy[:, 0:1] / (self.cell_size * np.cos(tbev_pose)))
        pixels = np.concatenate([image_x, image_y], axis=1)
        return pixels

    def interpolation(self, pixels, points, start_idx, end_idx):
        """

        :param pixels: (N,2) float
        :param points: (N,6) [rotated_points(x,y,z), height, reflence, normal_theta]
        :return:
        """
        imshape = [self.grid_shape, self.grid_shape, 3]
        valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < imshape[1] - 1) & (pixels[:, 1] >= 0) & (
                pixels[:, 1] < imshape[0] - 1)
        pixels = pixels[valid_mask, :]
        points = points[valid_mask, :]
        data = np.stack([np.floor(pixels[:, 0]), np.floor(pixels[:, 1]), np.ceil(pixels[:, 0]), np.ceil(pixels[:, 1])],
                        axis=1)
        quart_pixels = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2'])
        quart_pixels = quart_pixels.astype(int)
        quarter_columns = [['x1', 'y1'], ['x1', 'y2'], ['x2', 'y1'], ['x2', 'y2']]
        depthmap = np.zeros(imshape, dtype=np.float32)
        weightmap = np.zeros(imshape, dtype=np.float32)
        flpixels = pixels[:, :2]
        for quarter_col in quarter_columns:
            qtpixels = quart_pixels.loc[:, quarter_col]
            qtpixels = qtpixels.rename(columns={quarter_col[0]: 'col', quarter_col[1]: 'row'})
            diff = 1 - np.abs(flpixels - qtpixels.values)
            weights = diff[:, 0] * diff[:, 1]
            weights = np.expand_dims(weights, axis=1)
            weights = np.tile(weights, 3)

            step = 0
            while (len(qtpixels.index) > 0) and (step < 5):
                step += 1
                step_pixels = qtpixels.drop_duplicates(keep='first')
                rows = step_pixels['row'].values
                cols = step_pixels['col'].values
                inds = step_pixels.index.values
                depthmap[rows, cols, :] += points[inds, start_idx:end_idx] * weights[inds, :]
                weightmap[rows, cols, :] += weights[inds, :]
                qtpixels = qtpixels[~qtpixels.index.isin(step_pixels.index)]

        depthmap[depthmap > 0] = depthmap[depthmap > 0] / weightmap[depthmap > 0]
        depthmap[weightmap < 0.5] = 0
        return depthmap

def normalization(depthmap):
    height_scale = (0.0, 3.0)
    intensity_scale = (0.0, 1.0)
    normal_theta = (0, 2 * np.pi)
    normal_depthmap = np.zeros_like(depthmap)
    normal_depthmap[:, :, 0] = depthmap[:, :, 0] / height_scale[1]
    normal_depthmap[:, :, 1] = depthmap[:, :, 1] / intensity_scale[1]
    normal_depthmap[:, :, 2] = depthmap[:, :, 2] / normal_theta[1]
    return normal_depthmap


class SensorConfig:
    def __init__(self, cfgfile):
        self.sensor_config = self.read_calib(cfgfile)
        self.P = self.sensor_config["P2"]
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = self.sensor_config["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = self.sensor_config["R0_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib(self, cfgfile):
        data = {}
        if cfgfile:
            with open(cfgfile, "r") as f:
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) == 0:
                        continue
                    key, value = line.split(":", 1)
                    try:
                        data[key] = np.array([float(x) for x in value.split()])
                    except ValueError:
                        pass
        return data

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)


def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def get_box3d_corner(bbox_wlh):
    brl = np.asarray([+bbox_wlh[..., 0] / 2, -bbox_wlh[..., 1] / 2, -bbox_wlh[..., 2] / 2])
    bfl = np.asarray([+bbox_wlh[..., 0] / 2, +bbox_wlh[..., 1] / 2, -bbox_wlh[..., 2] / 2])
    bfr = np.asarray([-bbox_wlh[..., 0] / 2, +bbox_wlh[..., 1] / 2, -bbox_wlh[..., 2] / 2])
    brr = np.asarray([-bbox_wlh[..., 0] / 2, -bbox_wlh[..., 1] / 2, -bbox_wlh[..., 2] / 2])
    trl = np.asarray([+bbox_wlh[..., 0] / 2, -bbox_wlh[..., 1] / 2, +bbox_wlh[..., 2] / 2])
    tfl = np.asarray([+bbox_wlh[..., 0] / 2, +bbox_wlh[..., 1] / 2, +bbox_wlh[..., 2] / 2])
    tfr = np.asarray([-bbox_wlh[..., 0] / 2, +bbox_wlh[..., 1] / 2, +bbox_wlh[..., 2] / 2])
    trr = np.asarray([-bbox_wlh[..., 0] / 2, -bbox_wlh[..., 1] / 2, +bbox_wlh[..., 2] / 2])
    return np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])

def create_rotation_matrix(euler):
    (yaw, pitch, roll) = euler

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitch_matrix = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = np.dot(yaw_matrix, pitch_matrix, roll_matrix)

    return rotation_matrix
# ==================================================
import config as cfg

# def test_kitti_reader():
#     print("===== start test_kitti_reader")
#     dataset_cfg = cfg.Datasets.Kitti
#     drive_mngr = KittiDriveManager(dataset_cfg.PATH, "train")
#     drive_paths = drive_mngr.get_drive_paths()
#     reader = KittiReader(drive_paths[0], "train", dataset_cfg)
#     for i in range(reader.num_frames()):
#         image = reader.get_image(i)
#         bboxes = reader.get_2d_box(i)
#         print(f"frame {i}, bboxes:\n", bboxes)
#         boxed_image = tu.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
#         cv2.imshow("kitti", boxed_image)
#         key = cv2.waitKey()
#         if key == ord('q'):
#             break
#     print("!!! test_kitti_reader passed")
#

if __name__ == "__main__":
    test_kitti_reader()
