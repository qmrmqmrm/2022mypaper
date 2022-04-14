#!/home/falcon/.pyenv/versions/paper/bin/python
import numpy as np
import os
import glob
import cv2
import open3d as o3d
import pandas as pd
import time

import util.util_function as uf
import util.error_check as ec
import util.load_and_save as ls
import config as cfg

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class PreparationImage:
    def __init__(self, show_view=False, cell_size=0.05, grid_shape=1000):
        self.velo_path = cfg.Paths.VELO_ROOT
        self.label_path = cfg.Paths.LABEL_ROOT
        self.calib_path = cfg.Paths.CALIB_ROOT
        self.show_view = show_view
        self.cell_size = cell_size
        self.grid_shape = grid_shape
        self.limited_meter = cell_size * grid_shape

        # self.theta = theta
        # self.velodyne_points = ls.load_bin(self.velo_path)
        # self.img_labels = ls.load_txt(self.label_path)
        self.save_root = cfg.Paths.SAVE_DIR
        self.basic_plane = [-0.01958795, -0.00710267, 0.99978291, 1.755962]

    def make_dataset(self, tbev_pose=np.pi / 3):
        velo_files = sorted(glob.glob(os.path.join(self.velo_path, '*.bin')))
        label_flies = sorted(glob.glob(os.path.join(self.label_path, '*.txt')))
        calib_files = sorted(glob.glob(os.path.join(self.calib_path, '*.txt')))
        num = len(velo_files)
        count = 0
        for velo_file, label_flie, calib_file in zip(velo_files, label_flies, calib_files):
            count += 1
            start_time = time.time()
            file_num = velo_file.split('/')[-1].split('.')[0]
            velo = ls.load_velo(velo_file)  # numpy
            label = ls.load_label(label_flie)  # numpy
            calib = ls.read_calib_file(calib_file)  # dict
            self.get_calib(calib)
            points = velo[:, :3]
            plane_model = self.get_ground_plane(points)
            annotation = self.get_annotations(label, plane_model, file_num)
            if not bool(annotation[file_num]):
                continue

            # image_param shape : (N,6) :[rotated_points(x,y,z), points_z, reflence, normal_theta]
            image_param = self.get_image_param(velo, plane_model, tbev_pose)

            # pixels shape : (N,2)
            flpixels = self.pixel_coordinates(image_param[:, :2], tbev_pose)
            # depthmap shape : (grid_shape, grid_shape, 3)
            depthmap = self.interpolation(flpixels, image_param, 3, 6)
            normal_depthmap = uf.normalization(depthmap)
            deg = int(tbev_pose * (180 / np.pi))
            save_image_dir = f"{self.save_root}/{int(self.grid_shape * self.cell_size)}m/deg_{deg}/image"
            save_label_dir = f"{self.save_root}/{int(self.grid_shape * self.cell_size)}m/deg_{deg}/label"
            os.makedirs(save_image_dir, exist_ok=True)
            os.makedirs(save_label_dir, exist_ok=True)
            save_image_file = os.path.join(save_image_dir, f"{file_num}.jpg")
            result_depthmap = (normal_depthmap * 255).astype(np.uint8)
            cv2.imwrite(save_image_file, result_depthmap)
            show_map = self.draw_rotated_box(result_depthmap, annotation[file_num])
            save_label_img = os.path.join(save_label_dir, f"{file_num}.jpg")
            cv2.imwrite(save_label_img, show_map)
            ls.save_json(save_label_dir, annotation, file_num)

            end_time = time.time()
            step_time = end_time - start_time
            full_time = step_time * (num - count)

            uf.print_progress(f"-- Progress: deg:{deg}, {count}/{num} time:{step_time} fin_time:{full_time:.4f} ")

    def get_calib(self, calib):
        self.P = calib['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calib['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        # self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calib['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])
        self.C2V = inverse_rigid_trans(self.V2C)

    def get_annotations(self, labels, plane_model, file_num):

        box_id = 0
        if (labels.ndim < 1):
            labels = np.array(labels, ndmin=1)

        categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        categories = [categories[idx] for idx in range(5)]
        category_dict = {k: k for v, k in enumerate(categories)}
        annotations = {file_num: dict()}
        for obj in labels:
            t = obj['type']
            if isinstance(t, (bytes, np.bytes_)):
                t = t.decode("utf-8")
            label = category_dict.get(t, 'DontCare')
            if label != 5:
                location = np.array([[obj['location_1'], obj['location_2'], obj['location_3']]])  # cam xyz
                pts_3d_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(location)))
                yaw = obj['rotation_y']
                n = pts_3d_ref.shape[0]
                pts_3d_ref = np.hstack((pts_3d_ref, np.ones((n, 1))))
                he = np.array([0, 0, 1.73 / 2]).reshape([1, 3])
                centroid = np.dot(pts_3d_ref, np.transpose(self.C2V)) + he

                dimension = np.array([obj['dimensions_2'], obj['dimensions_3'], obj['dimensions_1']])  # wlh
                corners = uf.get_box3d_corner(dimension)  # wlh
                R = uf.create_rotation_matrix([yaw, 0, 0])
                corners = np.dot(corners, R) + centroid
                corners = np.concatenate([corners, centroid], axis=0)
                rotated_corners, normal_theta = self.get_rotation_and_normal_vector(corners, tbev_pose)
                height = self.cal_height(centroid, plane_model) * 2
                value_mask = (rotated_corners[:, 0] > 0) & (rotated_corners[:, 1] < self.limited_meter / 2) & (
                        rotated_corners[:, 1] > -self.limited_meter / 2)
                rotated_corners = rotated_corners[value_mask, :]

                pixels = self.pixel_coordinates(rotated_corners[:, :2], tbev_pose)

                imshape = [self.grid_shape, self.grid_shape, 3]
                valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < imshape[1] - 1) & (pixels[:, 1] >= 0) & (
                        pixels[:, 1] < imshape[0] - 1)
                pixels = pixels[valid_mask, :]
                if pixels.size < 18:
                    continue
                ann = dict()
                ann[box_id] = {'file_num': file_num, 'centroid': centroid.tolist(), 'dimension': dimension.tolist(),
                               'height': height.tolist(), 'corners': rotated_corners.tolist(),
                               'p_corners': pixels.tolist(), 'category': label,
                               'rotation_y': obj['rotation_y'], 'alpha': obj['alpha']}
                box_id += 1
                annotations[file_num].update(ann)
        return annotations

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
        if self.show_view:
            uf.show_points(crop_points)
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

    def get_ground_plane(self, points, plane_check=False):
        points_vaild = (points[:, 2] < -1.0)  # & (points[:, 2] < -1.5)
        rote_pcd = o3d.geometry.PointCloud()
        rote_pcd.points = o3d.utility.Vector3dVector(points[points_vaild, :])
        plane_model, inliers = rote_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=200)
        if plane_check:
            ec.plane_check(plane_model)
        plane_model = np.array(plane_model)
        return plane_model

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




def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


if __name__ == '__main__':
    cl = PreparationImage(show_view=False)
    theta = np.arange(0, np.pi / 2, np.pi / 12)
    for tbev_pose in theta:
        print('\ntbev_pose', tbev_pose)
        cl.make_dataset(tbev_pose=tbev_pose)
    # velodyne_points = ls.load_bin(root_path)
    # rotate_points(velodyne_points, pitch=0)
