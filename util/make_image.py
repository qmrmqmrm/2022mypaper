#!/home/dolphin/.pyenv/versions/paper/bin/python
import sys
import shutil
import math
import numpy as np
import os
import glob
import cv2
import open3d as o3d
import pandas as pd
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import copy
import time

import util.util_function as uf
import util.error_check as ec
import load_and_save as ls
import config as cfg


def normalization(depthmap):
    height_scale = (0.0, 3.0)
    intensity_scale = (0.0, 1.0)
    normal_theta = (0, 2 * np.pi)
    normal_depthmap = np.zeros_like(depthmap)
    normal_depthmap[:, :, 0] = depthmap[:, :, 0] / height_scale[1]
    normal_depthmap[:, :, 1] = depthmap[:, :, 1] / intensity_scale[1]
    normal_depthmap[:, :, 2] = depthmap[:, :, 2] / normal_theta[1]
    return normal_depthmap


class PreparationImage:
    def __init__(self, root_path, show_view=False):
        self.root_path = root_path
        self.show_view = show_view
        # self.theta = theta
        self.velodyne_points = ls.load_bin(root_path)
        self.save_root = cfg.Paths.SAVE_DIR
        self.basic_plane = [-0.01958795, -0.00710267, 0.99978291, 1.755962]

    def get_image(self, tbev_pose=np.pi / 3, cell_size=0.05, grid_shape=500):
        """

        :param tbev_pose: axis-y rotation angle
        :param cell_size: grid cell size
        :param grid_shape: image size / cell_size
        :return:
        """

        num = len(self.velodyne_points.items())
        count = 0
        for key, value in self.velodyne_points.items():
            print()
            start_time = time.time()

            # image_param shape : (N,6) :[rotated_points(x,y,z), points_z, reflence, normal_theta]
            image_param = self.get_image_param(value, tbev_pose, cell_size, grid_shape)
            # pixels shape : (N,2)
            flpixels = self.pixel_coordinates(image_param[:, :2], tbev_pose, cell_size, grid_shape)
            # depthmap shape : (grid_shape, grid_shape, 3)
            depthmap = self.interpolation(flpixels, image_param, grid_shape)
            normal_depthmap = normalization(depthmap)

            deg = int(tbev_pose * (180 / np.pi))
            save_dir = f"{self.save_root}/deg_{deg}"
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, f"{key}.jpg")
            result_depthmap = (normal_depthmap * 255).astype(np.uint8)
            cv2.imwrite(save_file, result_depthmap)
            count += 1
            end_time = time.time()
            step_time = end_time - start_time
            full_time = step_time * (num - count)
            uf.print_progress(f"-- Progress: deg:{deg}, {count}/{num} time:{step_time} fin_time:{full_time:.4f} ")

    def get_image_param(self, value, tbev_pose, cell_size, grid_shape):
        limited_meter = cell_size * grid_shape
        points = value[:, :3]
        rotated_points, normal_theta, height = self.get_rotation_and_normal_vector(points, tbev_pose)
        value_mask = (rotated_points[:, 0] > 0) & (rotated_points[:, 1] < limited_meter / 2) & (
                rotated_points[:, 1] > -limited_meter / 2)
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
        height = self.get_ground_height(pcd, points)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20)
        pcd.estimate_normals(search_param)
        points_normals = np.asarray(pcd.normals)[:, 0:3:2]
        normal_theta = np.arctan2(points_normals[:, 1:], points_normals[:, 0:1])
        normal_theta = normal_theta % (2 * np.pi)
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, tbev_pose, 0)))
        rotated_points = np.asarray(pcd.points)[:, :3]

        return rotated_points, normal_theta, height

    def get_ground_height(self, pcd, points, plane_check=False):
        points_vaild = (points[:, 2] < -1.0)  # & (points[:, 2] < -1.5)
        rote_pcd = o3d.geometry.PointCloud()
        rote_pcd.points = o3d.utility.Vector3dVector(points[points_vaild, :])
        plane_model, inliers = rote_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=200)
        if plane_check:
            ec.plane_check(plane_model)
        plane_model = np.array(plane_model)
        height = np.abs(plane_model[0:1] * points[:, 0:1] + plane_model[1:2] * points[:, 1:2] +
                        plane_model[2:3] * points[:, 2:3] + plane_model[3:4]) / np.sum(np.power(plane_model, 2))
        return height

    def pixel_coordinates(self, rotated_xy, tbev_pose, cell_size, grid_shape):
        """

        :param image_param: [rotated_points(x,y,z), height, reflence, normal_theta]
        :param cell_size: (default : 0.05)
        :param grid_shape: (default : 500)
        :return:
        """
        print((cell_size * np.cos(tbev_pose)))
        image_x = (grid_shape / 2) - (rotated_xy[:, 1:2] / cell_size)
        image_y = (grid_shape) - (rotated_xy[:, 0:1] / (cell_size * np.cos(tbev_pose)))
        pixels = np.concatenate([image_x, image_y], axis=1)
        return pixels

    def interpolation(self, pixels, points, grid_shape):
        """

        :param pixels: (N,2) float
        :param points: (N,6) [rotated_points(x,y,z), height, reflence, normal_theta]
        :param cell_size: default 0.05
        :param grid_shape: default 500
        :return:
        """
        start_time = time.time()
        imshape = [grid_shape, grid_shape, 3]
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
            # diff = (1-abs(x-xn), 1-abs(y-yn)) [N, 2]
            diff = 1 - np.abs(flpixels - qtpixels.values)
            # weights = (1-abs(x-xn)) * (1-abs(y-yn)) [N]
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
                depthmap[rows, cols, :] += points[inds, 3:6] * weights[inds, :]
                weightmap[rows, cols, :] += weights[inds, :]
                qtpixels = qtpixels[~qtpixels.index.isin(step_pixels.index)]

        depthmap[depthmap > 0] = depthmap[depthmap > 0] / weightmap[depthmap > 0]
        depthmap[weightmap < 0.5] = 0
        end_time = time.time()
        # print('interpolation time', end_time - start_time)

        return depthmap


if __name__ == '__main__':
    root_path = cfg.Paths.VELO_ROOT
    cl = PreparationImage(root_path, show_view=False)
    theta = np.arange(0, np.pi / 2, np.pi / 12)
    for tbev_pose in theta[0:1]:
        print('tbev_pose', tbev_pose)
        cl.get_image(tbev_pose=tbev_pose)
    # velodyne_points = ls.load_bin(root_path)
    # rotate_points(velodyne_points, pitch=0)
