#!/home/falcon/.pyenv/versions/paper/bin/python
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

import util.util_function as uf
import load_and_save as ls


class PreparationImage:
    def __init__(self, root_path, show_view=False):
        self.root_path = root_path
        self.show_view = show_view
        # self.theta = theta
        self.velodyne_points = ls.load_bin(root_path)

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
            # image_param shape : (N,6) :[rotated_points(x,y,z), points_z, reflence, normal_theta]
            image_param = self.get_image_param(value, 0, cell_size, grid_shape)
            # pixels shape : (N,2)
            flpixels = self.pixel_coordinates(image_param[:, :2], cell_size, grid_shape)
            # depthmap shape : (grid_shape, grid_shape, 3)
            result_depthmap = self.interporation(flpixels, image_param, grid_shape)

            # show_image = np.concatenate([bev_depthmap,depthmap_30,depthmap_45, depthmap_60],axis=1)
            # print(show_image.shape)

            cv2.imshow("depth", result_depthmap)
            cv2.waitKey(0)
            uf.print_progress(f"-- Progress: {++count}/{num}")

    def get_image_param(self, value, tbev_pose, cell_size, grid_shape):
        limited_meter = cell_size * grid_shape
        points = value[:, :3]
        rotated_points, normal_theta = self.get_rotation_and_normal_vector(points, tbev_pose)
        value_mask = (rotated_points[:, 0] > 0) & (rotated_points[:, 1] < limited_meter / 2) & (
                rotated_points[:, 1] > -limited_meter / 2) & (rotated_points[:, 0] < limited_meter)
        crop_points = rotated_points[value_mask, :]
        normal_theta = normal_theta[value_mask, :]
        points_z = value[value_mask, 2:3]
        reflence = value[value_mask, 3:4]
        if self.show_view:
            self.show_points(crop_points)
        # image_param shape : (N,6) :[rotated_points(x,y,z), points_z, reflence, normal_theta]
        image_param = np.concatenate([crop_points, points_z, reflence, normal_theta], axis=1)
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
        plane_model = self.get_ground_points(pcd)
        self.get_point_to_plane_dis(plane_model, points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=1000))
        points_normals = np.asarray(pcd.normals)[:, 0:3:2]
        normal_theta = np.arctan2(points_normals[:, 1:], points_normals[:, 0:1])
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, tbev_pose, 0)))
        rotated_points = np.asarray(pcd.points)[:, :3]

        return rotated_points, normal_theta

    def get_ground_points(self, pcd):
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                                 ransac_n=5,
                                                 num_iterations=5000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        if self.show_view:
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                              zoom=0.8,
                                              front=[-0.4999, -0.1659, -0.8499],
                                              lookat=[2.1813, 2.0619, 2.0999],
                                              up=[0.1204, -0.9852, 0.1215])
        return np.array(plane_model)

    def pixel_coordinates(self, rotated_xy, cell_size, grid_shape):
        """

        :param image_param: [rotated_points(x,y,z), points_z, reflence, normal_theta]
        :param cell_size: (default : 0.05)
        :param grid_shape: (default : 500)
        :return:
        """

        image_x = (grid_shape / 2) - (rotated_xy[:, 1:2] / cell_size)
        image_y = (grid_shape) - (rotated_xy[:, 0:1] / cell_size)
        pixels = np.concatenate([image_x, image_y], axis=1)
        return pixels

    def show_points(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6)
        o3d.visualization.draw_geometries([pcd, mesh_frame],
                                          zoom=0.3412,
                                          front=[0.0, -0.0, 1.0],
                                          lookat=[0.0, 0.0, 0.0],
                                          up=[-0.0694, -0.9768, 0.2024],
                                          point_show_normal=False)

    def interporation(self, pixels, points, grid_shape):
        """

        :param pixels: (N,2) float
        :param points: (N,6) [rotated_points(x,y,z), points_z, reflence, normal_theta]
        :param cell_size: default 0.05
        :param grid_shape: default 500
        :return:
        """
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

            step = 0
            while (len(qtpixels.index) > 0) and (step < 5):
                step += 1
                step_pixels = qtpixels.drop_duplicates(keep='first')
                rows = step_pixels['row'].values
                cols = step_pixels['col'].values
                inds = step_pixels.index.values

                depthmap[rows, cols, 0] += points[inds, 3] * weights[inds]
                depthmap[rows, cols, 1] += points[inds, 4] * weights[inds]
                depthmap[rows, cols, 2] += points[inds, 5] * weights[inds]
                weightmap[rows, cols, 0] += weights[inds]
                weightmap[rows, cols, 1] += weights[inds]
                weightmap[rows, cols, 2] += weights[inds]
                qtpixels = qtpixels[~qtpixels.index.isin(step_pixels.index)]

        depthmap[depthmap > 0] = depthmap[depthmap > 0] / weightmap[depthmap > 0]
        depthmap[weightmap < 0.5] = 0
        return depthmap


if __name__ == '__main__':
    root_path = '/media/falcon/IanBook8T/datasets/kitti_detection/data_object_velodyne/training/velodyne'
    cl = PreparationImage(root_path, show_view=False)
    cl.get_image()
    # velodyne_points = ls.load_bin(root_path)
    # rotate_points(velodyne_points, pitch=0)
