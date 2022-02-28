#!/home/falcon/.pyenv/versions/paper/bin/python
import sys
import shutil
import math
import numpy as np
import os
import glob
import util.util_function as uf
import open3d as o3d
import load_and_save as ls
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import copy


def rotate_points(points, show_view=True):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=6, origin=[-2, -2, -2])
    num = len(points.items())
    print(points)
    count = 0
    for key, value in points.items():
        points = value[:, :3]
        points_z = value[:, 2:3]
        reflence = value[:, 3:4]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        points_normals = np.asarray(pcd.normals)[:, 0:3:2]
        normal_theta = np.arctan2(points_normals[:, 1:], points_normals[:, 0:1])
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((-0.209440, -1.361357, 1.762783)))
        velodyne = np.concatenate([np.asarray(pcd.points)[:, :2], points_z, reflence, normal_theta], axis=1)
        a = pcd.imag

        print(velodyne)
        if show_view:
            o3d.visualization.draw_geometries([pcd, mesh_frame],
                                              zoom=0.3412,
                                              front=[0.0, -0.0, 1.0],
                                              lookat=[0.0, 0.0, 0.0],
                                              up=[-0.0694, -0.9768, 0.2024],
                                              point_show_normal=False)
        uf.print_progress(f"-- Progress: {++count}/{num}")




if __name__ == '__main__':
    root_path = '/media/falcon/IanBook8T/datasets/kitti_detection/data_object_velodyne/training/velodyne'
    velodyne_points = ls.load_bin(root_path)
    rotate_points(velodyne_points)
