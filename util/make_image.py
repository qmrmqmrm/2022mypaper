import sys
import shutil
import math
import numpy as np
import os
import glob
import util.util_function as uf
import open3d as o3d
import load_and_save as ls


def rotation_point(points, show_view=True):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=6, origin=[-2, -2, -2])
    for key, value in points.items():
        points = value[:, :3]
        points_z = value[:,2:3]
        test_rot = uf.create_rotation_matrix([0, -np.pi / 6, 0])
        points_rot = np.dot(test_rot, points.T)
        reflence = value[:, 3:4]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_rot.T)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        points_normals = np.asarray(pcd.normals)[:,0:3:2]

        # points_normals = np.concatenate(points_normals[:,0], points_normals[:,2])
        velodyne = np.concatenate([points_rot.T, points_z, reflence], axis=1)
        print("1",points_normals[:10])

        if show_view:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_rot.T)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            o3d.visualization.draw_geometries([pcd,mesh_frame],
                                              zoom=0.3412,
                                              front=[0.4257, -0.2125, -0.8795],
                                              lookat=[2.6172, 2.0475, 1.532],
                                              up=[-0.0694, -0.9768, 0.2024],
                                              point_show_normal=True)

if __name__ == '__main__':
    root_path = '/media/dolphin/intHDD/birdnet_data/kitti/training/velodyne'
    velodyne_points = ls.load_bin(root_path)
    rotation_point(velodyne_points)
