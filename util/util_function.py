import sys
import numpy as np
import pandas
import open3d as o3d


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def show_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6)
    o3d.visualization.draw_geometries([pcd, mesh_frame],
                                      zoom=0.3412,
                                      front=[0.0, -0.0, 1.0],
                                      lookat=[0.0, 0.0, 0.0],
                                      up=[-0.0694, -0.9768, 0.2024],
                                      point_show_normal=False)