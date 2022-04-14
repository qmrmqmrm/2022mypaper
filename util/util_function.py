import sys
import numpy as np
import pandas
import open3d as o3d
import cv2


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


def normalization(depthmap):
    height_scale = (0.0, 3.0)
    intensity_scale = (0.0, 1.0)
    normal_theta = (0, 2 * np.pi)
    normal_depthmap = np.zeros_like(depthmap)
    normal_depthmap[:, :, 0] = depthmap[:, :, 0] / height_scale[1]
    normal_depthmap[:, :, 1] = depthmap[:, :, 1] / intensity_scale[1]
    normal_depthmap[:, :, 2] = depthmap[:, :, 2] / normal_theta[1]
    return normal_depthmap


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

def draw_rotated_box(img, corners):
    """
    corners :
    """
    color = (255, 255, 255)
    for idx, corner in enumerate(corners.values()):
        corner = corner['p_corners']
        if int(corner[1][0]) - int(corner[0][0]) == 0 and int(corner[1][1]) - int(corner[0][1]) == 0:
            continue
        corner_idxs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6),
                       (3, 7)]
        for corner_idx in corner_idxs:
            cv2.line(img,
                     (int(corner[corner_idx[0]][0]),
                      int(corner[corner_idx[0]][1])),
                     (int(corner[corner_idx[1]][0]),
                      int(corner[corner_idx[1]][1])),
                     color, 2)
    return img