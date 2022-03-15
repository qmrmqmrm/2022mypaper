import sys
import numpy as np
import pandas


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


# rotation matrix 생성
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

    rotation_matrix_a = np.dot(pitch_matrix, roll_matrix)
    rotation_matrix = np.dot(yaw_matrix, rotation_matrix_a)

    return rotation_matrix


def point_cloud_to_depth_map(src_pcd, intrinsic, imshape):
    """
    :param src_pcd: source point cloud [N, 3] (X=right, Y=down, Z=front)
    :param intrinsic: [3, 3]
    :param imshape: height and width of output depth map
    :return: depth map
    """
    # print("[pcd2depth]", src_pcd.shape, intrinsic.shape, imshape)
    points = src_pcd[src_pcd[:, 2] > 1.].T  # [3, N]
    # project to camera, pixels: [3, N]
    pixels = np.dot(intrinsic, points) / points[2:3]
    assert np.isclose(pixels[2], 1.).all()
    # remove pixels out of image plane
    valid_mask = (pixels[0] >= 0) & (pixels[0] < imshape[1] - 1) & (pixels[1] >= 0) & (pixels[1] < imshape[0] - 1)
    pixels = pixels[:, valid_mask]
    points = points[:, valid_mask]
    # verify pixel-point relationship
    leftup = points[:, (pixels[1] > intrinsic[1, 2] - 20) & (pixels[1] < intrinsic[1, 2] - 10) & (pixels[0] < 50)]
    righdw = points[:,
             (pixels[1] > intrinsic[1, 2] + 30) & (pixels[1] < intrinsic[1, 2] + 40) & (pixels[0] > imshape[1] - 50)]
    if leftup.size > 0: assert (np.mean(leftup[:2], axis=1) < 0).all(), f"{leftup}"
    if righdw.size > 0: assert (np.mean(righdw[:2], axis=1) > 0).all(), f"{righdw}"
    # quarter pixels around `pixels`
    data = np.stack([np.floor(pixels[0]), np.floor(pixels[1]), np.ceil(pixels[0]), np.ceil(pixels[1])], axis=1)
    quart_pixels = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2'])
    quart_pixels = quart_pixels.astype(int)
    quarter_columns = [['x1', 'y1'], ['x1', 'y2'], ['x2', 'y1'], ['x2', 'y2']]
    depthmap = np.zeros(imshape, dtype=np.float32)
    weightmap = np.zeros(imshape, dtype=np.float32)
    flpixels = pixels[:2]

    for quarter_col in quarter_columns:

        qtpixels = quart_pixels.loc[:, quarter_col]
        qtpixels = qtpixels.rename(columns={quarter_col[0]: 'col', quarter_col[1]: 'row'})
        # diff = (1-abs(x-xn), 1-abs(y-yn)) [N, 2]
        diff = 1 - np.abs(flpixels.T - qtpixels.values)
        # weights = (1-abs(x-xn)) * (1-abs(y-yn)) [N]
        weights = diff[:, 0] * diff[:, 1]

        step = 0
        while (len(qtpixels.index) > 0) and (step < 5):
            step += 1
            step_pixels = qtpixels.drop_duplicates(keep='first')
            rows = step_pixels['row'].values
            cols = step_pixels['col'].values
            inds = step_pixels.index.values
            depthmap[rows, cols] += points[2, inds] * weights[inds]
            weightmap[rows, cols] += weights[inds]
            qtpixels = qtpixels[~qtpixels.index.isin(step_pixels.index)]

    depthmap[depthmap > 0] = depthmap[depthmap > 0] / weightmap[depthmap > 0]
    depthmap[weightmap < 0.5] = 0
    return depthmap
