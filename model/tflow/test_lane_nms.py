import numpy as np
import tensorflow as tf
import cv2

from model.framework.model_util import NonMaximumSuppressionLane

def test_lane_nms():
    NMS = NonMaximumSuppressionLane()
    image = np.zeros((512, 1280, 3))
    image_shape = np.array([512, 1280]).astype(np.int32)
    feat_shapes = np.array([32, 80]).astype(np.int32)


    # spts = np.array(
    #     [[510, 200], [510, 500], [510, 530], [510, 530], [510, 1000], [510, 1200], [510, 1100]]).astype(np.float32)
    # epts = np.array(
    #     [[250, 140], [250, 400], [250, 430], [250, 370], [250, 800], [250, 600], [250, 1200]]).astype(np.float32)

    spts = np.array(
        [[510, 700], [510, 720], [510, 730], [510, 750], [510, 900], [510, 1000], [510, 1000]]).astype(np.float32)
    epts = np.array(
        [[250, 600], [250, 620], [250, 630], [250, 650], [250, 800], [250, 600], [250, 10]]).astype(np.float32)


    # spts = np.array(
    #     [[510, 700], [510, 720], [510, 730], [510, 750], [510, 900], [510, 1000], [510, 1200]]).astype(np.float32)
    # epts = np.array(
    #     [[250, 700], [250, 720], [250, 730], [250, 750], [250, 900], [250, 1000], [250, 1200]]).astype(np.float32)

    spts /= image_shape
    epts /= image_shape

    fpoints = []
    for spt, ept in zip(spts, epts):
        lane = []
        for i in range(5):
            point = i/4 * spt + (4 - i) /4 * ept
            lane.append(point)
        lane = np.concatenate(lane, axis=-1)
        fpoints.append(lane)

    fpoints = np.stack(fpoints, axis=0)
    print(fpoints)
    centerness = np.arange(0.5, 0.9, 0.05)[:7]
    center_point = fpoints[..., 4:6]
    center_points_pixel = (center_point * feat_shapes).astype(np.int)
    flat_indices = center_points_pixel[:, 0] * 80 + center_points_pixel[:, 1]
    print('flat_indices', flat_indices)


    image_fpoints = draw_lanes(image, fpoints, centerness)

    fpoints_feature = np.zeros((feat_shapes[0], feat_shapes[1], 1, 10), dtype=np.float32)
    centerness_feature = np.zeros((feat_shapes[0], feat_shapes[1], 1, 1), dtype=np.float32)
    category_feature = np.zeros((feat_shapes[0], feat_shapes[1], 1, 3), dtype=np.float32)

    # fpoints_feature = fpoints_feature[center_points_pixel[:, 0]]
    # test = fpoints_feature[:, center_points_pixel[:, 1]]
    fpoints_feature[center_points_pixel[:, 0], center_points_pixel[:, 1], 0, :] = fpoints
    centerness_feature[center_points_pixel[:, 0], center_points_pixel[:, 1], 0, 0] = centerness
    category_feature[center_points_pixel[:, 0], center_points_pixel[:, 1], 0, :] = [0.6, 0.8, 0.6]


    fpoints_feature = fpoints_feature.reshape(1, 32 * 80, 10)
    centerness_feature = centerness_feature.reshape(1, 32 * 80, 1)
    category_feature = category_feature.reshape(1, 32 * 80, 3)
    lanes = {"lane_fpoints": fpoints_feature, "lane_centerness": centerness_feature, "lane_category": category_feature}


    nms, dist_test, overlap_test = NMS(lanes, feat_shapes, flat_indices)
    print(f"dist : \n {dist_test} \n\n overlap \n {overlap_test}")
    image_nms = draw_lanes(image, nms[0][:7].numpy(), nms[0][7].numpy())
    vlog_image = np.concatenate([image_fpoints, image_nms], axis=0)
    cv2.imshow("result", vlog_image)
    cv2.waitKey()


def draw_lanes(image, lanes, centerness):
    print(centerness)
    centerind = (np.squeeze(centerness) - 0.5) / 0.05

    image_d = image.copy()
    height, width = image.shape[:2]
    color = [[0,0,255], [0,255,0], [255,0,0], [255,255,0], [255,0,255], [0,255,255], [255, 255, 255], [153, 0, 102], [153, 102, 0], [0, 102, 255]]
    for num, lane in enumerate(lanes):
        five_points = lane[:10].reshape(-1, 2) * np.array([height, width])
        for i in range(five_points.shape[0]-1):
            cv2.line(image_d, (int(five_points[i, 1]), int(five_points[i, 0])), (int(five_points[i+1, 1]), int(five_points[i+1, 0])), color[int(num)], 10)


    return image_d


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    test_lane_nms()