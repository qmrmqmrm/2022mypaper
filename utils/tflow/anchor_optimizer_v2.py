import tensorflow as tf
import config as cfg
import os.path as op
from dataloader.framework.dataset_reader import DatasetReader
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import cv2


def plot_hws(dataset, clustered):
    total_hs = []
    total_ws = []
    for feature in dataset:
        # for scale in cfg.Model.Output.FEATURE_ORDER:
        height = feature["inst_box"][..., 2].numpy()
        width = feature["inst_box"][..., 3].numpy()
        center_x = feature["inst_box"][..., 1].numpy()
        center_y = feature["inst_box"][..., 0].numpy()
        xmin = np.maximum(center_x - width / 2, 0)
        xmax = np.minimum(center_x + width / 2, 1)
        ymin = np.maximum(center_y - height / 2, 0)
        ymax = np.minimum(center_y + height / 2, 1)

        height = ymax - ymin
        width = xmax - xmin
        ori_height = height
        height = height[height > 0]
        width = width[ori_height > 0]

        total_hs.append(height)
        total_ws.append(width)
    total_h = np.concatenate(total_hs, axis=0)
    total_w = np.concatenate(total_ws, axis=0)
    print('total h shape : ', total_h.shape)
    print('total w shape : ', total_w.shape)
    plt.scatter(total_h, total_w, s=1, c=[0, 1, 0])
    plt.scatter(clustered[:, 0], clustered[:, 1], s=1, c=[1, 0, 0])
    plt.savefig('./scatter_hw.png', dpi=200)


def log_clustering(dataset, no_rm=False):
    total_hs = []
    total_ws = []
    for feature in dataset:
        # for scale in cfg.Model.Output.FEATURE_ORDER:
        #     if no_rm:
        #         no_roadmarks = tf.cast(feature[scale][..., 5:6] < 18, dtype=tf.float32)
        #     else:
        #         no_roadmarks = 1
        #     feature[scale] = feature[scale] * no_roadmarks
        height = (feature["inst_box"][..., 2] * cfg.Datasets.Uplus.INPUT_RESOLUTION[0]).numpy()
        width = (feature["inst_box"][..., 3] * cfg.Datasets.Uplus.INPUT_RESOLUTION[1]).numpy()
        ori_height = height
        height = height[height > 0]
        width = width[ori_height > 0]
        total_hs.append(height)
        total_ws.append(width)
    print("end for")
    total_h = np.concatenate(total_hs, axis=0)
    total_w = np.concatenate(total_ws, axis=0)
    total_hws = np.stack([total_h, total_w], axis=-1)

    model = KMeans(n_clusters=9, algorithm='auto')

    model.fit(total_hws)
    results = np.round(model.cluster_centers_)
    squares = results[..., 0] * results[..., 1]
    results = results[np.argsort(squares)]
    sorted_squares = squares[np.argsort(squares)]
    for result, square in zip(results, sorted_squares):
        print(f"anchor: {result}, square: {square}")
    cluster = np.round(model.cluster_centers_)
    cluster /= np.array([[512, 1280]])
    return cluster


def log_tfrecord():
    dataset = DatasetReader(op.join(cfg.Paths.DATAPATH, "uplus21_train")).get_dataset()
    # for feature in dataset:
    #     print(feature.keys())
    # log_boxes_per_anchor(dataset)
    # log_num_per_category(dataset)
    with tf.device("/GPU:0"):
        clustered = log_clustering(dataset, no_rm=True)
        plot_hws(dataset, clustered)
    # image_test(dataset)


if __name__ == "__main__":
    log_tfrecord()