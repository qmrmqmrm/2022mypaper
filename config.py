import os
import sys
import glob


class Paths:
    class Kitti:
        Traing_ROOT = '/media/falcon/IanBook8T/datasets/kitti_detection/'
    VELO_ROOT = Kitti.Traing_ROOT + 'data_object_velodyne/training/velodyne'
    LABEL_ROOT = Kitti.Traing_ROOT + 'data_object_label_2/training/label_2'
    CALIB_ROOT = Kitti.Traing_ROOT + 'data_object_calib/training/calib'
    SAVE_DIR = "/media/falcon/IanBook8T/datasets/kim_result"
