import settings
import os
import numpy as np

from train.framework.train_plan import train_by_plan
import utils.framework.util_function as uf
import config as cfg


def train_main():
    uf.set_gpu_configs()
    end_epoch = 0
    for dataset_name, epochs, learning_rate, loss_weights, lr_hold in cfg.Train.TRAINING_PLAN:
        end_epoch += epochs
        train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, lr_hold)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    train_main()
