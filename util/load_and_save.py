import glob, json, os
import numpy as np
import util.util_function as uf


def save_bin(save_dirctory, data, file_name):
    if not os.path.exists(save_dirctory):
        os.makedirs(save_dirctory)
    save_bin_file_name = os.path.join(save_dirctory, f'{file_name}.bin')

    if not os.path.exists(save_bin_file_name):
        with open(save_bin_file_name, 'wb') as f:
            f.write(data)
            f.close()


def save_txt(save_dirctory, lines, file_name):
    if not os.path.exists(save_dirctory):
        os.makedirs(save_dirctory)
    save_txt_file_name = os.path.join(save_dirctory, f'{file_name}.txt')
    if not os.path.exists(save_txt_file_name):
        with open(save_txt_file_name, "w") as f:
            for line in lines:
                data = f"{line}\n"
                f.write(data)


def save_txt_dict(save_dirctory, dict, file_name):
    if not os.path.exists(save_dirctory):
        os.makedirs(save_dirctory)
    save_txt_file_name = os.path.join(save_dirctory, f'{file_name}.txt')

    if not os.path.exists(save_txt_file_name):
        with open(save_txt_file_name, "w") as f:
            for catagoly, value in dict.items():
                data = f"{catagoly}: {value}\n"
                f.write(data)

def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def load_label(root_path):
    with open(root_path, "r") as f:
        label = f.read().splitlines()
        pre_objs = np.genfromtxt(label, delimiter=' ',
                                 names=['type', 'truncated', 'occluded', 'alpha', 'bbox_xmin',
                                        'bbox_ymin',
                                        'bbox_xmax', 'bbox_ymax', 'dimensions_1', 'dimensions_2',
                                        'dimensions_3', 'location_1', 'location_2', 'location_3',
                                        'rotation_y'],
                                 dtype=None)
    return pre_objs

def load_velo(filepath):
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, np.float32)
        ss = int(data.shape[0] / 4)
        data = np.reshape(data[:ss * 4], (ss, 4))
        velodyne = data
    return velodyne


def load_kitti(velo_path, label_path, calib_path):
    print(velo_path, label_path, calib_path)
    velo_files = sorted(glob.glob(os.path.join(velo_path, '*.bin')))
    label_flies = sorted(glob.glob(os.path.join(label_path, '*.txt')))
    calib_files = sorted(glob.glob(os.path.join(calib_path, '*.txt')))
    print(velo_files)
    print(label_flies)
    print(calib_files)
    for velo_file, label_flie, calib_file in zip(velo_files, label_flies, calib_files):
        file_num = velo_file.split('/')[-1].split('.')[0]
        print(velo_file)
        print(label_flie)
        print(calib_file)
        velo = load_velo(velo_file) # numpy
        label = load_label(label_flie) # numpy
        calib = read_calib_file(calib_file) # dict
        print(file_num)
        print(type(velo))
        print(type(label))
        print(type(calib))

