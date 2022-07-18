import numpy as np

import config_dir.meta_config as meta
import config_dir.parameter_pool as params

np.set_printoptions(precision=5)


def save_config():
    read_file = open(meta.Paths.META_CFG_FILENAME, 'r')
    write_file = open(meta.Paths.CONFIG_FILENAME, "w")

    dataset = meta.Datasets.TARGET_DATASET
    set_dataset_and_get_config(dataset)

    set_anchors()
    space_count = 0
    data = "meta."
    write_file.write(f"import numpy as np\n")
    for i, line in enumerate(read_file):
        line = line.rstrip("\n")
        space_count, data = line_structure(line, write_file, space_count, data)
    write_file.close()


def line_structure(line, f, space_count=0, data=""):
    if "class " in line:  # 라인에 class가 있으면
        # skip = False
        if "clas" in line[space_count * 4:(space_count + 1) * 4]:  # space_count 다음에 바로 class가 오면
            if space_count != 0:  # space 가 0이 아니면 즉, 다른 내부 클래스면
                data = ".".join(data.split(".")[:-2]) + "."
                f.write(f"\n")
            else:  # 맨 처음 class 이면
                data = "meta."

                f.write(f"\n\n")
        elif "    " in line[space_count * 4:(space_count + 1) * 4]:  # 내부 클래스라 확인 되면
            space_count = space_count + 1
            f.write(f"\n")

        else:  #
            space_count = 0
            data = "meta."
            f.write(f"\n\n")

        class_name = line[space_count * 4:].replace("class ", "").replace(":", "")
        data = data + f"{class_name}."
        f.write(f"{line}\n")

    elif "#" in line:
        pass

    elif "=" in line:
        if not "    " in line[space_count * 4:(space_count + 1) * 4]:
            space_count = space_count - 1
            data = ".".join(data.split(".")[:-2]) + "."
        param_name = line[(space_count + 1) * 4:].split("=")[0].strip(" ")
        param = eval(f"{data}{param_name}")
        space = "    " * (space_count + 1)
        plan_space = " " * (len(f"{space}{param_name}") + 4)
        if param_name == "TRAINING_PLAN":
            f.write(f"{space}{param_name} = [\n")

            for plan in param:
                f.write(f"{plan_space}{plan},\n")
            f.write(f"{plan_space}]\n")

        elif isinstance(param, dict):
            count = 0
            f.write(f"{space}{param_name} =" + " {")

            for key, value in param.items():
                count += 1
                if isinstance(value, str):
                    f.write(f"\"{key}\": \"{value}\", ")
                else:
                    f.write(f"\"{key}\": {value}, ")
                if count % 3 == 0:
                    f.write(f"\n{plan_space}")

            f.write(f"\n{plan_space}" + "}" + "\n")

        elif isinstance(param, str):
            f.write(f"{space}{param_name} = \"{param}\"\n")
        elif isinstance(param, np.ndarray):
            f.write(f"{space}{param_name} = np.array({(np.round(param.tolist(), 5)).tolist()})\n")
        elif isinstance(param, type):
            f.write(f"{space}{param_name} = {param.__name__}\n")
        else:
            f.write(f"{space}{param_name} = {param}\n")
    elif "pass" in line:
        f.write(f"{line}\n")
    return space_count, data


def set_dataset_and_get_config(dataset):
    meta.Datasets.TARGET_DATASET = dataset
    dataset_cfg = getattr(meta.Datasets, dataset.capitalize())  # e.g. meta.Datasets.Uplus
    meta.Datasets.DATASET_CONFIG = dataset_cfg
    print(meta.Datasets.DATASET_CONFIG)

    if dataset == "uplus":
        basic_anchor = params.Anchor.UPLUS_YOLOv4
        anchor_resolution = np.array(params.Anchor.UPLUS_RESOLUTION, dtype=np.float32)
        meta.Dataloader.ANCHORS_LANE = params.Anchor.UPLUS_LANE

    elif dataset == "kitti" or dataset == "kittibev"  :
        basic_anchor = params.Anchor.KITTI
        anchor_resolution = np.array(params.Anchor.KITTI_RESOLUTION, dtype=np.float32)
    else:
        assert 0, f"{dataset} dataset is NOT expected"

    input_resolution = np.array(dataset_cfg.INPUT_RESOLUTION, dtype=np.float32)
    scale = np.min(input_resolution / anchor_resolution)
    meta.Dataloader.ANCHORS_PIXEL = np.around(basic_anchor * scale, 1)

    return meta.Datasets.DATASET_CONFIG


def set_anchors():
    anchor_mode = meta.AnchorGeneration.ANCHOR_STYLE
    anchor_class = getattr(meta.AnchorGeneration, anchor_mode)
    anchor_option = dict()
    for item in dir(anchor_class):
        if item.startswith("__"):
            continue
        anchor_option[item.lower()] = getattr(anchor_class, item)
    anchor = anchor_generator(**anchor_option)
    anchors = np.tile(anchor, (len(meta.ModelOutput.FEATURE_SCALES), 1))\
        .reshape((len(meta.ModelOutput.FEATURE_SCALES), len(anchor), 2))

    scale_anchor = []
    for anchor, scale in zip(anchors, meta.AnchorGeneration.MUL_SCALES):
        scale_anchor.append(anchor * scale)
    meta.AnchorGeneration.ANCHORS = np.array(scale_anchor, dtype=np.int32)


def anchor_generator(aspect_ratio, base_anchor, scales):
    anchor_hws = []
    for scale in scales:
        anchor_hw = [
            [base_anchor[0] * np.sqrt(aratio) * np.sqrt(scale), base_anchor[1] / np.sqrt(aratio) * np.sqrt(scale)] for
            aratio in aspect_ratio]
        anchor_hws.append(anchor_hw)
    anchors = np.array([anchor_hws], dtype=np.float32).reshape((len(aspect_ratio) * len(scales)), -1)
    return anchors


save_config()
