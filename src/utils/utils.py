import cv2 as cv, cv2
import os
from typing import List
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np
import random
import platform
from .opt import Opts, Config


def showInMovedWindow(winname, img, x, y):
    cv.namedWindow(winname)  # Create a named window
    cv.moveWindow(winname, x, y)  # Move it to (x,y)
    cv.imshow(winname, img)


def getCamCapture(data):
    """Returns the camera capture from parsing or a pre-existing video.

    Args:
      isParse: A boolean value denoting whether to parse or not.

    Returns:
      A video capture object to collect video sequences.
      Total video frames

    """
    total_frames = None
    if os.path.isdir(data):
        cap = cv.VideoCapture(data + "/input/in%06d.jpg")
        total_frames = len(os.listdir(os.path.join(data, "input")))
    else:
        cap = cv.VideoCapture(data)
    return cap, total_frames


def deep_update(mapping: dict, *updating_mappings: dict()) -> dict():
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def get_device(choose_device):
    if torch.cuda.is_available():
        device = "cuda:0"
    if choose_device == "cpu" or device == "cpu":
        return "cpu"
    return device


def get_dict_infor(_dict: dict) -> List[str]:
    res_list = []
    for k, v in _dict.items():
        if isinstance(v, dict):
            get_list = get_dict_infor(v)
            for val in get_list:
                res_list.append(str(k) + "." + str(val))
        else:
            res_list.append(str(k))
    return res_list


def update_cfg(cfg: List, value):
    if len(cfg) == 1:
        return {cfg[0]: value}
    return {cfg[0]: update_cfg(cfg[1:], value)}


def load_enviroment_path(cfg: dict):
    load_dotenv(Path(".env"))

    variables = get_dict_infor(cfg)

    for var in variables:
        variable_value = os.getenv(var)
        if variable_value is None:
            continue
        params = var.split(".")
        temp = update_cfg(params, variable_value)
        cfg = deep_update(cfg, temp)

    return cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device_name():
    return platform.node()


def config2object(config):
    """
    Convert dictionary into instance allowing access to dictionary keys using
    dot notation (attributes).
    """

    class ConfigObject(dict):
        """
        Represents configuration options' group, works like a dict
        """

        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)

        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, val):
            self[name] = val

    if isinstance(config, dict):
        result = ConfigObject()
        for key in config:
            result[key] = config2object(config[key])
        return result
    else:
        return config


def load_defaults(defaults_file: list = []):
    """
    Load default configuration from a list of file.
    """
    cfg = Config("configs/default.yaml")
    # cfg = cfg.update_config(Config("configs/dataset.yaml"))
    for file in defaults_file:
        print(file)
        cfg = deep_update(cfg, Config(file))
    
    cfg = Opts(cfg).parse_args()
   
    cfg = load_enviroment_path(cfg)
    return cfg
