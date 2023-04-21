import os
import json
import numpy as np
import PIL.Image as Image
from src.utils.opt import *
import src.utils.utils as util
import cv2
import os
from multiprocessing import Pool
from sys import stdout
import numpy as np

cfg = util.load_defaults()
ROOT_DATA_DIR = cfg["DATASETS"]["ROOT_DIR"]
print(ROOT_DATA_DIR)

fprint, endl = stdout.write, "\n"


ROLES = ["train", "validation", "test"]
IMAGE_FORMAT = ".jpg"  # ".png"


def video2image(parameter_set):
    role, scenario, camera, camera_dir = parameter_set
    fprint(f"[Processing] {role} {scenario} {camera}{endl}")
    imgs_dir = f"{camera_dir}/img1"
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    cap = cv2.VideoCapture(f"{camera_dir}/video.mp4")
    current_frame = 1
    ret, frame = cap.read()
    while ret:
        frame_file_name = f"{str(current_frame).zfill(6)}{IMAGE_FORMAT}"
        cv2.imwrite(f"{imgs_dir}/{frame_file_name}", frame)
        ret, frame = cap.read()
        current_frame += 1
    fprint(f"[Done] {role} {scenario} {camera}{endl}")


def main():
    parameter_sets = []
    for each_role in ROLES:
        role_dir = f"{ROOT_DATA_DIR}/{each_role}"
        scenarios = os.listdir(role_dir)
        for each_scenario in scenarios:
            scene = each_scenario
            scenario_dir = f"{role_dir}/{each_scenario}"
            cameras = os.listdir(scenario_dir)
            # cameras = ["c019"]
            for each_camera in cameras:
                cam = each_camera
                if "map" in each_camera:
                    continue
                camera_dir = f"{scenario_dir}/{each_camera}"                
                parameter_sets.append(
                    [each_role, each_scenario, each_camera, camera_dir]
                )

    pool = Pool(processes=15)
    pool.map(video2image, parameter_sets)
    pool.close()


if __name__ == "__main__":
    main()
