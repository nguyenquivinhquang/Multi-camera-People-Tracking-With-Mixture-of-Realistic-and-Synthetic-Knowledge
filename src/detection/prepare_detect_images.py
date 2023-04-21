import os
TARGET_DIR = '/mnt/ssd8tb/quang/detection'
ROOT_DATA_DIR = '/mnt/ssd8tb/quang/AIC23_Track1_MTMC_Tracking/'
ROLES = ["test"]
IMAGE_FORMAT = ".jpg"  # ".png"
from multiprocessing import Pool
from sys import stdout
from pathlib import Path

fprint, endl = stdout.write, "\n"

def prcocess_file(param):
    
    role, path, FOLDER = param
    fprint(f"[Process] {role}{endl}")

    for file in os.listdir(path):
        source_file = f"{path}/{file}"
        target = f"{TARGET_DIR}/{FOLDER}/images/{role}_{file}"
        # print(source_file, target)
        # Path(source_file).symlink_to(target)
        Path(target).symlink_to(source_file)

    fprint(f"[Done] {role}{endl}")

    return
def main():
    train_sets = []
    val_sets = []
    params_meter = []
    for each_role in ROLES:
        role_dir = f"{ROOT_DATA_DIR}/{each_role}"
        scenarios = os.listdir(role_dir)
        for each_scenario in scenarios:
            scenario_dir = f"{role_dir}/{each_scenario}"
            cameras = os.listdir(scenario_dir)
            for each_camera in cameras:
                if "map" in each_camera:
                    continue
                camera_dir = f"{scenario_dir}/{each_camera}/img1"
                params_meter.append(
                    [f"{each_scenario}_{each_camera}", camera_dir, "test"]
                )

    # print(params_meter)
    pool = Pool()
    pool.map(prcocess_file, params_meter)
    pool.close()

if __name__ == "__main__":
    main()