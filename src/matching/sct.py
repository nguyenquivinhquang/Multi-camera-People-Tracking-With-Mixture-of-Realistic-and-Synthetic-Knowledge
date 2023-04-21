"""Development of Single camera tracking algorithm
    By Vinh Quang Nguyen Qui
"""
import os
from src.utils.utils import load_defaults
import src.reid.processor.post_process.re_rank as re_rank
import pickle
import torch.nn as nn
import torch

import itertools
from multiprocessing import Pool
import numpy as np
import copy
from src.matching.tracklet import *
from sklearn.cluster import KMeans
from src.matching.matching.v3 import matching_v3


cam2scene = {
    "c001": "S001",
    "c002": "S001",
    "c003": "S001",
    "c004": "S001",
    "c005": "S001",
    "c006": "S001",
    "c007": "S001",
    "c014": "S003",
    "c015": "S003",
    "c016": "S003",
    "c017": "S003",
    "c018": "S003",
    "c019": "S003",
    "c047": "S009",
    "c048": "S009",
    "c049": "S009",
    "c050": "S009",
    "c051": "S009",
    "c052": "S009",
    "c076": "S014",
    "c077": "S014",
    "c078": "S014",
    "c079": "S014",
    "c080": "S014",
    "c081": "S014",
    "c100": "S018",
    "c101": "S018",
    "c102": "S018",
    "c103": "S018",
    "c104": "S018",
    "c105": "S018",
    "c124": "S022",
    "c125": "S022",
    "c126": "S022",
    "c127": "S022",
    "c128": "S022",
    "c129": "S022",
    "c118": "S021",
    "c119": "S021",
    "c120": "S021",
    "c121": "S021",
    "c122": "S021",
    "c123": "S021",
}
cfg = load_defaults(["configs/tracking.yaml"])

TRACKER_ROOT_DIR = f"src/SCMT/tmp"
FEATURE_DIR = "src/SCMT/tmp"


###### End approach using DB scan ###########
def matching_camera(scene_camera, n_clusters=8, isVisualize=False):
    print("[Processing]", scene_camera)

    cam = scene_camera.split("_")[1]

    # exit(0)
    trackers, uuids = process_tracklet_input_v2(f"{FEATURE_DIR}/{cam}.pkl", 729)
    track_matching, trackers = matching_v3(
        trackers, min_cluster=n_clusters, scene_camera=scene_camera
    )

    # track_matching = matching_GMM(trackers, scene_camera, n_clusters)
    f = open(f"{TRACKER_ROOT_DIR}/{cam}.txt", "r")
    f = f.readlines()

    result = []
    for line in f:
        line = line.split(",")
        frame_id = line[0]
        track_id, x, y, w, h = map(int, line[1:6])
        result.append(
            f"{frame_id},{track_matching[track_id]},{x},{y},{w},{h},-1,-1,-1\n"
        )
    print(
        "Saving output single camera matching at",
        f"outputs/matching/{scene_camera}.txt",
    )
    with open(f"outputs/matching/{scene_camera}.txt", "w") as f:
        f.writelines(result)

    print("[Done]", scene_camera)
    return


def matching_camera_multithreads(params):
    scene_camera, n_clusters, isVisualize = params
    matching_camera(scene_camera, n_clusters, isVisualize)
    return


if __name__ == "__main__":
    # scene_camera_list = [ (scene_camera.replace('.txt',''),8,False) for scene_camera in os.listdir(TRACKER_ROOT_DIR) if scene_camera.endswith('txt') and 'S003' in scene_camera]
    scene_camera_list = []

    for cam in sorted(os.listdir(TRACKER_ROOT_DIR)):
        if not cam.endswith("txt"):
            continue
        cam = cam.replace(".txt", "")
        scene = cam2scene[cam]
        if scene == "S001":
            continue
        scene_camera_list.append(
            (
                f"{scene}_{cam}",
                cfg["UNIQUE_PERSONS"][scene][cam],
                False,
            )
        )

    # with Pool(10) as p:
    #     p.map(matching_camera_multithreads, scene_camera_list)
    for i in range(len(scene_camera_list)):
        matching_camera_multithreads(scene_camera_list[i])
