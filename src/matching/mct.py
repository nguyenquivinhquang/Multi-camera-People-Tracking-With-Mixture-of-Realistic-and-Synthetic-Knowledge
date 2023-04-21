"""
Development of Multi camera tracking algorithm
By Vinh Quang Nguyen Qui
"""
import os
from src.utils.utils import load_defaults
import src.reid.processor.post_process.re_rank as re_rank
import pickle
import torch.nn as nn
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import itertools
from multiprocessing import Pool
import numpy as np
import copy
from src.matching.tracklet import *
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import json

cfg = load_defaults(["configs/tracking.yaml"])

SINGLE_CAMERA_MATCHING = f"outputs/matching/"
FEATURE_DIR = 'src/SCMT/tmp/'


def approach2(camera_trackers, num_clusters, scene):
    from sklearn.mixture import GaussianMixture # Gaussian Mixture Model
    uuids = []
    features = []

    params_sort = []
    center_feats = []
    _choose_center = True
    for cam in camera_trackers:
        for tracklet_id in camera_trackers[cam]:
            feat = torch.stack(camera_trackers[cam][tracklet_id])
            feat = torch.mean(feat, 0)
            features.append(feat.cpu().detach().numpy())
            uuids.append([cam, tracklet_id])
            if _choose_center:
                center_feats.append(feat.cpu().detach().numpy())
        if len(center_feats) == num_clusters:
            _choose_center = False
            center_feats = np.array(center_feats)
        else:
            center_feats = []
    params_sort.append((len(camera_trackers[cam]), cam))
    params_sort = sorted(params_sort, reverse=True)
    _camera_trackers = {}
    for _, cam in params_sort:
        _camera_trackers[cam] = camera_trackers[cam]
    camera_trackers = _camera_trackers
    # sort camera_trackers by the length of tracklets
    
    matrix_distance = cdist(features, features, metric="cosine")
     
    
    print("Using Agglomerative Clustering")
    if scene == 'S021':
        for i in range(len(uuids)):
            for j in range(len(uuids)):
                if i == j: matrix_distance[i][j] = 1
                if uuids[i][0] == uuids[j][0]: 
                    matrix_distance[i][j] = 1
        cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete')
    else:   
        cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
    cluster.fit(matrix_distance)
    cluster_labels = cluster.labels_
    
    
    tracklet_mct__matching = {}
    
    
    for uuid, label in zip(uuids, cluster_labels):
        old_tracklet_id = f"{uuid[0]}_{uuid[1]}"
        tracklet_mct__matching[old_tracklet_id] = int(label)
        
    print(tracklet_mct__matching)
    with open(f"outputs/matching_tracklet/multicamera/tracklet_mct_matching_{scene}.json", "w") as f:
        json.dump(tracklet_mct__matching, f, indent=4)
    
    
    return tracklet_mct__matching
    

def multi_camera_tracking_multi_thread(params):
    scene_camera, camera_lists, clusters = params
    print("[Process]", scene_camera)
    multi_camera_tracking(camera_lists, clusters, scene_camera)
    print("[Done]", scene_camera)
    return


def multi_camera_tracking(camera_lists, ncluster, scene):
    camera_trackers = {}
    for scene_camera in camera_lists:
        scene_camera = scene_camera.split(".")[0]  # remove .txt
        cam = scene_camera.split("_")[1]
        trackers, uuids = process_tracklet_input_matching(
        f"{FEATURE_DIR}/{cam}.pkl",
        f"outputs/matching_tracklet/{scene_camera}_debug_result.json",
        729
        )
        camera_trackers[scene_camera] = trackers

    matching_res = approach2(camera_trackers, ncluster, scene)
    
    scene = scene_camera.split("_")[0]
    result = []
    for camera in camera_lists:
        scene_camera = camera.split(".")[0]  # remove .txt
        camera = scene_camera.split("_")[1]
        camera = int(camera[1:])  # remove c in c001 -> 1
        f = open(f"{SINGLE_CAMERA_MATCHING}/{scene_camera}.txt", "r")
        f = f.readlines()
        for line in f:
            line = line.split(",")
            frame_id = int(line[0])
            track_id = int(line[1])
            x1, y1, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
            track_id = f"{scene_camera}_{track_id}"
            new_track_id = matching_res[track_id]
            result.append(
                f"{camera},{new_track_id},{frame_id},{x1},{y1},{w},{h},-1,-1\n"
            )
    with open(f"outputs/multi_matching/{scene}.txt", "w") as f:
        f.writelines(result)
    return


if __name__ == "__main__":
    scenes = {}
    
    for file in os.listdir(SINGLE_CAMERA_MATCHING):
        scene, camera = file.split(".")[0].split("_")
        if scene not in scenes:
            scenes[scene] = {
                "camera_lists": [],
                "ncluster": cfg["UNIQUE_PERSONS"][scene]["TOTAL"],
            }
        scenes[scene]["camera_lists"].append(file)

    camera_process = []
    for scn in scenes:
        camera_process.append(
            [scn, scenes[scn]["camera_lists"], scenes[scn]["ncluster"]]
        )
    print(camera_process)
    with Pool(8) as p:
        p.map(multi_camera_tracking_multi_thread, camera_process)
    # multi_camera_tracking_multi_thread(camera_process[0])