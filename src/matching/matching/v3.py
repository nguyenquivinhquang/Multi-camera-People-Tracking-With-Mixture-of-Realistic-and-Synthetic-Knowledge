import os
from src.utils.utils import load_defaults
import pickle
import torch.nn as nn
import torch
from sklearn.cluster import KMeans
from multiprocessing import Pool
import numpy as np
import copy
from src.matching.tracklet import *
from typing import List
from src.reid.processor.post_process.re_rank import re_ranking
import json
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering


WIDTH_THRESHOLD = 30
HEIGHT_THRESHOLD = 30
AREA_THRESHOLD = 900
DISTANCE_THRESHOLD = 5


def get_param(scene_camera):
    k1 = 10
    if scene_camera == "S001_c001":
        return 1000, "GMM", k1
    if scene_camera == "S001_c003":
        return 100, "GMM", k1
    if "c077" in scene_camera:
        return 10, "Agglomerative", 3
    if "c080" in scene_camera:
        return 10, "GMM", 3
    if "c004" in scene_camera:
        return 10, "Agglomerative", k1
    if "S001" in scene_camera:
        return 500, "GMM", k1

    if scene_camera == "S003_c014":
        return 100, "GMM", k1

    if "S003" in scene_camera:
        return 100, "Agglomerative", k1

    if scene_camera == "S014_c081" or scene_camera == "S014_c079":
        return 10, "GMM", 3

    if "c118" in scene_camera or "c123" in scene_camera:
        return 100, "Agglomerative", k1
    if "c120" in scene_camera:
        return 10, "Agglomerative", 3
    if "c119" in scene_camera:
        return 10, "Agglomerative", 3
    if "c122" in scene_camera:
        return 10, "Agglomerative", 3
    if "c127" in scene_camera:
        return 30, "Agglomerative", k1
    if "c124" in scene_camera:
        return 113, "GMM", 3
    if "c129" in scene_camera:
        return 60, "GMM", 3

    if "c100" in scene_camera:
        return 100, "Agglomerative", k1
    if "c102" in scene_camera:
        return 100, "Agglomerative", k1
    if "c103" in scene_camera:
        return 100, "Agglomerative", k1
    if "c104" in scene_camera:
        return 100, "Agglomerative", k1
    if "c105" in scene_camera:
        return 100, "Agglomerative", k1
    if "c047" in scene_camera:
        return 50, "Agglomerative", k1
    if "c048" in scene_camera:
        return 3, "Agglomerative", 3

    if "c049" in scene_camera:
        return 50, "Agglomerative", 3

    if "c050" in scene_camera:
        return 30, "Agglomerative", 3
    if "c052" in scene_camera:
        return 30, "Agglomerative", 3
    if "c051" in scene_camera:
        return 5, "Agglomerative", 3

    if "c125" in scene_camera:
        return 5, "Agglomerative", 3

    if "S009" in scene_camera:
        return 100, "GMM", k1

    if "S022" in scene_camera:
        return 100, "GMM", 3
    if "c076" in scene_camera:
        return 10, "GMM", 3
    if "c078" in scene_camera:
        return 10, "GMM", 5
    if "c118" in scene_camera:
        return 10, "GMM", 5
    if "c119" in scene_camera:
        return 10, "GMM", 5
    if "c121" in scene_camera:
        return 10, "Agglomerative", 5
    if "c122" in scene_camera:
        return 100, "GMM", 5

    return 10, "GMM", k1


def merge_label(label_match, org_tracklet):
    cluster_features = {}
    for trackid in label_match:
        new_cluster = label_match[trackid]
        if new_cluster not in cluster_features:
            cluster_features[new_cluster] = []

        feat = torch.stack(org_tracklet[trackid])
        feat = torch.mean(feat, 0)
        cluster_features[new_cluster].append(feat.cpu().detach().numpy())

    for cluster in cluster_features:
        cluster_features[cluster] = np.array(cluster_features[cluster])
        cluster_features[cluster] = np.mean(cluster_features[cluster], axis=0)

    return cluster_features


def matching_v3(tracklets: Dict[str, Tracklet], min_cluster, scene_camera):
    """
    Matching tracklets using DBSCAN
    """
    TRACKLET_LENGTH, Method, k1 = get_param(scene_camera)
    print(TRACKLET_LENGTH, Method, k1)
    label_match = {}

    track_uuid = []
    features = []

    pending_tracklets = {}
    for uuid in tracklets:
        if len(tracklets[uuid]) < TRACKLET_LENGTH:
            pending_tracklets[uuid] = tracklets[uuid]
            continue
        feat = torch.stack(tracklets[uuid])
        feat = torch.mean(feat, 0)
        features.append(feat.cpu().detach().numpy())
        track_uuid.append(uuid)
    features = np.asarray(features)

    if Method == "GMM":
        gm = GaussianMixture(
            n_components=min_cluster,
            covariance_type="full",
            max_iter=100000,
            random_state=0,
        ).fit(features)
        labels = gm.predict(features)

    elif Method == "Agglomerative":
        ag = AgglomerativeClustering(
            n_clusters=min_cluster, affinity="euclidean", linkage="average"
        ).fit(features)
        labels = ag.labels_

    label_match = {}
    for i in range(len(track_uuid)):
        label_match[track_uuid[i]] = int(labels[i])

    if len(pending_tracklets) == 0:
        debug_result = {}
        for uuid in label_match:
            if label_match[uuid] not in debug_result:
                debug_result[label_match[uuid]] = []
            debug_result[label_match[uuid]].append(int(uuid))
        with open(
            f"outputs/matching_tracklet/{scene_camera}_debug_result.json", "w"
        ) as f:
            json.dump(debug_result, f)
        return label_match, tracklets

    print("Handle pending tracklets")
    cluster_features = merge_label(label_match, tracklets)

    query_features_list = []
    query_tracks_id = []
    for uuid in pending_tracklets:
        feat = torch.stack(tracklets[uuid])
        feat = torch.mean(feat, 0)
        query_features_list.append(feat.cpu().detach().numpy())
        query_tracks_id.append(uuid)

    gallery_features_list = []
    gallery_tracks_id = []

    for uuid in cluster_features:
        gallery_features_list.append(cluster_features[uuid])
        gallery_tracks_id.append(uuid)

    query_features_list = np.asarray(query_features_list)
    gallery_features_list = np.asarray(gallery_features_list)
    query_features_list = torch.from_numpy(query_features_list).cuda()
    gallery_features_list = torch.from_numpy(gallery_features_list).cuda()

    distmat = re_ranking(
        query_features_list, gallery_features_list, k1=k1, k2=6, lambda_value=0.3
    )
    for i in range(len(query_tracks_id)):
        query_track_id = query_tracks_id[i]
        dist = distmat[i]
        # min_dist = np.min(dist)
        min_index = np.argmin(dist)
        gallery_track_id = gallery_tracks_id[min_index]
        label_match[query_track_id] = int(gallery_track_id)

    # Visualiing debug the result
    debug_result = {}
    for uuid in label_match:
        if label_match[uuid] not in debug_result:
            debug_result[label_match[uuid]] = []
        debug_result[label_match[uuid]].append(int(uuid))

    print(
        "Saving relabel debug at",
        f"outputs/matching_tracklet/{scene_camera}_debug_result.json",
    )
    with open(f"outputs/matching_tracklet/{scene_camera}_debug_result.json", "w") as f:
        json.dump(debug_result, f)

    return label_match, tracklets
