import os
from src.utils.utils import load_defaults
import pickle
import torch
from scipy.spatial.distance import cdist
import numpy as np
import copy

from src.matching.tracklet_v2 import Tracklet, process_tracklet_input_s001
import json
import torch.nn.functional as F
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from src.utils.matching_utils import load_roi_mask
SINGLE_CAMERA_MATCHING = f"outputs/single_matching_S001/"
ensemble_feature_path = [
    "output/transformer_feat/",
    "output/cls_hrnet_w48_feat",
    "output/transformer_local_feat",
]


out_bbox, out_zone = load_roi_mask()
camera_list = ["c001", "c002", "c003", "c004", "c005", "c006", "c007"]
camera_tracklets = {}
track2cluster = {}
for cam in camera_list:
    tracklets = process_tracklet_input_s001(
        f"S001_{cam}",
        f"outputs/tracking_results/S001_{cam}.txt",
        ensemble_feature_path,
    )
    camera_tracklets[cam] = tracklets


camera_cluster_features = {}
print("[Process input camera]: ")
for cam in camera_list:
    if cam == "c004":
        continue
    tracklets = camera_tracklets[cam]
    good_tracklets = json.load(
        open(f"{SINGLE_CAMERA_MATCHING}/good_tracklet_S001_{cam}.json")
    )
    # print(good_tracklets)
    track_ids = []
    cluster_feature = {}
    for cluster in good_tracklets:
        for track in good_tracklets[cluster]:
            
            tracklets[track].restore()
            tracklets[track].outPolygonList = out_bbox[f"{cam}"]
            tracklets[track].refine_tracklets()
            feat = tracklets[track].mean_features()
            if cluster not in cluster_feature:
                cluster_feature[cluster] = []
           
            cluster_feature[cluster].append(feat)
    for cluster in cluster_feature:
        cluster_feature[cluster] = torch.stack(cluster_feature[cluster]).mean(dim=0)
    camera_cluster_features[cam] = cluster_feature

print("[Process Done]")

uuids, features,center_features = [], [], []

print("Select c001 as center features")
for cam in camera_cluster_features:
    if "c004" in cam:
        continue
    for cluster_id in camera_cluster_features[cam]:
        uuids.append([cam, cluster_id])
        features.append(camera_cluster_features[cam][cluster_id].cpu().detach().numpy())
        if cam == "c001":
            center_features.append(
                camera_cluster_features[cam][cluster_id].cpu().detach().numpy()
            )


center_features = np.array(center_features)
features = np.array(features)

index_feat_cam = {}
for cam in camera_cluster_features:
    start, end = -1, -1
    for idx, uuid in enumerate(uuids):
        if uuid[0] == cam:
            if start == -1:
                start = idx
            end = idx
    index_feat_cam[cam] = [start, end]

### Matching process ###
print("[Start matching]")
matching = {}
camera_process = ["c002", "c003", "c005", "c006", "c007"]
preCam = "c001"
for cam in camera_process:
    preCam_x = index_feat_cam[preCam][0]
    preCam_y = index_feat_cam[preCam][1]

    cam_x = index_feat_cam[cam][0]
    cam_y = index_feat_cam[cam][1]

    pre_uuid = uuids[preCam_x : preCam_y + 1]
    cur_uuid = uuids[cam_x : cam_y + 1]
    distmat = cdist(
        features[index_feat_cam[preCam][0] : index_feat_cam[preCam][1] + 1],
        features[index_feat_cam[cam][0] : index_feat_cam[cam][1] + 1],
        "cosine",
    )
    row_ind, col_ind = linear_sum_assignment(distmat)
    
    for pre, cur in zip(row_ind, col_ind):
        matching[
            f"{pre_uuid[pre][0]}_{pre_uuid[pre][1]}"
        ] = f"{cur_uuid[cur][0]}_{cur_uuid[cur][1]}"

    preCam = cam
    
tracklet_mct_matching, _tracklet_mct_matching = {}, {}
_count = 0
debug_vis = {}
for uuid in uuids:
    uuid = f"{uuid[0]}_{uuid[1]}"
    tracklet_mct_matching[uuid] = -1
    if "c001" in uuid:
        tracklet_mct_matching[uuid] = _count
        debug_vis[_count] = [str(uuid)]
        _count += 1


for source in matching:
    target = matching[source]
    tracklet_mct_matching[target] = tracklet_mct_matching[source]
    if tracklet_mct_matching[target] not in debug_vis:
        debug_vis[tracklet_mct_matching[target]] = []
    debug_vis[tracklet_mct_matching[target]].append(target)

for uuid in tracklet_mct_matching:
    _tracklet_mct_matching["S001_" + uuid] = tracklet_mct_matching[uuid]

print(debug_vis)
print("[Matching Done]")

## Matching C004 to cluster #####
query_features_list, query_tracks_id = [], []
gallery_features_list, gallery_tracks_id = [], []

for tracklet_id in camera_tracklets["c004"]:
    query_tracks_id.append(["S001_c004", tracklet_id])
    query_features_list.append(camera_tracklets["c004"][tracklet_id].mean_features())
for cam in camera_cluster_features:
    for track_id in camera_cluster_features[cam]:
        gallery_features_list.append(camera_cluster_features[cam][track_id])
        gallery_tracks_id.append(tracklet_mct_matching[f"{cam}_{track_id}"])

query_features_list = torch.stack(query_features_list).cuda()
gallery_features_list = torch.stack(gallery_features_list).cuda()
distmat = cdist(
    query_features_list.cpu().detach().numpy(),
    gallery_features_list.cpu().detach().numpy(),
    "cosine",
)

_debug_s004 = {}
for i in range(len(query_tracks_id)):
    query_track_id = query_tracks_id[i]
    query_track_id = f"{query_track_id[0]}_{query_track_id[1]}"
    # print(query_track_id)
    dist = distmat[i]
    min_dist = np.min(dist)
    min_index = np.argmin(dist)
    gallery_track_id = gallery_tracks_id[min_index]
    _tracklet_mct_matching[query_track_id] = int(gallery_track_id)
    if gallery_track_id not in _debug_s004:
        _debug_s004[gallery_track_id] = []
    _debug_s004[gallery_track_id].append(int(query_track_id.replace("S001_c004_", "")))

# debug only
_write_res = {}
for uuid in _debug_s004:
    _write_res[str(uuid)] = _debug_s004[uuid]
with open(f"outputs/matching_tracklet/S001_c004_debug_result.json", "w") as f:
    json.dump(_write_res, f, indent=4)


_res_debug = {}

for uuid in tracklet_mct_matching:
    _res_debug["S001_" + uuid] = tracklet_mct_matching[uuid]

for x in _debug_s004:
    _res_debug["S001_c004_" + str(x)] = x
_res_debug
with open(
    f"outputs/matching_tracklet/multicamera/tracklet_mct_matching_S001.json", "w"
) as f:
    json.dump(_res_debug, f, indent=4)


### Writig result
result = []
for camera in camera_list:
    if "good" in camera or "c004" in camera:
        continue
    print(camera)
    scene_camera = camera.split(".")[0]  # remove .txt
    # camera = scene_camera.split("_")[1]
    camera = int(camera[1:])  # remove c in c001 -> 1
    f = open(f"outputs/single_matching_S001/S001_{scene_camera}.txt", "r")
    f = f.readlines()
    for line in f:
        line = line.split(",")
        frame_id = int(line[0])
        track_id = int(line[1])
        x1, y1, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
        track_id = f"{scene_camera}_{track_id}"
        new_track_id = tracklet_mct_matching[track_id]
        if (
            tracklet_mct_matching[track_id]
            != _tracklet_mct_matching[f"S001_{track_id}"]
        ):
            print("asas")
        result.append(f"{camera},{new_track_id},{frame_id},{x1},{y1},{w},{h},-1,-1\n")


# handle cam004
for tracklet_id in camera_tracklets["c004"]:
    new_cluster = _tracklet_mct_matching[f"S001_c004_{tracklet_id}"]
    for frame, bbox in zip(
        camera_tracklets["c004"][tracklet_id].frames,
        camera_tracklets["c004"][tracklet_id].bboxes,
    ):
        x, y, w, h = bbox
        result.append(f"{4},{new_cluster},{frame},{x},{y},{w},{h},-1,-1\n")


with open(f"outputs/multi_matching/S001.txt", "w") as f:
    f.writelines(result)
