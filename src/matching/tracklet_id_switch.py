import os
import json
import numpy as np
from src.matching.tracklet_v2 import *
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from src.reid.processor.post_process.re_rank import re_ranking
from scipy.spatial import distance
from multiprocessing import Pool

TRACKER_ROOT_DIR = "outputs/tracking_results/"

TRACK2MATCH = "outputs/matching_tracklet/"
MULTICAM_MATCH = "outputs/matching_tracklet/multicamera"
COS_THRESHOLD = 0.4

def longest_subarray(arr):
    start = end = 0
    max_start = 0
    max_length = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            end += 1
        else:
            if end - start + 1 > max_length:
                max_length = end - start + 1
                max_start = start
            start = end = i
    if end - start + 1 > max_length:
        return start
    else:
        return max_start


def process_split_tracks(pending_tracks_list, split_tracklets_frames):
    new_tracklets = []
    new2old = {}
    total_cur_tracks = 0
    for track in pending_tracks_list:
        bboxes = track.bboxes
        features = track.features
        track_id = track.track_id
        frames = track.frames
        scores = track.scores
        scene_camera = track.scene_camera
        trackA = Tracklet(total_cur_tracks, track.scene_camera)
        trackB = Tracklet(total_cur_tracks + 1, track.scene_camera)
        new2old

        trackA.features = features[: split_tracklets_frames[track_id]]
        trackA.bboxes = bboxes[: split_tracklets_frames[track_id]]
        trackA.frames = frames[: split_tracklets_frames[track_id]]
        trackA.scores = scores[: split_tracklets_frames[track_id]]

        trackB.features = features[split_tracklets_frames[track_id] :]
        trackB.bboxes = bboxes[split_tracklets_frames[track_id] :]
        trackB.frames = frames[split_tracklets_frames[track_id] :]
        trackB.scores = scores[split_tracklets_frames[track_id] :]

        if len(trackA.features) > 0:
            new_tracklets.append(trackA)
            new2old[total_cur_tracks] = track_id
            total_cur_tracks += 1
        if len(trackB.features) > 0:
            new_tracklets.append(trackB)
            new2old[total_cur_tracks] = track_id
            new2old[total_cur_tracks + 1] = track_id
            total_cur_tracks += 1
    return new_tracklets, new2old


def merge_label(label_match, org_tracklet):
    cluster_features = {}
    for trackid in label_match:
        new_cluster = label_match[trackid]
        if new_cluster not in cluster_features:
            cluster_features[new_cluster] = []

        feat = org_tracklet[trackid].mean_features()
        cluster_features[new_cluster].append(feat.cpu().detach().numpy())

    for cluster in cluster_features:
        cluster_features[cluster] = np.array(cluster_features[cluster])
        cluster_features[cluster] = np.mean(cluster_features[cluster], axis=0)

    return cluster_features


def detect_swich(tracklet, sim_thresh):
    feat = torch.stack(tracklet.features)
    cluster = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    cluster.fit(feat)
    _cos = distance.cosine(cluster.means_[0], cluster.means_[1])

    print(tracklet.track_id, _cos)
    if _cos < sim_thresh:
        return (_cos, [])

    cluster_labels = cluster.predict(feat)
    return [_cos, cluster_labels]


def find_split_track(track_id, cluster_labels):
    prefix_zero = [0 for i in range(len(cluster_labels))]
    prefix_one = [0 for i in range(len(cluster_labels))]

    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] == 0:
            prefix_zero[i] = prefix_zero[i - 1] + 1
        else:
            prefix_zero[i] = prefix_zero[i - 1]

        prefix_one[i] = prefix_one[i - 1] + cluster_labels[i]
    _zero = longest_subarray(prefix_zero)
    _one = longest_subarray(prefix_one)
    _split_track = max(_zero, _one)
    # print(track_id, _zero, _one)

    return (track_id, _split_track)


def find_split_track_multi_thread(params):
    return find_split_track(*params)


def process(SCENE, CAM, enssemble=False):
    
    COS_THRESHOLD = 0.4
    print("[Process]: ", SCENE, CAM, COS_THRESHOLD)
    multi_cam_label = json.load(
        open(f"{MULTICAM_MATCH}/tracklet_mct_matching_{SCENE}.json")
    )
    old2new = {}
    file = json.load(open(f"{TRACK2MATCH}/{SCENE}_{CAM}_debug_result.json", "r"))
    for new in file:
        for old in file[new]:
            old2new[int(old)] = int(new)
    ensemble_feature_path = [
        "output/transformer_feat/",
        "output/cls_hrnet_w48_feat",
        "output/transformer_local_feat",
    ]

    tracklets = process_tracklet_input_enssemble(
        f"{SCENE}_{CAM}",
        f"outputs/tracking_results/{SCENE}_{CAM}.txt",
        ensemble_feature_path,
    )
    

    unstable_tracklets = []
    for track_id in tracklets:
        _cos, _label = detect_swich(tracklets[track_id], COS_THRESHOLD)
        if len(_label) == 0:
            continue
        unstable_tracklets.append((track_id, _label))

    split_tracklets = {}
    results = []
    for u in unstable_tracklets:
        results.append(find_split_track_multi_thread(u))

    for track_id, _split_track in results:
        split_tracklets[track_id] = _split_track

    pending_tracklets = []
    keep_tracklets = []
    print(split_tracklets)
    for track_id in tracklets:
        track = tracklets[track_id]
        if track_id in split_tracklets:
            pending_tracklets.append(track)
        else:
            keep_tracklets.append(track)

    new_split_tracklets, new2old = process_split_tracks(
        pending_tracklets, split_tracklets
    )

    label_match = {}
    for track in keep_tracklets:
        label_match[track.track_id] = old2new[track.track_id]

    cluster_features = merge_label(label_match, tracklets)
    query_features_list = []
    query_tracks_id = []

    gallery_features_list = []
    gallery_tracks_id = []
    query_features_list = []
    query_tracks_id = []

    gallery_features_list = []
    gallery_tracks_id = []

    for tracklet in new_split_tracklets:
        # feat = tracklet.mean_features()
        feat = tracklet.mean_features()
        query_features_list.append(feat.cpu().detach().numpy())
        query_tracks_id.append(tracklet.track_id)

    gallery_features_list = []
    gallery_tracks_id = []

    for uuid in cluster_features:
        gallery_features_list.append(cluster_features[uuid])
        gallery_tracks_id.append(uuid)
    query_features_list = np.asarray(query_features_list)
    gallery_features_list = np.asarray(gallery_features_list)
    query_features_list = torch.from_numpy(query_features_list).cuda()
    gallery_features_list = torch.from_numpy(gallery_features_list).cuda()

    result = []
    cam_id = int(CAM.replace("c", ""))
    final_cluster = {}
    label_match_next = {}
    if len(query_features_list) > 0:
        distmat = cdist(
            query_features_list.cpu().detach().numpy(),
            gallery_features_list.cpu().detach().numpy(),
            "cosine",
        )

        for i in range(len(query_tracks_id)):
            query_track_id = query_tracks_id[i]
            dist = distmat[i]
            min_dist = np.min(dist)
            min_index = np.argmin(dist)
            gallery_track_id = gallery_tracks_id[min_index]
            label_match_next[query_track_id] = int(gallery_track_id)
            if gallery_track_id not in final_cluster:
                final_cluster[gallery_track_id] = []
            final_cluster[gallery_track_id].append(query_track_id)

        cam_id = int(CAM.replace("c", ""))
        for tracklet in new_split_tracklets:
            new_cluster = multi_cam_label[
                f"{SCENE}_{CAM}_{label_match_next[tracklet.track_id]}"
            ]
            # print(tracklet.track_id)
            for (frame, bbox) in zip(tracklet.frames, tracklet.bboxes):
                x, y, w, h = bbox
                result.append(f"{cam_id},{new_cluster},{frame},{x},{y},{w},{h},-1,-1\n")
    for tracklet in keep_tracklets:
        new_cluster = multi_cam_label[f"{SCENE}_{CAM}_{label_match[tracklet.track_id]}"]
        # print(tracklet.track_id)
        for (frame, bbox) in zip(tracklet.frames, tracklet.bboxes):
            x, y, w, h = bbox
            result.append(f"{cam_id},{new_cluster},{frame},{x},{y},{w},{h},-1,-1\n")
    print("[DONE]: ", SCENE, CAM)
    return result


def process_multi_threads(prams):
    scene, cam, enssemble = prams
    result = process(scene, cam, enssemble)
    return result


def main():
    global COS_THRESHOLD
    data = {
        "S001": ["c001", "c002", "c003", "c004", "c005", "c006", "c007"],
        "S003": ["c014", "c015", "c016", "c017", "c018", "c019"],
        "S009": ["c047", "c048", "c049", "c050", "c051", "c052"],
        "S014": ["c076", "c077", "c078", "c079", "c080", "c081"],
        "S018": ["c100", "c101", "c102", "c103", "c104", "c105"],
        "S022": ["c124", "c125", "c126", "c127", "c128", "c129"],
        "S021": ["c118", "c119", "c120", "c121", "c122", "c123"],
    }

    for SCENE in data:
        if SCENE == "S001":
            continue
        params = []
        for cam in data[SCENE]:
            params.append((SCENE, cam, True))
        print(params)
        results = []
        for param in params:
            results += process_multi_threads(param)
            # break
 
        with open(
            f"outputs/id_switch/{SCENE}.txt", "w"
        ) as f:
            f.writelines(results)


if __name__ == "__main__":
    main()
