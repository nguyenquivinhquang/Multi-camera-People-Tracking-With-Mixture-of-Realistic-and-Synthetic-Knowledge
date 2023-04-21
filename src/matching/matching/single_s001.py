import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy
import json
from scipy.spatial.distance import cdist
from src.reid.processor.post_process.re_rank import re_ranking
from src.matching.tracklet_v2 import Tracklet, process_tracklet_input_s001
from src.utils.matching_utils import *
_init_frames = {
    'c005': 17286,
    'c001': 6281,
    'c002': 19568,
    'c003': 17233,
    'c006': 40007,
    'c007': 32904
}

def check_intersect_time(trackA, trackB):
    setA = set(trackA.frames)
    setB = set(trackB.frames)
    
    return len(setA.intersection(setB)) > 3


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


ensemble_feature_path = ["/mnt/ssd8tb/quang/AIC23_Track1_MTMC_Tracking/outputs/features/cls_hrnet_w48_feat/"]



def load_roi(CAM):
    outzone = json.load(open(f"datasets/ROI/out_zone_tracklet_{CAM}.json"))
    out_bbox = json.load(open(f"datasets/ROI/out_bbox_{CAM}.json"))
    return outzone, out_bbox

class Single_camera_matching:
    def __init__(self, cam) -> None:
        self.cam = cam
        self.tracklets = process_tracklet_input_s001(
            f"S001_{cam}",
            f"outputs/tracking_results/S001_{cam}.txt",
            ensemble_feature_path
        )
        self.outzone, self.out_bbox = load_roi(cam)

    def run(self):
        tracklets_list = list(self.tracklets.values())
        tracklets_list = sorted(tracklets_list, key=lambda x: x.frames[0], reverse=False)

        
        pending_tracklet, keep_tracklet = [], []
        for tracklet in tracklets_list:
            prob = check_inside(tracklet.bboxes, self.outzone)
            if prob < 0.8: 
                keep_tracklet.append(tracklet)
            else:
                pending_tracklet.append(tracklet)
        print("pending", len(pending_tracklet), "; kept", len(keep_tracklet))
        points = self.out_bbox
        ANCHOR_FRAME = _init_frames[self.cam]

        anchors_cluster = []
        occur_frames = [ANCHOR_FRAME]
        big_cluster,tracklets ={}, {}
        
        
        for track in tracklets_list:
            tracklets[track.track_id] = track
            if occur_frames[0] in track.frames:
                anchors_cluster.append(track.track_id)
        for idx, track_id in enumerate(anchors_cluster):
            big_cluster[idx] = [track_id]
 
        self.tracklets = tracklets
        ## Get prefix, postfix tracklets
        prefix_tracklet, postfix_tracklet = [], []
        for track in tracklets_list:
            if track in pending_tracklet: continue
            if track.track_id in anchors_cluster: continue
            track.outPolygonList = points
            track.refine_tracklets()
            if len(track.frames) == 0:
                track.restore()
                pending_tracklet.append(track)
                # print(track.track_id)
                continue
            if track.frames[-1] < ANCHOR_FRAME:
                prefix_tracklet.append(track.track_id)
            else:
                postfix_tracklet.append(track.track_id)
                
        print(len(prefix_tracklet), len(postfix_tracklet))


        #### Start r-matching ------ #### 
        prefix_tracklet = sorted(prefix_tracklet, key=lambda x: tracklets[x].frames[-1], reverse=True)
        postfix_tracklet = sorted(postfix_tracklet, key=lambda x: tracklets[x].frames[0], reverse=True)

        cluster_group_pre, prefix_uncertain = self.matching_tracklet(prefix_tracklet, big_cluster)
        cluster_group_post, postfix_uncertain = self.matching_tracklet(postfix_tracklet, big_cluster)

        for track_id in prefix_uncertain:
            pending_tracklet.append(tracklets[track_id])
        for track_id in postfix_uncertain:
            pending_tracklet.append(tracklets[track_id])
            
        final_cluster = {}
        for cluster in cluster_group_pre:
            final_cluster[cluster] = set(cluster_group_pre[cluster])
            final_cluster[cluster].update(set(cluster_group_post[cluster]))

        label_match = {}
        for cluster in final_cluster:
            for track_id in final_cluster[cluster]:
                label_match[track_id] = cluster


        good_cluster = copy.deepcopy(final_cluster)
        cluster_features = merge_label(label_match, tracklets)

        query_features_list, query_tracks_id = [], []
        gallery_features_list, gallery_tracks_id = [], []
        
        for tracklet in pending_tracklet:
            feat = tracklets[tracklet.track_id].mean_features()
            query_features_list.append(feat.cpu().detach().numpy())
            query_tracks_id.append(tracklet.track_id)
        for uuid in cluster_features:
            gallery_features_list.append(cluster_features[uuid])
            gallery_tracks_id.append(uuid)
        

        query_features_list = torch.from_numpy(np.asarray(query_features_list)).cuda()
        gallery_features_list = torch.from_numpy(np.asarray(gallery_features_list)).cuda()
        if self.cam == 'c002' or self.cam == 'c003':
            distmat = cdist(query_features_list.cpu().detach().numpy(), gallery_features_list.cpu().detach().numpy(),'cosine')
        else:
            distmat = re_ranking(
                query_features_list, gallery_features_list, k1=10, k2=6, lambda_value=0.3
            )

        for i in range(len(query_tracks_id)):
            query_track_id = query_tracks_id[i]
            dist = distmat[i]
            min_index = np.argmin(dist)
            gallery_track_id = gallery_tracks_id[min_index]
            label_match[query_track_id] = int(gallery_track_id)
            final_cluster[gallery_track_id].add(query_track_id)

        result = []
        for cluster in final_cluster:
            for track_id in final_cluster[cluster]:
                tracklet = tracklets[track_id]
                tracklet.restore()
                for (frame, bbox) in zip(tracklet.frames, tracklet.bboxes):
                    _line = f"{frame},{cluster},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},-1,-1,-1\n"
                    result.append(_line)
        
        cam = self.cam
        with open(f"outputs/single_matching_S001/S001_{cam}.txt", "w") as f:
                f.writelines(result)
                
      
                
        for cluster in good_cluster:
            good_cluster[cluster] = list(good_cluster[cluster])    
        for cluster in final_cluster:
            final_cluster[cluster] = list(final_cluster[cluster])    
        
        # Writing to sample.json
        with open(f"outputs/single_matching_S001/good_tracklet_S001_{cam}.json", "w") as outfile:
            outfile.write(json.dumps(good_cluster, indent=4))

        with open(f"outputs/matching_tracklet/S001_{cam}_debug_result.json", "w") as outfile:
            outfile.write(json.dumps(final_cluster, indent=4))
        return
    
    def matching_tracklet(self, tracklets_list, clusters_init):
        cos = nn.CosineSimilarity(dim = 0)
        cluster_group = copy.deepcopy(clusters_init)
        uncertain = []
        cluster_feats = {}
        
        for cluster in cluster_group:
            cluster_feats[cluster] = self.update_mean_features(cluster_group[cluster])
        
        for track_id in tracklets_list:
            candidate_clusters = []
            for cluster in cluster_group:
                if self.check_intersec_time(track_id, cluster_group[cluster]):
                    continue
                cosin_sim = cos(self.tracklets[track_id].mean_features(), torch.from_numpy(cluster_feats[cluster]))
                candidate_clusters.append([float(1-cosin_sim),cluster])
            if len(candidate_clusters) == 0:
                uncertain.append(track_id)
                continue
            candidate_clusters = sorted(candidate_clusters, key=lambda x: x[0], reverse=False)
            
            if candidate_clusters[0][0] > 0.2: 
                uncertain.append(track_id)
                continue
         
            threshold = 23
            if self.cam == 'c005':
                threshold = 27
            if len(candidate_clusters) > 1 and calculate_err(candidate_clusters[0][0], candidate_clusters[1][0]) < threshold:
                uncertain.append(track_id)
                continue
            chosen_cluster = candidate_clusters[0][1]
            
            # recalculate the mean feats
            cluster_group[chosen_cluster].append(track_id)
            cluster_feats[chosen_cluster] = self.update_mean_features(cluster_group[chosen_cluster])
        
        
        return cluster_group, uncertain
    def check_intersec_time(self, trackId, track_list):
        _count = 0
        for track in track_list:
            if check_intersect_time(self.tracklets[trackId], self.tracklets[track]):
                _count += 1
        return _count > 1

    def update_mean_features(self, members):
        mem_features = []
        for mem in members:
            feat = self.tracklets[mem].mean_features()
            mem_features.append(feat.cpu().detach().numpy())
        feats = np.array(mem_features)
        feats = np.mean(feats, axis=0)
        return feats
    

