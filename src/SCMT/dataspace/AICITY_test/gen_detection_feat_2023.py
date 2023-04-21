import gc
import os
import pickle

import numpy as np
from tqdm import tqdm

feat_list = []

FEATURE_DIR = "/mnt/Data/dataset/ReiD/AIC23_Track1_MTMC_Tracking/outputs/Features/trans_feat_yolo"
DETECTION_DIR = "/mnt/Data/CVIP-Lab-Work/Multi-Camera-People-Tracking/datasets/detections/Yolo_pretrain"


def process_cam(scene_cam):
    result = []
    feats = pickle.load(open(f"{FEATURE_DIR}/{scene_cam}_feature.pkl", "rb"))
    print("Load feat done")
    f = open(f"{DETECTION_DIR}/{scene_cam}.txt", "r")
    f = f.readlines()
    f = sorted(f, key=lambda x: int(x.split(",")[0]))
    # print(f[0])
    for (idx, line) in  tqdm(enumerate(f),total=len(f)):
        f = line.split(",")
        fid = f[0].zfill(6)
        x = int(f[2])
        y = int(f[3])
        w = int(f[4])
        h = int(f[5])
        conf = float(f[6])

        feat_name = f"{scene_cam}_{fid}_{x}_{y}_{w}_{h}"
        feat_value = feats.pop(feat_name)
        _line = [fid, -1, x, y, w, h, conf, -1, -1, -1]
        _line.extend(feat_value.tolist())
        result.append(_line)
        
    return result

for scene in os.listdir(DETECTION_DIR):
    scene = scene.replace(".txt", "")
    result = process_cam(scene)
    det_feat_npy = np.array(result)
    # np.save('{}.npy'.format(det_feat_npy), det_feat_npy)
    with open(f"{scene}.pkl", "wb") as f:
        pickle.dump(det_feat_npy, f)
    del result, det_feat_npy
    gc.collect()

# print(result[0])