import os
import cv2 
import numpy as np
from src.utils.utils import load_defaults
cfg = load_defaults(['configs/baseline.yaml'])
print(cfg['DATASETS']['ROOT_DIR'])
ROOT_DIR = os.path.join(cfg['DATASETS']['ROOT_DIR'], 'test')
print(ROOT_DIR)
# ROOT_DIR = "/mnt/ssd8tb/quang/AIC23_Track1_MTMC_Tracking/test"

cam_info = ""
frame_rate=30
length = 0
width=1920
height=1080
ext = ".jpg"
infor = f"[Sequence]\nname={cam_info}\nimgDir=img1\nframeRate={frame_rate}\nSeqLength={length}\nimWidth={width}\nimHeight={height}\nimgExt={ext}\n"
scene_infor = dict()
for scene in os.listdir(ROOT_DIR):
    if len(scene.split('.')) == 2: continue
    scene_infor[scene] = []
    for cam in os.listdir(os.path.join(ROOT_DIR, scene)):
        if cam == 'map.png': continue
        if len(cam.split('.')) == 2: continue
        cam_dir = os.path.join(ROOT_DIR, scene, cam)
        length = len(os.listdir(cam_dir +'/img1'))
        infor = f"[Sequence]\nname={cam}\nimgDir=img1\nframeRate={frame_rate}\nSeqLength={length}\nimWidth={width}\nimHeight={height}\nimgExt={ext}\n"

        with open(f"{cam_dir}/seqinfo.ini", "w") as f:
            f.write(infor)
        scene_infor[scene].append(cam)

img = np.ones((1080, 1920, 3), np.uint8) * 255.0
cv2.imwrite("src/SCMT/track_roi/roi.png", img)
# print(scene_infor)