import os

MULTICAM_TRACKING_RESULT_PATH = 'outputs/id_switch'

person_id_convert = {}
res = []
def process_scene(scene):
    global res
    f = open(scene, 'r')
    f = f.readlines()
    scene = scene.split('/')[-1].split('.')[0]
    for line in f:
        cam, person_id, frame_id, x, y, w, h, _,_ = line.split(',')
        frame_id = int(frame_id) - 1
        # cam = f"c{cam.zfill(3)}"
        uuid = f"{scene}_{person_id}"
        if uuid not in person_id_convert:
            person_id_convert[uuid] = len(person_id_convert)
        
        res.append(f"{cam} {person_id_convert[uuid]} {frame_id} {x} {y} {w} {h} -1 -1\n")
for scene in os.listdir(MULTICAM_TRACKING_RESULT_PATH):
    if scene.endswith('txt'):
        process_scene(os.path.join(MULTICAM_TRACKING_RESULT_PATH, scene))
with open('outputs/track1.txt', 'w') as f:
    f.writelines(res)