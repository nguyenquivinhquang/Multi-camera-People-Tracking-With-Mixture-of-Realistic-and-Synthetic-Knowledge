import pickle
import numpy as np
import gc

cam_det = {'c041': [],
           'c042': [],
           'c043': [],
           'c044': [],
           'c045': [],
           'c046': []}
box_feat = {}
print('* reading ReID results, it may cost a lot of time')
feat_list = ['aic22_1_test_infer_v2_HR48_eps.pkl', 'aic22_1_test_infer_v2_Convnext.pkl', 'aic22_1_test_infer_v2_Res2Net200.pkl', 'aic22_1_test_infer_v2_R50.pkl', 'aic22_1_test_infer_v2_ResNext101.pkl']

for res_file in feat_list[:1]:
    res_file = '/mnt/bk_data/track1/' + res_file
    with open(res_file, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
        _count = 0
        for k, v in obj.items():
            if 'ipynb_checkpoints' in k:
                continue
            cam = k.split('/')[-1].split('_')[0]
            if cam != 'c042': continue
            if k not in box_feat.keys():
                box_feat[k] = []
            box_feat[k].append(v)
            _count += 1
            # if _count > 10: break

gc.collect()

for k, vs in box_feat.items():
    file_name = k.split('/')[-1].replace('.png', '')
    cam_name = file_name.split('_')[0]
    fid = int(file_name.split('_')[1])
    x = int(file_name.split('_')[2])
    y = int(file_name.split('_')[3])
    w = int(file_name.split('_')[4])
    h = int(file_name.split('_')[5])
    conf = float(file_name.split('_')[6])
    feat = []
    for v in vs:
        feat += v.tolist()
    line = [fid, -1, x, y, w, h, conf, -1, -1, -1] + feat
    cam_det[cam_name].append(line)

del box_feat
gc.collect()

for cam_name in cam_det.keys():
    print('* start to save', cam_name)
    det_feat_npy = np.array(sorted(cam_det[cam_name]))
    #print(det_feat_npy.shape)
    np.save('{}.npy'.format(cam_name), det_feat_npy)

del cam_det
gc.collect()
