"""
@Filename: run_aicity.py
@Discription: Run AICity Track
"""
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app import run
import time
from multiprocessing import Pool
import traceback

FEAT_DIR = '/home/synh/workspace/quang/Multi-Camera-People-Tracking/output/transformer_feat'
DETECTION_DIR = '/home/synh/workspace/quang/Multi-Camera-People-Tracking/datasets/detections/Yolo_pretrain'

def process(seq):
    
    start_time = time.time()
    
    scene = opt.scene
    FEAT_DIR = opt.feature_dir
    DETECTION_DIR = opt.detection_dir

    print('processing the video {}...'.format(seq))
    path_save = join(opt.dir_save, seq + '.txt')
    try:
        run(
            sequence_dir=join(opt.dir_dataset, seq),
            detection_file=join(DETECTION_DIR, scene + "_"+ seq + '.txt'),
            features_file=join(FEAT_DIR, scene + "_"+ seq + '_feature.pkl'),
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=False
        )
        print('the video {} is done!. Took {}s'.format(seq, time.time() - start_time))
    
    except Exception:
        print("Err on video {}".format(seq))
        traceback.print_exc()
    return
if __name__ == '__main__':
    #print(opt)
    seqs = []
    FEAT_DIR = opt.feature_dir
    DETECTION_DIR = opt.detection_dir
    print("\t Feature Dir:", FEAT_DIR)
    print("\t Detection Dir:", DETECTION_DIR)
    
    for i, seq in enumerate(opt.sequences, start=1):
        print(seq)
        seqs.append(seq)
    
    pool = Pool(processes=10)
    pool.map(process, seqs)
    pool.close()

    print("Tracking DONE!!!!")