"""
@Filename: opts.py
@Discription: opts
"""
import json
import argparse
from os.path import join

data = {
    "S001": ["c001", "c002", "c003", "c004", "c005", "c006", "c007"],
    "S003": ["c014", "c015", "c016", "c017", "c018", "c019"],
    "S009": ["c047", "c048", "c049", "c050", "c051", "c052"],
    "S014": ["c076", "c077", "c078", "c079", "c080", "c081"],
    "S018": ["c100", "c101", "c102", "c103", "c104", "c105"],
    "S022": ["c124", "c125", "c126", "c127", "c128", "c129"],
    "S021": ["c118", "c119", "c120", "c121", "c122", "c123"],
}


class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="AICITY",
        )
        self.parser.add_argument(
            "--scene",
            type=str,
            default="S001",
        )
        self.parser.add_argument(
            "--NSA", action="store_true", help="NSA Kalman filter", default=True
        )
        self.parser.add_argument(
            "--EMA",
            action="store_true",
            help="EMA feature updating mechanism",
            default=True,
        )
        self.parser.add_argument(
            "--MC",
            action="store_true",
            help="Matching with both appearance and motion cost",
            default=True,
        )
        self.parser.add_argument(
            "--woC",
            action="store_true",
            help="Replace the matching cascade with vanilla matching",
            default=True,
        )
        self.parser.add_argument(
            "--root_dataset", default="/mnt/ssd8tb/quang/AIC23_Track1_MTMC_Tracking/test/"
        )
        self.parser.add_argument(
            "--feature_dir", default="/home/synh/workspace/quang/Multi-Camera-People-Tracking/output/features"
        )
        self.parser.add_argument(
            "--detection_dir", default="/home/synh/workspace/quang/Multi-Camera-People-Tracking/datasets/detections/Yolo"
        )

        self.parser.add_argument("--dir_save", default="./tmp")
        self.parser.add_argument("--EMA_alpha", default=0.9)
        self.parser.add_argument("--MC_lambda", default=0.98)

    def parse(self, args=""):
        # if args == '':
        #   opt = self.parser.parse_args()
        # else:
        #   opt = self.parser.parse_args(args)
        opt = self.parser.parse_args()
        opt.min_confidence = 0.1
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        opt.max_cosine_distance = 0.3
                
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        opt.sequences = data[opt.scene]
        opt.dir_dataset = join(
            opt.root_dataset, opt.scene
        )
        return opt


opt = opts().parse()
