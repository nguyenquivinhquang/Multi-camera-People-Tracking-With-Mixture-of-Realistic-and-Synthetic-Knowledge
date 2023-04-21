import torch.nn.functional as F
import numpy as np
from shapely.geometry import Polygon
import copy
import torch
import pickle


class Tracklet(object):
    IMG_WIDTH = 1920
    IMG_HEIGHT = 1080
    THRESHOLD = 10
    """ 
        Note:
            BB: [x1, y1, w, h]
    """

    def __init__(self, track_id, scene_camera):
        self.track_id = track_id
        self.scene_camera = scene_camera
        self.bboxes = []
        self.features = []
        self.scores = []
        self.frames = []
        self._small_object = None
        self._mean_features = None
        self.inPolygon = None
        self.outPolygonList = None

    def mean_features(self):
        if self._mean_features is None:
            self._mean_features = torch.mean(torch.stack(self.features), axis=0)
        return self._mean_features

    def add(self, frames, bbox, feature, score):
        """
        Add new frame to tracklet
            + frames: frame id
            + bbox: [x1, y1, w, h]
        """
        self.frames.append(frames)
        self.bboxes.append(bbox)
        self.features.append(feature)
        self.scores.append(score)
        # self.labels.append(label)

        (
            self.backup_bboxes,
            self.backup_features,
            self.backup_scores,
            self.backup_frames,
        ) = (None, None, None, None)

    @property
    def name(self):
        return f"{self.scene_camera}_{self.track_id}"

    def sortFrame(self):
        self.frames, self.bboxes, self.features, self.scores = zip(
            *sorted(zip(self.frames, self.bboxes, self.features, self.scores))
        )

    def refine_tracklets(self):
        """
        Remove bboxes that are in outzone, remove its feature also
        """
        self.backup_bboxes = copy.deepcopy(self.bboxes)
        self.backup_features = copy.deepcopy(self.features)
        self.backup_scores = copy.deepcopy(self.scores)
        self.backup_frames = copy.deepcopy(self.frames)

        new_boxes, new_features, new_scores, new_frames = [], [], [], []
        for (i, bbox) in enumerate(self.bboxes):
            if self.is_outzone(bbox):
                continue
            new_boxes.append(bbox)
            new_features.append(self.features[i])
            new_scores.append(self.scores[i])
            new_frames.append(self.frames[i])
        self.bboxes = new_boxes
        self.features = new_features
        self.scores = new_scores
        self.frames = new_frames
        self._mean_features = None

    def is_outzone(self, bbox):
        """
        Check if bbox is in outzone/uncertain zone
        """
        if self.outPolygonList is None:
            return False
        for outPolygon in self.outPolygonList:
            if self.get_iou(bbox, outPolygon) > 0.5:
                return True
        return False

    def get_iou(self, bbox, pts):
        """The intersection over union (IoU) between a set of bboxes and a set of points//polygons.

        Args:
            bbox ([x1,y1,w,h]):
            pts ([List[[x1,y1]]]): Polygon coordinates

        Returns:
            _type_: _description_
        """
        x, y, w, h = bbox
        new_bbox = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        polygon1 = Polygon(new_bbox)
        polygon2 = Polygon(pts)
        intersect = polygon1.intersection(polygon2).area
        return intersect / polygon1.area

    def convert2numpy(self):
        _feat_list = []
        for feat in self.features:
            feat = feat.numpy()
            _feat_list.append(feat)
        self.features = _feat_list

    def restore(self):
        if self.backup_bboxes is None:
            return
        self.bboxes = self.backup_bboxes
        self.features = self.backup_features
        self.scores = self.backup_scores
        self.frames = self.backup_frames
        self._mean_features = None


def process_tracklet_input_v3(
    scene_camera, features_path: str, tracklet_path: str, isNormalize=True
):
    tracklets, trackers, uuids, bboxes = {}, {}, {}, {}

    print(features_path)
    features = pickle.load(open(features_path, "rb"))
    # Load tracker results file
    f = open(tracklet_path, "r")
    f = f.readlines()
    for line in f:
        line = line.split(",")
        frames = int(line[0])
        track_id = int(line[1])
        x1, y1, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
        prob = float(line[6])
        uuid = (
            scene_camera
            + "_"
            + line[0].zfill(6)
            + "_"
            + line[2]
            + "_"
            + line[3]
            + "_"
            + line[4]
            + "_"
            + line[5]
        )
        uuid = uuid.replace(" ", "")
        if track_id not in trackers:
            trackers[track_id] = []
            uuids[track_id] = []
            bboxes[track_id] = []
            tracklets[track_id] = Tracklet(track_id, scene_camera)
        if isNormalize:
            feat = F.normalize(features[uuid], dim=0)
        else:
            feat = features[uuid]
        tracklets[track_id].add(frames, [x1, y1, w, h], feat, prob)
    for track_id in tracklets:
        tracklets[track_id].sortFrame()
    return tracklets


def process_tracklet_input_enssemble(
    scene_camera: str, tracklet_path: str, ensemble_feature_path
):
    tracklets = {}
    trackers = {}
    uuids = {}
    bboxes = {}
    features = {}
    
    for feat_path in ensemble_feature_path:
        print("Loading feat:", feat_path)
        feat_path = feat_path + "/" + scene_camera + "_feature.pkl"
        features[feat_path] = pickle.load(open(feat_path, "rb"))

    print("[Loaded feat]")
    # Load tracker results file
    f = open(tracklet_path, "r")
    f = f.readlines()
    for line in f:
        line = line.split(",")
        frames = int(line[0])
        track_id = int(line[1])
        x1, y1, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
        prob = float(line[6])
        uuid = (
            scene_camera
            + "_"
            + line[0].zfill(6)
            + "_"
            + line[2]
            + "_"
            + line[3]
            + "_"
            + line[4]
            + "_"
            + line[5]
        )

        uuid = uuid.replace(" ", "")
        if track_id not in trackers:
            trackers[track_id] = []
            uuids[track_id] = []
            bboxes[track_id] = []
            tracklets[track_id] = Tracklet(track_id, scene_camera)
        feat_list = []
        for feat_path in features:
            feat = F.normalize(features[feat_path][uuid], dim=0)
            feat_list.append(feat)
            # print(feat.shape)
        feat = torch.cat(feat_list, 0)
        feat = F.normalize(feat, dim=0)

        tracklets[track_id].add(frames, [x1, y1, w, h], feat, prob)
    for track_id in tracklets:
        tracklets[track_id].sortFrame()
    return tracklets


def process_tracklet_input_s001(
    scene_camera,
    tracklet_path: str,
    ensemble_feature_path,
):
    tracklets = {}
    trackers = {}
    uuids = {}
    bboxes = {}
    features = {}
    
    for feat_path in ensemble_feature_path:
        feat_path = feat_path + "/" + scene_camera + "_feature.pkl"
        print("[Loading]", feat_path)
        features[feat_path] = pickle.load(open(feat_path, "rb"))
        print("[Loaded]", feat_path)

    # Load tracker results file
    f = open(tracklet_path, "r")
    f = f.readlines()
    for line in f:
        line = line.split(",")
        frames = int(line[0])
        track_id = int(line[1])
        x1, y1, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
        prob = float(line[6])
        uuid = (
            scene_camera
            + "_"
            + line[0].zfill(6)
            + "_"
            + line[2]
            + "_"
            + line[3]
            + "_"
            + line[4]
            + "_"
            + line[5]
        )

        uuid = uuid.replace(" ", "")
        if track_id not in trackers:
            trackers[track_id] = []
            uuids[track_id] = []
            bboxes[track_id] = []
            tracklets[track_id] = Tracklet(track_id, scene_camera)
        feat_list = []
        for feat_path in features:
            feat = F.normalize(features[feat_path][uuid], dim=0)
            feat_list.append(feat)

        feat = torch.cat(feat_list, 0)
        feat = F.normalize(feat, dim=0)

        tracklets[track_id].add(frames, [x1, y1, w, h], feat, prob)
    for track_id in tracklets:
        tracklets[track_id].sortFrame()
    return tracklets