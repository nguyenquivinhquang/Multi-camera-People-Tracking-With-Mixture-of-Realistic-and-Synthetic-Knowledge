import numpy as np
import torch
import torch.nn as nn
import copy
from multiprocessing import Pool
import itertools
from scipy.spatial.distance import cdist
import pickle
import torch.nn.functional as F
from typing import Dict, List
import json
from shapely.geometry import Polygon


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
        self.frame_id = []
        self._small_object = None

    def add(self, frame_id, bbox, feature, score):
        """
        Add new frame to tracklet
            + frame_id: frame id
            + bbox: [x1, y1, w, h]
        """
        self.frame_id.append(frame_id)
        self.bboxes.append(bbox)
        self.features.append(feature)
        self.scores.append(score)
        # self.labels.append(label)

    @property
    def name(self):
        return f"{self.scene_camera}_{self.track_id}"

    @staticmethod
    def tracklet_intersec(trackletA, trackletB):
        """
        Compute intersection between two tracklets
        """
        framesA = set(trackletA.frame_id)
        framesB = set(trackletB.frame_id)
        intersec = framesA.intersection(framesB)
        return len(intersec) > 3  # 3 frames
        # return len(intersec) > 0.2 * min(len(framesA), len(framesB))

    def isOutScene(self):
        """
        Check if tracklet is out of scene
        """
        return any(
            bbox[0] < self.THRESHOLD
            or bbox[1] < self.THRESHOLD
            or bbox[2] + bbox[0] > self.IMG_WIDTH - self.THRESHOLD
            or bbox[3] + bbox[1] > self.IMG_HEIGHT - self.THRESHOLD
            for bbox in reversed(self.bboxes)
        )

    def _isSmallObject(
        self, area_threshold=400, width_threshold=50, height_threshold=50
    ):
        """
        Get area of tracklet
        """
        _num_small_objects = 0
        for box in self.bboxes:
            x, y, w, h = box
            area = w * h
            if area < area_threshold or (w) < width_threshold or (h) < height_threshold:
                _num_small_objects += 1
                if _num_small_objects > 5:
                    return True
        return _num_small_objects / len(self.bboxes) > 0.4

    def isSmallObject(
        self, area_threshold=400, width_threshold=50, height_threshold=50
    ):
        if self._small_object is None:
            self._small_object = self._isSmallObject(
                area_threshold, width_threshold, height_threshold
            )
        return self._small_object

    def unCertain(self):
        """
        Check if tracklet is uncertain
        """
        _count = 0
        for score in self.scores:
            if score < 0.80:
                _count += 1
        return _count / len(self.scores) > 0.1 or _count > 5


def calculate_distance(trackletA, trackletB, isMean=True):
    if Tracklet.tracklet_intersec(trackletA, trackletB):
        return 0

        # featuresA = torch.stack(trackletA.features, dim=0)
        # featuresB = torch.stack(trackletB.features, dim=0)
    featuresA = np.array(trackletA.features)
    featuresB = np.array(trackletB.features)
    if isMean:
        meanFeaturesA = np.mean(featuresA, axis=0)
        meanFeaturesB = np.mean(featuresB, axis=0)
        distance = 1 - cdist(
            meanFeaturesA.reshape(1, -1), meanFeaturesB.reshape(1, -1), metric="cosine"
        )
    else:
        cos_similarity_matrix = cdist(featuresA, featuresB, metric="cosine")
        distance = 1 - cos_similarity_matrix.max().item()
    return distance


def calculate_tracklet_distance_matrix(tracklets, isMean=True, metric="cosine"):
    """
    metric: cosine, euclidean
    """
    cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    distance_matrix = []
    import time

    if isMean is False:
        with Pool(processes=6) as pool:
            for i, trackletA in enumerate(tracklets.values()):
                start_time = time.time()
                distance_matrix.append(
                    [0] * i
                )  # add placeholders for previously computed distances
                distances = pool.starmap(
                    calculate_distance,
                    (
                        (trackletA, trackletB)
                        for trackletB in itertools.islice(
                            tracklets.values(), i + 1, None
                        )
                    ),
                )
                distance_matrix[i].extend(distances)
                print("[Done]", i, "/", len(tracklets))
                print("Time: ", time.time() - start_time)
                print("----------------------------------")

        # fill in missing values
        for i in range(len(distance_matrix)):
            for j in range(i):
                distance_matrix[i][j] = distance_matrix[j][i]
    else:
        # Using mean tracklet feature for calculate the distance
        featuresA = []
        featuresB = []
        uuidA = []
        uuidB = []
        for i, trackletA in enumerate(tracklets.values()):
            featuresA.append(np.mean(trackletA.features, axis=0))
            uuidA.append(trackletA.track_id)
        featuresB = copy.deepcopy(featuresA)
        uuidB = copy.deepcopy(uuidA)
        distance_matrix = cdist(featuresA, featuresB, metric=metric)

        for i in range(len(uuidA)):
            distance_matrix[i][i] = 10000000
            for j in range(i):
                trackletA = tracklets[uuidA[i]]
                trackletB = tracklets[uuidB[j]]
                # The distance between two tracklets is 1 if they have intersection in time frame
                # if Tracklet.tracklet_intersec(trackletA, trackletB):
                #     distance_matrix[i][j] = 10000000
                #     distance_matrix[j][i] = 10000000
                # if distance_matrix[i][j] > 0.6:
                #     distance_matrix[i][j] = 10000000
                #     distance_matrix[j][i] = 10000000

    # with open(f"{TRACKER_ROOT_DIR}/{scene_camera}_distance_matrix.pkl", "wb") as f:
    #     pickle.dump(distance_matrix, f)
    return distance_matrix, uuidA


def process_tracklet_input(
    scene_camera, features_path: str, tracklet_path: str, isNormalize=True
):
    """This function read a feature vector and a tracklet file and return a dictionary of tracklets
    The key of the dictionary is the track_id
    Args:
        scene_camera (str): the name of the scene and camera. Eg: "S009_C013"
        features_path (str): the path to the feature vector file
        tracklet_path (str): the path to the tracklet file
    """
    trackers = {}
    print(features_path)
    features = pickle.load(open(features_path, "rb"))
    # Load tracker results file
    f = open(tracklet_path, "r")
    f = f.readlines()
    for line in f:
        line = line.split(",")
        frame_id = int(line[0])
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
            trackers[track_id] = Tracklet(track_id, scene_camera)
        if isNormalize:
            feat = F.normalize(features[uuid], dim=0)
        else:
            feat = features[uuid]
        trackers[track_id].add(frame_id, [x1, y1, w, h], feat.numpy(), prob)

    return trackers


def process_tracklet_input_v2(
    features_path: str,
    feat_dim: int = 792,
):
    """This function read a feature vector and a tracklet file and return a dictionary of tracklets
    The key of the dictionary is the track_id
    Args:
        features_path (str): the path to the feature vector file

    """
    trackers = {}
    print(features_path)
    file = pickle.load(open(features_path, "rb"))
    tracklets = {}
    uuids = {}
    for uuid in file:
        cam, frame, trackid = uuid.split("_")
        trackid = int(trackid)
        if len(file[uuid]) < 50:
            continue  # bad detector feature
        if trackid not in tracklets:
            tracklets[trackid] = []
        if trackid not in uuids:
            uuids[trackid] = []

        feat = np.array(file[uuid])
        feat = torch.tensor(feat)
        feat = F.normalize(feat, dim=0)
        tracklets[trackid].append(feat)
        uuids[trackid].append(uuid)

    return tracklets, uuids


def process_tracklet_input_matching(
    features_path: str,
    single_matching_path: str,
    feat_dim: int = 792,
):
    """This function read a feature vector and a tracklet file and return a dictionary of tracklets
    The key of the dictionary is the track_id
    Args:
        features_path (str): the path to the feature vector file

    """
    old2new = {}

    file = json.load(open(single_matching_path, "r"))
    for new in file:
        for old in file[new]:
            old2new[int(old)] = int(new)

    print(features_path)
    file = pickle.load(open(features_path, "rb"))
    tracklets = {}
    uuids = {}
    for uuid in file:
        cam, frame, trackid = uuid.split("_")
        trackid = int(trackid)
        trackid = old2new[trackid]
        if len(file[uuid]) < 50:
            continue  # bad detector feature
        if trackid not in tracklets:
            tracklets[trackid] = []
        if trackid not in uuids:
            uuids[trackid] = []

        feat = np.array(file[uuid])
        feat = torch.tensor(feat)
        feat = F.normalize(feat, dim=0)
        tracklets[trackid].append(feat)
        uuids[trackid].append(uuid)

    return tracklets, uuids


def split_track(trackers):
    trackersA = {}
    trackersB = {}
    for track_id, track in trackers.items():
        if track.track_id < 6094:
            trackersA[track_id] = track
        else:
            trackersB[track_id] = track

    return trackersA, trackersB


def split_tracklet_len(trackers: Dict[int, Tracklet]):
    total_tracklet = len(trackers)
    ouptut_tracklets = {}
    for track_id, track in trackers.items():
        if len(track.frame_id) < 1000:
            ouptut_tracklets[track_id] = track
            continue
        _count = 0
        for i in range(0, len(track.frame_id), 1000):
            new_track = Tracklet(track_id, track.scene_camera)
            nxt_get = min(i + 1000, len(track.frame_id))

            new_track.frame_id = track.frame_id[i:nxt_get]
            new_track.bboxes = track.bboxes[i:nxt_get]
            new_track.features = track.features[i:nxt_get]
            new_track.scores = track.scores[i:nxt_get]
            new_track.scene_camera = track.scene_camera

            if _count == 0:
                new_track.track_id = track.track_id
                ouptut_tracklets[track_id] = new_track
            else:
                total_tracklet += 1
                new_track.track_id = total_tracklet
                ouptut_tracklets[total_tracklet] = new_track

            _count += 1

        print("Track_id: ", track_id, "Split: ", _count, "times")
    return ouptut_tracklets


def get_iou(bbox, pts):
    polygon1 = Polygon(bbox)
    polygon2 = Polygon(pts)
    intersect = polygon1.intersection(polygon2).area
    return intersect / polygon1.area


def load_uncertain_zone(path):
    points = json.load(open(path, "r"))

    return points


def check_inside(bboxes, points):
    # points is polygon of uncertain zone
    _count = 0
    for box in bboxes:
        x, y, w, h = box
        for point in points:
            _iou = get_iou([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], point)
            if _iou > 0.9:
                _count += 1
                break

    return _count / len(bboxes)
