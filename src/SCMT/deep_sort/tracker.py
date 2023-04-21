# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import math
import numpy as np
import cv2
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from deep_sort.kalman_filter import KalmanFilter
from opts import opt
from .detection import Detection
from scipy.spatial import distance
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, high_score=0.6, cam_name=None, mask=None, image_filenames=None):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.tracks = []
        self._next_id = 1

        self.tracks_all = []
        self.cam_name = cam_name
        self.high_score = high_score
        self.mask = mask
        self.image_filenames = image_filenames
        if cam_name == 'c041':
            self.high_score = 0.5

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # self.tracks = [t for t in self.tracks if t.to_tlwh()[2] > 0 and t.to_tlwh()[3] > 0]
        # self.tracks_all += [t for t in self.tracks if ((t.to_tlwh()[2] <= 0 or t.to_tlwh()[3] <= 0) and t.hits > t._n_init)]
        # Run matching cascade.
        # detections = self._huge_object_nms(detections)
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            # if self.tracks[track_idx].is_still():
            #     self.tracks[track_idx].time_since_update -= 1
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks_all += [t for t in self.tracks if (t.is_deleted() and (t.hits > t._n_init))]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        self._check_still()

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            if not opt.EMA:
                track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def postprocess(self):
        self.tracks_all += [t for t in self.tracks if t.is_confirmed()]
        for track in self.tracks_all:
            track.storage.sort(key=lambda d : d.frame_idx)
        self._update_tracklet_info()
        self.tracks_all = [t for t in self.tracks_all if not t.delete]
        self._delete_tracklets()
        self.tracks_all = [t for t in self.tracks_all if not t.delete]
        self._merge_similar()
        self.tracks_all = [t for t in self.tracks_all if not t.delete]
        self._update_tracklet_info()
        self._linear_interpolation()
        self._gaussian_smooth()
        self._update_tracklet_info()
        self._merge_overlap()
        self.tracks_all = [t for t in self.tracks_all if not t.delete]
        self._update_tracklet_info()
        self._linear_interpolation()
        self._gaussian_smooth()
        self._update_tracklet_info()
        
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices, enable_motion_shape=False, color_gate=True):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            if enable_motion_shape:
                for i in range(len(cost_matrix)):
                    for j in range(len(cost_matrix[0])):
                        cost_matrix[i, j] += self._motion_shape_distance(tracks[track_indices[i]].to_tlwh(), dets[detection_indices[j]].tlwh)
            gating_threshold = 50.0 if self.cam_name not in ['c043', 'c044'] else kalman_filter.chi2inv95[4]
            add_identity = True if self.cam_name not in ['c043', 'c044'] else False
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices, detection_indices, 
                gating_threshold=gating_threshold, add_identity=add_identity)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks, detection set into high score and low score detections.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        high_score_det_indices = [
            i for i, d in enumerate(detections) if d.confidence >= self.high_score]
        low_score_det_indices = [
            i for i, d in enumerate(detections) if d.confidence < self.high_score]

        # Associate confirmed tracks with high score detections using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks, high_score_det_indices)

        second_track_candidates = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        # Associate remaining tracks with low score detections using IOU.
        match_b_thresh = 0.4
        matches_b, unmatched_tracks_b, unmatched_detections_b = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, match_b_thresh, self.tracks,
                detections, second_track_candidates, low_score_det_indices)

        third_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_b if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_b = [
            k for k in unmatched_tracks_b if
            self.tracks[k].time_since_update != 1]

        # Associate remaining tracks and unconfirmed tracks with remaining high score detections using IOU.
        matches_c, unmatched_tracks_c, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, third_track_candidates, unmatched_detections_a)

        matches = matches_a + matches_b + matches_c
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b + unmatched_tracks_c))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age,
            detection))
        self._next_id += 1

    def _check_still(self):
        if self.cam_name == 'c041':
            still_tracks_count = 0
            for track in self.tracks:
                if track.is_confirmed() and track.time_since_update <= 1 and track.is_still() and (560 < track.to_tlwh()[0] < 900) and track.to_tlwh()[1] < 175:
                    still_tracks_count += 1
            if still_tracks_count > 5:
                for track in self.tracks:
                    if track.is_confirmed() and track.time_since_update == 0:
                        if track.is_still(iou_thresh=0.8) and track.storage[-1].confidence < track.storage[-2].confidence:
                            track.storage[-1].tlwh = track.storage[-2].tlwh
                            track.storage[-1].confidence = track.storage[-2].confidence
                            track.storage[-1].feature = track.storage[-2].feature
                            track.storage[-1].color_hist = track.storage[-2].color_hist
                            track.mean[:4] = track.storage[-2].to_xyah()

    def _update_tracklet_info(self):
        for track in self.tracks_all:
            track.start_frame = track.storage[0].frame_idx
            track.end_frame = track.storage[-1].frame_idx

    def _merge_overlap(self):
        for t1 in self.tracks_all:
            t1_len = len(t1.storage)
            if t1.delete or t1_len < 3:
                continue
            t1_det1 = t1.storage[0]
            t1_det2 = t1.storage[1]
            t1_det3 = t1.storage[2]
            t1_start_frame = t1.storage[0].frame_idx
            t1_end_frame = t1.storage[-1].frame_idx

            for t2 in self.tracks_all:
                t2_len = len(t2.storage)
                t2_start_frame = t2.storage[0].frame_idx
                t2_end_frame = t2.storage[-1].frame_idx
                if t1 == t2 or t2.delete or t2_len < 3:
                    continue
                if t1_start_frame < t2_start_frame or t1_start_frame > t2_end_frame - 2:
                    continue
                if t1_end_frame > t2_end_frame:
                    t1_det4 = None
                    for det in t1.storage:
                        if det.frame_idx == t2_end_frame:
                            t1_det4 = det
                    t2_det4 = t2.storage[-1]
                    if t1_det4 is None or self._det_iou(t1_det4, t2_det4) <= 0.8:
                        continue
                else:
                    t1_det4 = t1.storage[-1]
                    t2_det4 = None
                    for det in t2.storage:
                        if det.frame_idx == t1_end_frame:
                            t2_det4 = det
                    if t2_det4 is None or self._det_iou(t1_det4, t2_det4) <= 0.8:
                        continue
                k = t1_start_frame - t2_start_frame
                t2_det1 = None
                for det in t2.storage:
                    if det.frame_idx == t1_start_frame:
                        t2_det1 = det
                if t2_det1 is not None and self._det_iou(t1_det1, t2_det1) > 0.8:
                    t1.delete = True
                    for p in range(t1_len):
                        if t1.storage[p].frame_idx > t2.storage[-1].frame_idx:
                            t2.storage += t1.storage[p:]
                            break
                    break

    def _merge_similar(self):
        if self.cam_name in ['c041']:
            mid_end_track = []
            mid_start_track = []
            mid_end_huge_track = []
            huge_track = []
            for track in self.tracks_all:
                if 450 < track.storage[-1].tlwh[0] < 900 and track.storage[-1].tlwh[1] < 175 and len(track.storage) > 5:
                    mid_end_track.append(track)
                if 350 < track.storage[0].tlwh[0] < 900 and track.storage[0].tlwh[1] < 175 and track.storage[0].frame_idx > 30 and len(track.storage) > 5:
                    mid_start_track.append(track)
                if 260 < track.storage[-1].tlwh[0] < 330:
                    for det in track.storage:
                        if det.tlwh[2] * det.tlwh[3] > 140000:
                            mid_end_huge_track.append(track)
                            break
                for det in track.storage:
                    if det.tlwh[2] * det.tlwh[3] > 260000:
                        huge_track.append(track)
                        break
            for track2 in mid_end_huge_track:
                track2.delete = True
                huge_track[0].storage += track2.storage
                huge_track[0].storage.sort(key=lambda d : d.frame_idx)

            for track1 in mid_end_track:
                id_2_match = []
                for i, track2 in enumerate(mid_start_track):
                    if not track2.delete:
                        time_diff = track2.storage[0].frame_idx - track1.storage[-1].frame_idx
                        if -2 < time_diff < 150:
                            id_2_match.append((distance.cosine(track1.features[0], track2.features[0]), distance.cosine(track1.color_hists[0], track2.color_hists[0]), i, track2.track_id))
                if id_2_match:
                    id_2_match.sort()
                    if id_2_match[0][0] < 0.4:
                        idx = id_2_match[0][2]
                        mid_start_track[idx].delete = True
                        track1.storage += mid_start_track[idx].storage

        if self.cam_name in ['c042']:
            mid_end_track = []
            mid_start_track = []
            for track in self.tracks_all:
                if 510 < track.storage[-1].tlwh[0] < 900 and 70 < track.storage[-1].tlwh[1] < 175 and len(track.storage) > 5:
                    mid_end_track.append(track)
                if 450 < track.storage[0].tlwh[0] < 900 and 70 < track.storage[0].tlwh[1] < 175 and track.storage[0].frame_idx > 30 and len(track.storage) > 5 and track not in mid_end_track:
                    mid_start_track.append(track)
                
            for track1 in mid_end_track:
                id_2_match = []
                for i, track2 in enumerate(mid_start_track):
                    if not track2.delete:
                        time_diff = track2.storage[0].frame_idx - track1.storage[-1].frame_idx
                        x_diff = track2.storage[0].tlwh[0] - track1.storage[-1].tlwh[0]
                        y_diff = track2.storage[0].tlwh[1] - track1.storage[-1].tlwh[1]
                        if (0 < time_diff < 110) and x_diff < 2 and y_diff > -2:
                            id_2_match.append((distance.cosine(track1.features[0], track2.features[0]) 
                            + distance.cosine(track1.color_hists[0], track2.color_hists[0]), i, track2.track_id))
                if id_2_match:
                    id_2_match.sort()
                    if id_2_match[0][0] < 0.4:
                        idx = id_2_match[0][1]
                        mid_start_track[idx].delete = True
                        track1.storage += mid_start_track[idx].storage

        if self.cam_name in ['c043']:
            mid_end_track = []
            mid_start_track = []
            mid_end_huge_track = []
            huge_track = []
            for track in self.tracks_all:
                if 510 < track.storage[-1].tlwh[0] < 1100 and 70 < track.storage[-1].tlwh[1] < 175 and len(track.storage) > 5:
                    mid_end_track.append(track)
                if 450 < track.storage[0].tlwh[0] < 1100 and 70 < track.storage[0].tlwh[1] < 175 and track.storage[0].frame_idx > 30 and len(track.storage) > 5 and track not in mid_end_track:
                    mid_start_track.append(track)
                if track.storage[-1].tlwh[0] < 640:
                    for det in track.storage:
                        if det.tlwh[2] * det.tlwh[3] > 90000:
                            mid_end_huge_track.append(track)
                            break
                else:
                    for det in track.storage:
                        if det.tlwh[2] * det.tlwh[3] > 280000:
                            huge_track.append(track)
                            break
            for track2 in mid_end_huge_track:
                track2.delete = True
                huge_track[0].storage += track2.storage
                huge_track[0].storage.sort(key=lambda d : d.frame_idx)

            for track1 in mid_end_track:
                id_2_match = []
                for i, track2 in enumerate(mid_start_track):
                    if not track2.delete:
                        time_diff = track2.storage[0].frame_idx - track1.storage[-1].frame_idx
                        x_diff = track2.storage[0].tlwh[0] - track1.storage[-1].tlwh[0]
                        y_diff = track2.storage[0].tlwh[1] - track1.storage[-1].tlwh[1]
                        if (-3 < time_diff < 75):
                            id_2_match.append((distance.cosine(track1.features[0], track2.features[0]), distance.cosine(track1.color_hists[0], track2.color_hists[0]), i, track2.track_id))
                if id_2_match:
                    id_2_match.sort()
                    if id_2_match[0][0] < 0.6:
                        idx = id_2_match[0][2]
                        mid_start_track[idx].delete = True
                        track1.storage += mid_start_track[idx].storage

        if self.cam_name in ['c044']:
            mid_end_track = []
            mid_start_track = []
            mid_end_huge_track = []
            huge_track = []
            for track in self.tracks_all:
                if track.storage[-1].tlwh[0] < 400 and 180 < track.storage[-1].tlwh[1] < 400 and len(track.storage) > 5:
                    mid_end_track.append(track)
                if 110 < track.storage[0].tlwh[0] < 400 and 180 < track.storage[0].tlwh[1] < 400 and track.storage[0].frame_idx > 30 and len(track.storage) > 5 and track not in mid_end_track:
                    mid_start_track.append(track)
                if 550 < track.storage[-1].tlwh[0] < 650 and track.storage[-1].tlwh[1] < 300:
                    for det in track.storage:
                        if det.tlwh[2] * det.tlwh[3] > 13800:
                            mid_end_huge_track.append(track)
                            break
                elif track.storage[-1].tlwh[1] < 300:
                    for det in track.storage:
                        if det.tlwh[2] * det.tlwh[3] > 80000:
                            huge_track.append(track)
                            break
            for track2 in mid_end_huge_track:
                track2.delete = True
                huge_track[0].storage += track2.storage
                huge_track[0].storage.sort(key=lambda d : d.frame_idx)

            for track1 in mid_end_track:
                id_2_match = []
                for i, track2 in enumerate(mid_start_track):
                    if not track2.delete:
                        time_diff = track2.storage[0].frame_idx - track1.storage[-1].frame_idx
                        x_diff = track2.storage[0].tlwh[0] - track1.storage[-1].tlwh[0]
                        y_diff = track2.storage[0].tlwh[1] - track1.storage[-1].tlwh[1]
                        if (-2 < time_diff < 10):
                            id_2_match.append((distance.cosine(track1.features[0], track2.features[0]), distance.cosine(track1.color_hists[0], track2.color_hists[0]), i, track2.track_id))
                if id_2_match:
                    id_2_match.sort()
                    if id_2_match[0][0] < 0.45:
                        idx = id_2_match[0][2]
                        mid_start_track[idx].delete = True
                        track1.storage += mid_start_track[idx].storage

        if self.cam_name in ['c046']:
            mid_end_track = []
            mid_start_track = []
            for track in self.tracks_all:
                if 650 < track.storage[-1].tlwh[0] < 1200 and 70 < track.storage[-1].tlwh[1] < 170 and len(track.storage) > 5:
                    mid_end_track.append(track)
                if 650 < track.storage[0].tlwh[0] < 1000 and 70 < track.storage[0].tlwh[1] < 170 and track.storage[0].frame_idx > 30 and len(track.storage) > 5 and track not in mid_end_track:
                    mid_start_track.append(track)

            for track1 in mid_end_track:
                id_2_match = []
                for i, track2 in enumerate(mid_start_track):
                    if not track2.delete:
                        time_diff = track2.storage[0].frame_idx - track1.storage[-1].frame_idx
                        x_diff = track2.storage[0].tlwh[0] - track1.storage[-1].tlwh[0]
                        y_diff = track2.storage[0].tlwh[1] - track1.storage[-1].tlwh[1]
                        if (-2 < time_diff < 250 and x_diff < 3):
                            id_2_match.append((distance.cosine(track1.features[0], track2.features[0]), distance.cosine(track1.color_hists[0], track2.color_hists[0]), i, track2.track_id))
                if id_2_match:
                    id_2_match.sort()
                    if id_2_match[0][0] < 0.4:
                        idx = id_2_match[0][2]
                        mid_start_track[idx].delete = True
                        track1.storage += mid_start_track[idx].storage

    def _delete_tracklets(self, delete_less_than = 5, tracklet_trust_threshold = 0.5):
        for track in self.tracks_all:
            length = len(track.storage)
            if length <= delete_less_than:
                track.delete = True
            else:
                average_width = (track.storage[0].tlwh[2] + track.storage[length - 1].tlwh[2]) / 4
                move_distance = np.sqrt((track.storage[length - 1].tlwh[0] - track.storage[0].tlwh[0]) \
                    * (track.storage[length - 1].tlwh[0] - track.storage[0].tlwh[0]) + \
                    (track.storage[length - 1].tlwh[1] - track.storage[0].tlwh[1]) * \
                    (track.storage[length - 1].tlwh[1] - track.storage[0].tlwh[1]))
                if self.cam_name in ['c043', 'c045'] and move_distance < average_width:
                    track.delete = True
            if not track.delete and self.cam_name == 'c042':
                for det in track.storage:
                    if det.tlwh[0] < 905:
                        break
                else:
                    track.delete = True

            if not track.delete and self.cam_name == 'c044':
                for det in track.storage:
                    if det.tlwh[0] < 910 or det.tlwh[1] < 640:
                        break
                else:
                    track.delete = True

            if not track.delete and self.cam_name == 'c045':
                for det in track.storage:
                    if det.tlwh[0] < 1115:
                        break
                else:
                    track.delete = True
    
    def _linear_interpolation(self, interval=20):
        '''线性插值'''
        for track in self.tracks_all:
            f_pre = track.storage[0].frame_idx            
            tlwh_pre = track.storage[0].tlwh
            interpolation_storage = []
            for det in track.storage:
                f_curr = det.frame_idx
                tlwh_curr = det.tlwh
                if f_pre + 1 < f_curr < f_pre + interval:
                    for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                        step = (tlwh_curr - tlwh_pre) / (f_curr - f_pre) * i
                        tlwh_new = tlwh_pre + step
                        img = cv2.imread(self.image_filenames[f])
                        color_hist = []
                        H, W, _ = img.shape
                        x1 = int(tlwh_new[0])
                        y1 = int(tlwh_new[1])
                        w = int(tlwh_new[2])
                        h = int(tlwh_new[3])
                        x2 = x1 + w
                        y2 = y1 + h
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(W - 1, x2)
                        y2 = min(H - 1, y2)
                        for i in range(3):
                            color_hist += cv2.calcHist([img[y1 : y2, x1 : x2]], [i], None, [8], [0.0,255.0]).T.tolist()[0]
                        color_hist = np.array(color_hist)
                        norm = np.linalg.norm(color_hist)
                        color_hist /= norm
                        interpolation_storage.append(Detection(tlwh_new, 0.3, None, f, color_hist))
                f_pre = f_curr
                tlwh_pre = tlwh_curr
            track.storage += interpolation_storage
            track.storage.sort(key=lambda d : d.frame_idx)

    def _gaussian_smooth(self, tau=10):
        for track in self.tracks_all:
            len_scale = np.clip(tau * np.log(tau ** 3 / len(track.storage)), tau ** -1, tau ** 2)
            gpr = GPR(RBF(len_scale, 'fixed'))
            ftlwh = []
            for det in track.storage:
                ftlwh.append([det.frame_idx, det.tlwh[0], det.tlwh[1], det.tlwh[2], det.tlwh[3]])
            ftlwh = np.array(ftlwh)
            t = ftlwh[:, 0].reshape(-1, 1)
            x = ftlwh[:, 1].reshape(-1, 1)
            y = ftlwh[:, 2].reshape(-1, 1)
            w = ftlwh[:, 3].reshape(-1, 1)
            h = ftlwh[:, 4].reshape(-1, 1)
            gpr.fit(t, x)
            xx = gpr.predict(t)[:, 0]
            gpr.fit(t, y)
            yy = gpr.predict(t)[:, 0]
            gpr.fit(t, w)
            ww = gpr.predict(t)[:, 0]
            gpr.fit(t, h)
            hh = gpr.predict(t)[:, 0]
            for i in range(len(track.storage)):
                track.storage[i].tlwh = np.array([xx[i], yy[i], ww[i], hh[i]])

    def _det_iou(self, det1, det2):
        ltx1 = det1.tlwh[0]
        lty1 = det1.tlwh[1]
        rdx1 = det1.tlwh[0] + det1.tlwh[2]
        rdy1 = det1.tlwh[1] + det1.tlwh[3]
        ltx2 = det2.tlwh[0]
        lty2 = det2.tlwh[1]
        rdx2 = det2.tlwh[0] + det2.tlwh[2]
        rdy2 = det2.tlwh[1] + det2.tlwh[3]

        W = min(rdx1, rdx2) - max(ltx1, ltx2)
        H = min(rdy1, rdy2) - max(lty1, lty2)
        cross = W * H

        if(W <= 0 or H <= 0):
            return 0

        SA = (rdx1 - ltx1) * (rdy1 - lty1)
        SB = (rdx2 - ltx2) * (rdy2 - lty2)
        if min(SA, SB) <= 0:
            return 0
        return cross / min(SA, SB)

    # not used
    def _motion_shape_distance(self, track_box, det_box):
        _weight_motion = -0.5
        _weight_shape = -1.5
        if track_box[2] <= 0 or track_box[3] <= 0:
            return 1e+5
        # motion cost
        factor_motion_1 = math.pow((track_box[0] - det_box[0]) / det_box[2], 2)
        factor_motion_2 = math.pow((track_box[1] - det_box[1]) / det_box[3], 2)
        motion_cost = math.exp(_weight_motion * (factor_motion_1 + factor_motion_2))
        # shape cost
        factor_shape_1 = abs(track_box[3] - det_box[3]) / (track_box[3] + det_box[3])
        factor_shape_2 = abs(track_box[2] - det_box[2]) / (track_box[2] + det_box[2])
        shape_cost = math.exp(_weight_shape * (factor_shape_1 + factor_shape_2))

        return (1 - motion_cost) + (1 - shape_cost)
