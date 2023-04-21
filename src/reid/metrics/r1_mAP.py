# encoding: utf-8

"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# from ignite.metrics import Metric

# from .re_ranking import re_ranking
from processor.post_process.re_rank import re_ranking
from .metric import Metric_Interface
from scipy.spatial.distance import cdist
import pickle


def eval_func(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, isVisualize=False
):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        keep = ~(g_pids[order] == q_pid) | ~(g_camids[order] == q_camid)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if np.nonzero(orig_cmc)[0].size == 0:
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    if isVisualize:
        filter = (q_pids[:, np.newaxis] == g_pids).astype(bool)
        pos_score = distmat[filter]
        neg_score = distmat[np.invert(filter)]
        pos_scores_hist = pos_score.flatten()
        neg_scores_hist = neg_score.flatten()
        plt.hist(1 - pos_scores_hist, bins=100, alpha=0.5, label="positive pair")
        plt.hist(1 - neg_scores_hist, bins=100, alpha=0.5, label="negative pair")
        plt.legend(loc="upper right")
        plt.savefig("Distribution.png")
        plt.show()

    return all_cmc, mAP


class R1_mAP(Metric_Interface):
    def __init__(
        self,
        num_query,
        max_rank=50,
        feat_norm="yes",
        is_cross_cam=True,
        isVisualize=False,
    ):
        """1xN rank metric
        is_cross_cam: bool
            Controls cross-camera evaluation. If False, the camera of the query is assumed to be 1, and the camera of the gallery is assumed to be 0.
        """
        super(R1_mAP, self).__init__()

        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

        self.reset()
        self.isVisualize = isVisualize
        self.is_cross_cam = is_cross_cam

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == "yes":
            print("Normalize test features")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[: self.num_query]
        q_pids = np.asarray(self.pids[: self.num_query])
        q_camids = np.asarray(self.camids[: self.num_query])

        # gallery
        gf = feats[self.num_query :]
        g_pids = np.asarray(self.pids[self.num_query :])
        g_camids = np.asarray(self.camids[self.num_query :])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = (
        #     torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        #     + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # )
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        distmat = calculate_similarity(qf, gf, useCuda=True)
        if self.is_cross_cam == False:
            """
            'is_cross_cam' == True:
                + By default, the evaluation will remove the gallery items
            that have the same ID and camera ID as the query,
            but only .
            'is_cross_cam' == False:
                the evaluation will not remove these items."
            """
            g_camids = np.ones_like(g_camids)
            q_camids = np.zeros_like(q_camids)

        cmc, mAP = eval_func(
            distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank, self.isVisualize
        )

        return cmc, mAP


class R1_mAP_extend(R1_mAP):
    """
    NxN rank metric
    """

    def __init__(self, num_query, max_rank=50, feat_norm="yes", is_cross_cam=True):
        print("Using R1_mAP_extend")
        super(R1_mAP_extend, self).__init__(
            num_query, max_rank, feat_norm, is_cross_cam
        )

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == "yes":
            print("Normalize test features")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats
        q_pids = np.asarray(self.pids)
        q_camids = np.asarray(self.camids)
        # gallery
        gf = feats
        g_pids = np.asarray(self.pids)
        g_camids = np.asarray(self.camids)

        # distmat = cdist(qf, gf, metric="cosine")
        distmat = calculate_similarity(qf, gf, useCuda=True)
        del qf, gf
        print("Dismat shape", distmat.shape)
        if self.is_cross_cam == False:
            """
            'is_cross_cam' == True:
                + By default, the evaluation will remove the gallery items
            that have the same ID and camera ID as the query,
            but only .
            'is_cross_cam' == False:
                the evaluation will not remove these items."
            """
            g_camids = np.ones_like(g_camids)
            q_camids = np.zeros_like(q_camids)
        cmc, mAP = eval_func(
            distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank, self.isVisualize
        )

        return cmc, mAP


class R1_mAP_reranking(Metric_Interface):
    def __init__(self, num_query, max_rank=50, feat_norm="yes", is_cross_cam=True):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()
        self.is_cross_cam = is_cross_cam
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == "yes":
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[: self.num_query]
        q_pids = np.asarray(self.pids[: self.num_query])
        q_camids = np.asarray(self.camids[: self.num_query])
        # gallery
        gf = feats[self.num_query :]
        g_pids = np.asarray(self.pids[self.num_query :])
        g_camids = np.asarray(self.camids[self.num_query :])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        if self.is_cross_cam == False:
            """
            'is_cross_cam' == True:
                + By default, the evaluation will remove the gallery items
            that have the same ID and camera ID as the query,
            but only .
            'is_cross_cam' == False:
                the evaluation will not remove these items."
            """
            g_camids = np.ones_like(g_camids)
            q_camids = np.zeros_like(q_camids)
        

        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=10, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank)

        return cmc, mAP

def calculate_similarity(qf, gf, metric = 'euclidean', useCuda = False):

    if useCuda and metric == 'euclidean' and torch.cuda.is_available():
        print("\t->Calculating similarity using CUDA and Euclidean metric.")
        qf = qf.to("cuda:0")
        gf = gf.to("cuda:0")
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        return distmat
    distmat = cdist(qf, gf, metric=metric)
    return distmat