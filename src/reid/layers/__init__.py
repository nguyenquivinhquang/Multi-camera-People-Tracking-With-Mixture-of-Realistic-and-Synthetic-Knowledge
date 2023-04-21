# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss


class Vehicle_Reid_Loss(object):
    def __init__(self, cfg, num_classes):
        self.cfg = cfg
        self.triplet = TripletLoss(cfg["SOLVER"]["MARGIN"])
        self.loss_fn = self.make_loss(cfg, num_classes)

    def __call__(self, score, feat, target):
        return self.loss_fn(score, feat, target)

    def make_loss(self, cfg, num_classes):  # modified by gu
        # @todo debug
        sampler = cfg["DATALOADER"]["SAMPLER"]
        if cfg["MODEL"]["METRIC_LOSS_TYPE"] == "triplet":
            triplet = TripletLoss(cfg["SOLVER"]["MARGIN"])  # triplet loss
        else:
            print(
                "expected METRIC_LOSS_TYPE should be triplet"
                "but got {}".format(cfg["MODEL"]["METRIC_LOSS_TYPE"])
            )

        if cfg["MODEL"]["IF_LABELSMOOTH"] == "on":
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, num classes:", num_classes)

        if sampler == "softmax":

            def loss_func(score, feat, target):
                return F.cross_entropy(score, target)

        elif cfg["DATALOADER"]["SAMPLER"] == "triplet":

            def loss_func(score, feat, target):
                return triplet(feat, target)[0]

        elif cfg["DATALOADER"]["SAMPLER"] == "softmax_triplet":

            def loss_func(score, feat, target):
                if cfg["MODEL"]["METRIC_LOSS_TYPE"] == "triplet":
                    if cfg["MODEL"]["IF_LABELSMOOTH"] == "on":
                        if isinstance(score, list):
                            ID_LOSS = [xent(scor, target) for scor in score[1:]]
                            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                            ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                        else:
                            ID_LOSS = xent(score, target)
                        
                        if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                        else:
                            TRI_LOSS = triplet(feat, target)[0]
                        
                        return cfg["MODEL"]["ID_LOSS_WEIGHT"] * ID_LOSS + \
                               cfg["MODEL"]["TRIPLET_LOSS_WEIGHT"] * TRI_LOSS
                    
                    else:
                        if isinstance(score, list):
                            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                            ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                        else:
                            ID_LOSS = F.cross_entropy(score, target)

                        if isinstance(feat, list):
                                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                        else:
                                TRI_LOSS = triplet(feat, target)[0]

                        return cfg["MODEL"]["ID_LOSS_WEIGHT"] * ID_LOSS + \
                                cfg["MODEL"]["TRIPLET_LOSS_WEIGHT"] * TRI_LOSS
                else:
                    print(
                        "expected METRIC_LOSS_TYPE should be triplet"
                        "but got {}".format(cfg["MODEL"]["METRIC_LOSS_TYPE"])
                    )

        else:
            print(
                "expected sampler should be softmax, triplet or softmax_triplet, "
                "but got {}".format(cfg["DATALOADER"]["SAMPLER"])
            )
        return loss_func


def make_loss(cfg, num_classes):
    return Vehicle_Reid_Loss(cfg, num_classes)


class Vehicle_Reid_Center_Loss(object):
    def __init__(self, cfg, num_classes):
        self.loss_fn = self.make_loss_with_center(cfg, num_classes)
        return

    def make_loss_with_center(cfg, num_classes):  # modified by gu
        if cfg["MODEL"]["NAME"] == "resnet18" or cfg["MODEL"]["NAME"] == "resnet34":
            feat_dim = 512
        else:
            feat_dim = 2048

        if cfg["MODEL"]["METRIC_LOSS_TYPE"] == "center":
            center_criterion = CenterLoss(
                num_classes=num_classes, feat_dim=feat_dim, use_gpu=True
            )  # center loss

        elif cfg["MODEL"]["METRIC_LOSS_TYPE"] == "triplet_center":
            triplet = TripletLoss(cfg["SOLVER"]["MARGIN"])  # triplet loss
            center_criterion = CenterLoss(
                num_classes=num_classes, feat_dim=feat_dim, use_gpu=True
            )  # center loss

        else:
            print(
                "expected METRIC_LOSS_TYPE with center should be center, triplet_center"
                "but got {}".format(cfg["MODEL"]["METRIC_LOSS_TYPE"])
            )

        if cfg["MODEL"]["IF_LABELSMOOTH"] == "on":
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, numclasses:", num_classes)

        def loss_func(score, feat, target):
            if cfg["MODEL"]["METRIC_LOSS_TYPE"] == "center":
                if cfg["MODEL"]["IF_LABELSMOOTH"] == "on":
                    return xent(score, target) + cfg["SOLVER"][
                        "CENTER_LOSS_WEIGHT"
                    ] * center_criterion(feat, target)
                else:
                    return F.cross_entropy(score, target) + cfg["SOLVER"][
                        "CENTER_LOSS_WEIGHT"
                    ] * center_criterion(feat, target)

            elif cfg["MODEL"]["METRIC_LOSS_TYPE"] == "triplet_center":
                if cfg["MODEL"]["IF_LABELSMOOTH"] == "on":
                    return (
                        xent(score, target)
                        + triplet(feat, target)[0]
                        + cfg["SOLVER"]["CENTER_LOSS_WEIGHT"]
                        * center_criterion(feat, target)
                    )
                else:
                    return (
                        F.cross_entropy(score, target)
                        + triplet(feat, target)[0]
                        + cfg["SOLVER"]["CENTER_LOSS_WEIGHT"]
                        * center_criterion(feat, target)
                    )

            else:
                print(
                    "expected METRIC_LOSS_TYPE with center should be center, triplet_center"
                    "but got {}".format(cfg["MODEL"]["METRIC_LOSS_TYPE"])
                )

            return loss_func, center_criterion

        def __call__(self, score, feat, target):
            return self.loss_fn(score, feat, target)
