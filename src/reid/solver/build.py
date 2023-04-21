#
# Created on Tue Nov 15 2022 by Nguyen Qui Vinh Quang
#
# @licensed: Computer Vision & Image Processing Lab
# @project: VehicleReid
#

import torch


def make_optimizer(cfg, model):
    # @todo: change make optimizer to cfg = cfg['SOLVER'] when call make_optimizer

    cfg = cfg["SOLVER"]
    ### Remove/update the line above later

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg["BASE_LR"]
        weight_decay = cfg["WEIGHT_DECAY"]
        if "bias" in key:
            lr = cfg["BASE_LR"] * cfg["BIAS_LR_FACTOR"]
            weight_decay = cfg["WEIGHT_DECAY_BIAS"]
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg["OPTIMIZER_NAME"] == "SGD":
        optimizer = getattr(torch.optim, cfg["OPTIMIZER_NAME"])(
            params, momentum=cfg["MOMENTUM"]
        )
    else:
        optimizer = getattr(torch.optim, cfg["OPTIMIZER_NAME"])(params)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    # @todo: change make optimizer to cfg = cfg['SOLVER'] when call make_optimizer

    cfg = cfg["SOLVER"]

    ### Remove/update the line above later
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg["SOLVER"]["BASE_LR"]
        weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
        if "bias" in key:
            lr = cfg["SOLVER"]["BASE_LR"] * cfg["SOLVER"]["BIAS_LR_FACTOR"]
            weight_decay = cfg["SOLVER"]["WEIGHT_DECAY_BIAS"]
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg["SOLVER"]["OPTIMIZER_NAME"] == "SGD":
        optimizer = getattr(torch.optim, cfg["SOLVER"]["OPTIMIZER_NAME"])(
            params, momentum=cfg["SOLVER"]["MOMENTUM"]
        )
    else:
        optimizer = getattr(torch.optim, cfg["SOLVER"]["OPTIMIZER_NAME"])(params)
    optimizer_center = torch.optim.SGD(
        center_criterion.parameters(), lr=cfg["SOLVER"]["CENTER_LR"]
    )
    return optimizer, optimizer_center
