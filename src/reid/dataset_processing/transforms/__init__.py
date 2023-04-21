#
# Created on Tue Nov 15 2022 by Nguyen Qui Vinh Quang
#
# @licensed: Computer Vision & Image Processing Lab
# @project: VehicleReid
#

import torchvision.transforms as T


def build_transforms(cfg, is_train=True):
    cfg = cfg["INPUT"]
    normalize_transform = T.Normalize(mean=cfg["PIXEL_MEAN"], std=cfg["PIXEL_STD"])
    if is_train:
        transform = T.Compose(
            [
                T.Resize(cfg["SIZE_TRAIN"]),
                T.RandomHorizontalFlip(p=cfg["PROB"]),
                T.Pad(cfg["PADDING"]),
                T.RandomCrop(cfg["SIZE_TRAIN"]),
                T.ToTensor(),
                normalize_transform,
                T.RandomErasing(p=cfg["RE_PROB"]),
            ]
        )
    else:
        transform = T.Compose(
            [T.Resize(cfg["SIZE_TEST"]), T.ToTensor(), normalize_transform]
        )

    return transform
