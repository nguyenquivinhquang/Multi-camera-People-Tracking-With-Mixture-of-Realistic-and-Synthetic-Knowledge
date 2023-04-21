from .build_baseline import Baseline
from .build_transformer import build_transformer, build_transformer_local
import os


def build_model(cfg, num_classes, camera_num=0, view_num=0):
    print(cfg["MODEL"])
    cfg["MODEL"]["PRETRAIN_PATH"] = os.path.join(
        cfg["PRETRAIN_ROOT"], cfg["MODEL"]["PRETRAIN_NAME"]
    )
    if cfg["MODEL"]["NAME"] == "transformer":
        model = build_transformer(num_classes, camera_num, view_num, cfg=cfg)
    elif cfg["MODEL"]["NAME"] == "transformer_local":
        model = build_transformer_local(num_classes, camera_num, view_num, cfg=cfg, rearrange=cfg["MODEL"]["RE_ARRANGE"])
    else:
        model = Baseline(
            num_classes,
            cfg["MODEL"]["LAST_STRIDE"],
            cfg["MODEL"]["PRETRAIN_PATH"],
            cfg["MODEL"]["NECK"],
            cfg["TEST"]["NECK_FEAT"],
            cfg["MODEL"]["NAME"],
            cfg["MODEL"]["PRETRAIN_CHOICE"],
            cfg=cfg
        )
    return model
