import os
from processor.trainers.do_train_vehicle_reid import Vehicle_Reid
from metrics.r1_mAP import R1_mAP, R1_mAP_reranking
from model import *
from dataset_processing import make_inference_data_loader, make_data_loader
from src.utils.opt import *
import src.utils.utils as util
from layers import *
from solver import make_optimizer, make_scheduler

import torch
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor


############################# Main ##################################################################################
def inference(cfg):
    # print(cfg["MODEL"])
    checkpoint_path = None if cfg["CHECKPOINT_PATH"] == "" else cfg["CHECKPOINT_PATH"]

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)  # num_classes = 726

    # loss_fn = make_loss(cfg, num_classes)
    # optimizer = make_optimizer(cfg["SOLVER"], model)
    # scheduler = make_scheduler(cfg["SOLVER"], optimizer)
    # evaluator = R1_mAP(num_query=num_query, max_rank=50, feat_norm="yes")
    val_loader = make_inference_data_loader(cfg)

    print(checkpoint_path)
    vehicle_reid = Vehicle_Reid.load_from_checkpoint(
        checkpoint_path,
        model=model,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        evaluator=None,
        cfg=cfg,
    )

    trainer = Trainer(
        accelerator=cfg["ACCELERATOR"],
        devices=[1],
    )
    
    OUTPUT_FEAT_DIR = f'output/{cfg["MODEL"]["NAME"]}_feat'
    if os.path.exists(OUTPUT_FEAT_DIR) == False:
        os.makedirs(OUTPUT_FEAT_DIR)
    print("FEATURE Saved at:", OUTPUT_FEAT_DIR)
    for scene_cam in val_loader:
        if 'S001' not in scene_cam: continue
        print("[Process]", scene_cam)
        features = dict()
        predictions = trainer.predict(vehicle_reid, val_loader[scene_cam])
        for prediction in predictions:
            features.update(prediction)

        with open(f"{OUTPUT_FEAT_DIR}/{scene_cam}_feature.pkl", "wb") as fid:
            pickle.dump(features, fid)
        
        print("[Extract done]", scene_cam)
if __name__ == "__main__":
    #############################  Config processing  ##################################################################################
    cfg = util.load_defaults(["configs/dataset_AIC.yaml", "configs/baseline.yaml"])

    # set seed
    util.set_seed(cfg["SOLVER"]["SEED"])
    #############################################################

    inference(cfg)
