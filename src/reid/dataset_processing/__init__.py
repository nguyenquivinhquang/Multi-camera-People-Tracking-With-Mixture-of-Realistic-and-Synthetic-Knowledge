from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Union

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset, AICDataset
from .datasets.aic_reid import AIC_reid, AIC_Feature_Extractor
from .samplers import (
    RandomIdentitySampler,
    RandomIdentitySampler_alignedreid,
)  # New add by gu
from .transforms import build_transforms
from tqdm import tqdm


def make_data_loader(cfg: Dict[str, Any]) -> Union[DataLoader, DataLoader, int, int]:
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg["DATALOADER"]["NUM_WORKERS"]

    aic_train = AIC_reid(
        cfg["DATASETS"]["ROOT_DIR"], cfg["DATASETS"]["TRAIN_FILE"], relabel=True
    )
    dataset_train, num_classes = aic_train.dataset, aic_train.num_persons
    dataset_query = AIC_reid(
        cfg["DATASETS"]["ROOT_DIR"], cfg["DATASETS"]["GALLERY"]
    ).dataset
    dataset_gallery = AIC_reid(
        cfg["DATASETS"]["ROOT_DIR"], cfg["DATASETS"]["QUERY"]
    ).dataset

    # num_classes = dataset_train.num_persons
    train_set = ImageDataset(dataset_train, train_transforms)
    val_set = ImageDataset(dataset_query + dataset_gallery, val_transforms)

    if cfg["DATALOADER"]["SAMPLER"] == "softmax":
        train_loader = DataLoader(
            train_set,
            batch_size=cfg["SOLVER"]["IMS_PER_BATCH"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg["TEST"]["IMS_PER_BATCH"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
        )
    else:
        print("Using RandomIdentitySampler")
        sampler = RandomIdentitySampler(
            dataset_train,
            cfg["SOLVER"]["IMS_PER_BATCH"],
            cfg["DATALOADER"]["NUM_INSTANCE"],
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg["SOLVER"]["IMS_PER_BATCH"],
            sampler=sampler,
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg['DATALOADER']['NUM_INSTANCE']),      # new add by gu
            num_workers=num_workers,
            drop_last=False,
            collate_fn=train_collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg["TEST"]["IMS_PER_BATCH"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
        )

    return train_loader, val_loader, len(dataset_query), num_classes


def make_inference_data_loader(cfg: dict)-> dict:
    """
    Returns a dictionary of data loaders for performing inference on an AIC dataset.

    Args:
        cfg (dict): A dictionary of configuration parameters.

    Returns:
        dict: A dictionary containing data loaders for each camera view in the test set.

    """
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg["DATALOADER"]["NUM_WORKERS"]

    dataset_query = AIC_Feature_Extractor(
        cfg["DATASETS"]["ROOT_DIR"] + "/test/", cfg["DATASETS"]["LABEL_FOLDER"]
    ).dataset
    
    # Create a data loader for each camera view
    AIC_dataloader = {}
    for scene_cam in dataset_query:
        val_set = AICDataset(dataset_query[scene_cam], val_transforms)
        AIC_dataloader[scene_cam] = DataLoader(
            val_set,
            batch_size=cfg["TEST"]["IMS_PER_BATCH"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
        )
    return AIC_dataloader
