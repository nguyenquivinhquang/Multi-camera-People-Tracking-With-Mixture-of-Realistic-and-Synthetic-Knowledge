# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path
                )
            )
            pass
    return img


class ImageDataset(Dataset):
    """Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class AICDataset(Dataset):
    """AIC Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, camid, bbox = self.dataset[index]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        img = read_image(img_path)
        img = img.crop((x, y, x + w, y + h))
        if self.transform is not None:
            img = self.transform(img)
        frame_idx = img_path.split("/")[-1].split(".")[0]
        save_name = (
            camid
            + "_"
            + frame_idx
            + "_"
            + str(x)
            + "_"
            + str(y)
            + "_"
            + str(w)
            + "_"
            + str(h)
        )
        return img, save_name, camid, img_path
