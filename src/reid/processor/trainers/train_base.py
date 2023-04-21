import logging
import os
import sys
from src.utils.utils import get_device
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from metrics.metric import Metric_Interface
from torch import nn
import time

# See:
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
# https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks
# to understand more.
class Trainner_Base(LightningModule):
    def __init__(
        self,
        model,
        train_dataloader,
        total_train_epochs,
        loss_fn,
        optimizer,
        validation_per_epoch=2,
        val_dataloader=None,
        device="cpu",
    ):
        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = get_device(device)
        self.total_train_epoch = total_train_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def init_validation(self, val_dataloader, validation_per_epoch):
        self.val_loader = val_dataloader
        self.validation_per_epoch = validation_per_epoch
        return

    def do_train_per_epoch(self, epoch_idx):
        for batch_idx, batch in enumerate(self.train_loader):
            self.do_train_per_batch(batch, batch_idx)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return

    def do_train_per_batch(self, batch, batch_idx):
        raise NotImplementedError
        return

    def train(self):
        for epoch in range(self.total_train_epoch):
            self.model.train()
            self.do_train_per_epoch(epoch)

        return

    def validate(self):
        for batch_idx, batch in enumerate(self.val_loader):
            self.do_validate_per_epoch(batch, batch_idx)
        return

    def do_validate_per_epoch(self, batch, batch_idx):
        return

    def validation_epoch_end(self, outputs):
        """
        If you need to do something with all the outputs of each validation_step(),
        override the validation_epoch_end() method.
        Note that this method is called before training_epoch_end().

        For example:

            loss = torch.mean(torch.stack([o for o in outputs], dim=0))
        """

        return


class Vehicle_Reid_old(object):
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer,
        scheduler,
        evaluator: Metric_Interface,
        total_epoch=10,
        device="cuda",
    ):
        super(Vehicle_Reid_old, self).__init__()
        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.total_epoch = total_epoch
        self.device = device
        self.model.to(device)
        # self.save_hyperparameters()
        return

    def train(self):
        self.model.train()

        for epoch in range(self.total_epoch):
            total = 0
            for n_iter, (batch) in enumerate((self.train_loader)):
                img, pid, camid, img_path = batch
                total += img.shape[0]
                img = img.to(self.device)
                pid = pid.to(self.device)
                score, feat = self.model(img)
                self.optimizer.zero_grad()
                loss = self.loss_fn(score, feat, pid)
                loss.backward()
                self.optimizer.step()
            print("Done epoch:", total)
        return

    def eval(self):
        print("validation")
