import os
from pyparsing import Any
from pytorch_lightning import LightningModule, Trainer
from torchmetrics.functional import accuracy
from metrics.metric import Metric_Interface
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import time

class Vehicle_Reid(LightningModule):
    def __init__(
        self,
        model,
        loss_fn: nn.Module,
        optimizer,
        scheduler,
        evaluator: Metric_Interface,
        cfg: None,
    ):
        super(Vehicle_Reid, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.cfg = cfg
        self.predict_ouputs = dict
        self.start_time = time.time()
        return

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, pid = batch
        score, feat = self.model(img)
        loss = self.loss_fn(score, feat, pid)
        
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == pid).float().mean()
        else:
            acc = (score.max(1)[1] == pid).float().mean()
        
        self.log(f"Train loss", loss, prog_bar=True)
        self.log(f"Train accuracy", acc, prog_bar=True)

        return {"loss": loss, "acc": acc}

    def evaluate(self, batch, stage):
        img, pid, camid, img_path = batch
        feat = self.model(img)
        self.evaluator.update([feat.cpu(), pid, camid])
        return

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return dict(
            zip(batch[1], super().predict_step(batch[0], batch_idx, dataloader_idx))
        )

    def on_train_epoch_start(self) -> None:
        print("Start training on epoch:", self.current_epoch)

    def training_epoch_end(self, outputs) -> None:
        loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        acc = torch.mean(torch.stack([o["acc"] for o in outputs], dim=0)) * 100
        print(
            "Epoch: {:.1f},Train accuracy: {:.5f}, Train loss: {:.5f}".format(
                self.current_epoch, acc, loss
            )
        )
        print("Learning rate:", self.scheduler.get_lr()[0])

    def validation_epoch_end(self, outputs):
        print("Calulating the acc, cmc, mAP")

        cmc, mAP = self.evaluator.compute()

        self.log("Val_CMC@rank1", cmc[0])
        self.log("Val_CMC@rank5", cmc[4])
        self.log("Val_mAP", mAP)

        print("Validation result:")
        print(
            "Validation epoch: {:.1f}, CMC@rank1: {:.5f}%, CMC@rank5: {:.5f}%, mAP: {:.5f}%".format(
                self.current_epoch, cmc[0] * 100, cmc[4] * 100, mAP * 100
            )
        )
        self.evaluator.reset()
        # if time.time() - self.start_time > 36000:
        #     print("OUT of training time, stop!!")
        #     exit(0)
        return

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def on_validation_epoch_start(self):
        print("--------")
        print("Start validation")

    def on_train_start(self):
        print("Saving directory: ", self.logger.log_dir)
        import ruamel.yaml as yaml

        if not os.path.exists(self.logger.log_dir):
            os.makedirs(self.logger.log_dir)

        with open(self.logger.log_dir + "/settings.yaml", "w+") as yml:
            yaml.dump(self.cfg, yml, allow_unicode=True, Dumper=yaml.RoundTripDumper)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.cfg["SOLVER"]["WARMUP_METHOD"] == "cosine":
            scheduler.step(
                epoch=self.current_epoch
            )  # timm's scheduler need the epoch value
        else:
            scheduler.step()
