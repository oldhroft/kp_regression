import typing as T

from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import ndarray

import torch.nn as nn
from torch import from_numpy, Tensor
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningModule, Trainer

from kp_regression.utils import safe_mkdir
from kp_regression.data_pipe import Dataset


@dataclass
class TorchModelParams:
    model_params: dict
    data_params: dict
    train_params: dict
    checkpoint_cfg: dict
    early_stopping_cfg: dict
    epochs: int
    accelerator: T.Literal["cpu", "tpu"] = "cpu"
    val_frac: float = 0.1


class TrainingModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.03,
        lr_reduce_factor: float = 0.2,
        lr_reduce_patience: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_patience = lr_reduce_patience
        self.loss = nn.MSELoss()

    def training_step(self, batch: Tensor, batch_idx: T.Any) -> Tensor:
        x, y = batch

        y_pred: Tensor = self.model(x)

        loss = self.loss(y_pred, y)

        metrics = {"train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(
        self, batch: T.Tuple[Tensor, Tensor], batch_idx: T.Any
    ) -> Tensor:
        x, y = batch
        y_pred: Tensor = self.model(x)

        loss = self.loss(y_pred, y)

        metrics = {"val_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch[0])

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = opt.Adam(
            self.parameters(),
            lr=self.lr,
        )

        lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_reduce_factor,
            patience=self.lr_reduce_patience,
            verbose=True,
        )

        lr_dict = {
            # The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
        }

        return [optimizer], [lr_dict]


class TrainingModule3Inputs(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.03,
        lr_reduce_factor: float = 0.2,
        lr_reduce_patience: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_patience = lr_reduce_patience
        self.loss = nn.MSELoss()

    def training_step(self, batch: Tensor, batch_idx: T.Any) -> Tensor:
        x1, x2, x3, y = batch

        y_pred: Tensor = self.model(x1, x2, x3)

        loss = self.loss(y_pred, y)

        metrics = {"train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(
        self, batch: T.Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: T.Any
    ) -> Tensor:
        x1, x2, x3, y = batch
        y_pred: Tensor = self.model(x1, x2, x3)

        loss = self.loss(y_pred, y)

        metrics = {"val_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x1, x2, x3 = batch
        return self.model(x1, x2, x3)

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = opt.Adam(
            self.parameters(),
            lr=self.lr,
        )

        lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_reduce_factor,
            patience=self.lr_reduce_patience,
            verbose=True,
        )

        lr_dict = {
            # The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
        }

        return [optimizer], [lr_dict]


class TrainingModule4Inputs(TrainingModule3Inputs):

    def training_step(self, batch: Tensor, batch_idx: T.Any) -> Tensor:
        x1, x2, x3, x4, y = batch

        y_pred: Tensor = self.model(x1, x2, x3, x4)

        loss = self.loss(y_pred, y)

        metrics = {"train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(
        self, batch: T.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: T.Any
    ) -> Tensor:
        x1, x2, x3, x4, y = batch
        y_pred: Tensor = self.model(x1, x2, x3, x4)

        loss = self.loss(y_pred, y)

        metrics = {"val_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x1, x2, x3, x4 = batch
        return self.model(x1, x2, x3, x4)

def get_dataloader_from_dataset(
    X: NDArray,
    y: T.Optional[NDArray],
    shuffle: bool,
    batch_size: int,
) -> DataLoader:

    if y is not None:
        ds = TensorDataset(
            from_numpy(X.astype("float32")), from_numpy(y.astype("float32"))
        )
    else:
        ds = TensorDataset(from_numpy(X.astype("float32")))

    dl = DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=0)

    return dl


def get_dataloader_from_dataset_tuple(
    data: T.Tuple[NDArray], shuffle: bool, batch_size: int
) -> DataLoader:
    tensor_list = [from_numpy(x.astype("float32")) for x in list(data)]
    ds = TensorDataset(*tensor_list)
    dl = DataLoader(ds, shuffle=shuffle, batch_size=batch_size, num_workers=0)
    return dl


def build_callbacks(folder: str, cfg: TorchModelParams) -> T.List[Callback]:

    safe_mkdir(folder)

    TrainingModuleCheckpoint = ModelCheckpoint(dirpath=folder, **cfg.checkpoint_cfg)

    EarlyStoppingCallback = EarlyStopping(**cfg.early_stopping_cfg)

    return [TrainingModuleCheckpoint, EarlyStoppingCallback]
