import typing as T
import os

import torch

from numpy.typing import NDArray
import torch.nn as nn
from torch import from_numpy, Tensor, save
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as opt
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.preprocessing import StandardScaler

from torchsummary import summary


from numpy import concatenate

from pytorch_lightning import LightningModule, Trainer
from dataclasses import dataclass

from joblib import dump

from kp_regression.base_model import BaseModel
from kp_regression.utils import safe_mkdir

import logging


@dataclass
class TorchModelParams:
    model_params: dict
    data_params: dict
    train_params: dict
    checkpoint_cfg: dict
    early_stopping_cfg: dict
    epochs: int
    accelerator: T.Literal["cpu", "tpu"] = "cpu"


class MLP(nn.Module):

    def __init__(self, input_shape: T.Tuple[int], layers: T.List[int]) -> None:
        super().__init__()

        assert len(input_shape) == 1, "MLP only accepts 1D data"

        n_inputs = input_shape[0]

        layers_list = []

        for n_outputs in layers[:-1]:
            layers_list.append(nn.Linear(n_inputs, n_outputs))
            layers_list.append(nn.ReLU())
            n_inputs = n_outputs

        layers_list.append(nn.Linear(n_inputs, layers[-1]))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


class TrainingModule(LightningModule):
    def __init__(self, model: nn.Module, lr: float = 0.03) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = nn.MSELoss()

    def training_step(self, batch: Tensor, batch_idx: T.Any) -> Tensor:
        x, y = batch

        y_pred: Tensor = self.model(x)

        loss = self.loss(y_pred, y)

        metrics = {"train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: T.Any) -> Tensor:
        x, y = batch
        y_pred: Tensor = self.model(x)

        loss = self.loss(y_pred, y)

        metrics = {"val_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch[0])

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = opt.Adam(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer


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


def build_callbacks(folder: str, cfg: TorchModelParams) -> T.List[Callback]:

    checkpoints_folder = os.path.join(folder, "checkpoints")

    TrainingModuleCheckpoint = ModelCheckpoint(
        dirpath=checkpoints_folder, **cfg.checkpoint_cfg
    )

    EarlyStoppingCallback = EarlyStopping(**cfg.early_stopping_cfg)

    return [TrainingModuleCheckpoint, EarlyStoppingCallback]


class MLPClass(BaseModel):

    def build(self) -> None:
        self.model_params = TorchModelParams(**self.model_params)
        self.model = MLP(self.shape, **self.model_params.model_params)
        self.scaler = StandardScaler()
        summary(self.model, self.shape)

    def train(
        self,
        X: NDArray,
        y: NDArray,
        X_val: T.Optional[NDArray] = None,
        y_val: T.Optional[NDArray] = None,
    ) -> None:

        logging.info("X shape %s", X.shape)
        logging.info("y shape %s", y.shape)

        X = self.scaler.fit_transform(X)
        X_val = self.scaler.transform(X_val)

        dl_train = get_dataloader_from_dataset(
            X, y, shuffle=True, **self.model_params.data_params
        )

        if X_val is not None:
            dl_val = get_dataloader_from_dataset(
                X_val, y_val, shuffle=False, **self.model_params.data_params
            )
        else:
            dl_val = None

        logging.info("Testing dataloader")

        for x1, y1 in dl_train:
            break

        logging.info("Built data")

        callbacks = build_callbacks(self.model_dir, self.model_params)

        training_module = TrainingModule(self.model, **self.model_params.train_params)

        trainer = Trainer(
            max_epochs=self.model_params.epochs,
            accelerator=self.model_params.accelerator,
            devices=1,
            default_root_dir=self.model_dir,
            callbacks=callbacks,
            enable_progress_bar=True,
        )

        if dl_val is not None:
            trainer.fit(training_module, dl_train, dl_val)
        else:
            trainer.fit(training_module, dl_train)

        best_model: str = callbacks[0].best_model_path

        training_module = TrainingModule.load_from_checkpoint(
            best_model, model=self.model, **self.model_params.train_params
        )

        self.model = training_module.model

    def cv(self, cv_params: T.Dict, X: NDArray, y: NDArray):
        raise NotImplementedError("CV not implemented")

    def predict(self, X: NDArray) -> NDArray:

        X = self.scaler.transform(X)

        module = TrainingModule(self.model, **self.model_params.train_params)

        trainer = Trainer(accelerator=self.model_params.accelerator)

        dl_test = get_dataloader_from_dataset(
            X, y=None, shuffle=False, **self.model_params.data_params
        )
        preds_list = trainer.predict(module, dl_test)

        preds_concat = concatenate(preds_list, axis=0)

        return preds_concat

    def save(self, file_path: str) -> None:

        safe_mkdir(file_path)

        path = os.path.join(file_path, "weights.pth")
        save(self.model.state_dict(), path)

        path_scaler = os.path.join(file_path, "scaler.sav")
        dump(self.scaler, path_scaler)
