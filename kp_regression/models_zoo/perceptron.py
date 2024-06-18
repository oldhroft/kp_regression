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
from sklearn.model_selection import train_test_split

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
    val_frac: float = 0.1


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

    def validation_step(self, batch: Tensor, batch_idx: T.Any) -> Tensor:
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

    safe_mkdir(folder)

    TrainingModuleCheckpoint = ModelCheckpoint(dirpath=folder, **cfg.checkpoint_cfg)

    EarlyStoppingCallback = EarlyStopping(**cfg.early_stopping_cfg)

    return [TrainingModuleCheckpoint, EarlyStoppingCallback]


class MLPClassMulti(BaseModel):

    def build(self) -> None:
        self.model_params = TorchModelParams(**self.model_params)

        self.models = [
            MLP(self.shape, **self.model_params.model_params)
            for i in range(self.output_shape[0])
        ]
        self.scaler = StandardScaler()

    def train(
        self,
        X: NDArray,
        y: NDArray,
        X_val: T.Optional[NDArray] = None,
        y_val: T.Optional[NDArray] = None,
    ) -> None:

        if X_val is None and self.model_params.val_frac is not None:
            logging.info("Received val frac, creating val split")

            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.model_params.val_frac,
                random_state=17,
                shuffle=True,
            )

        logging.info("X shape %s", X.shape)
        logging.info("y shape %s", y.shape)

        X = self.scaler.fit_transform(X)

        if X_val is not None:
            X_val = self.scaler.transform(X_val)

            logging.info("X val shape %s", X_val.shape)
            logging.info("y val shape %s", y_val.shape)

        for dim_i in range(self.output_shape[0]):
            logging.info("Fitting MLP for dim %s", dim_i)

            dl_train = get_dataloader_from_dataset(
                X,
                y[:, dim_i].reshape(-1, 1),
                shuffle=True,
                **self.model_params.data_params,
            )

            if X_val is not None:
                dl_val = get_dataloader_from_dataset(
                    X_val,
                    y_val[:, dim_i].reshape(-1, 1),
                    shuffle=False,
                    **self.model_params.data_params,
                )
            else:
                dl_val = None

            logging.info("Built data for %s", dim_i)

            checkpoints_folder = os.path.join(self.model_dir, f"checkpoints")

            safe_mkdir(checkpoints_folder)

            checkpoints_folder_i = os.path.join(
                checkpoints_folder, f"checkpoints{dim_i}"
            )

            callbacks = build_callbacks(checkpoints_folder_i, self.model_params)

            training_module = TrainingModule(
                self.models[dim_i], **self.model_params.train_params
            )

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
                best_model, model=self.models[dim_i], **self.model_params.train_params
            )

            self.models[dim_i] = training_module.model

    def predict(self, X: NDArray) -> NDArray:

        X = self.scaler.transform(X)

        total_preds = []

        for dim_i in range(self.output_shape[0]):
            logging.info("Predicting for dim %s", dim_i)

            module = TrainingModule(
                self.models[dim_i], **self.model_params.train_params
            )

            trainer = Trainer(accelerator=self.model_params.accelerator)

            dl_test = get_dataloader_from_dataset(
                X, y=None, shuffle=False, **self.model_params.data_params
            )
            preds_list = trainer.predict(module, dl_test)

            preds_concat = concatenate(preds_list, axis=0)

            total_preds.append(preds_concat)

        return concatenate(total_preds, axis=1)

    def cv(self, cv_params: T.Dict, X: NDArray, y: NDArray):
        raise NotImplementedError("CV not implemented")

    def save(self, file_path: str) -> None:

        safe_mkdir(file_path)

        for i, model in enumerate(self.models):
            path = os.path.join(file_path, f"weights{i}.pth")
            save(model.state_dict(), path)

        path_scaler = os.path.join(file_path, "scaler.sav")
        dump(self.scaler, path_scaler)


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

        if X_val is None and self.model_params.val_frac is not None:
            logging.info("Received val frac, creating val split")

            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.model_params.val_frac,
                random_state=17,
                shuffle=True,
            )

        logging.info("X shape %s", X.shape)
        logging.info("y shape %s", y.shape)

        X = self.scaler.fit_transform(X)

        if X_val is not None:
            X_val = self.scaler.transform(X_val)

            logging.info("X val shape %s", X_val.shape)
            logging.info("y val shape %s", y_val.shape)

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

        checkpoints_folder = os.path.join(self.model_dir, "checkpoints")

        callbacks = build_callbacks(checkpoints_folder, self.model_params)

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
