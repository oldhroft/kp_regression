import typing as T
import os

import torch

from numpy.typing import NDArray
import torch.nn as nn
from torch import save

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torchsummary import summary


from numpy import concatenate, ndarray

from pytorch_lightning import Trainer

from joblib import dump

from kp_regression.base_model import BaseModel
from kp_regression.utils import safe_mkdir
from kp_regression.data_pipe import Dataset
from kp_regression.models_zoo.torch_common import (
    TorchModelParams,
    TrainingModule,
    get_dataloader_from_dataset,
    build_callbacks,
)

import logging

import typing as T


def conv1d_block(n_inputs: int, n_outputs: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            padding="same",
        ),
        nn.ReLU(),
        nn.AvgPool1d(2, 2),
    )


def fc_layer(n_inputs: int, n_outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_inputs, n_outputs), nn.ReLU())


def get_fc_net(input_shape: T.Tuple[int], layers: T.List[int]) -> nn.Sequential:
    assert len(input_shape) == 1, "FCNet only accepts 1D data"
    n_inputs = input_shape[0]

    layers_list = []

    for n_outputs in layers:
        layers_list.append(fc_layer(n_inputs=n_inputs, n_outputs=n_outputs))
        n_inputs = n_outputs

    return nn.Sequential(*layers_list)


def get_conv1d_backbone(
    input_shape: T.Tuple[int], layers: T.List[int], kernel_size: int = 3
) -> T.Tuple[nn.Sequential, int]:
    assert len(input_shape) == 2, "Conv1Net only accepts 2D data"

    n_inputs, n_features = input_shape

    layers_list = []

    for n_outputs in layers:

        layers_list.append(
            conv1d_block(
                n_inputs=n_inputs, n_outputs=n_outputs, kernel_size=kernel_size
            )
        )
        n_inputs = n_outputs
        n_features = n_features // 2

    layers_list.append(nn.Flatten())

    return nn.Sequential(*layers_list), n_features * n_outputs


class Conv1DNet3inputs(nn.Module):
    def __init__(
        self,
        input_shape1: T.Tuple[int],
        input_shape2: T.Tuple[int],
        input_shape3: T.Tuple[int],
        conv_layers1: T.List[int],
        conv_layers2: T.List[int],
        layers: T.List[int],
        layers_head: T.List[int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.cv_net1, n1 = get_conv1d_backbone(
            input_shape=input_shape1, layers=conv_layers1, kernel_size=kernel_size
        )
        self.cv_net2, n2 = get_conv1d_backbone(
            input_shape=input_shape2, layers=conv_layers2, kernel_size=kernel_size
        )

        self.net3 = get_fc_net(input_shape=input_shape3, layers=layers)

        input_shape_fc = n1 + n2 + layers[-1]

        self.head = get_fc_net(input_shape=(input_shape_fc,), layers=layers_head)

    def forward(self, x1, x2, x3):
        y1 = self.cv_net1(x1)
        y2 = self.cv_net2(x2)
        y3 = self.net3(x3)

        flattaned = torch.concat([y1, y2, y3], dim=1)

        return self.head(flattaned)


class Conv1DNet3InputsMulti(BaseModel):

    def build(self) -> None:
        self.model_params = TorchModelParams(**self.model_params)

        self.models = [
            Conv1DNet3inputs(
                input_shape1=self.shape[0], 
                input_shape2=self.shape[1], 
                input_shape3=self.shape[1], 
                **self.model_params.model_params)
            for i in range(self.output_shape[0])
        ]

        summary(self.models[0], self.shape, device=self.model_params.accelerator)


    def _check(self, ds: T.Optional[Dataset]) -> None:

        assert (
            ds is None or (
                isinstance(ds.X, tuple)
                and len(ds.X) == 3
                and isinstance(ds.X[0], ndarray)
                and isinstance(ds.X[1], ndarray)
                and isinstance(ds.X[2], ndarray)
            )
        ), "Wrong dataset for conv net"

    def train(
        self,
        ds: Dataset,
        ds_val: T.Optional[Dataset] = None,
    ) -> None:
        
        self._check(ds)
        self._check(ds_val)

        X, y = ds.X, ds.y

        X_val = None if ds_val is None else ds_val.X
        y_val = None if ds_val is None else ds_val.y

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

            hist_dir = os.path.join(self.model_dir, "hist")
            safe_mkdir(hist_dir)

            hist_dir_i = os.path.join(hist_dir, str(dim_i))
            safe_mkdir(hist_dir_i)

            trainer = Trainer(
                max_epochs=self.model_params.epochs,
                accelerator=self.model_params.accelerator,
                devices=1,
                default_root_dir=hist_dir_i,
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

    def predict(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), "For MLP dataset should be Numpy"

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