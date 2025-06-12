import logging
import os
import typing as T

import torch
import torch.nn as nn
from joblib import dump, load
from numpy import concatenate, ndarray
from numpy.typing import NDArray
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import save
from torchsummary import summary

from kp_regression.base_model import BaseModel
from kp_regression.data_pipe import Dataset
from kp_regression.models_zoo.torch_common import (
    TorchModelParams,
    TrainingModule,
    build_callbacks,
    get_dataloader_from_dataset,
)
from kp_regression.utils import safe_mkdir


class MLP(nn.Module):

    def __init__(
        self,
        input_shape: T.Tuple[int],
        layers: T.List[int],
        dropout: T.Optional[float] = None,
    ) -> None:
        super().__init__()

        assert len(input_shape) == 1, "MLP only accepts 1D data"

        n_inputs = input_shape[0]

        layers_list = []

        for n_outputs in layers[:-1]:
            layers_list.append(nn.Linear(n_inputs, n_outputs))
            if dropout is not None:
                layers_list.append(nn.Dropout(p=dropout))
            layers_list.append(nn.ReLU())
            n_inputs = n_outputs

        layers_list.append(nn.Linear(n_inputs, layers[-1]))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


class MLPClassMulti(BaseModel):

    def build(self) -> None:
        self.model_params = TorchModelParams(**self.model_params)

        self.models = [
            MLP(self.shape, **self.model_params.model_params)
            for i in range(self.output_shape[0])
        ]
        self.scaler = StandardScaler()

        summary(self.models[0], self.shape, device=self.model_params.accelerator)

    def train(
        self,
        ds: Dataset,
        ds_val: T.Optional[Dataset] = None,
    ) -> None:

        assert isinstance(ds.X, ndarray), "For MLP dataset should be Numpy"
        assert ds_val is None or isinstance(
            ds_val.X, ndarray
        ), "For MLP dataset should be Numpy"

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

        X = ds.X

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

    def load(self, dirpath: str) -> None:
        for i, model in enumerate(self.models):
            path = os.path.join(dirpath, f"weights{i}.pth")
            model.load_state_dict(torch.load(path))

        path_scaler = os.path.join(dirpath, "scaler.sav")
        self.scaler = load(path_scaler)


class MLPClass(BaseModel):

    def build(self) -> None:
        self.model_params = TorchModelParams(**self.model_params)
        self.model = MLP(self.shape, **self.model_params.model_params)
        self.scaler = StandardScaler()
        summary(self.model, self.shape, device=self.model_params.accelerator)

    def train(self, ds: Dataset, ds_val: Dataset) -> None:

        assert isinstance(ds.X, ndarray), "For MLP dataset should be Numpy"
        assert ds_val is None or isinstance(
            ds_val.X, ndarray
        ), "For MLP dataset should be Numpy"

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

        hist_dir = os.path.join(self.model_dir, "hist")
        safe_mkdir(hist_dir)
        trainer = Trainer(
            max_epochs=self.model_params.epochs,
            accelerator=self.model_params.accelerator,
            devices=1,
            default_root_dir=hist_dir,
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

    def predict(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), "For MLP dataset should be Numpy"
        X = ds.X

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

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        path_scaler = os.path.join(path, "scaler.sav")
        self.scaler = load(path_scaler)
