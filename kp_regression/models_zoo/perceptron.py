import logging
import os
import typing as T

import torch
import torch.nn as nn
from joblib import dump, load  # type: ignore
from numpy import concatenate, ndarray
from numpy.typing import NDArray
from pytorch_lightning import Trainer
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch import save
from torchsummary import summary  # type: ignore

from kp_regression.base_model import BaseModel
from kp_regression.data_pipe import Dataset
from kp_regression.models_zoo.torch_common import (
    TorchModelParams,
    TrainingModule,
    build_callbacks,
    get_dataloader_from_dataset,
)
from kp_regression.models_zoo.utils import check_data_and_get_train_val_plain_input
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

        layers_list: T.List[nn.Module] = []

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

        assert len(self.shape) == 1, "MLP accepts only 1d input"
        assert isinstance(self.shape[0], int), "MLP accepts only 1d input"
        self.shape = T.cast(T.Tuple[int], self.shape)

        self.torch_model_params = TorchModelParams(**self.model_params)

        self.models = [
            MLP(self.shape, **self.torch_model_params.model_params)
            for i in range(self.output_shape[0])
        ]
        self.scaler = StandardScaler()

        summary(self.models[0], self.shape, device=self.torch_model_params.accelerator)

    def train(
        self,
        ds: Dataset,
        ds_val: T.Optional[Dataset] = None,
    ) -> None:
        X, y, X_val, y_val = check_data_and_get_train_val_plain_input(
            ds, ds_val, self.torch_model_params.val_frac, error_if_both_absent=False
        )

        X = self.scaler.fit_transform(X)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)
        for dim_i in range(self.output_shape[0]):
            logging.info("Fitting MLP for dim %s", dim_i)
            dl_train = get_dataloader_from_dataset(
                X,
                y[:, dim_i].reshape(-1, 1),
                shuffle=True,
                **self.torch_model_params.data_params,
            )
            if X_val is not None and y_val is not None:
                dl_val = get_dataloader_from_dataset(
                    X_val,
                    y_val[:, dim_i].reshape(-1, 1),
                    shuffle=False,
                    **self.torch_model_params.data_params,
                )
            else:
                dl_val = None
            logging.info("Built data for %s", dim_i)
            checkpoints_folder = os.path.join(self.model_dir, f"checkpoints")
            safe_mkdir(checkpoints_folder)
            checkpoints_folder_i = os.path.join(
                checkpoints_folder, f"checkpoints{dim_i}"
            )
            callbacks = build_callbacks(checkpoints_folder_i, self.torch_model_params)
            training_module = TrainingModule(
                self.models[dim_i], **self.torch_model_params.train_params
            )
            hist_dir = os.path.join(self.model_dir, "hist")
            safe_mkdir(hist_dir)
            hist_dir_i = os.path.join(hist_dir, str(dim_i))
            safe_mkdir(hist_dir_i)
            trainer = Trainer(
                max_epochs=self.torch_model_params.epochs,
                accelerator=self.torch_model_params.accelerator,
                devices=1,
                default_root_dir=hist_dir_i,
                callbacks=list(callbacks),
                enable_progress_bar=True,
            )
            if dl_val is not None:
                trainer.fit(training_module, dl_train, dl_val)
            else:
                trainer.fit(training_module, dl_train)
            checkpoint = callbacks[0]
            logging.info(
                "loading the best model from path %s", checkpoint.best_model_path
            )
            training_module_restored = TrainingModule.load_from_checkpoint(
                checkpoint.best_model_path,
                model=self.models[dim_i],
                **self.torch_model_params.train_params,
            )
            self.models[dim_i] = training_module_restored.model  # type: ignore

    def predict(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), "For MLP dataset should be Numpy"

        X = ds.X

        X = self.scaler.transform(X)

        total_preds = []

        for dim_i in range(self.output_shape[0]):
            logging.info("Predicting for dim %s", dim_i)

            module = TrainingModule(
                self.models[dim_i], **self.torch_model_params.train_params
            )

            trainer = Trainer(accelerator=self.torch_model_params.accelerator)

            dl_test = get_dataloader_from_dataset(
                X, y=None, shuffle=False, **self.torch_model_params.data_params
            )
            preds_list = trainer.predict(module, dl_test)

            assert preds_list is not None, "No output predicted"

            preds_concat = concatenate(preds_list, axis=0)

            total_preds.append(preds_concat)

        return concatenate(total_preds, axis=1)

    def cv(self, cv_params: T.Dict[str, T.Any], ds: Dataset):
        raise NotImplementedError("Not implemented")

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

        assert len(self.shape) == 1, "MLP accepts only 1d input"
        assert len(self.shape) == 1, "MLP accepts only 1d input"
        assert isinstance(self.shape[0], int), "MLP accepts only 1d input"
        self.shape = T.cast(T.Tuple[int], self.shape)
        
        self.torch_model_params = TorchModelParams(**self.model_params)

        assert (
            self.output_shape[0] == self.torch_model_params.model_params["layers"][-1]
        ), "For predicting all steps at once neurons in the last layers should be exactly equal to the number of targets"

        self.model = MLP(self.shape, **self.torch_model_params.model_params)
        self.scaler = StandardScaler()
        summary(self.model, self.shape, device=self.torch_model_params.accelerator)

    def train(
        self,
        ds: Dataset,
        ds_val: T.Optional[Dataset] = None,
    ) -> None:
        X, y, X_val, y_val = check_data_and_get_train_val_plain_input(
            ds, ds_val, self.torch_model_params.val_frac, error_if_both_absent=False
        )
        X = self.scaler.fit_transform(X)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)
        dl_train = get_dataloader_from_dataset(
            X, y, shuffle=True, **self.torch_model_params.data_params
        )
        if X_val is not None:
            dl_val = get_dataloader_from_dataset(
                X_val, y_val, shuffle=False, **self.torch_model_params.data_params
            )
        else:
            dl_val = None
        logging.info("Built data")
        checkpoints_folder = os.path.join(self.model_dir, "checkpoints")
        callbacks = build_callbacks(checkpoints_folder, self.torch_model_params)
        training_module = TrainingModule(
            self.model, **self.torch_model_params.train_params
        )
        hist_dir = os.path.join(self.model_dir, "hist")
        safe_mkdir(hist_dir)
        trainer = Trainer(
            max_epochs=self.torch_model_params.epochs,
            accelerator=self.torch_model_params.accelerator,
            devices=1,
            default_root_dir=hist_dir,
            callbacks=list(callbacks),
            enable_progress_bar=True,
        )
        if dl_val is not None:
            trainer.fit(training_module, dl_train, dl_val)
        else:
            trainer.fit(training_module, dl_train)
        best_model = callbacks[0].best_model_path
        training_module_restored = TrainingModule.load_from_checkpoint(
            best_model, model=self.model, **self.torch_model_params.train_params
        )
        self.model = training_module_restored.model  # type: ignore

    def cv(self, cv_params: T.Dict[str, T.Any], ds: Dataset):
        raise NotImplementedError("Not implemented")

    def predict(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), "For MLP dataset should be Numpy"
        X = ds.X

        X = self.scaler.transform(X)

        module = TrainingModule(self.model, **self.torch_model_params.train_params)

        trainer = Trainer(accelerator=self.torch_model_params.accelerator)

        dl_test = get_dataloader_from_dataset(
            X, y=None, shuffle=False, **self.torch_model_params.data_params
        )
        preds_list = trainer.predict(module, dl_test)

        assert preds_list is not None, "Model output when predicting was None"

        preds_concat = concatenate(preds_list, axis=0)

        logging.info("Preds shape %s", preds_concat.shape)

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
