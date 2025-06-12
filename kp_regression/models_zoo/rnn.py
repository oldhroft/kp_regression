import logging
import os
import typing as T

import torch
import torch.nn as nn
from numpy import concatenate
from numpy.typing import NDArray
from pytorch_lightning import Trainer
from torch import save

from kp_regression.base_model import BaseModel
from kp_regression.data_pipe import Dataset
from kp_regression.models_zoo.torch_common import (
    TorchModelParams, TrainingModuleNInputs, build_callbacks,
    get_dataloader_from_dataset_tuple)
from kp_regression.utils import safe_mkdir


def fc_layer(n_inputs: int, n_outputs: int, use_relu: bool = False) -> nn.Module:
    if use_relu:
        return nn.Sequential(nn.Linear(n_inputs, n_outputs), nn.ReLU())
    else:
        return nn.Linear(n_inputs, n_outputs)


def get_fc_net(
    input_shape: T.Tuple[int, ...], layers: T.List[int], last: bool = False
) -> nn.Sequential:
    assert len(input_shape) == 1, "FCNet only accepts 1D data"
    n_inputs = input_shape[0]

    layers_list = []

    for i, n_outputs in enumerate(layers):
        use_relu = i < len(layers) - 1 or not last
        layers_list.append(
            fc_layer(n_inputs=n_inputs, n_outputs=n_outputs, use_relu=use_relu)
        )
        n_inputs = n_outputs

    return nn.Sequential(*layers_list)


class LSTM(nn.Module):
    def __init__(
        self,
        input_shape1: T.Tuple[int, ...],
        input_shape2: T.Tuple[int, ...],
        input_shape3: T.Tuple[int, ...],
        hidden_size1: int,
        hidden_size2: int,
        num_layers1: int,
        num_layers2: int,
        layers: T.List[int],
        layers_head: T.List[int],
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.rnn1 = nn.LSTM(
            input_size=input_shape1[0],
            hidden_size=hidden_size1,
            num_layers=num_layers1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.LSTM(
            input_size=input_shape2[0],
            hidden_size=hidden_size2,
            num_layers=num_layers2,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.net3 = get_fc_net(input_shape=input_shape3, layers=layers)

        mult = 2 if bidirectional else 1

        input_shape_fc = mult * hidden_size1 + mult * hidden_size2 + layers[-1]

        self.head = get_fc_net(
            input_shape=(input_shape_fc,), layers=layers_head, last=True
        )

    def forward(self, x1, x2, x3):
        y1, _ = self.rnn1(x1.transpose(2, 1))
        y2, _ = self.rnn2(x2.transpose(2, 1))
        y3 = self.net3(x3)

        flattaned = torch.concat([y1[:, -1, :], y2[:, -1, :], y3], dim=1)

        return self.head(flattaned)


class LSTM5m(nn.Module):
    def __init__(
        self,
        input_shape1: T.Tuple[int, ...],
        input_shape2: T.Tuple[int, ...],
        input_shape3: T.Tuple[int, ...],
        input_shape4: T.Tuple[int, ...],
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        num_layers1: int,
        num_layers2: int,
        num_layers3: int,
        layers: T.List[int],
        layers_head: T.List[int],
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.rnn1 = nn.LSTM(
            input_size=input_shape1[0],
            hidden_size=hidden_size1,
            num_layers=num_layers1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.LSTM(
            input_size=input_shape2[0],
            hidden_size=hidden_size2,
            num_layers=num_layers2,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.rnn3 = nn.LSTM(
            input_size=input_shape3[0],
            hidden_size=hidden_size3,
            num_layers=num_layers3,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.net4 = get_fc_net(input_shape=input_shape4, layers=layers)

        mult = 2 if bidirectional else 1

        input_shape_fc = (
            mult * hidden_size1 + mult * hidden_size2 + mult * hidden_size3 + layers[-1]
        )

        self.head = get_fc_net(
            input_shape=(input_shape_fc,), layers=layers_head, last=True
        )

    def forward(self, x1, x2, x3, x4):
        y1, _ = self.rnn1(x1.transpose(2, 1))
        y2, _ = self.rnn2(x2.transpose(2, 1))
        y3, _ = self.rnn3(x3.transpose(2, 1))

        y4 = self.net4(x4)

        flattaned = torch.concat([y1[:, -1, :], y2[:, -1, :], y3[:, -1, :], y4], dim=1)

        return self.head(flattaned)


class LSTM3Inputs(BaseModel):

    def build(self) -> None:

        n_inputs = 3
        self.torch_model_params = TorchModelParams(**self.model_params)
        assert len(self.shape) == n_inputs, "Shape should contain 3 inputs"
        assert isinstance(self.shape[0], tuple), f"Shape 0 should be tuple"
        assert isinstance(self.shape[1], tuple), f"Shape 1 should be tuple"
        assert isinstance(self.shape[2], tuple), f"Shape 2 should be tuple"

        self.models: T.List[nn.Module] = [
            LSTM(
                input_shape1=self.shape[0],
                input_shape2=self.shape[1],
                input_shape3=self.shape[2],
                **self.torch_model_params.model_params,
            )
            for i in range(self.output_shape[0])
        ]

        self.n_inputs = n_inputs

        logging.info("Shape %s", self.shape)

    def train(
        self,
        ds: Dataset,
        ds_val: T.Optional[Dataset] = None,
    ) -> None:

        assert ds.y is not None, "Dataset should contain y"
        assert isinstance(ds.X, tuple), "Dataset for the model should be tuple"

        X, y = ds.X, ds.y

        if ds_val is None:
            X_val, y_val = None, None
        else:
            assert ds_val.y is not None, "Val dataset should contain y"
            X_val, y_val = ds_val.X, ds_val.y

        for dim_i in range(self.output_shape[0]):
            logging.info("Fitting MLP for dim %s", dim_i)

            dl_train = get_dataloader_from_dataset_tuple(
                (*X, y[:, dim_i].reshape(-1, 1)),
                shuffle=True,
                **self.torch_model_params.data_params,
            )

            if X_val is not None and y_val is not None:
                dl_val = get_dataloader_from_dataset_tuple(
                    (*X_val, y_val[:, dim_i].reshape(-1, 1)),
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

            training_module = TrainingModuleNInputs(
                self.models[dim_i],
                n_inputs=self.n_inputs,
                **self.torch_model_params.train_params,
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

            best_model: str = callbacks[0].best_model_path

            training_module_restored = TrainingModuleNInputs.load_from_checkpoint(
                best_model,
                n_inputs=self.n_inputs,
                model=self.models[dim_i],
                **self.torch_model_params.train_params,
            )
            assert hasattr(
                training_module_restored, "model"
            ), "Training module should have model attr"

            self.models[dim_i] = training_module_restored.model  # type: ignore

    def predict(self, ds: Dataset) -> NDArray:

        assert isinstance(ds.X, tuple), "Dataset for the model should be tuple"

        total_preds = []

        for dim_i in range(self.output_shape[0]):
            logging.info("Predicting for dim %s", dim_i)

            module = TrainingModuleNInputs(
                self.models[dim_i],
                n_inputs=self.n_inputs,
                **self.torch_model_params.train_params,
            )

            trainer = Trainer(accelerator=self.torch_model_params.accelerator)

            dl_test = get_dataloader_from_dataset_tuple(
                ds.X, shuffle=False, **self.torch_model_params.data_params
            )
            preds_list = trainer.predict(module, dl_test)

            assert preds_list is not None, "Returned empty output"

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

    def load(self, path) -> None:
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(path, f"weights{i}.pth")))


class LSTM4Inputs(LSTM3Inputs):

    def build(self) -> None:

        n_inputs = 4

        self.torch_model_params = TorchModelParams(**self.model_params)
        assert len(self.shape) == self.n_inputs, "Shape should contain 4 inputs"

        assert len(self.shape) == 4, "Shape should contain 3 inputs"
        assert isinstance(self.shape[0], tuple), f"Shape 0 should be tuple"
        assert isinstance(self.shape[1], tuple), f"Shape 1 should be tuple"
        assert isinstance(self.shape[2], tuple), f"Shape 2 should be tuple"
        assert isinstance(self.shape[3], tuple), f"Shape 3 should be tuple"

        self.models = [
            LSTM5m(
                input_shape1=self.shape[0],
                input_shape2=self.shape[1],
                input_shape3=self.shape[2],
                input_shape4=self.shape[3],
                **self.torch_model_params.model_params,
            )
            for i in range(self.output_shape[0])
        ]

        self.n_inputs = n_inputs
