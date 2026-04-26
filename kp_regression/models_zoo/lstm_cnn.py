import logging
import os
import typing as T

import torch
import torch.nn as nn
from joblib import dump, load
from numpy import concatenate, ndarray
from numpy.typing import NDArray
from pandas import DataFrame
from pytorch_lightning import Trainer
from sklearn.preprocessing import StandardScaler
from torch import Tensor, save
from torch.utils.data import DataLoader

from kp_regression.base_model import BaseModel
from kp_regression.data_pipe import Dataset
from kp_regression.models_zoo.blocks import vgg2d_block
from kp_regression.models_zoo.torch_common import (
    TorchModelParams,
    TrainingModuleImageTabular,
    build_callbacks,
)
from kp_regression.models_zoo.torch_datasets import KpImageTabularDataset
from kp_regression.utils import safe_mkdir


class CNN2DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_filters: list[int],
        img_size: tuple[int, int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        channels = [in_channels, *num_filters]
        self.blocks = nn.ModuleList(
            [
                vgg2d_block(kernel_size, channels[i], channels[i + 1])
                for i in range(len(num_filters))
            ]
        )

        h, w = img_size
        for _ in num_filters:
            h = (h - 2 * (kernel_size - 1)) // 2
            w = (w - 2 * (kernel_size - 1)) // 2

        self.flat_dim = num_filters[-1] * h * w
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.flatten(x)


class LSTMCNNTabularModel(nn.Module):
    def __init__(
        self,
        n_image_channels: int,
        cnn_num_filters: list[int],
        img_size: tuple[int, int],
        tabular_input_dim: int,
        n_targets: int,
        cnn_kernel_size: int = 3,
        projection_dim: int = 128,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        bidirectional: bool = False,
        tabular_layers: list[int] | None = None,
        head_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        if tabular_layers is None:
            tabular_layers: list[int] = [128]
        if head_layers is None:
            head_layers = [64]

        self.encoder = CNN2DEncoder(
            n_image_channels, cnn_num_filters, img_size, cnn_kernel_size
        )

        self.projection = nn.Sequential(
            nn.Linear(self.encoder.flat_dim, projection_dim),
            nn.ReLU(),
        )

        lstm_dir_mult = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        tab_layers: list[nn.Module] = []
        tab_in = tabular_input_dim
        for tab_out in tabular_layers:
            tab_layers.append(nn.Linear(tab_in, tab_out))
            tab_layers.append(nn.ReLU())
            tab_in = tab_out
        self.tabular_fc = nn.Sequential(*tab_layers)

        combined_dim = lstm_hidden * lstm_dir_mult + tabular_layers[-1]
        head: list[nn.Module] = []
        head_in = combined_dim
        for h_out in head_layers:
            head.append(nn.Linear(head_in, h_out))
            head.append(nn.ReLU())
            head_in = h_out
        head.append(nn.Linear(head_in, n_targets))
        self.head = nn.Sequential(*head)

    def forward(self, tabular_x: Tensor, image_seq: Tensor) -> Tensor:
        batch, seq_len, c, h, w = image_seq.shape

        x_flat = image_seq.reshape(batch * seq_len, c, h, w)
        features = self.encoder(x_flat)
        features = self.projection(features)
        features = features.reshape(batch, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]

        tabular_out = self.tabular_fc(tabular_x)

        combined = torch.cat([last_out, tabular_out], dim=1)
        return self.head(combined)


class LSTMCNNImageModel(BaseModel):
    def build(self) -> None:
        self.torch_model_params = TorchModelParams(**self.model_params)
        mp = self.torch_model_params.model_params

        self.image_categories: list[str] = mp["image_categories"]
        self.crop: int = mp.get("crop", 25)
        self.img_resize: int = mp.get("img_resize", 128)

        img_h = self.img_resize - 2 * self.crop
        img_w = self.img_resize

        assert isinstance(self.shape, tuple)
        assert isinstance(self.shape[0], int), "Expected flat tabular shape"
        tabular_dim: int = self.shape[0]

        self.model = LSTMCNNTabularModel(
            n_image_channels=len(self.image_categories) * 3,
            cnn_num_filters=mp.get("cnn_num_filters", [32, 64]),
            img_size=(img_h, img_w),
            tabular_input_dim=tabular_dim,
            n_targets=self.output_shape[0],
            cnn_kernel_size=mp.get("cnn_kernel_size", 3),
            projection_dim=mp.get("projection_dim", 128),
            lstm_hidden=mp.get("lstm_hidden", 64),
            lstm_layers=mp.get("lstm_layers", 1),
            lstm_dropout=mp.get("lstm_dropout", 0.0),
            bidirectional=mp.get("bidirectional", False),
            tabular_layers=mp.get("tabular_layers"),
            head_layers=mp.get("head_layers"),
        )

        self.scaler = StandardScaler()

        logging.info("Built LSTMCNNTabularModel with shape %s", self.shape)

    def _build_path_columns(self, image_paths: DataFrame) -> list[list[str]]:
        all_columns = list(image_paths.columns)
        return [
            sorted(c for c in all_columns if c.startswith(f"{cat}_lag_"))
            for cat in self.image_categories
        ]

    def _extract_tabular_x(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), (
            f"Expected ndarray tabular X, got {type(ds.X).__name__}"
        )
        return ds.X

    def _build_dataloader(
        self, ds: Dataset, tabular_x: NDArray, shuffle: bool
    ) -> DataLoader:
        assert ds.image_paths is not None, "Dataset must have image_paths"
        path_columns = self._build_path_columns(ds.image_paths)

        n_image_lags = len(path_columns[0])
        dataset = KpImageTabularDataset(
            tabular_x=tabular_x,
            image_path_columns=path_columns,
            image_paths_df=ds.image_paths,
            y=ds.y,
            n_categories=len(self.image_categories),
            n_lags=n_image_lags,
            crop=self.crop,
            img_resize=self.img_resize,
        )
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.torch_model_params.data_params["batch_size"],
            num_workers=self.torch_model_params.data_params.get("num_workers", 0),
        )

    def train(self, ds: Dataset, ds_val: Dataset | None = None) -> None:
        tabular_x_train = self.scaler.fit_transform(self._extract_tabular_x(ds))
        dl_train = self._build_dataloader(ds, tabular_x_train, shuffle=True)

        if ds_val is not None:
            tabular_x_val = self.scaler.transform(self._extract_tabular_x(ds_val))
            dl_val = self._build_dataloader(ds_val, tabular_x_val, shuffle=False)
        else:
            dl_val = None

        checkpoints_folder = os.path.join(self.model_dir, "checkpoints")
        safe_mkdir(checkpoints_folder)
        callbacks = build_callbacks(checkpoints_folder, self.torch_model_params)

        training_module = TrainingModuleImageTabular(
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

        best_model_path: str = callbacks[0].best_model_path
        if best_model_path:
            restored = TrainingModuleImageTabular.load_from_checkpoint(
                best_model_path,
                model=self.model,
                **self.torch_model_params.train_params,
            )
            self.model = restored.model

    def predict(self, ds: Dataset) -> NDArray:
        tabular_x = self.scaler.transform(self._extract_tabular_x(ds))
        dl_test = self._build_dataloader(ds, tabular_x, shuffle=False)

        training_module = TrainingModuleImageTabular(
            self.model, **self.torch_model_params.train_params
        )
        trainer = Trainer(accelerator=self.torch_model_params.accelerator, devices=1)
        preds_list = trainer.predict(training_module, dl_test)
        assert preds_list is not None, "Returned empty output"
        return concatenate(preds_list, axis=0)

    def save(self, file_path: str) -> None:
        safe_mkdir(file_path)
        path = os.path.join(file_path, "weights.pth")
        save(self.model.state_dict(), path)
        dump(self.scaler, os.path.join(file_path, "scaler.sav"))

    def load(self, path: str) -> None:
        weights_path = os.path.join(path, "weights.pth")
        self.model.load_state_dict(torch.load(weights_path))
        self.scaler = load(os.path.join(path, "scaler.sav"))

    def cv(self, cv_params: dict[str, T.Any], ds: Dataset) -> None:
        raise NotImplementedError("CV not implemented for LSTMCNNImageModel")
