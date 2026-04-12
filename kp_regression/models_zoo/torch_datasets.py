import logging
import typing as T

import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, from_numpy
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Compose, Resize, ToTensor


class KpImageTabularDataset(TorchDataset):
    def __init__(
        self,
        tabular_x: NDArray,
        image_path_columns: list[list[str]],
        image_paths_df: T.Any,
        y: NDArray | None,
        n_categories: int,
        n_lags: int,
        crop: int = 25,
        img_resize: int = 128,
    ) -> None:
        self.tabular_x = tabular_x
        self.image_path_columns = image_path_columns
        self.image_paths_df = image_paths_df
        self.y = y
        self.n_categories = n_categories
        self.n_lags = n_lags
        self.crop = crop
        self.img_resize = img_resize

        self.transform = Compose(
            [
                Resize((img_resize, img_resize)),
                ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.tabular_x)

    def _load_image(self, path: str | None) -> Tensor:
        if path is None or (isinstance(path, float) and path != path):
            return torch.zeros(3, self.img_resize - 2 * self.crop, self.img_resize)

        try:
            img = Image.open(path).convert("RGB")
        except (OSError, SyntaxError):
            logging.warning("Failed to load image: %s", path)
            return torch.zeros(3, self.img_resize - 2 * self.crop, self.img_resize)

        tensor: Tensor = self.transform(img)
        if self.crop > 0:
            tensor = tensor[:, self.crop : -self.crop, :]
        return tensor

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:  # ty: ignore[invalid-method-override]
        tab = from_numpy(self.tabular_x[idx].astype("float32"))

        lag_tensors = []
        for lag in range(self.n_lags):
            cat_tensors = []
            for cat_cols in self.image_path_columns:
                col = cat_cols[lag]
                path = self.image_paths_df[col].iloc[idx]
                cat_tensors.append(self._load_image(path))
            lag_tensors.append(torch.cat(cat_tensors, dim=0))

        image_seq = torch.stack(lag_tensors, dim=0)

        if self.y is not None:
            y = from_numpy(self.y[idx].astype("float32"))
            return tab, image_seq, y
        return tab, image_seq
