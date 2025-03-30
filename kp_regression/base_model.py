import typing as T
from numpy.typing import NDArray

from abc import ABC, abstractmethod

from kp_regression.data_pipe import Dataset


class BaseModel(ABC):

    def __init__(
        self,
        shape: tuple,
        features: T.Optional[T.List[str]],
        output_shape: tuple,
        model_params: T.Dict,
        model_dir: str,
    ) -> None:

        self.features = features
        self.model_params = model_params
        self.shape = shape
        self.model_dir = model_dir
        self.output_shape = output_shape

        self.build()

    @abstractmethod
    def build(self) -> None:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def train(self, ds: Dataset, ds_val: T.Optional[Dataset] = None) -> None:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def save(self, file_path: str) -> None:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def predict(self, ds: Dataset) -> NDArray:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def cv(self, cv_params: T.Dict, ds: Dataset):
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplemented("Method not implemented")
