import typing as T
from numpy.typing import NDArray

from abc import ABC, abstractmethod

import logging


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
    def train(
        self,
        X: NDArray,
        y: NDArray,
        X_val: T.Optional[NDArray] = None,
        y_val: T.Optional[NDArray] = None,
    ) -> None:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def save(self, file_path: str) -> None:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def cv(self, cv_params: T.Dict, X: NDArray, y: NDArray):
        raise NotImplemented("Method not implemented")