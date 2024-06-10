import typing as T

from abc import ABC, abstractmethod

from numpy.typing import NDArray
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    TimeSeriesSplit,
)
from sklearn.base import BaseEstimator

from joblib import dump
import os

from kp_regression.utils import dump_json


class BaseModel(ABC):

    def __init__(
        self,
        shape: tuple,
        features: T.Optional[T.List[str]],
        model_params: T.Dict,
        model_dir: str,
    ) -> None:

        self.features = features
        self.model_params = model_params
        self.shape = shape
        self.model_dir = model_dir

        self.build()

    @abstractmethod
    def build(self) -> None:
        raise NotImplemented("Method not implemented")

    @abstractmethod
    def train(self, X: NDArray, y: NDArray) -> None:
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


class SklearnModel(BaseModel):

    @abstractmethod
    def get_model(self) -> BaseEstimator: ...

    def build(self) -> None:
        self.model = self.get_model()
        self.model.set_params(**self.model_params)

    def save(self, file_path: str) -> None:
        dump(self.model, file_path)

    def train(self, X: NDArray, y: NDArray):
        self.model.fit(X, y)

    def predict(self, X: NDArray) -> NDArray:
        return self.model.predict(X)

    def cv(self, cv_params: T.Dict, X: NDArray, y: NDArray):

        if cv_params["cv_split_type"] == "fold":
            kf = KFold(**cv_params["cv_split_params"])
        elif cv_params["cv_split_type"] == "tss":
            kf = TimeSeriesSplit(**cv_params["cv_split_params"])
        else:
            raise ValueError("Param not specified")

        if cv_params["search_type"] == "gs":
            self.gcv = GridSearchCV(self.model, cv=kf, **cv_params["search_cfg"])
        elif cv_params["search_type"] == "rs":
            self.gcv = RandomizedSearchCV(self.model, cv=kf, **cv_params["search_cfg"])

        self.gcv.fit(X, y)
        self.model = self.gcv.best_estimator_

        results_path = os.path.join(self.model_dir, "cv_results.json")

        dump_json(self.model, results_path)
