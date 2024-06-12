import typing as T
from numpy.typing import NDArray

import datetime
import os
from pandas import DataFrame, read_csv

from numpy import savez_compressed

from abc import ABC, abstractmethod

from kp_regression.utils import dump_json


class BaseData(ABC):

    def __init__(
        self, input_path: str, save_data: bool, pipe_params: dict, exp_dir: str
    ) -> None:
        self.input_path = input_path
        self.save_data = save_data
        self.pipe_params = pipe_params
        self.exp_dir = exp_dir

    @abstractmethod
    def get_train_test(
        self, **kwargs
    ) -> T.Tuple[NDArray, NDArray, NDArray, NDArray]: ...

    @abstractmethod
    def get_train_test_val(
        self, **kwargs
    ) -> T.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]: ...

    @abstractmethod
    def get_features(self) -> T.List[str]: ...

    def get_data(self, val: bool, **kwargs) -> T.Any:

        if val:
            return self.get_train_test_val(**kwargs)
        else:
            return self.get_train_test(**kwargs)


def read_data(path: str) -> DataFrame:
    data = read_csv(path, encoding="cp1251", na_values="N")

    data["dttm"] = data.apply(
        lambda y: datetime.datetime(
            int(y.year), int(y.month), int(y.day), int(y["hour from"]), 0
        ),
        axis=1,
    )

    return data.drop("Unnamed: 62", axis=1)


class KpData(BaseData):

    def _read_data(self) -> DataFrame:
        self.raw_data = read_data(self.input_path)

    @abstractmethod
    def process_data(
        self, df: DataFrame, params: dict
    ) -> T.Tuple[T.List[str], NDArray, NDArray]: ...

    def get_train_test(self, year_test) -> T.Tuple[NDArray, NDArray, NDArray, NDArray]:

        raw_data_train = self.raw_data[self.raw_data.year < year_test].reset_index(
            drop=True
        )
        raw_data_test = self.raw_data[self.raw_data.year >= year_test].reset_index(
            drop=True
        )

        features, X_train, y_train = self.process_data(
            raw_data_train, params=self.pipe_params
        )
        _, X_test, y_test = self.process_data(raw_data_test, params=self.pipe_params)

        self.features = features

        features_path = os.path.join(self.exp_dir, "features.json")

        dump_json(self.features, features_path)

        if self.save_data:
            file_path = os.path.join(self.exp_dir, "data_train_test.npz")
            savez_compressed(
                file_path,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        return X_train, y_train, X_test, y_test

    def get_train_test_val(
        self, year_test, year_val
    ) -> T.Tuple[NDArray, NDArray, NDArray, NDArray]:

        raw_data_train = self.raw_data[self.raw_data.year < year_val].reset_index(
            drop=True
        )
        raw_data_val = self.raw_data[
            (self.raw_data.year <= year_val) & (self.raw_data.year < year_test)
        ].reset_index(drop=True)
        raw_data_test = self.raw_data[(self.raw_data.year >= year_test)].reset_index(
            drop=True
        )

        features, X_train, y_train = self.process_data(
            raw_data_train, params=self.pipe_params
        )
        _, X_test, y_test = self.process_data(raw_data_test, params=self.pipe_params)
        _, X_val, y_val = self.process_data(raw_data_val, params=self.pipe_params)

        self.features = features

        features_path = os.path.join(self.exp_dir, "features.json")

        dump_json(self.features, features_path)

        if self.save_data:
            file_path = os.path.join(self.exp_dir, "data_train_test_val.npz")
            savez_compressed(
                file_path,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X_val=X_val,
                y_val=y_val,
            )

        return X_train, y_train, X_test, y_test, X_val, y_val
