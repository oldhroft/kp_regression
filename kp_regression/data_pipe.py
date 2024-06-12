import typing as T
from numpy.typing import NDArray

import datetime
import os
from pandas import DataFrame, read_csv

from numpy import savez_compressed
from dataclasses import dataclass

from abc import ABC, abstractmethod

from kp_regression.utils import dump_json, safe_mkdir

import logging


@dataclass
class Dataset:
    X: NDArray
    y: NDArray
    feature_names: T.List[str]
    target_names: T.List[str]
    meta: DataFrame

    def save(self, path: str, names_only: bool = True):

        safe_mkdir(path)

        features_path = os.path.join(path, "features.json")
        target_path = os.path.join(path, "targets.json")
        dump_json(self.feature_names, features_path)
        dump_json(self.target_names, target_path)

        if not names_only:
            out_path = os.path.join(path, "data.npz")
            savez_compressed(out_path, X=self.X, y=self.y)

            meta_path = os.path.join(path, "meta.csv")
            self.meta.to_csv(meta_path, index=None)

    def log(self, name):
        logging.info(
            "Dataset %s, X shape = %s, y shape = %s", name, self.X.shape, self.y.shape
        )
        logging.info(
            "Dataset %s, Min dttm %s, Max dttm %s",
            name,
            self.meta.dttm.dt.date.min(),
            self.meta.dttm.dt.date.max(),
        )


class BaseData(ABC):

    def __init__(
        self, input_path: str, save_data: bool, pipe_params: dict, exp_dir: str
    ) -> None:
        self.input_path = input_path
        self.save_data = save_data
        self.pipe_params = pipe_params
        self.exp_dir = exp_dir

    @abstractmethod
    def get_train_test(self, **kwargs) -> T.Tuple[Dataset, Dataset]: ...

    @abstractmethod
    def get_train_test_val(self, **kwargs) -> T.Tuple[Dataset, Dataset, Dataset]: ...


def read_data(path: str) -> DataFrame:
    data = read_csv(path, encoding="cp1251", na_values="N")

    data["dttm"] = data.apply(
        lambda y: datetime.datetime(
            int(y.year), int(y.month), int(y.day), int(y["hour from"]), 0
        ),
        axis=1,
    )

    return (
        data.drop("Unnamed: 62", axis=1)
        .sort_values(by="dttm")
        .reset_index(drop=True)
    )


class KpData(BaseData):

    def _read_data(self) -> DataFrame:
        self.raw_data = read_data(self.input_path)

    @abstractmethod
    def process_data(self, df: DataFrame, params: dict) -> Dataset: ...

    def get_train_test(self, year_test, year_val) -> T.Tuple[Dataset, Dataset]:

        self._read_data()

        raw_data_train = self.raw_data[self.raw_data.year < year_test].reset_index(
            drop=True
        )
        raw_data_test = self.raw_data[self.raw_data.year >= year_test].reset_index(
            drop=True
        )

        data_train = self.process_data(raw_data_train, **self.pipe_params)
        data_train.log("Train")
        data_test = self.process_data(raw_data_test, **self.pipe_params)
        data_test.log("Test")

        data_train.save(
            os.path.join(self.exp_dir, "data_train"), names_only=not self.save_data
        )
        data_test.save(
            os.path.join(self.exp_dir, "data_test"), names_only=not self.save_data
        )

        return data_train, data_test

    def get_train_test_val(
        self, year_test, year_val
    ) -> T.Tuple[Dataset, Dataset, Dataset]:

        self._read_data()

        raw_data_train = self.raw_data[self.raw_data.year < year_val].reset_index(
            drop=True
        )
        raw_data_val = self.raw_data[
            (self.raw_data.year >= year_val) & (self.raw_data.year < year_test)
        ].reset_index(drop=True)
        raw_data_test = self.raw_data[(self.raw_data.year >= year_test)].reset_index(
            drop=True
        )

        data_train = self.process_data(raw_data_train, **self.pipe_params)
        data_train.log("Train")
        data_test = self.process_data(raw_data_test, **self.pipe_params)
        data_test.log("Test")
        data_val = self.process_data(raw_data_val, **self.pipe_params)
        data_val.log("Val")

        data_train.save(
            os.path.join(self.exp_dir, "data_train"), names_only=not self.save_data
        )
        data_test.save(
            os.path.join(self.exp_dir, "data_test"), names_only=not self.save_data
        )
        data_val.save(
            os.path.join(self.exp_dir, "data_val"), names_only=not self.save_data
        )

        return data_train, data_test, data_val
