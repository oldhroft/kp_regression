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
    X: T.Union[NDArray, T.Tuple[NDArray, ...]]
    y: T.Optional[NDArray]
    feature_names: T.Any
    target_names: T.Any
    meta: DataFrame
    shape: tuple

    def save(self, path: str, names_only: bool = True):

        safe_mkdir(path)

        features_path = os.path.join(path, "features.json")
        target_path = os.path.join(path, "targets.json")
        dump_json(self.feature_names, features_path)
        dump_json(self.target_names, target_path)

        if not names_only:
            out_path = os.path.join(path, "data.npz")
            if not isinstance(self.X, tuple):
                savez_compressed(out_path, X=self.X, y=self.y)
            else:
                save_args = {f"X{i}": x for i, x in enumerate(self.X)}
                savez_compressed(out_path, y=self.y, **save_args)

            meta_path = os.path.join(path, "meta.csv")
            self.meta.to_csv(meta_path, index=None)

    def log(self, name):

        if not isinstance(self.X, tuple):
            logging.info(
                "Dataset %s, X shape = %s, y shape = %s",
                name,
                self.X.shape,
                self.y.shape,
            )
        else:
            log_string = ", ".join(f"X{i} shape = %s" for i in range(len(self.X)))
            logging.info(
                "Dataset %s, y shape %s, " + log_string,
                name,
                self.y.shape,
                *(x.shape for x in self.X),
            )
        logging.info(
            "Dataset %s, Min dttm %s, Max dttm %s",
            name,
            self.meta.dttm.dt.date.min(),
            self.meta.dttm.dt.date.max(),
        )


class BaseData(ABC):

    def __init__(
        self,
        input_path: T.Union[str, T.Dict[str, str]],
        save_data: bool,
        pipe_params: dict,
        exp_dir: str,
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

    # anchor to the beginning of the interval
    data["dttm"] = data.apply(
        lambda y: datetime.datetime(
            int(y.year), int(y.month), int(y.day), int(y["hour from"]), 0
        ),
        axis=1,
    )

    return (
        data.drop("Unnamed: 62", axis=1, errors="ignore")
        .sort_values(by="dttm")
        .reset_index(drop=True)
    )


class KpData(BaseData):

    def _read_data(self) -> DataFrame:
        if isinstance(self.input_path, str):
            self.raw_data = read_data(self.input_path)
        else:
            raise ValueError("Only single input path supported")

    @abstractmethod
    def process_data(self, df: DataFrame, is_train: bool, **kwargs) -> Dataset: ...

    def get_train_test(
        self, year_test: int, year_val: int
    ) -> T.Tuple[Dataset, Dataset]:

        self._read_data()

        raw_data_train = self.raw_data[self.raw_data.year < year_test].reset_index(
            drop=True
        )
        raw_data_test = self.raw_data[self.raw_data.year >= year_test].reset_index(
            drop=True
        )

        data_train = self.process_data(
            raw_data_train, is_train=True, **self.pipe_params
        )
        data_train.log("Train")
        data_test = self.process_data(raw_data_test, is_train=False, **self.pipe_params)
        data_test.log("Test")

        data_train.save(
            os.path.join(self.exp_dir, "data_train"), names_only=not self.save_data
        )
        data_test.save(
            os.path.join(self.exp_dir, "data_test"), names_only=not self.save_data
        )

        return data_train, data_test

    def get_train_test_val(
        self, year_test: int, year_val: int
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

        data_train = self.process_data(
            raw_data_train, is_train=True, **self.pipe_params
        )
        data_train.log("Train")
        data_test = self.process_data(raw_data_test, is_train=False, **self.pipe_params)
        data_test.log("Test")
        data_val = self.process_data(raw_data_val, is_train=False, **self.pipe_params)
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


@dataclass
class KpData5mConfig:
    path_base: str
    path_5m: str
    path_1h: str


class KpData5m(BaseData):

    def _read_data(self):

        from pandas import read_parquet

        if isinstance(self.input_path, dict):
            path_cfg = KpData5mConfig(**self.input_path)

            self.raw_data_base = read_data(path_cfg.path_base)
            self.raw_data_5m = read_parquet(path_cfg.path_5m)
            self.raw_data_1h = read_parquet(path_cfg.path_1h)
            self.raw_data_5m["year"] = self.raw_data_5m.dttm.dt.year
            self.raw_data_1h["year"] = self.raw_data_1h.dttm.dt.year
        else:
            raise ValueError("Path should be config KpData5mAggConfig")

    @abstractmethod
    def process_data(
        self,
        df: DataFrame,
        df_1h: DataFrame,
        df_5min: DataFrame,
        is_train: bool,
        **kwargs,
    ) -> Dataset: ...

    def get_train_test(
        self, year_test: int, year_val: int
    ) -> T.Tuple[Dataset, Dataset]:

        self._read_data()

        raw_data_base_train = self.raw_data_base[
            self.raw_data_base.year < year_test
        ].reset_index(drop=True)
        raw_data_base_test = self.raw_data_base[
            self.raw_data_base.year >= year_test
        ].reset_index(drop=True)

        raw_data_5m_train = self.raw_data_5m[
            self.raw_data_5m.year < year_test
        ].reset_index(drop=True)
        raw_data_5m_test = self.raw_data_5m[
            self.raw_data_5m.year >= year_test
        ].reset_index(drop=True)

        raw_data_1h_train = self.raw_data_1h[
            self.raw_data_1h.year < year_test
        ].reset_index(drop=True)
        raw_data_1h_test = self.raw_data_1h[
            self.raw_data_1h.year >= year_test
        ].reset_index(drop=True)

        data_train = self.process_data(
            raw_data_base_train,
            raw_data_1h_train,
            raw_data_5m_train,
            is_train=True,
            **self.pipe_params,
        )
        data_train.log("Train")
        data_test = self.process_data(
            raw_data_base_test,
            raw_data_1h_test,
            raw_data_5m_test,
            is_train=False,
            **self.pipe_params,
        )
        data_test.log("Test")

        data_train.save(
            os.path.join(self.exp_dir, "data_train"), names_only=not self.save_data
        )
        data_test.save(
            os.path.join(self.exp_dir, "data_test"), names_only=not self.save_data
        )

        return data_train, data_test

    def get_train_test_val(
        self, year_test: int, year_val: int
    ) -> T.Tuple[Dataset, Dataset, Dataset]:

        self._read_data()

        raw_data_train = self.raw_data_base[
            self.raw_data_base.year < year_val
        ].reset_index(drop=True)
        raw_data_val = self.raw_data_base[
            (self.raw_data_base.year >= year_val)
            & (self.raw_data_base.year < year_test)
        ].reset_index(drop=True)
        raw_data_test = self.raw_data_base[
            (self.raw_data_base.year >= year_test)
        ].reset_index(drop=True)

        raw_data_5m_train = self.raw_data_5m[
            self.raw_data_5m.year < year_test
        ].reset_index(drop=True)
        raw_data_5m_val = self.raw_data_5m[
            (self.raw_data_5m.year >= year_val) & (self.raw_data_5m.year < year_test)
        ].reset_index(drop=True)
        raw_data_5m_test = self.raw_data_5m[
            self.raw_data_5m.year >= year_test
        ].reset_index(drop=True)

        raw_data_1h_train = self.raw_data_1h[
            self.raw_data_1h.year < year_val
        ].reset_index(drop=True)
        raw_data_1h_val = self.raw_data_1h[
            (self.raw_data_1h.year >= year_val) & (self.raw_data_1h.year < year_test)
        ].reset_index(drop=True)
        raw_data_1h_test = self.raw_data_1h[
            self.raw_data_1h.year >= year_test
        ].reset_index(drop=True)

        data_train = self.process_data(
            raw_data_train, raw_data_1h_train, raw_data_5m_train, is_train=True, **self.pipe_params
        )
        data_train.log("Train")
        data_test = self.process_data(
            raw_data_test, raw_data_1h_test, raw_data_5m_test, is_train=False, **self.pipe_params
        )
        data_test.log("Test")
        data_val = self.process_data(
            raw_data_val, raw_data_1h_val, raw_data_5m_val, is_train=False, **self.pipe_params
        )
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
