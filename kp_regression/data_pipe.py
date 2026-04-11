import datetime
import logging
import os
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy import savez_compressed
from numpy.typing import NDArray
from pandas import DataFrame, read_csv

from kp_regression.utils import dump_json, safe_mkdir


@dataclass
class Dataset:
    X: NDArray | tuple[NDArray, ...]
    y: NDArray | None
    feature_names: T.Any
    target_names: T.Any
    meta: DataFrame
    shape: tuple
    image_paths: DataFrame | None = None

    def save(self, path: str, names_only: bool = True) -> None:
        safe_mkdir(path)

        features_path = os.path.join(path, "features.json")
        target_path = os.path.join(path, "targets.json")
        dump_json(self.feature_names, features_path)
        dump_json(self.target_names, target_path)

        if not names_only:
            out_path = os.path.join(path, "data.npz")

            if isinstance(self.X, tuple):
                save_args = {f"X{i}": x for i, x in enumerate(self.X)}
            else:
                save_args = {"X": self.X}
            if self.y is not None:
                save_args["y"] = self.y

            savez_compressed(out_path, **save_args)  # ty: ignore[invalid-argument-type]

            meta_path = os.path.join(path, "meta.csv")
            self.meta.to_csv(meta_path, index=False)

            if self.image_paths is not None:
                image_paths_path = os.path.join(path, "image_paths.csv")
                self.image_paths.to_csv(image_paths_path, index=False)

    def log(self, name: str):
        y_shape: tuple | None = None
        if self.y is not None:
            y_shape = self.y.shape

        if not isinstance(self.X, tuple):
            logging.info(
                "Dataset %s, X shape = %s, y shape = %s", name, self.X.shape, y_shape
            )
        else:
            log_string = ", ".join(f"X{i} shape = %s" for i in range(len(self.X)))
            logging.info(
                "Dataset %s, y shape %s, " + log_string,
                name,
                y_shape,
                *(x.shape for x in self.X),
            )

        logging.info(
            "Dataset %s, Min dttm %s, Max dttm %s",
            name,
            self.meta.dttm.dt.date.min(),
            self.meta.dttm.dt.date.max(),
        )

        if self.image_paths is not None:
            logging.info(
                "Dataset %s, image_paths shape = %s",
                name,
                self.image_paths.shape,
            )


class BaseData(ABC):
    def __init__(
        self,
        input_path: str | dict[str, str],
        save_data: bool,
        pipe_params: dict,
        exp_dir: str,
    ) -> None:
        self.input_path = input_path
        self.save_data = save_data
        self.pipe_params = pipe_params
        self.exp_dir = exp_dir

    @abstractmethod
    def get_train_test(
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset]: ...

    @abstractmethod
    def get_train_test_val(
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset, Dataset]: ...


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
    def _read_data(self) -> None:
        if isinstance(self.input_path, str):
            self.raw_data = read_data(self.input_path)
        else:
            raise ValueError("Only single input path supported")

    @abstractmethod
    def process_data(self, df: DataFrame, is_train: bool, **kwargs) -> Dataset: ...

    def get_train_test(
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset]:
        self._read_data()

        mask_train = self.raw_data.year < year_test
        if train_date_from is not None:
            mask_train = mask_train & (self.raw_data.dttm >= train_date_from)

        raw_data_train = self.raw_data[mask_train].reset_index(drop=True)
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
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset, Dataset]:
        self._read_data()

        mask_train = self.raw_data.year < year_val
        if train_date_from is not None:
            mask_train = mask_train & (self.raw_data.dttm >= train_date_from)

        raw_data_train = self.raw_data[mask_train].reset_index(drop=True)
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
            self.raw_data_5m = (
                read_parquet(path_cfg.path_5m)
                .sort_values(by="dttm")
                .reset_index(drop=True)
            )
            self.raw_data_1h = (
                read_parquet(path_cfg.path_1h)
                .sort_values(by="dttm")
                .reset_index(drop=True)
            )

            self.raw_data_5m["year"] = self.raw_data_5m.dttm.dt.year
            self.raw_data_1h["year"] = self.raw_data_1h.dttm.dt.year
        else:
            raise ValueError("Path should be config KpData5mAggConfig")

    @abstractmethod
    def process_data(
        self,
        df: DataFrame,
        df_1h: DataFrame,
        df_5m: DataFrame,
        is_train: bool,
        **kwargs,
    ) -> Dataset: ...

    def get_train_test(
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset]:
        self._read_data()

        mask_base_train = self.raw_data_base.year < year_test
        mask_5m_train = self.raw_data_5m.year < year_test
        mask_1h_train = self.raw_data_1h.year < year_test

        if train_date_from is not None:
            mask_base_train = mask_base_train & (
                self.raw_data_base.dttm >= train_date_from
            )
            mask_5m_train = mask_5m_train & (
                self.raw_data_5m.dttm >= train_date_from
            )
            mask_1h_train = mask_1h_train & (
                self.raw_data_1h.dttm >= train_date_from
            )

        raw_data_base_train = self.raw_data_base[mask_base_train].reset_index(drop=True)
        raw_data_base_test = self.raw_data_base[
            self.raw_data_base.year >= year_test
        ].reset_index(drop=True)

        raw_data_5m_train = self.raw_data_5m[mask_5m_train].reset_index(drop=True)
        raw_data_5m_test = self.raw_data_5m[
            self.raw_data_5m.year >= year_test
        ].reset_index(drop=True)

        raw_data_1h_train = self.raw_data_1h[mask_1h_train].reset_index(drop=True)
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
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset, Dataset]:
        self._read_data()

        mask_train = self.raw_data_base.year < year_val
        if train_date_from is not None:
            mask_train = mask_train & (self.raw_data_base.dttm >= train_date_from)

        raw_data_train = self.raw_data_base[mask_train].reset_index(drop=True)
        raw_data_val = self.raw_data_base[
            (self.raw_data_base.year >= year_val)
            & (self.raw_data_base.year < year_test)
        ].reset_index(drop=True)
        raw_data_test = self.raw_data_base[
            (self.raw_data_base.year >= year_test)
        ].reset_index(drop=True)

        mask_5m_train = self.raw_data_5m.year < year_val
        mask_1h_train = self.raw_data_1h.year < year_val
        if train_date_from is not None:
            mask_5m_train = mask_5m_train & (
                self.raw_data_5m.dttm >= train_date_from
            )
            mask_1h_train = mask_1h_train & (
                self.raw_data_1h.dttm >= train_date_from
            )

        raw_data_5m_train = self.raw_data_5m[mask_5m_train].reset_index(drop=True)
        raw_data_5m_val = self.raw_data_5m[
            (self.raw_data_5m.year >= year_val) & (self.raw_data_5m.year < year_test)
        ].reset_index(drop=True)
        raw_data_5m_test = self.raw_data_5m[
            self.raw_data_5m.year >= year_test
        ].reset_index(drop=True)

        raw_data_1h_train = self.raw_data_1h[mask_1h_train].reset_index(drop=True)
        raw_data_1h_val = self.raw_data_1h[
            (self.raw_data_1h.year >= year_val) & (self.raw_data_1h.year < year_test)
        ].reset_index(drop=True)
        raw_data_1h_test = self.raw_data_1h[
            self.raw_data_1h.year >= year_test
        ].reset_index(drop=True)

        data_train = self.process_data(
            raw_data_train,
            raw_data_1h_train,
            raw_data_5m_train,
            is_train=True,
            **self.pipe_params,
        )
        data_train.log("Train")
        data_test = self.process_data(
            raw_data_test,
            raw_data_1h_test,
            raw_data_5m_test,
            is_train=False,
            **self.pipe_params,
        )
        data_test.log("Test")
        data_val = self.process_data(
            raw_data_val,
            raw_data_1h_val,
            raw_data_5m_val,
            is_train=False,
            **self.pipe_params,
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


@dataclass
class KpData5mWithImagesConfig:
    path_base: str
    path_5m: str
    path_1h: str
    path_images: str


class KpData5mWithImages(BaseData):
    def _read_data(self) -> None:
        from pandas import Timedelta, read_parquet

        if not isinstance(self.input_path, dict):
            raise ValueError("Path should be config KpData5mWithImagesConfig")

        path_cfg = KpData5mWithImagesConfig(**self.input_path)

        self.raw_data_base = read_data(path_cfg.path_base)
        self.raw_data_5m = (
            read_parquet(path_cfg.path_5m).sort_values(by="dttm").reset_index(drop=True)
        )
        self.raw_data_1h = (
            read_parquet(path_cfg.path_1h).sort_values(by="dttm").reset_index(drop=True)
        )
        self.raw_data_images = read_parquet(path_cfg.path_images)
        self.raw_data_images["dttm"] = self.raw_data_images["datetime"] - Timedelta(
            hours=1
        )
        self.raw_data_images["year"] = self.raw_data_images["dttm"].dt.year

        self.raw_data_5m["year"] = self.raw_data_5m.dttm.dt.year
        self.raw_data_1h["year"] = self.raw_data_1h.dttm.dt.year

    @abstractmethod
    def process_data(
        self,
        df: DataFrame,
        df_1h: DataFrame,
        df_5m: DataFrame,
        df_images: DataFrame,
        is_train: bool,
        **kwargs: T.Any,
    ) -> Dataset: ...

    def _split_by_year(
        self, year_from: int | None, year_to: int | None
    ) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        if year_from is not None and year_to is not None:
            mask_base = (self.raw_data_base.year >= year_from) & (
                self.raw_data_base.year < year_to
            )
            mask_5m = (self.raw_data_5m.year >= year_from) & (
                self.raw_data_5m.year < year_to
            )
            mask_1h = (self.raw_data_1h.year >= year_from) & (
                self.raw_data_1h.year < year_to
            )
            mask_img = (self.raw_data_images.year >= year_from) & (
                self.raw_data_images.year < year_to
            )
        elif year_from is not None:
            mask_base = self.raw_data_base.year >= year_from
            mask_5m = self.raw_data_5m.year >= year_from
            mask_1h = self.raw_data_1h.year >= year_from
            mask_img = self.raw_data_images.year >= year_from
        elif year_to is not None:
            mask_base = self.raw_data_base.year < year_to
            mask_5m = self.raw_data_5m.year < year_to
            mask_1h = self.raw_data_1h.year < year_to
            mask_img = self.raw_data_images.year < year_to
        else:
            raise ValueError("At least one of year_from or year_to must be set")

        return (
            self.raw_data_base[mask_base].reset_index(drop=True),
            self.raw_data_5m[mask_5m].reset_index(drop=True),
            self.raw_data_1h[mask_1h].reset_index(drop=True),
            self.raw_data_images[mask_img].reset_index(drop=True),
        )

    def get_train_test(
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset]:
        self._read_data()

        base_tr, d5m_tr, d1h_tr, img_tr = self._split_by_year(None, year_test)
        if train_date_from is not None:
            base_tr = base_tr[base_tr.dttm >= train_date_from].reset_index(drop=True)
            d5m_tr = d5m_tr[d5m_tr.dttm >= train_date_from].reset_index(drop=True)
            d1h_tr = d1h_tr[d1h_tr.dttm >= train_date_from].reset_index(drop=True)
            img_tr = img_tr[img_tr.dttm >= train_date_from].reset_index(drop=True)
        base_te, d5m_te, d1h_te, img_te = self._split_by_year(year_test, None)

        data_train = self.process_data(
            base_tr, d1h_tr, d5m_tr, img_tr, is_train=True, **self.pipe_params
        )
        data_train.log("Train")
        data_test = self.process_data(
            base_te, d1h_te, d5m_te, img_te, is_train=False, **self.pipe_params
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
        self, year_test: int, year_val: int, train_date_from: str | None = None
    ) -> tuple[Dataset, Dataset, Dataset]:
        self._read_data()

        base_tr, d5m_tr, d1h_tr, img_tr = self._split_by_year(None, year_val)
        if train_date_from is not None:
            base_tr = base_tr[base_tr.dttm >= train_date_from].reset_index(drop=True)
            d5m_tr = d5m_tr[d5m_tr.dttm >= train_date_from].reset_index(drop=True)
            d1h_tr = d1h_tr[d1h_tr.dttm >= train_date_from].reset_index(drop=True)
            img_tr = img_tr[img_tr.dttm >= train_date_from].reset_index(drop=True)
        base_val, d5m_val, d1h_val, img_val = self._split_by_year(year_val, year_test)
        base_te, d5m_te, d1h_te, img_te = self._split_by_year(year_test, None)

        data_train = self.process_data(
            base_tr, d1h_tr, d5m_tr, img_tr, is_train=True, **self.pipe_params
        )
        data_train.log("Train")
        data_test = self.process_data(
            base_te, d1h_te, d5m_te, img_te, is_train=False, **self.pipe_params
        )
        data_test.log("Test")
        data_val = self.process_data(
            base_val, d1h_val, d5m_val, img_val, is_train=False, **self.pipe_params
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
