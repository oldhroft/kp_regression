import typing as T

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator

from numpy.typing import NDArray
from numpy import ndarray
from numpy import concatenate

from joblib import dump
import os
from abc import abstractmethod
from dataclasses import dataclass

from kp_regression.utils import dump_json, safe_mkdir, serialize_params
from kp_regression.base_model import BaseModel
from kp_regression.data_pipe import Dataset

import logging


class SklearnMultiOutputModel(BaseModel):

    @abstractmethod
    def get_model(self, i: int) -> BaseEstimator: ...

    def build(self) -> None:
        self.model = self.get_model()
        self.model.set_params(**self.model_params)
        self.multi_model = MultiOutputRegressor(self.model)

    def save(self, file_path: str) -> None:
        try:
            check_is_fitted(self.multi_model)
        except NotFittedError:
            logging.warn("Model is not fiited, not saving")
            return

        safe_mkdir(file_path)

        for i, model in enumerate(self.multi_model.estimators_):

            path = os.path.join(file_path, f"{i}.sav")
            dump(model, path)

        params_path = os.path.join(file_path, "params.json")
        dump_json(serialize_params(self.multi_model.estimators_[0]), params_path)

    def train(
        self,
        ds: Dataset,
        ds_val: T.Optional[Dataset] = None,
        # X: NDArray,
        # y: NDArray,
        # X_val: T.Optional[NDArray] = None,
        # y_val: T.Optional[NDArray] = None,
    ):
        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"
        self.multi_model.fit(ds.X, ds.y)

    def predict(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"
        return self.multi_model.predict(ds.X)

    def cv(self, cv_params: T.Dict, ds: Dataset):
        """A very hacky type of CV"""
        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"

        X = ds.X
        y = ds.y

        if cv_params["cv_split_type"] == "fold":
            kf = KFold(**cv_params["cv_split_cfg"])
        elif cv_params["cv_split_type"] == "tss":
            kf = TimeSeriesSplit(**cv_params["cv_split_cfg"])
        else:
            raise ValueError("Param not specified")

        if cv_params["search_type"] == "gs":
            self.gcv = GridSearchCV(self.model, cv=kf, **cv_params["search_cfg"])
        elif cv_params["search_type"] == "rs":
            self.gcv = RandomizedSearchCV(self.model, cv=kf, **cv_params["search_cfg"])

        logging.info("Started cross-validation")
        self.gcv.fit(X, y[:, 0])

        logging.info(
            "Best params %s, best score %s", self.gcv.best_params_, self.gcv.best_score_
        )
        results_path = os.path.join(self.model_dir, "cv_results.json")
        dump_json(self.gcv.cv_results_, results_path)
        self.model_params = self.gcv.best_params_
        logging.info("Rebuilding...")
        self.build()


@dataclass
class BoostingEvalConfig:
    model_params: dict
    val_frac: float
    early_stopping_rounds: T.Optional[int] = None


class BoostingValModel(BaseModel):
    early_stopping_in_fit = True

    @abstractmethod
    def get_model(self) -> BaseEstimator: ...

    def build(self) -> None:
        self.model_params = BoostingEvalConfig(**self.model_params)
        self.models = [
            self.get_model(i).set_params(**self.model_params.model_params)
            for i in range(self.output_shape[0])
        ]

    def train(self, ds: Dataset, ds_val: T.Optional[Dataset] = None):

        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"
        assert ds_val is None or isinstance(
            ds_val.X, ndarray
        ), "For sklearn models dataset should be Numpy"

        X, y = ds.X, ds.y

        X_val = None if ds_val is None else ds_val.X
        y_val = None if ds_val is None else ds_val.y

        if X_val is None and self.model_params.val_frac is not None:
            logging.info("Received val frac, creating val split")

            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.model_params.val_frac,
                random_state=17,
                shuffle=True,
            )

        logging.info("X shape %s", X.shape)
        logging.info("y shape %s", y.shape)

        if X_val is not None:
            logging.info("X val shape %s", X_val.shape)
            logging.info("y val shape %s", y_val.shape)
        else:
            raise ValueError("This is for val set only, sorry")

        for dim_i in range(self.output_shape[0]):

            logging.info("Fitting boosting for level %s", dim_i)

            params = {}

            if self.early_stopping_in_fit:
                params["early_stopping_rounds"] = (
                    self.model_params.early_stopping_rounds
                )

            self.models[dim_i].fit(
                X, y[:, dim_i], eval_set=[(X_val, y_val[:, dim_i])], **params
            )

    def save(self, file_path: str) -> None:
        safe_mkdir(file_path)
        for i, model in enumerate(self.models):
            path = os.path.join(file_path, f"{i}.sav")
            dump(model, path)

            params_path = os.path.join(file_path, f"params{i}.json")
            dump_json(serialize_params(model), params_path)

    def predict(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"

        total_preds = []

        for dim_i in range(self.output_shape[0]):
            logging.info("Predicting for dim %s", dim_i)

            total_preds.append(
                self.models[dim_i].predict(ds.X)[:, None],
            )

        return concatenate(total_preds, axis=1)

    def cv(self, cv_params: T.Dict, X: NDArray, y: NDArray):
        raise NotImplementedError("CV not implemented")
