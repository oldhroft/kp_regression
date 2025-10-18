import logging
import os
import typing as T
from abc import abstractmethod
from dataclasses import dataclass

from joblib import dump, load  # type: ignore
from numpy import concatenate, ndarray
from numpy.typing import NDArray
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore
from sklearn.model_selection import (  # type: ignore
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from kp_regression.base_model import BaseModel
from kp_regression.data_pipe import Dataset
from kp_regression.utils import dump_json, safe_mkdir, serialize_params


class SklearnMultiOutputModel(BaseModel):

    @abstractmethod
    def get_model(self) -> BaseEstimator: ...

    def build(self) -> None:
        self.model = self.get_model()
        self.model.set_params(**self.model_params)
        self.multi_model = MultiOutputRegressor(self.model)

    def save(self, file_path: str) -> None:
        try:
            check_is_fitted(self.multi_model)
        except NotFittedError:
            logging.warning("Model is not fiited, not saving")
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
    ):
        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"
        assert ds.y is not None and isinstance(
            ds.y, ndarray
        ), "For outputs should be present and be numpy"

        self.multi_model.fit(ds.X, ds.y)

    def predict(self, ds: Dataset) -> NDArray:
        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"
        return self.multi_model.predict(ds.X)

    def cv(self, cv_params: T.Dict[str, T.Any], ds: Dataset):
        """A very hacky type of CV"""
        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"

        X = ds.X
        y = ds.y

        assert y is not None, "Y should be not none"

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

    def load(self, dirpath: str) -> None:

        paths = [os.path.join(dirpath, f"{i}.sav") for i in range(self.output_shape[0])]
        models = map(load, paths)

        for i, model in enumerate(models):
            self.multi_model.estimators_[i] = model


@dataclass
class BoostingEvalConfig:
    model_params: dict
    val_frac: float
    early_stopping_rounds: T.Optional[int] = None


class BoostingValModel(BaseModel):
    early_stopping_in_fit = True

    @abstractmethod
    def get_model(self, i: int) -> BaseEstimator: ...

    def build(self) -> None:
        self.boosting_params = BoostingEvalConfig(**self.model_params)
        self.models = [
            self.get_model(i).set_params(**self.boosting_params.model_params)
            for i in range(self.output_shape[0])
        ]

    def train(self, ds: Dataset, ds_val: T.Optional[Dataset] = None):

        assert isinstance(ds.X, ndarray), "For sklearn models dataset should be Numpy"
        assert hasattr(self, "boosting_params"), "Model should be built prior to train"
        assert ds.y is not None and isinstance(
            ds.y, ndarray
        ), "For outputs should be present and be numpy"

        X, y = ds.X, ds.y

        # if ds_val is None:
        #     X_val =

        # X_val = None if ds_val is None else ds_val.X
        y_val = None if ds_val is None else ds_val.y

        if ds_val is None and self.boosting_params.val_frac is not None:
            logging.info("Received val frac, creating val split")

            split_result = tuple(
                train_test_split(
                    X,
                    y,
                    test_size=self.boosting_params.val_frac,
                    random_state=17,
                    shuffle=True,
                )
            )

            assert (
                len(split_result) == 4
            ), "Result of train-test split should contain exactly 4 items"

            split_result = T.cast(T.Tuple[NDArray, ...], split_result)

            X, X_val, y, y_val = split_result

        elif ds_val is not None:
            assert isinstance(
                ds_val.X, ndarray
            ), "For sklearn models dataset X should be Numpy"
            assert isinstance(
                ds_val.y, ndarray
            ), "For sklearn models dataset y should be Numpy"

            X_val, y_val = ds_val.X, ds_val.y

            assert isinstance(
                y_val, ndarray
            ), "Outputs should be present and be numpy (val set)"

        else:
            raise ValueError(
                "Either validation set should be provided, or parameter val_frac"
            )

        for dim_i in range(self.output_shape[0]):

            logging.info("Fitting boosting for level %s", dim_i)

            params = {}

            if self.early_stopping_in_fit:
                params["early_stopping_rounds"] = (
                    self.boosting_params.early_stopping_rounds
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

    def cv(self, cv_params: dict[str, T.Any], ds: Dataset):
        raise NotImplementedError("CV not implemented")

    def load(self, dirpath: str) -> None:

        paths = [os.path.join(dirpath, f"{i}.sav") for i in range(self.output_shape[0])]
        models = map(load, paths)

        for i, model in enumerate(models):
            self.models[i] = model
