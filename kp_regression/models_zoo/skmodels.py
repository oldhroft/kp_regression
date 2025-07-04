from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import os

from kp_regression.utils import safe_mkdir

from kp_regression.models_zoo.column_estimator import ColumnEstimator
from kp_regression.models_zoo.sklearn_models import (
    SklearnMultiOutputModel,
    BoostingValModel,
)


class LGBMRegressorClass(SklearnMultiOutputModel):
    def get_model(self) -> BaseEstimator:
        return LGBMRegressor()


class LGBMRegressorValClass(BoostingValModel):
    early_stopping_in_fit = False

    def get_model(self, i: int) -> BaseEstimator:
        return LGBMRegressor()


class CatBoostRegressorClass(SklearnMultiOutputModel):
    def get_model(self) -> BaseEstimator:
        dirpath = os.path.join(self.model_dir, "hist")

        safe_mkdir(dirpath)
        return CatBoostRegressor(train_dir=dirpath)


class CatBoostRegressorValClass(BoostingValModel):
    def get_model(self, i: int) -> BaseEstimator:
        dirpath = os.path.join(self.model_dir, "hist")

        safe_mkdir(dirpath)

        histpath = os.path.join(dirpath, str(i))

        safe_mkdir(histpath)

        return CatBoostRegressor(train_dir=histpath)


class RandomForestRegressorClass(SklearnMultiOutputModel):
    def get_model(self) -> BaseEstimator:
        return RandomForestRegressor()


class RidgeClass(SklearnMultiOutputModel):
    def get_model(self) -> None:

        steps = [("scaler", StandardScaler()), ("ridge", Ridge())]

        return Pipeline(steps=steps)


class LassoClass(SklearnMultiOutputModel):
    def get_model(self) -> None:
        return Lasso()


class ColumnEstimatorClass(SklearnMultiOutputModel):

    def get_model(self) -> BaseEstimator:
        return ColumnEstimator()
