from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from kp_regression.base_model import SklearnModel


class LGBMRegressorClass(SklearnModel):
    def get_model(self) -> BaseEstimator:
        return LGBMRegressor()


class CatBoostRegressorClass(SklearnModel):
    def get_model(self) -> BaseEstimator:
        return CatBoostRegressor()


class RandomForestRegressorClass(SklearnModel):
    def get_model(self) -> BaseEstimator:
        return RandomForestRegressor()


class RidgeClass(SklearnModel):
    def get_model(self) -> None:

        steps = [("scaler", StandardScaler()), ("ridge", Ridge())]

        return Pipeline(steps=steps)


class LassoClass(SklearnModel):
    def get_model(self) -> None:
        self.model = Lasso()
