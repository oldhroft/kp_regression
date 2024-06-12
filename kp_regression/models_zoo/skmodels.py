from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from kp_regression.base_model import SklearnMultiOutputModel


class LGBMRegressorClass(SklearnMultiOutputModel):
    def get_model(self) -> BaseEstimator:
        return LGBMRegressor()


class CatBoostRegressorClass(SklearnMultiOutputModel):
    def get_model(self) -> BaseEstimator:
        return CatBoostRegressor()


class RandomForestRegressorClass(SklearnMultiOutputModel):
    def get_model(self) -> BaseEstimator:
        return RandomForestRegressor()


class RidgeClass(SklearnMultiOutputModel):
    def get_model(self) -> None:

        steps = [("scaler", StandardScaler()), ("ridge", Ridge())]

        return Pipeline(steps=steps)


class LassoClass(SklearnMultiOutputModel):
    def get_model(self) -> None:
        self.model = Lasso()
