import typing as T

from kp_regression.models_zoo.skmodels import *
from kp_regression.base_model import BaseModel

from kp_regression.models_zoo.perceptron import MLPClass

MODEL_FACTORY: T.Dict[str, T.Type[BaseModel]] = {
    "lgbm_regressor": LGBMRegressorClass,
    "catboost_regressor": CatBoostRegressorClass,
    "ridge_regressor": RidgeClass,
    "lasso_regressor": LassoClass,
    "rf_regressor": RandomForestRegressorClass,
    "column_estimator": ColumnEstimatorClass,
    "mlp": MLPClass
}
