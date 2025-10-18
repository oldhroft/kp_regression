import typing as T

from kp_regression.base_model import BaseModel
from kp_regression.models_zoo.conv import Conv1DNet3InputsMulti
from kp_regression.models_zoo.perceptron import MLPClass, MLPClassMulti
from kp_regression.models_zoo.rnn import LSTM3Inputs, LSTM4Inputs
from kp_regression.models_zoo.skmodels import *

MODEL_FACTORY: T.Dict[str, T.Type[BaseModel]] = {
    "lgbm_regressor": LGBMRegressorClass,
    "catboost_regressor": CatBoostRegressorClass,
    "ridge_regressor": RidgeClass,
    "lasso_regressor": LassoClass,
    "rf_regressor": RandomForestRegressorClass,
    "column_estimator": ColumnEstimatorClass,
    "mlp": MLPClass,
    "mlp_multi": MLPClassMulti,
    "lgbm_regressor_val": LGBMRegressorValClass,
    "catboost_regressor_val": CatBoostRegressorValClass,
    "conv1d": Conv1DNet3InputsMulti,
    "lstm": LSTM3Inputs,
    "lstm_5m": LSTM4Inputs
}
