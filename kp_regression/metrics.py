import typing as T

from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy.typing import NDArray
from pandas import DataFrame


def calculate_regression_metrics(
    preds: NDArray, y_true: NDArray, meta: DataFrame
) -> T.List[dict]:

    results = []

    for i in range(y_true.shape[1]):

        metrics = {"horizon": i}

        pred_i = preds[:, i]
        y_true_i = y_true[:, i]

        metrics["MAE"] = mean_absolute_error(y_true_i, pred_i)
        metrics["MSE"] = mean_squared_error(y_true_i, pred_i)

        mask_t0 = (meta["hour_type"] == "T0").values
        mask_t1 = (meta["hour_type"] == "T1").values
        mask_t2 = (meta["hour_type"] == "T2").values

        metrics["MAE_T0"] = mean_absolute_error(y_true_i[mask_t0], pred_i[mask_t0])
        metrics["MSE_T0"] = mean_squared_error(y_true_i[mask_t0], pred_i[mask_t0])

        metrics["MAE_T1"] = mean_absolute_error(y_true_i[mask_t1], pred_i[mask_t1])
        metrics["MSE_T1"] = mean_squared_error(y_true_i[mask_t1], pred_i[mask_t1])

        metrics["MAE_T2"] = mean_absolute_error(y_true_i[mask_t2], pred_i[mask_t2])
        metrics["MSE_T2"] = mean_squared_error(y_true_i[mask_t2], pred_i[mask_t2])

        results.append(metrics)

    return results
