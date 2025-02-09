from numpy.typing import NDArray

import os
import logging

from kp_regression.models_zoo import MODEL_FACTORY
from kp_regression.config import Config, ModelConfig
from kp_regression.utils import load_json, dump_json
from kp_regression.data import DATA_FACTORY
from kp_regression.data_pipe import Dataset
from kp_regression.logging_utils import config_logger

from kp_regression.metrics import calculate_regression_metrics

logger = logging.getLogger()

config_logger(logger)


def get_stats(ds_test: Dataset, preds: NDArray) -> dict:

    metrics_by_year = []

    for year in (2022, 2023):
        mask = ds_test.meta.dttm.dt.year == year
        metrics = calculate_regression_metrics(
            preds[mask], ds_test.y[mask], meta=ds_test.meta[mask].reset_index(drop=True)
        )
        metrics_by_year.append({"year": year, "metrics": metrics})

    return metrics_by_year


def rerun_single_exp(path: str, model_dir: str):

    full_model_path = os.path.join(path, "models", model_dir)
    result = load_json(os.path.join(full_model_path, "result.json"))
    cfg = Config.from_file(os.path.join(path, "config.yaml"))
    model_type: str = result["model"]["model_type"]

    model_cfg = ModelConfig(**result["model"])

    data_builder = DATA_FACTORY.get(cfg.data_config.pipe_name)

    exp_folder = os.path.join(path, "rerun")
    os.makedirs(exp_folder, exist_ok=True)

    data = data_builder(
        input_path=cfg.data_config.input_path,
        save_data=False,
        pipe_params=cfg.data_config.pipe_params,
        exp_dir=exp_folder,
    )
    logger.info("Loading data")
    data_train, data_test, data_val = data.get_train_test_val(
        **cfg.data_config.split_params
    )

    model_bld = MODEL_FACTORY[model_type]

    os.makedirs(os.path.join(exp_folder, "models"), exist_ok=True)

    model = model_bld(
        data_train.shape,
        data_train.feature_names,
        data_train.y.shape[1:],
        model_params=model_cfg.model_config,
        model_dir=os.path.join(exp_folder, "models"),
    )
    logger.info("Loading model")

    model.load(os.path.join(full_model_path, "model"))

    logger.info("Predicting")

    preds = model.predict(data_test)

    logger.info("Calculating metrics")

    stats = get_stats(data_test, preds)

    models_folder = os.path.join(exp_folder, "rerun_models")
    os.makedirs(path, exist_ok=True)

    model_folder = os.path.join(models_folder, model_dir)

    os.makedirs(model_folder, exist_ok=True)

    result = {
        "model": result["model"],
        "metrics": stats,
    }
    logger.info("Saving results")

    dump_json(result, os.path.join(model_folder, "result.json"))


if __name__ == "__main__":
    rerun_single_exp(
        path="./experiments/2024-06-19_val_depth0",
        model_dir="catboost_val_2024-06-20T03:47:20_b6c9",
    )
