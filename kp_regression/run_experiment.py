import typing as T

import click
import os
from numpy import savez_compressed

from dataclasses import asdict

from kp_regression.config import Config
from kp_regression.utils import safe_mkdir, add_unique_suffix, dump_json
from kp_regression.logging_utils import config_logger
from kp_regression.models_zoo import MODEL_FACTORY
from kp_regression.base_model import BaseModel
from kp_regression.data import DATA_FACTORY, POST_PROCESS_FACTORY
from kp_regression.metrics import calculate_regression_metrics

import logging

logger = logging.getLogger()


@click.command()
@click.option("--config_path", help="Path to config", type=click.STRING, required=True)
@click.option(
    "--exp_folder", help="Path to exp folder", type=click.STRING, required=True
)
def run(config_path: str, exp_folder: str, report: bool = False) -> None:

    safe_mkdir(exp_folder)

    config_logger(logger)

    logger.info("Starting experiment in %s", exp_folder)

    config = Config.from_file(config_path)

    logger.info("Got data config %s", config.data_config)

    data_builder = DATA_FACTORY.get(config.data_config.pipe_name)

    if data_builder is None:
        raise ValueError("No such data pipeline %s", config.data_config.pipe_name)

    data_folder = os.path.join(exp_folder, "data")
    safe_mkdir(data_folder)

    data = data_builder(
        input_path=config.data_config.input_path,
        save_data=config.data_config.save_file,
        pipe_params=config.data_config.pipe_params,
        exp_dir=data_folder,
    )

    if config.data_config.use_val:
        data_train, data_test, data_val = data.get_train_test_val(
            **config.data_config.split_params
        )
    else:
        data_train, data_test = data.get_train_test(**config.data_config.split_params)
        data_val = None

    logger.info("Verifying building from config...")

    built_models: T.Dict[str, BaseModel] = {}
    model_dirs: T.Dict[str, str] = {}

    model_folder = os.path.join(exp_folder, "models")
    safe_mkdir(model_folder)

    # Build all models first to fast fail on errors

    names = []
    for model_cfg in config.models:

        logger.info(
            "Building model %s of %s", model_cfg.model_name, model_cfg.model_type
        )
        builder = MODEL_FACTORY[model_cfg.model_type]
        model_dir = os.path.join(model_folder, add_unique_suffix(model_cfg.model_name))
        safe_mkdir(model_dir)
        model_dirs[model_cfg.model_name] = model_dir

        built_models[model_cfg.model_name] = builder(
            shape=data_train.X.shape[1:],
            features=data_train.feature_names,
            output_shape=data_train.y.shape[1:],
            model_params=model_cfg.model_config,
            model_dir=model_dir,
        )

        names.append(model_cfg.model_name)

    if len(set(names)) < len(names):
        raise ValueError("Model names should be unique!")

    for model_cfg in config.models:

        logger.info("=" * 50)
        logger.info("Model config %s", model_cfg)
        model_dir = model_dirs[model_cfg.model_name]
        logger.info("Creating model dir %s", model_dir)
        safe_mkdir(model_dir)

        logger.info(
            "Getting model %s of type %s", model_cfg.model_name, model_cfg.model_type
        )
        model = built_models[model_cfg.model_name]

        if model_cfg.use_cv:
            logger.info("Performing CV for model %s", model_cfg.model_name)
            model.cv(model_cfg.cv_config, data_train.X, data_train.y)

        logger.info("Training model %s", model_cfg.model_name)
        if config.data_config.use_val:
            model.train(data_train.X, data_train.y, data_val.X, data_val.y)
        else:
            model.train(data_train.X, data_train.y)

        save_path = os.path.join(model_dir, "model")
        logger.info("Saving model to %s", save_path)
        model.save(save_path)

        logger.info("Predicting model %s", model_cfg.model_type)

        preds = model.predict(X=data_test.X)

        postproc = POST_PROCESS_FACTORY[model_cfg.postprocess_name]

        logger.info("Applying postproc %s", model_cfg.postprocess_name)

        preds = postproc(preds)

        if config.data_config.save_preds:
            logger.info("Saving preds")
            preds_path = os.path.join(model_dir, "preds.npz")
            savez_compressed(preds_path, preds=preds)

        metrics = calculate_regression_metrics(
            preds, y_true=data_test.y, meta=data_test.meta
        )

        result = {"model": asdict(model_cfg), "metrics": metrics}

        logger.info("Dumping result for %s", model_cfg.model_name)

        dump_json(result, os.path.join(model_dir, "result.json"))

        for level, metric_dict in enumerate(metrics):
            logger.info(
                "Model %s, Horizon %s, metric %s = %s",
                model_cfg.model_type,
                level,
                "MSE",
                metric_dict["MSE"],
            )
