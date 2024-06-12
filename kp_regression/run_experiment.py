import click
import os

from kp_regression.config import Config
from kp_regression.utils import safe_mkdir, add_unique_suffix
from kp_regression.logging_utils import config_logger
from kp_regression.models_zoo import MODEL_FACTORY
from kp_regression.data import DATA_FACTORY

import logging

logger = logging.getLogger()


def run(config_path: str, exp_folder: str, report: bool = False) -> None:

    safe_mkdir(exp_folder)

    config_logger(logger, folder=exp_folder)

    logger.info("Starting experiment in %s", exp_folder)

    config = Config.from_file(config_path)

    # data_builder = DATA_FACTORY.get(config.data_config.pipe_name)

    # X, y = data_builder(config.data_config.input_path, **config.data_config.pipe_params)

    for model_cfg in config.models:
        logger.info("Model config %s", model_cfg)

        model_dir = os.path.join(exp_folder, add_unique_suffix(model_cfg.model_name))

        logger.info("Creating model dir %s", model_dir)
        safe_mkdir(model_dir)

        builder = MODEL_FACTORY[model_cfg.model_type]
        logger.info("Building model %s", model_cfg.model_type)
        model = builder(
            shape=None,
            features=None,
            model_params=model_cfg.model_config,
            model_dir=dir,
        )

        save_path = os.path.join(model_dir, "model")
        logger.info("Saving model %s", save_path)
        model.save(save_path)
