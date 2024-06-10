import logging
import sys

import datetime
import os

from kp_regression.utils import safe_mkdir


class StreamToLogger(object):  # pragma: no cover
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def config_logger(logger: logging.Logger, folder: str) -> None:

    handler = logging.StreamHandler(sys.stdout)

    log_folder = os.path.join(folder, "logs")
    safe_mkdir(log_folder)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(log_folder, f"log_{now}.log")
    file_handler = logging.FileHandler(filename, mode="w", encoding="utf-8")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
