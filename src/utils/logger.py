import logging.config
import os
from pathlib import Path

import yaml


def setup_logger(config_path: str = "src/config/logger_setting.yml"):
    """
    Setup logger using the provided YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    config_file_path = Path(config_path).resolve()
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            f"Logging configuration file not found at {config_file_path}"
        )

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

    logger = logging.getLogger("root")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a StreamHandler if none is attached yet."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger
