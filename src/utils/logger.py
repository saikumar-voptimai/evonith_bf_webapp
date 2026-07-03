import logging
import logging.config
import logging.handlers
import os
import time
from pathlib import Path

import yaml

from furnace_data.runtime_paths import get_logs_dir


class WindowsSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that retries rotate() on Windows PermissionError.

    On Windows, a brief race between threads/processes can leave the log file
    momentarily locked when a rollover is attempted.  This subclass retries the
    rename up to 5 times with increasing back-off before silently skipping the
    rotation (which is safer than crashing the logging thread).
    """

    def rotate(self, source: str, dest: str) -> None:
        for attempt in range(5):
            try:
                super().rotate(source, dest)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        # Final attempt – skip rather than raise so logging never crashes.
        try:
            super().rotate(source, dest)
        except PermissionError:
            pass


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
        logs_dir = get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        for handler in (config.get("handlers") or {}).values():
            filename = handler.get("filename") if isinstance(handler, dict) else None
            if filename:
                handler["filename"] = str(logs_dir / Path(filename).name)
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
