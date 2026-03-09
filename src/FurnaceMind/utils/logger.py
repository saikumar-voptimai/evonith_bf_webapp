# FurnaceMind/utils/logger.py
# Purpose: Centralized logging utility for FurnaceMind
# Fixed: Forces UTF-8 output stream to prevent UnicodeEncodeError
#        on Windows terminals (cp1252 can't handle emoji in filenames)

import sys
import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Force UTF-8 stream to prevent Windows cp1252 encoding crashes
        # when log messages contain emoji (e.g. from filenames like 7_🧠_FurnaceMind.py)
        try:
            utf8_stream = open(
                sys.stdout.fileno(),
                mode="w",
                encoding="utf-8",
                closefd=False,
            )
        except (OSError, ValueError):
            # Fallback if stdout fileno is unavailable (e.g. some IDEs, notebooks)
            utf8_stream = sys.stdout

        handler = logging.StreamHandler(stream=utf8_stream)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger