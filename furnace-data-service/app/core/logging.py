"""Lightweight backend logging setup."""

from __future__ import annotations

import logging

from app.core.config import BackendSettings


def configure_logging(settings: BackendSettings) -> None:
    """Configure standard logging for backend startup and errors."""
    level = getattr(logging, settings.backend_log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(level)
