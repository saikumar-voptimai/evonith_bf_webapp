"""Structured backend logging setup."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.services.redaction_service import redact_text, safe_log_extra
from furnace_data.runtime_paths import runtime_path


class JsonFormatter(logging.Formatter):
    """Small JSON formatter for backend and access logs."""

    _EXTRA_FIELDS = (
        "service",
        "environment",
        "request_id",
        "user_id",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "error_code",
        "event",
        "query",
    )

    def __init__(self, settings: BackendSettings) -> None:
        super().__init__()
        self.settings = settings

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": "evonith-backend-api",
            "environment": self.settings.backend_env,
            "message": redact_text(record.getMessage()) if self.settings.log_redaction_enabled else record.getMessage(),
        }
        for field in self._EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(safe_log_extra(payload), default=str, sort_keys=True)


class RedactingTextFormatter(logging.Formatter):
    """Text formatter that redacts the rendered message."""

    def __init__(self, settings: BackendSettings) -> None:
        super().__init__("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        self.settings = settings

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return redact_text(rendered) if self.settings.log_redaction_enabled else rendered


def configure_logging(settings: BackendSettings) -> None:
    """Configure standard logging for backend startup, access, and errors."""
    level = getattr(logging, settings.backend_log_level.upper(), logging.INFO)
    formatter: logging.Formatter
    if settings.log_format == "json":
        formatter = JsonFormatter(settings)
    else:
        formatter = RedactingTextFormatter(settings)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if settings.log_file_enabled:
        try:
            log_dir = runtime_path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "backend.log",
                maxBytes=max(1, settings.log_max_file_mb) * 1024 * 1024,
                backupCount=max(1, settings.log_backup_count),
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("File logging could not be initialized: %s", exc)
