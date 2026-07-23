"""JSON-safety helpers for V-Sense results."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def json_safe_result(value: Any) -> Any:
    """Return *value* with NaN/infinity converted to null-compatible values."""

    if isinstance(value, dict):
        return {str(key): json_safe_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_result(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe_result(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
