"""Value, readout and IST formatting."""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from domain.furnace_status.readings import _to_ist
from domain.furnace_status.types import ParameterReading

NOT_AVAILABLE = "Not available"


def format_value(value: float | None, decimals: int) -> str:
    if value is None or not math.isfinite(value):
        return NOT_AVAILABLE
    if round(value, decimals) == 0:
        value = 0.0
    return f"{value:,.{decimals}f}"


def has_display_value(reading: ParameterReading) -> bool:
    return reading.value is not None or reading.setpoint is not None


def format_reading(reading: ParameterReading) -> str:
    actual = format_value(reading.value, reading.spec.decimals)
    if not reading.spec.setpoint_field or not has_display_value(reading):
        return actual
    return f"{format_value(reading.setpoint, reading.spec.decimals)} / {actual}"


def format_ist(
    timestamp: pd.Timestamp | datetime | None, *, with_date: bool = True
) -> str:
    if timestamp is None:
        return "—"
    local = _to_ist(pd.Timestamp(timestamp))
    pattern = "%d %b %Y, %H:%M:%S IST" if with_date else "%H:%M:%S IST"
    return local.strftime(pattern)
