"""Pure trend transformation and statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.furnace_status.readings import series_for_spec, setpoint_series
from domain.furnace_status.types import ParameterSpec, TrendData, TrendStats


def build_trend_data(spec: ParameterSpec, frame: pd.DataFrame | None) -> TrendData:
    if not spec.has_source:
        return TrendData(spec, "no_source", None, spec.unavailable_reason)
    actual = series_for_spec(frame, spec)
    if actual is None or actual[np.isfinite(actual)].empty:
        return TrendData(
            spec, "no_data", None, "No data was returned for the selected interval."
        )
    setpoint = setpoint_series(frame, spec)
    if setpoint is not None and setpoint[np.isfinite(setpoint)].empty:
        setpoint = None
    return TrendData(spec, "ok", actual, setpoint=setpoint)


def compute_trend_stats(series: pd.Series | None) -> TrendStats | None:
    if series is None:
        return None
    finite = series[np.isfinite(series)]
    if finite.empty:
        return None
    return TrendStats(
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        mean=float(finite.mean()),
    )
