"""Timestamp normalization and current-reading resolution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.furnace_status.catalogue import SETTINGS
from domain.furnace_status.types import ParameterReading, ParameterSpec, ReadingIssue

IST = SETTINGS.timezone


def _to_ist(timestamp: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    return (
        timestamp.tz_localize(IST)
        if timestamp.tzinfo is None
        else timestamp.tz_convert(IST)
    )


def normalize_timestamps(
    frame: pd.DataFrame | None,
    *,
    assume_timezone: ZoneInfo | timezone = IST,
) -> pd.DataFrame:
    """Copy a frame onto a sorted, timezone-aware IST index."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    raw_time = result.pop("time") if "time" in result.columns else result.index
    timestamps = pd.DatetimeIndex(pd.to_datetime(raw_time, errors="coerce"))
    valid = ~timestamps.isna()
    result = result.loc[valid].copy()
    timestamps = timestamps[valid]
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(assume_timezone)
    result.index = timestamps.tz_convert(IST).rename("time (IST)")
    return result.sort_index()


def _field_series(
    frame: pd.DataFrame | None,
    field: str | None,
    *,
    name: str | None = None,
) -> pd.Series | None:
    normalized = normalize_timestamps(frame)
    if field is None or normalized.empty or field not in normalized.columns:
        return None
    series = pd.to_numeric(normalized[field], errors="coerce").astype("float64")
    return series.where(np.isfinite(series)).rename(name or field)


def series_for_spec(
    frame: pd.DataFrame | None,
    spec: ParameterSpec,
    *,
    field: str | None = None,
    name: str | None = None,
) -> pd.Series | None:
    """Build a scaled direct or derived series while preserving gaps."""
    normalized = normalize_timestamps(frame)
    if normalized.empty:
        return None
    components = () if field is not None else spec.components
    if components:
        if not set(components) <= set(normalized.columns):
            return None
        parts = normalized.loc[:, list(components)].apply(
            pd.to_numeric, errors="coerce"
        )
        parts = parts.where(np.isfinite(parts))
        series = parts.sum(axis=1, min_count=len(components))
        if spec.aggregate == "mean":
            series = series / len(components)
    else:
        series = _field_series(normalized, field or spec.field)
        if series is None:
            return None
    return (series.astype("float64") * spec.scale).rename(name or spec.title)


def setpoint_series(
    frame: pd.DataFrame | None, spec: ParameterSpec
) -> pd.Series | None:
    if spec.setpoint_field is None:
        return None
    return series_for_spec(frame, spec, field=spec.setpoint_field, name="Setpoint")


def finite_series(
    frame: pd.DataFrame | None,
    column: str | None,
    *,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
) -> pd.Series | None:
    series = _field_series(frame, column)
    if series is None:
        return None
    if start is not None:
        series = series[series.index >= _to_ist(pd.Timestamp(start))]
    if end is not None:
        series = series[series.index <= _to_ist(pd.Timestamp(end))]
    series = series[np.isfinite(series)].sort_index()
    return series if not series.empty else None


def latest_finite_point(
    frame: pd.DataFrame | None,
    column: str | None,
    *,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
) -> tuple[float, pd.Timestamp] | None:
    series = finite_series(frame, column, start=start, end=end)
    if series is None:
        return None
    return float(series.iloc[-1]), _to_ist(series.index[-1])


def resolve_reading(
    spec: ParameterSpec,
    frames: Mapping[str, pd.DataFrame | None],
    now: datetime,
    *,
    range_start: datetime | None = None,
    stale_after: timedelta | None = None,
) -> ParameterReading:
    """Resolve actual and setpoint independently from raw telemetry."""
    if not spec.has_source:
        return ParameterReading(spec, issue="no_source")
    frame = frames.get(spec.measurement)
    if frame is None:
        return ParameterReading(spec, issue="fetch_failed")
    endpoint = _to_ist(pd.Timestamp(now))
    maximum_age = stale_after or SETTINGS.stale_after

    def fresh(series: pd.Series | None) -> tuple[float, pd.Timestamp] | ReadingIssue:
        if series is None:
            return "no_data"
        if range_start is not None:
            series = series[series.index >= _to_ist(pd.Timestamp(range_start))]
        series = series[series.index <= endpoint]
        finite = series[np.isfinite(series)]
        if finite.empty:
            return "no_data"
        value = float(finite.iloc[-1])
        timestamp = _to_ist(finite.index[-1])
        if endpoint.to_pydatetime() - timestamp.to_pydatetime() > maximum_age:
            return "stale"
        return value, timestamp

    setpoint_result = (
        fresh(setpoint_series(frame, spec)) if spec.setpoint_field else None
    )
    setpoint = setpoint_result[0] if isinstance(setpoint_result, tuple) else None
    setpoint_timestamp = (
        setpoint_result[1] if isinstance(setpoint_result, tuple) else None
    )
    actual = fresh(series_for_spec(frame, spec))
    if not isinstance(actual, tuple):
        return ParameterReading(
            spec,
            issue=actual,
            setpoint=setpoint,
            setpoint_timestamp=setpoint_timestamp,
        )
    value, timestamp = actual
    return ParameterReading(
        spec,
        value=value,
        timestamp=timestamp,
        setpoint=setpoint,
        setpoint_timestamp=setpoint_timestamp,
    )
