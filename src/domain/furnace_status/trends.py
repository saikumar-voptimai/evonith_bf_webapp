"""Pure trend transformation and statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.furnace_status.catalogue import SETTINGS
from domain.furnace_status.readings import (
    normalize_timestamps,
    series_for_spec,
    setpoint_series,
)
from domain.furnace_status.types import ParameterSpec, TrendData, TrendStats


def _clean_raw_series(series: pd.Series | None) -> pd.Series | None:
    """Drop invalid raw points and add one Plotly line break across outages."""
    if series is None:
        return None
    cleaned = series[np.isfinite(series)].sort_index()
    if cleaned.empty:
        return cleaned
    gap_limit = pd.Timedelta(seconds=SETTINGS.raw_gap_break_seconds)
    gaps = cleaned.index.to_series().diff() > gap_limit
    if not gaps.any():
        return cleaned
    breaks = pd.Series(
        np.nan,
        index=pd.DatetimeIndex(
            [
                cleaned.index[index - 1]
                + (cleaned.index[index] - cleaned.index[index - 1]) / 2
                for index in np.flatnonzero(gaps.to_numpy())
            ],
            name=cleaned.index.name,
        ),
        name=cleaned.name,
        dtype="float64",
    )
    return pd.concat([cleaned, breaks]).sort_index(kind="stable")


def _align_component_samples(
    frame: pd.DataFrame | None, spec: ParameterSpec
) -> pd.DataFrame | None:
    """Align asynchronously written derived fields to the latest prior sample."""
    if not spec.components:
        return frame
    normalized = normalize_timestamps(frame)
    if normalized.empty or not set(spec.components) <= set(normalized.columns):
        return normalized
    numeric = normalized.loc[:, list(spec.components)].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.where(np.isfinite(numeric))
    populated_indexes = [
        numeric.index[numeric[column].notna()] for column in spec.components
    ]
    if not populated_indexes or all(
        index.equals(populated_indexes[0]) for index in populated_indexes[1:]
    ):
        return normalized

    aligned = pd.DataFrame({"time": numeric.index.unique().sort_values()})
    for column in spec.components:
        source = numeric[column].dropna().rename(column).reset_index()
        source.columns = ["time", column]
        aligned = pd.merge_asof(
            aligned,
            source.sort_values("time"),
            on="time",
            direction="backward",
            tolerance=pd.Timedelta(seconds=60),
        )
    if spec.setpoint_field and spec.setpoint_field in normalized.columns:
        setpoint = pd.to_numeric(
            normalized[spec.setpoint_field], errors="coerce"
        ).where(lambda values: np.isfinite(values))
        source = setpoint.dropna().rename(spec.setpoint_field).reset_index()
        source.columns = ["time", spec.setpoint_field]
        source = source.drop_duplicates(subset="time", keep="last")
        aligned = aligned.merge(source, on="time", how="left")
    result = aligned.set_index("time")
    result.index = result.index.rename(normalized.index.name)
    return result


def build_trend_data(spec: ParameterSpec, frame: pd.DataFrame | None) -> TrendData:
    if not spec.has_source:
        return TrendData(spec, "no_source", None, spec.unavailable_reason)
    trend_frame = _align_component_samples(frame, spec)
    actual = _clean_raw_series(series_for_spec(trend_frame, spec))
    if actual is None or actual.empty:
        return TrendData(
            spec, "no_data", None, "No data was returned for the selected interval."
        )
    setpoint = _clean_raw_series(setpoint_series(trend_frame, spec))
    if setpoint is not None and setpoint.empty:
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
