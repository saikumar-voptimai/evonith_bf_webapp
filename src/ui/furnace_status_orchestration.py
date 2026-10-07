"""Thin data coordinator for the V-Board Furnace Status section.

All reads use the existing generic V-Board chain:
``TimeSeriesDataFetcher.fetch_data`` → ``BaseDataFetcher.fetch_averaged_data``
→ the shared Influx query builder. Business rules stay in
``domain.furnace_status``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.fetchers.ts_data_fetcher import TimeSeriesDataFetcher
from domain import furnace_status as fs
from utils.logger import get_logger

log = get_logger(__name__)

STATUS_LOOKBACK = "last 15 minutes"
SELECTED_RANGE = "over selected range"
CACHE_TTL_SECONDS = 60
_SAFE_SOURCE_ERROR = "The data source could not be reached. Try Refresh."


def _normalise_frame(
    frame: pd.DataFrame | None,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    """Normalize transport output without changing any measurement values."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    if "time" in result.columns:
        timestamps = pd.to_datetime(result.pop("time"), utc=True, errors="coerce")
    else:
        timestamps = pd.to_datetime(result.index, utc=True, errors="coerce")

    valid = ~pd.isna(timestamps)
    result = result.loc[valid].copy()
    timestamps = pd.DatetimeIndex(timestamps[valid]).tz_convert(fs.IST)
    result.index = timestamps.rename("time (IST)")

    # Selecting from the validated requested tuple drops only transport/tag
    # metadata and preserves canonical field names and raw values unchanged.
    present = [field for field in fields if field in result.columns]
    return result.loc[:, present].sort_index()


def _fetch_measurement(
    measurement: str,
    fields: tuple[str, ...],
    time_interval: str,
    start_utc: datetime | None,
    end_utc: datetime | None,
    *,
    request_type: str,
    window_by: str | None,
) -> pd.DataFrame:
    """Delegate one measurement read to the generic V-Board fetcher."""
    fetcher = TimeSeriesDataFetcher(measurement, debug=False, source="historical")
    frame = fetcher.fetch_data(
        time_interval,
        start_utc,
        end_utc,
        request_type=request_type,
        window_by=window_by,
        fields=fields,
    )
    return _normalise_frame(frame, fields)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_current_measurement(
    measurement: str,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    """Return recent raw samples for one measurement (one cached request)."""
    return _fetch_measurement(
        measurement,
        fields,
        STATUS_LOOKBACK,
        None,
        None,
        request_type="ts",
        window_by=None,
    )


def _collect_current_frames() -> dict[str, pd.DataFrame | None]:
    """Collect each dashboard measurement once, isolating source failures."""
    frames: dict[str, pd.DataFrame | None] = {}
    for measurement in fs.measurements_for():
        fields = fs.source_fields_for_measurement(measurement)
        try:
            frames[measurement] = _load_current_measurement(measurement, fields)
        except Exception:  # noqa: BLE001 - one source must not blank the dashboard
            log.exception("Furnace Status: raw fetch failed for %s", measurement)
            frames[measurement] = None
    return frames


def load_status_snapshot(now: datetime | None = None) -> fs.StatusSnapshot:
    """Resolve the dashboard from cached raw samples at the current endpoint."""
    endpoint = now or datetime.now(timezone.utc)
    return fs.build_status_snapshot(_collect_current_frames(), endpoint)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_trend_frame(
    measurement: str,
    fields: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> pd.DataFrame:
    """Fetch one range-keyed, windowed measurement frame for a chart."""
    return _fetch_measurement(
        measurement,
        fields,
        SELECTED_RANGE,
        start_utc,
        end_utc,
        request_type="windowed-average",
        window_by=window_by,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_historical_current_frame(
    measurement: str,
    fields: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Fetch raw samples in the freshness-sized tail of a historical range."""
    lookup_start = max(start_utc, end_utc - fs.STALE_AFTER)
    return _fetch_measurement(
        measurement,
        fields,
        SELECTED_RANGE,
        lookup_start,
        end_utc,
        request_type="ts",
        window_by=None,
    )


def load_trend(
    spec: fs.ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> fs.TrendData:
    """Load one aggregated chart request, independently of its raw Current."""
    if not spec.has_source:
        return fs.TrendData(spec, "no_source", None, spec.unavailable_reason)
    try:
        frame = _load_trend_frame(
            spec.measurement,
            spec.source_fields,
            start_utc,
            end_utc,
            window_by,
        )
    except Exception:  # noqa: BLE001 - connection details stay in server logs
        log.exception(
            "Furnace Status: trend fetch failed for %s.%s",
            spec.measurement,
            spec.field,
        )
        return fs.TrendData(spec, "error", None, _SAFE_SOURCE_ERROR)
    return fs.build_trend_data(spec, frame)


def load_current_reading(
    spec: fs.ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    *,
    use_live_cache: bool,
) -> fs.ParameterReading:
    """Resolve Current from raw samples bounded by the selected interval."""
    if not spec.has_source:
        return fs.ParameterReading(spec, issue="no_source")
    try:
        if use_live_cache:
            frame = _load_current_measurement(
                spec.measurement,
                fs.source_fields_for_measurement(spec.measurement),
            )
        else:
            frame = _load_historical_current_frame(
                spec.measurement,
                spec.source_fields,
                start_utc,
                end_utc,
            )
    except Exception:  # noqa: BLE001 - Current and chart failures are independent
        log.exception(
            "Furnace Status: raw Current fetch failed for %s.%s",
            spec.measurement,
            spec.field,
        )
        frame = None
    return fs.resolve_reading(
        spec,
        {spec.measurement: frame},
        end_utc,
        range_start=start_utc,
    )


def load_trend_view(
    spec: fs.ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
    *,
    use_live_current: bool,
) -> fs.TrendViewData:
    """Return independent raw Current, aggregated trend, and range statistics."""
    current = load_current_reading(
        spec,
        start_utc,
        end_utc,
        use_live_cache=use_live_current,
    )
    trend = load_trend(spec, start_utc, end_utc, window_by)
    return fs.TrendViewData(current, trend, fs.compute_trend_stats(trend.series))


def clear_furnace_status_caches() -> None:
    """Clear only Furnace Status cached results."""
    _load_current_measurement.clear()
    _load_trend_frame.clear()
    _load_historical_current_frame.clear()
