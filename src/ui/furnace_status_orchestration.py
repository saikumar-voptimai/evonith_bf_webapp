"""Live-data orchestration for the V-Board Furnace Status section.

The shared :func:`furnace_data.influx.online.fetch_online_df` function is the
only data-access entry point used here. Caching is deliberately feature-local:
one cached dashboard collection and one range-keyed trend operation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from domain import furnace_status as fs
from furnace_data.influx.online import fetch_online_df
from utils.logger import get_logger

log = get_logger(__name__)

STATUS_LOOKBACK = "last 15 minutes"
STATUS_WINDOW = "1 minute"
TREND_TIME_RANGE_PLACEHOLDER = "last 1 hour"
CACHE_TTL_SECONDS = 60


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_dashboard_frames() -> dict[str, pd.DataFrame | None]:
    """Fetch each catalogue measurement once and isolate measurement failures."""
    frames: dict[str, pd.DataFrame | None] = {}
    for measurement in fs.measurements_for():
        try:
            frames[measurement] = fetch_online_df(
                selected_measurements=[measurement],
                time_range=STATUS_LOOKBACK,
                request_type="windowed-average",
                window_by=STATUS_WINDOW,
                column_naming="field",
            )
        except Exception:  # noqa: BLE001 - one source must not blank the dashboard
            log.exception("Furnace Status: fetch failed for %s", measurement)
            frames[measurement] = None
    return frames


def load_status_snapshot(now: datetime | None = None) -> fs.StatusSnapshot:
    """Return the cached live frames resolved against the current wall clock."""
    resolved_now = now or datetime.now(timezone.utc)
    return fs.build_status_snapshot(_load_dashboard_frames(), resolved_now)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_trend_data(
    spec: fs.ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> fs.TrendData:
    """Fetch and transform one parameter trend with a range-aware cache key."""
    try:
        frame = fetch_online_df(
            selected_measurements=[spec.measurement],
            time_range=TREND_TIME_RANGE_PLACEHOLDER,
            request_type="windowed-average",
            window_by=window_by,
            start_time_override=start_utc,
            end_time_override=end_utc,
            column_naming="field",
        )
    except Exception:  # noqa: BLE001 - connection details stay in server logs
        log.exception(
            "Furnace Status: trend fetch failed for %s.%s",
            spec.measurement,
            spec.field,
        )
        return fs.TrendData(
            spec,
            "error",
            None,
            "The data source could not be reached. Try Refresh.",
        )

    # The shared measurement fetch returns canonical fields. Keep the feature
    # result compact while retaining every actual/setpoint/derived component.
    if frame is not None and not frame.empty:
        present = [field for field in spec.source_fields if field in frame.columns]
        frame = frame[present] if present else pd.DataFrame(index=frame.index)
    return fs.build_trend_data(spec, frame)


def load_trend(
    spec: fs.ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> fs.TrendData:
    """Load a source-backed trend, or explain an intentionally unsourced item."""
    if not spec.has_source:
        return fs.TrendData(spec, "no_source", None, spec.unavailable_reason)
    return _load_trend_data(spec, start_utc, end_utc, window_by)


def clear_furnace_status_caches() -> None:
    """Clear only Furnace Status cached results."""
    _load_dashboard_frames.clear()
    _load_trend_data.clear()
