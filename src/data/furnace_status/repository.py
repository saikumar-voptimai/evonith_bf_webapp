"""Influx/cache boundary for Furnace Status telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.fetchers.ts_data_fetcher import TimeSeriesDataFetcher
from data.furnace_status.catalogue import (
    PARAMETERS,
    SETTINGS,
    measurements_for,
    source_fields_for_measurement,
)
from domain.furnace_status.readings import normalize_timestamps
from domain.furnace_status.types import ParameterSpec
from utils.logger import get_logger

log = get_logger(__name__)


@st.cache_data(ttl=SETTINGS.cache_ttl_seconds, show_spinner=False)
def _fetch_cached(
    measurement: str,
    fields: tuple[str, ...],
    time_interval: str,
    start_utc: datetime | None,
    end_utc: datetime | None,
    request_type: str,
    window_by: str | None,
) -> pd.DataFrame:
    """Run one validated request and normalize its timestamps."""
    fetcher = TimeSeriesDataFetcher(measurement, debug=False, source="historical")
    frame = fetcher.fetch_data(
        time_interval,
        start_utc,
        end_utc,
        request_type=request_type,
        window_by=window_by,
        fields=fields,
    )
    normalized = normalize_timestamps(frame, assume_timezone=timezone.utc)
    present = [field for field in fields if field in normalized.columns]
    return normalized.loc[:, present]


def current_frames(
    specs: Sequence[ParameterSpec] = PARAMETERS,
) -> dict[str, pd.DataFrame | None]:
    """Read every required measurement once and isolate failures by source."""
    frames: dict[str, pd.DataFrame | None] = {}
    for measurement in measurements_for(specs):
        fields = source_fields_for_measurement(measurement, specs)
        try:
            frames[measurement] = _fetch_cached(
                measurement,
                fields,
                SETTINGS.status_lookback,
                None,
                None,
                "ts",
                None,
            )
        except Exception:  # noqa: BLE001 - source details stay in server logs
            log.exception("Furnace Status: raw fetch failed for %s", measurement)
            frames[measurement] = None
    return frames


def trend_frame(
    spec: ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> pd.DataFrame:
    return _fetch_cached(
        spec.measurement,
        spec.source_fields,
        SETTINGS.selected_range_label,
        start_utc,
        end_utc,
        "windowed-average",
        window_by,
    )


def current_frame(
    spec: ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    *,
    use_live_cache: bool,
) -> pd.DataFrame:
    if use_live_cache:
        time_interval = SETTINGS.status_lookback
        start = end = None
        fields = source_fields_for_measurement(spec.measurement)
    else:
        time_interval = SETTINGS.selected_range_label
        start = max(start_utc, end_utc - SETTINGS.stale_after)
        end = end_utc
        fields = spec.source_fields
    return _fetch_cached(
        spec.measurement,
        fields,
        time_interval,
        start,
        end,
        "ts",
        None,
    )


def clear_cache() -> None:
    _fetch_cached.clear()
