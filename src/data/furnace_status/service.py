"""Application operations composing repository reads with domain logic."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from data.furnace_status import repository
from data.furnace_status.catalogue import PARAMETERS, SETTINGS
from domain.furnace_status.readings import resolve_reading
from domain.furnace_status.status import build_status_snapshot
from domain.furnace_status.trends import build_trend_data, compute_trend_stats
from domain.furnace_status.types import (
    ParameterReading,
    ParameterSpec,
    StatusSnapshot,
    TrendData,
    TrendViewData,
)
from utils.logger import get_logger

log = get_logger(__name__)
_SAFE_SOURCE_ERROR = "The data source could not be reached. Try Refresh."


def load_status_snapshot(
    now: datetime | None = None,
    specs: Sequence[ParameterSpec] = PARAMETERS,
) -> StatusSnapshot:
    endpoint = now or datetime.now(timezone.utc)
    return build_status_snapshot(repository.current_frames(specs), endpoint, specs)


def _trend_result(
    spec: ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> TrendData:
    if not spec.has_source:
        return TrendData(spec, "no_source", None, spec.unavailable_reason)
    try:
        frame = repository.trend_frame(spec, start_utc, end_utc, window_by)
    except Exception:  # noqa: BLE001 - details belong in server logs only
        log.exception(
            "Furnace Status: trend fetch failed for %s.%s",
            spec.measurement,
            spec.field,
        )
        return TrendData(spec, "error", None, _SAFE_SOURCE_ERROR)
    return build_trend_data(spec, frame)


def _current_reading(
    spec: ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    *,
    use_live_cache: bool,
) -> ParameterReading:
    if not spec.has_source:
        return ParameterReading(spec, issue="no_source")
    try:
        frame = repository.current_frame(
            spec,
            start_utc,
            end_utc,
            use_live_cache=use_live_cache,
        )
    except Exception:  # noqa: BLE001 - Current and trend fail independently
        log.exception(
            "Furnace Status: raw Current fetch failed for %s.%s",
            spec.measurement,
            spec.field,
        )
        frame = None
    return resolve_reading(
        spec,
        {spec.measurement: frame},
        end_utc,
        range_start=start_utc,
        stale_after=SETTINGS.stale_after,
    )


def load_trend_view(
    spec: ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
    *,
    use_live_current: bool,
) -> TrendViewData:
    current = _current_reading(
        spec,
        start_utc,
        end_utc,
        use_live_cache=use_live_current,
    )
    trend = _trend_result(spec, start_utc, end_utc, window_by)
    return TrendViewData(current, trend, compute_trend_stats(trend.series))


def clear_cache() -> None:
    repository.clear_cache()
