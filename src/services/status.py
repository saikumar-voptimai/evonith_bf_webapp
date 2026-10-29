"""Furnace status for the twin-api and the worker (#271, SV-2.1 #293): snapshot, trends and the live-status tags.

The feature is layered: pure ``domain.furnace_status`` (readings, telemetry and operating state, trend series), data
access in ``data.furnace_status`` (catalogue ``config/furnace_status.yml``, InfluxDB behind ``utils.memo``). The
Streamlit V-Board renders ``data.furnace_status.load_status_snapshot`` directly; this module turns the same
``StatusSnapshot`` into the API view, so ``GET /api/status/snapshot`` equals the page for the same timestamp
(``tests/test_services_status.py``). Nothing is composed twice: values, display text, telemetry and operating state are
the domain's; this module adds the limit status, the zones and UTC times.

Uncached building blocks: ``snapshot_view`` (domain snapshot -> view), ``build_snapshot_view`` (one snapshot query:
one InfluxDB request per measurement), ``build_trend`` (raw series of a range, min/max-decimated), ``latest_hot_metal``
(Neon), ``live_status_fields`` (plant part of the ``live.status`` event). Memoised entry point:
``current_snapshot_view`` (at most one snapshot query per 10-s slot and process; the API's answer while the worker's
live snapshot in Redis is not fresh, ``services.live_status``).

Limits: a catalogue parameter may name parameters of a fact of the process fact register (``limits: {fact, low,
high}``); the reading's ``state`` is ``low`` / ``high`` outside them, compared unrounded like the operating state.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from data.furnace_status import PARAMETERS, PARAMETERS_BY_KEY, SECTIONS, SETTINGS, UI
from data.furnace_status.service import clear_cache as _clear_repository_cache
from data.furnace_status.service import load_status_snapshot, load_trend_view
from domain.furnace_status.formatting import format_reading
from domain.furnace_status.time_ranges import FIXED_INTERVALS, RangeError, resolve_custom_range, resolve_fixed_range
from domain.furnace_status.types import ParameterReading, ParameterSpec, StatusSnapshot, TrendViewData
from services.events import TagValue
from utils.memo import memo

__all__ = [
    "PARAMETERS",
    "PARAMETERS_BY_KEY",
    "SETTINGS",
    "RangeError",
    "StatusSnapshotView",
    "StatusTrend",
    "build_snapshot_view",
    "build_trend",
    "clear_cache",
    "current_snapshot_view",
    "hot_metal_samples",
    "latest_hot_metal",
    "live_status_fields",
    "load_status_snapshot",
    "load_trend_view",
    "snapshot_view",
]

log = logging.getLogger(__name__)

PLANT_TZ = "Asia/Kolkata"
PERIOD_S = 10.0  # one snapshot query per slot of this length (the worker's heartbeat, the API fallback)
CUSTOM = "custom"
TREND_INTERVALS: tuple[str, ...] = (*FIXED_INTERVALS, CUSTOM)
MAX_TREND_PARAMETERS = 8
DEFAULT_TREND_POINTS = 2000
HOT_METAL_LOOKBACK = timedelta(hours=24)

ParameterState = Literal["ok", "low", "high", "stale", "no_data", "fetch_failed", "no_source"]


def clock() -> datetime:
    """Current time (UTC); tests replace it."""
    return datetime.now(UTC)


# ---------------------------------------------------------------- views


class StatusLimits(BaseModel):
    low: float | None = Field(None, description="Lower limit; a reading below it is low, in the parameter's unit")
    high: float | None = Field(None, description="Upper limit; a reading above it is high, in the parameter's unit")
    fact: str = Field(..., description="Fact of the process fact register the limits come from, e.g. ASM-11")
    basis: str = Field(..., description="Statement of that fact")


class StatusParameter(BaseModel):
    key: str = Field(..., description="Catalogue key (config/furnace_status.yml)")
    label: str
    title: str = Field(..., description="Title of the parameter's trend")
    section: str
    unit: str = Field(..., description="Unit of value, setpoint and limits; empty when the plant unit is not confirmed")
    decimals: int = Field(..., description="Decimals the V-Board shows, count")
    value: float | None = Field(
        None, description="Latest reading within stale_after_minutes (scaled as shown, unrounded), in unit"
    )
    at: datetime | None = Field(None, description="Time of value (UTC)")
    setpoint: float | None = Field(None, description="Latest setpoint within stale_after_minutes, in unit")
    setpoint_at: datetime | None = Field(None, description="Time of setpoint (UTC)")
    display: str = Field(
        ..., description="Reading as the V-Board shows it, without unit: 'setpoint / actual' for SP parameters"
    )
    state: ParameterState = Field(
        ...,
        description=(
            "ok: fresh reading (inside its limits when it has limits); low / high: outside its limits; stale: newest "
            "reading older than stale_after_minutes; no_data: no reading in the window; fetch_failed: its measurement "
            "could not be read; no_source: no live field"
        ),
    )
    limits: StatusLimits | None = Field(None, description="Limits of the parameter; null when it has none")
    note: str = Field("", description="Source note, or why the parameter has no live source")


class StatusZone(BaseModel):
    name: str
    lower_m: float = Field(..., description="Lower edge, elevation above the hearth bottom, m")
    upper_m: float = Field(..., description="Upper edge, elevation above the hearth bottom, m")
    readings: list[str] = Field(..., description="Catalogue keys of the readings related to the zone (its callouts)")


class StatusOperatingState(BaseModel):
    level: Literal["normal", "abnormal", "unknown"] = Field(..., description="Operating state of the V-Board")
    state: Literal["normal", "reduced"] | None = Field(
        None, description="normal at or above normal_min, reduced below it, null when unknown (live.status wording)"
    )
    label: str
    detail: str = Field(..., description="Rule and current value as the V-Board shows them")
    parameter: str = Field(..., description="Catalogue key of the deciding parameter")
    normal_min: float = Field(
        ..., description="Lowest value of normal operation, in the unit of the deciding parameter (Nm³/h)"
    )
    value: float | None = Field(
        None, description="Fresh reading of the deciding parameter, in the unit of the deciding parameter (Nm³/h)"
    )


class StatusFreshness(BaseModel):
    computed_at: datetime = Field(..., description="Time the snapshot was taken (UTC)")
    data_at: datetime | None = Field(None, description="Newest reading or setpoint of the snapshot (UTC)")
    age_s: float | None = Field(None, description="computed_at minus the newest reading, s")
    stale: bool = Field(..., description="No parameter has a reading within stale_after_minutes")
    telemetry: Literal["live", "partial", "offline"] = Field(
        ...,
        description=(
            "Telemetry availability as the V-Board header (not furnace health): live when the newest reading is "
            "within live_max_age_minutes and coverage >= live_min_coverage, offline without fresh readings"
        ),
    )
    telemetry_label: str
    detail: str = Field(..., description="Mapped parameters reporting, e.g. '40 of 41 mapped parameters reporting'")
    coverage: float = Field(..., description="Share of mapped parameters with a fresh reading, fraction 0-1")
    stale_after_minutes: float = Field(..., description="Age from which a reading is not shown, min")
    live_max_age_minutes: float = Field(..., description="Largest age of the newest reading for live telemetry, min")
    live_min_coverage: float = Field(..., description="Smallest coverage for live telemetry, fraction 0-1")
    failed_sources: list[str] = Field(default_factory=list, description="InfluxDB measurements that could not be read")
    source: Literal["producer", "request"] = Field(
        ..., description="producer: the worker's live snapshot (Redis, every 10 s); request: taken for this request"
    )


class StatusSnapshotView(BaseModel):
    """Latest 15-min snapshot of every furnace-status catalogue parameter, as the Streamlit V-Board shows it."""

    parameters: list[StatusParameter] = Field(..., description="Every catalogue parameter, in display order")
    sections: list[str] = Field(..., description="Section names, in display order")
    zones: list[StatusZone] = Field(..., description="Furnace zones, top to bottom, with their related readings")
    operating_state: StatusOperatingState
    freshness: StatusFreshness


class StatusTrendStats(BaseModel):
    minimum: float = Field(..., description="Lowest reading of the range, in unit")
    maximum: float = Field(..., description="Highest reading of the range, in unit")
    mean: float = Field(..., description="Mean reading of the range, in unit")


class StatusTrendSeries(BaseModel):
    key: str = Field(..., description="Catalogue key")
    label: str
    title: str
    unit: str = Field(..., description="Unit of values, setpoints and stats; empty when not confirmed")
    decimals: int = Field(..., description="Decimals the V-Board shows, count")
    status: Literal["ok", "no_source", "no_data", "error"]
    message: str = Field("", description="Why there is no series (no_source, no_data, error)")
    times: list[datetime] = Field(default_factory=list, description="Sample times (UTC)")
    values: list[float | None] = Field(
        default_factory=list, description="Readings; null marks a gap in the data (break the line there), in unit"
    )
    setpoint_times: list[datetime] = Field(default_factory=list, description="Setpoint sample times (UTC)")
    setpoints: list[float | None] = Field(default_factory=list, description="Setpoints; null marks a gap, in unit")
    raw_points: int = Field(0, description="Samples of the range before decimation (gap marks included), count")
    stats: StatusTrendStats | None = Field(None, description="Minimum, maximum and mean of the raw range")
    current: StatusParameter = Field(..., description="Current reading, as the V-Board trend header")


class StatusTrend(BaseModel):
    interval: str = Field(..., description="1h, 4h, 8h, 16h, 24h or custom")
    start: datetime = Field(..., description="Start of the range (UTC)")
    end: datetime = Field(..., description="End of the range (UTC)")
    max_points: int = Field(..., description="Most points per series after decimation, count")
    series: list[StatusTrendSeries]


# ---------------------------------------------------------------- limits and readings


def _utc(value: pd.Timestamp | datetime | None) -> datetime | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(PLANT_TZ) if stamp.tzinfo is None else stamp
    return stamp.tz_convert("UTC").to_pydatetime(warn=False)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@functools.cache
def _register() -> Any:
    from utils.burden_engine.facts import load_register

    return load_register()


@functools.cache
def parameter_limits(key: str) -> StatusLimits | None:
    """Limits of a catalogue parameter from the process fact register; None when it has none or they do not resolve
    (logged; ``tests/test_services_status.py`` checks that every reference resolves)."""
    ref = PARAMETERS_BY_KEY[key].limits
    if ref is None:
        return None
    try:
        fact = _register().active(ref.fact)
        low = _finite(fact.params[ref.low]) if ref.low else None
        high = _finite(fact.params[ref.high]) if ref.high else None
    except Exception:  # noqa: BLE001 - a broken reference must not take the snapshot down
        log.exception("furnace status: limits of %s (%s) do not resolve in the fact register", key, ref)
        return None
    if low is None and high is None:
        return None
    return StatusLimits(low=low, high=high, fact=fact.id, basis=fact.statement)


def reading_state(reading: ParameterReading, limits: StatusLimits | None) -> ParameterState:
    """The reading's freshness issue, else its place against its limits (unrounded)."""
    if reading.issue is not None:
        return reading.issue
    if reading.value is None:
        return "no_data"
    if limits is not None:
        if limits.low is not None and reading.value < limits.low:
            return "low"
        if limits.high is not None and reading.value > limits.high:
            return "high"
    return "ok"


def parameter_view(reading: ParameterReading) -> StatusParameter:
    spec: ParameterSpec = reading.spec
    limits = parameter_limits(spec.key)
    return StatusParameter(
        key=spec.key,
        label=spec.label,
        title=spec.title,
        section=spec.section,
        unit=spec.unit,
        decimals=spec.decimals,
        value=reading.value,
        at=_utc(reading.timestamp),
        setpoint=reading.setpoint,
        setpoint_at=_utc(reading.setpoint_timestamp),
        display=format_reading(reading),
        state=reading_state(reading, limits),
        limits=limits,
        note=spec.source_note or spec.unavailable_reason,
    )


# ---------------------------------------------------------------- snapshot


def live_operating_state(level: str) -> Literal["normal", "reduced"] | None:
    """The V-Board level in the wording of the live.status event: abnormal (below normal_min) is reduced."""
    if level == "normal":
        return "normal"
    if level == "abnormal":
        return "reduced"
    return None


def snapshot_view(
    snapshot: StatusSnapshot, computed_at: datetime, *, source: Literal["producer", "request"] = "request"
) -> StatusSnapshotView:
    """The API view of a V-Board snapshot taken at ``computed_at``."""
    by_key = {reading.spec.key: reading for reading in snapshot.readings}
    policy = SETTINGS.operating_state
    deciding = by_key.get(policy.parameter)
    status = snapshot.status
    state = snapshot.operating_state
    return StatusSnapshotView(
        parameters=[parameter_view(reading) for reading in snapshot.readings],
        sections=list(SECTIONS),
        zones=[
            StatusZone(name=zone.name, lower_m=zone.lower_m, upper_m=zone.upper_m, readings=list(zone.readings))
            for zone in UI.zones
        ],
        operating_state=StatusOperatingState(
            level=state.level,
            state=live_operating_state(state.level),
            label=state.label,
            detail=state.detail,
            parameter=policy.parameter,
            normal_min=policy.normal_min,
            value=deciding.value if deciding is not None else None,
        ),
        freshness=StatusFreshness(
            computed_at=_utc(computed_at),
            data_at=_utc(snapshot.last_updated),
            age_s=status.age.total_seconds() if status.age is not None else None,
            stale=status.level == "offline",
            telemetry=status.level,
            telemetry_label=status.label,
            detail=status.detail,
            coverage=status.coverage,
            stale_after_minutes=SETTINGS.stale_after.total_seconds() / 60,
            live_max_age_minutes=SETTINGS.live_max_age.total_seconds() / 60,
            live_min_coverage=SETTINGS.live_min_coverage,
            failed_sources=list(snapshot.failed_measurements),
            source=source,
        ),
    )


def build_snapshot_view(
    now: datetime | None = None, *, source: Literal["producer", "request"] = "request"
) -> StatusSnapshotView:
    """One snapshot query (the V-Board's ``load_status_snapshot``: one InfluxDB request per measurement) as the view."""
    at = now or clock()
    return snapshot_view(load_status_snapshot(at), at, source=source)


_current_lock = threading.Lock()


@memo(max_entries=2)
def _snapshot_of_slot(slot: int) -> StatusSnapshotView:
    return build_snapshot_view(clock())


def current_snapshot_view() -> StatusSnapshotView:
    """The snapshot of the current 10-s slot, taken once per slot and process: concurrent callers wait for the one
    query and share its result (do not mutate it)."""
    slot = int(clock().timestamp() // PERIOD_S)
    with _current_lock:
        return _snapshot_of_slot(slot)


# ---------------------------------------------------------------- live status


def live_status_fields(view: StatusSnapshotView, hot_metal: Mapping[str, TagValue] | None = None) -> dict[str, Any]:
    """Plant part of the ``live.status`` event (``services.events.LiveStatusData``) from a snapshot view: a tag per
    sourced catalogue parameter (value null without a fresh reading), the hot-metal tags, the operating state, the data
    time and the stale flag."""
    tags = {p.key: TagValue(value=p.value, unit=p.unit, at=p.at) for p in view.parameters if p.state != "no_source"}
    tags.update(hot_metal or {})
    fresh = view.freshness
    return {
        "tags": tags,
        "operating_state": view.operating_state.state,
        "data_at": fresh.data_at,
        "stale": fresh.stale,
        "telemetry": fresh.telemetry,
        "stale_after_minutes": fresh.stale_after_minutes,
        "live_max_age_minutes": fresh.live_max_age_minutes,
    }


def hot_metal_samples(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """HMT per tap (°C, the pipeline's tap filter ``utils.coke_energy.hmt_forecast.taps``, >= 1,350 °C) and Si per
    sample (%, finite and above zero) of Neon ``offline_feed.hot_metal_slag_analysis`` between ``start`` and ``end``
    (IST, tz-naive; ``data.coke_energy.sources.fetch_hm_slag``), oldest first, indexed by sample time (IST). Raises
    when Neon is not reachable."""
    from utils.coke_energy.hmt_forecast import taps

    if fetch is None:
        from data.coke_energy.sources import fetch_hm_slag as fetch
    hm = fetch(start, end)
    empty = pd.Series(dtype=float)
    if hm is None or hm.empty or "t" not in hm:
        return empty, empty
    temps = taps(hm).sort_index()
    si = empty
    if "chem_pct_si" in hm:
        si = pd.Series(pd.to_numeric(hm["chem_pct_si"], errors="coerce").to_numpy(), index=pd.to_datetime(hm["t"]))
        si = si[np.isfinite(si) & (si > 0)].sort_index()
    return temps, si


def latest_hot_metal(
    now: datetime | None = None, fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None
) -> dict[str, TagValue]:
    """``hmt`` (°C) and ``hm_si`` (%) of the latest valid hot-metal samples of the last 24 h in Neon
    (``hot_metal_samples``; ``at`` = sample time). Empty when Neon is not reachable or holds no sample."""
    end = pd.Timestamp(now or clock()).tz_convert(PLANT_TZ).tz_localize(None)
    try:
        temps, si = hot_metal_samples(end - pd.Timedelta(HOT_METAL_LOOKBACK), end + pd.Timedelta(minutes=1), fetch)
    except Exception as exc:  # noqa: BLE001 - Neon unreachable: the tags are left out
        log.warning("furnace status: hot-metal samples not read (%s)", type(exc).__name__)
        return {}
    out: dict[str, TagValue] = {}
    if not temps.empty:
        out["hmt"] = TagValue(value=float(temps.iloc[-1]), unit="°C", at=_utc(temps.index[-1]))
    if not si.empty:
        out["hm_si"] = TagValue(value=float(si.iloc[-1]), unit="%", at=_utc(si.index[-1]))
    return out


# ---------------------------------------------------------------- trends


def trend_range(
    interval: str, start: datetime | None, end: datetime | None, now: datetime
) -> tuple[datetime, datetime]:
    """UTC range of a fixed interval ending now, or of a custom range (naive times are IST; at most
    ``max_custom_range_days``; end clamped to now), as the V-Board trend view. Raises ``RangeError``."""
    if interval == CUSTOM:
        if start is None or end is None:
            raise RangeError("A custom interval needs start and end.")
        return resolve_custom_range(start, end, now)
    return resolve_fixed_range(interval, now)


def decimate(series: pd.Series | None, max_points: int) -> pd.Series | None:
    """At most ``max_points`` points of a trend series.

    A series that fits is returned as it is (with the V-Board's gap marks). A longer one is cut into
    ``max_points // 2`` equal time buckets; each keeps its lowest and highest sample in time order, so peaks survive. A
    gap mark (NaN) goes between consecutive kept samples further apart than two buckets and the raw gap limit: a hole
    in the data wider than the drawing resolves. Every hole spans an empty bucket, so kept samples plus marks stay
    within ``max_points``.
    """
    if series is None or len(series) <= max_points:
        return series
    finite = series[np.isfinite(series.to_numpy(dtype="float64"))]
    buckets = max(1, max_points // 2)
    t = finite.index.asi8
    if len(finite) <= 2 or t[-1] <= t[0]:
        return finite.iloc[[0, -1]] if len(finite) > 2 else finite
    width = (t[-1] - t[0]) / buckets
    bucket = np.minimum(((t - t[0]) / width).astype(np.int64), buckets - 1)
    grouped = pd.Series(finite.to_numpy(dtype="float64")).groupby(bucket)
    keep = np.unique(np.concatenate([grouped.idxmin().to_numpy(), grouped.idxmax().to_numpy()]))
    kept = finite.iloc[keep]
    stamps = kept.index.asi8
    holes = np.flatnonzero(np.diff(stamps) > max(2 * width, SETTINGS.raw_gap_break_seconds * 1e9))
    if holes.size == 0:
        return kept
    middles = pd.to_datetime(stamps[holes] + (stamps[holes + 1] - stamps[holes]) // 2, utc=True)
    marks = pd.Series(
        np.nan,
        index=pd.DatetimeIndex(middles, name=kept.index.name).tz_convert(kept.index.tz),
        name=kept.name,
        dtype="float64",
    )
    return pd.concat([kept, marks]).sort_index(kind="stable")


def _points(series: pd.Series | None) -> tuple[list[datetime], list[float | None]]:
    if series is None or series.empty:
        return [], []
    index = pd.DatetimeIndex(series.index)
    index = index.tz_localize(PLANT_TZ) if index.tz is None else index
    times = list(index.tz_convert("UTC").to_pydatetime())
    values = [float(v) if math.isfinite(v) else None for v in series.to_numpy(dtype="float64")]
    return times, values


def trend_series_view(view: TrendViewData, max_points: int) -> StatusTrendSeries:
    """One parameter's trend (the V-Board's ``load_trend_view``) for the API: decimated series, raw-range stats."""
    trend = view.trend
    spec = trend.spec
    times, values = _points(decimate(trend.series, max_points))
    setpoint_times, setpoints = _points(decimate(trend.setpoint, max_points))
    stats = view.stats
    return StatusTrendSeries(
        key=spec.key,
        label=spec.label,
        title=spec.title,
        unit=spec.unit,
        decimals=spec.decimals,
        status=trend.status,
        message=trend.message,
        times=times,
        values=values,
        setpoint_times=setpoint_times,
        setpoints=setpoints,
        raw_points=0 if trend.series is None else len(trend.series),
        stats=None
        if stats is None
        else StatusTrendStats(minimum=stats.minimum, maximum=stats.maximum, mean=stats.mean),
        current=parameter_view(view.current),
    )


def build_trend(
    keys: Sequence[str],
    interval: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    max_points: int = DEFAULT_TREND_POINTS,
    now: datetime | None = None,
) -> StatusTrend:
    """Trends of catalogue parameters over a fixed interval ending now or a custom range: the V-Board's
    ``load_trend_view`` per parameter (one InfluxDB request each, cached 5 s). Raises ``KeyError`` for an unknown
    key, ``RangeError`` for an unknown interval or a bad range."""
    if interval not in TREND_INTERVALS:
        raise RangeError(f"Unknown interval: {interval!r}")
    unknown = [key for key in keys if key not in PARAMETERS_BY_KEY]
    if unknown:
        raise KeyError(", ".join(unknown))
    at = now or clock()
    start_utc, end_utc = trend_range(interval, start, end, at)
    series = [
        trend_series_view(
            load_trend_view(PARAMETERS_BY_KEY[key], start_utc, end_utc, use_live_current=interval != CUSTOM),
            max_points,
        )
        for key in keys
    ]
    return StatusTrend(interval=interval, start=start_utc, end=end_utc, max_points=max_points, series=series)


def clear_cache() -> None:
    """Empty the InfluxDB request cache and the slot snapshot."""
    _clear_repository_cache()
    _snapshot_of_slot.clear()
