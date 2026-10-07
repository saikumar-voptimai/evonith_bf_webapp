"""Data layer for the Furnace Status page.

Holds the declarative parameter catalogue, value resolution, time-range helpers
and the cached fetchers.  Everything except the two ``st.cache_data`` wrappers at
the bottom is plain Python, so it can be unit-tested without a Streamlit
runtime or a live InfluxDB.  All live data goes through the existing
:func:`furnace_data.influx.online.fetch_online_df`; this module never talks to
InfluxDB itself.

Conventions
-----------
* Displayed timestamps are Asia/Kolkata (IST); values handed to the fetcher's
  ``start_time_override`` / ``end_time_override`` are UTC.
* A value that cannot be shown honestly is ``None`` and renders as
  :data:`NOT_AVAILABLE`.  Zero is a legitimate value and is never treated as
  missing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from furnace_data.influx.online import fetch_online_df
from utils.logger import get_logger

log = get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

NOT_AVAILABLE = "Not available"

# ── Fetch / freshness policy ──────────────────────────────────────────────────
STATUS_LOOKBACK = "last 15 minutes"
STATUS_WINDOW = "1 minute"
CACHE_TTL_SECONDS = 60

#: A value older than this is treated as missing.  Equal to the status lookback,
#: so anything the fetch returns is eligible; the ring (below) is what tells the
#: operator the data is getting old.
STALE_AFTER = timedelta(minutes=15)
#: Telemetry no older than this counts as "live".
LIVE_MAX_AGE = timedelta(minutes=5)
#: Share of source-backed parameters that must have a value to be "live".
LIVE_MIN_COVERAGE = 0.70

# ── Trend ranges ──────────────────────────────────────────────────────────────
FIXED_INTERVALS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "8h": timedelta(hours=8),
    "16h": timedelta(hours=16),
    "24h": timedelta(hours=24),
}
CUSTOM_INTERVAL = "Custom"
INTERVAL_OPTIONS: tuple[str, ...] = (*FIXED_INTERVALS, CUSTOM_INTERVAL)
DEFAULT_INTERVAL = "8h"
MAX_CUSTOM_RANGE = timedelta(days=90)

#: ``(longest duration served, aggregation window)``; first match wins.  Chosen so
#: every range stays at or below ~150 points, and so the fixed intervals map to
#: 1h→1m, 4h→5m, 8h→5m, 16h→10m, 24h→15m.
_WINDOW_BY_DURATION: tuple[tuple[timedelta, str], ...] = (
    (timedelta(hours=1), "1 minute"),
    (timedelta(hours=8), "5 minutes"),
    (timedelta(hours=16), "10 minutes"),
    (timedelta(hours=24), "15 minutes"),
    (timedelta(hours=72), "30 minutes"),
)
_WIDEST_WINDOW = "1 hour"

#: ``fetch_online_df`` requires ``time_range`` positionally but ignores it once
#: ``start_time_override`` is given.
_IGNORED_TIME_RANGE = "last 1 hour"


# ══════════════════════════════════════════════════════════════════════════════
# Parameter catalogue
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParameterSpec:
    """One display parameter and where (if anywhere) its live value comes from.

    A spec with no ``measurement``/``field`` has no confirmed live source; it is
    still shown, as :data:`NOT_AVAILABLE`, with ``unavailable_reason`` explaining
    why.
    """

    key: str
    label: str
    section: str
    measurement: str | None = None
    field: str | None = None
    unit: str = ""
    decimals: int = 1
    scale: float = 1.0
    #: Title used on the trend view, when it differs from ``label``.
    trend_label: str | None = None
    #: Setpoint field in the same measurement; the dashboard then shows
    #: ``<setpoint> / <actual>`` and the trend charts both.
    setpoint_field: str | None = None
    #: Derived parameter: the value combines these fields (same measurement)
    #: with ``aggregate``; ``field`` is only the name of the computed column.
    components: tuple[str, ...] = ()
    aggregate: Literal["sum", "mean"] = "sum"
    source_note: str = ""
    unavailable_reason: str = ""

    @property
    def has_source(self) -> bool:
        return self.measurement is not None and self.field is not None

    @property
    def source_fields(self) -> tuple[str, ...]:
        """Fields fetched for this parameter: actual (or its components), then setpoint."""
        actual = self.components or (self.field,)
        return tuple(f for f in (*actual, self.setpoint_field) if f)

    @property
    def title(self) -> str:
        return self.trend_label or self.label


SECTION_PRODUCTION = "Production"
SECTION_BLAST = "Blast and Gas"
SECTION_INJECTION = "Steam and O₂"
SECTION_PERFORMANCE = "Furnace Performance"
SECTION_UPTAKE = "Uptake Temperatures"
SECTION_HEAT_LOAD = "Total Heat Load"
SECTION_HEARTH = "Hearth Temperatures"

SECTIONS: tuple[str, ...] = (
    SECTION_PRODUCTION,
    SECTION_BLAST,
    SECTION_UPTAKE,
    SECTION_INJECTION,
    SECTION_PERFORMANCE,
    SECTION_HEAT_LOAD,
    SECTION_HEARTH,
)

_PROC = "process_params"
_MISC = "miscellaneous"
_HEAT = "heatload_delta_t"
_TEMP = "temperature_profile"

#: Hearth pads A–D: live InfluxDB fields of ``temperature_profile`` (the same
#: fields setting_ds_dv.yml maps to hearth_pad_*_c for the ML dataset).
_HEARTH_PADS = {
    "A": "temp_4373_a",
    "B": "temp_5411_b",
    "C": "temp_5757_c",
    "D": "temp_6103_d",
}

_STAVE_ROWS = range(6, 11)  # R6–R10
_QUADRANT_STAVES = {1: "1–8", 2: "9–16", 3: "17–24", 4: "25–32"}


def _quadrant_fields(quadrant: int) -> tuple[str, ...]:
    """Per-row heat-load fields of one quadrant, R6–R10."""
    return tuple(f"heat_load_r{row}_q{quadrant}" for row in _STAVE_ROWS)


def _uptake_spec(key: str, label: str, field: str, title: str) -> ParameterSpec:
    return ParameterSpec(
        key,
        label,
        SECTION_UPTAKE,
        _PROC,
        field,
        unit="°C",
        decimals=0,
        trend_label=title,
    )


def _heat_load_spec(quadrant: int) -> ParameterSpec:
    staves = _QUADRANT_STAVES[quadrant]
    return ParameterSpec(
        f"heat_load_q{quadrant}",
        f"Q{quadrant} (Staves {staves})",
        SECTION_HEAT_LOAD,
        _HEAT,
        f"heat_load_q{quadrant}_sum",
        unit="MW",
        decimals=2,
        trend_label=f"Heat Load Q{quadrant} (Staves {staves}, R6–R10)",
        components=_quadrant_fields(quadrant),
        source_note=f"Sum of rows R6–R10, staves {staves}.",
    )


# Order is the display order of the reference screen; do not re-sort.
PARAMETERS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        "production_theoretical",
        "Production (Theor.)",
        SECTION_PRODUCTION,
        _PROC,
        "theoretical_production_per_day",
        unit="t/day",
        decimals=0,
    ),
    ParameterSpec(
        "production_rate",
        "Production Rate",
        SECTION_PRODUCTION,
        _PROC,
        "production_per_hour",
        unit="TPH",
        decimals=1,
    ),
    ParameterSpec(
        "hot_blast_volume",
        "Hot Blast Volume",
        SECTION_BLAST,
        _PROC,
        "hot_blast_vol_nm3h",
        unit="Nm³/h",
        decimals=0,
    ),
    ParameterSpec(
        "blast_pressure",
        "Blast Pressure",
        SECTION_BLAST,
        _PROC,
        "hot_blast_press",
        unit="bar",
        decimals=2,
    ),
    ParameterSpec(
        "top_pressure",
        "Top Pressure",
        SECTION_BLAST,
        _PROC,
        "top_press_avg",
        unit="bar",
        decimals=2,
    ),
    ParameterSpec(
        "co_utilization",
        "CO Utilization",
        SECTION_BLAST,
        _PROC,
        "body_etaco",
        decimals=5,
        source_note="ETA CO as stored; the raw scale is preserved, not converted to %.",
    ),
    ParameterSpec(
        "top_gas_h2",
        "Top Gas H₂",
        SECTION_BLAST,
        _PROC,
        "h2_pct",
        unit="%",
        decimals=2,
    ),
    ParameterSpec(
        "hbt",
        "HBT",
        SECTION_BLAST,
        _PROC,
        "hot_blast_temp",
        unit="°C",
        decimals=0,
    ),
    ParameterSpec(
        "raft",
        "RAFT",
        SECTION_BLAST,
        _PROC,
        "body_raft",
        unit="°C",
        decimals=0,
    ),
    _uptake_spec("uptake_t1", "T1", "top_temp_1", "Uptake Temperature T1"),
    _uptake_spec("uptake_t2", "T2", "top_temp_2", "Uptake Temperature T2"),
    _uptake_spec("uptake_t3", "T3", "top_temp_3", "Uptake Temperature T3"),
    _uptake_spec("uptake_t4", "T4", "top_temp_4", "Uptake Temperature T4"),
    _uptake_spec("uptake_avg", "T Avg", "top_temp_avg", "Uptake Temperature Average"),
    ParameterSpec(
        "steam_injection",
        "Steam Injection",
        SECTION_INJECTION,
        _PROC,
        "steam_injection",
        unit="TPH",
        decimals=2,
        scale=0.001,
        source_note="Source field is in kg/h; shown in TPH (÷ 1000).",
    ),
    ParameterSpec(
        "steam_bypass_flow",
        "Steam Bypass Flow",
        SECTION_INJECTION,
        unavailable_reason="No confirmed live source is configured for steam bypass flow.",
    ),
    ParameterSpec(
        "o2_injection",
        "O₂ Injection",
        SECTION_INJECTION,
        _PROC,
        "oxygen_enrichment_pct",
        unit="%",
        decimals=2,
        source_note="Represents oxygen enrichment (%) of the blast.",
    ),
    ParameterSpec(
        "oxygen_flow",
        "Oxygen Flow",
        SECTION_INJECTION,
        _PROC,
        "oxygen_flow",
        unit="Nm³/h",
        decimals=0,
    ),
    ParameterSpec(
        "fuel_rate",
        "Fuel Rate",
        SECTION_PERFORMANCE,
        _PROC,
        "fuel_rate",
        unit="kg/THM",
        decimals=0,
    ),
    ParameterSpec(
        "pci_rate",
        "PCI Rate (SP/ACT.)",
        SECTION_PERFORMANCE,
        _PROC,
        "coal_rate_actual_value",
        unit="kg/THM",
        decimals=0,
        trend_label="PCI Rate (SP / Actual)",
        setpoint_field="coal_rate_set_value",
    ),
    ParameterSpec(
        "slag_rate",
        "Slag Rate",
        SECTION_PERFORMANCE,
        unavailable_reason=(
            "Slag rate exists only in the daily production report. There is no "
            "high-frequency live field, so no 1–24 hour trend can be built."
        ),
    ),
    ParameterSpec(
        "permeability",
        "Permeability",
        SECTION_PERFORMANCE,
        _PROC,
        "body_perm",
        decimals=1,
        source_note="Unit not confirmed in the application, so none is shown.",
    ),
    ParameterSpec(
        "tuyere_velocity",
        "Tuyere Velocity",
        SECTION_PERFORMANCE,
        _PROC,
        "tuyere_velocity",
        unit="m/s",
        decimals=1,
    ),
    ParameterSpec(
        "furnace_level",
        "Furnace Level",
        SECTION_PERFORMANCE,
        _MISC,
        "stock_rod_radar_level",
        unit="m",
        decimals=2,
    ),
    ParameterSpec(
        "heat_load_total",
        "Total Heat Load",
        SECTION_HEAT_LOAD,
        _HEAT,
        "heat_load_total_sum",
        unit="MW",
        decimals=2,
        components=tuple(f for q in _QUADRANT_STAVES for f in _quadrant_fields(q)),
        source_note="Q1 + Q2 + Q3 + Q4 (rows R6–R10).",
    ),
    *(_heat_load_spec(q) for q in _QUADRANT_STAVES),
    *(
        ParameterSpec(
            f"hearth_temp_{pad.lower()}",
            f"Pad {pad}",
            SECTION_HEARTH,
            _TEMP,
            field,
            unit="°C",
            decimals=0,
            trend_label=f"Hearth Temp {pad}",
            source_note=f"Hearth pad {pad} (HEARTH_TEMP_{pad}).",
        )
        for pad, field in _HEARTH_PADS.items()
    ),
    ParameterSpec(
        "hearth_temp_avg",
        "Pad Avg",
        SECTION_HEARTH,
        _TEMP,
        "hearth_temp_avg",
        unit="°C",
        decimals=0,
        components=tuple(_HEARTH_PADS.values()),
        aggregate="mean",
        trend_label="Hearth Temp Avg",
        source_note="Mean of hearth pads A–D (HEARTH_TEMP_AVG).",
    ),
)

PARAMETERS_BY_KEY: dict[str, ParameterSpec] = {p.key: p for p in PARAMETERS}


def parameters_in_section(
    section: str, specs: Sequence[ParameterSpec] = PARAMETERS
) -> tuple[ParameterSpec, ...]:
    """Return the specs of one section, in catalogue order."""
    return tuple(p for p in specs if p.section == section)


def measurements_for(specs: Sequence[ParameterSpec] = PARAMETERS) -> tuple[str, ...]:
    """Return the distinct source measurements, in first-use order."""
    return tuple(dict.fromkeys(p.measurement for p in specs if p.measurement))


# ══════════════════════════════════════════════════════════════════════════════
# View state (query parameters)
# ══════════════════════════════════════════════════════════════════════════════

VIEW_STATUS = "status"
VIEW_TREND = "trend"


@dataclass(frozen=True)
class ViewState:
    """Which full-page view to render, resolved from the query string."""

    view: Literal["status", "trend"]
    spec: ParameterSpec | None = None
    #: The query string was malformed or named an unknown parameter and should
    #: be rewritten to ``?view=status``.
    needs_reset: bool = False


def _single_value(raw: object) -> str | None:
    """Reduce a query-parameter value (str, or list of str) to one string."""
    if isinstance(raw, (list, tuple)):
        raw = raw[-1] if raw else None
    return raw if isinstance(raw, str) else None


def parse_view_state(params: Mapping[str, object]) -> ViewState:
    """Validate ``?view=...&parameter=...`` against the catalogue.

    The parameter key is only ever *looked up* in :data:`PARAMETERS_BY_KEY`; the
    raw string is never returned, so nothing user-supplied reaches HTML or a
    query.  Anything unrecognised falls back to the status view.
    """
    view = _single_value(params.get("view"))
    key = _single_value(params.get("parameter"))

    if view == VIEW_TREND:
        spec = PARAMETERS_BY_KEY.get(key) if key is not None else None
        if spec is not None:
            return ViewState(VIEW_TREND, spec)
        return ViewState(VIEW_STATUS, needs_reset=True)

    if view is None or view == VIEW_STATUS:
        return ViewState(VIEW_STATUS, needs_reset=key is not None)
    return ViewState(VIEW_STATUS, needs_reset=True)


# ══════════════════════════════════════════════════════════════════════════════
# Value resolution
# ══════════════════════════════════════════════════════════════════════════════

ReadingIssue = Literal["no_source", "fetch_failed", "no_data", "stale"]


def _to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)


def finite_series(frame: pd.DataFrame | None, column: str | None) -> pd.Series | None:
    """Return ``frame[column]`` as float with non-finite values dropped.

    ``None`` when the frame/column is absent or nothing finite remains.  Zero is
    finite and is kept.
    """
    if frame is None or column is None or frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    series = series[np.isfinite(series)].sort_index()
    return series if not series.empty else None


def latest_finite_point(
    frame: pd.DataFrame | None, column: str | None
) -> tuple[float, pd.Timestamp] | None:
    """Return ``(value, IST timestamp)`` of the newest finite sample, or ``None``.

    No forward-fill: if the newest samples are null, the newest *real* sample is
    returned together with its own (older) timestamp.
    """
    series = finite_series(frame, column)
    if series is None:
        return None
    return float(series.iloc[-1]), _to_ist(series.index[-1])


@dataclass(frozen=True)
class ParameterReading:
    """Resolved dashboard value for one parameter (already scaled)."""

    spec: ParameterSpec
    value: float | None = None
    timestamp: pd.Timestamp | None = None
    issue: ReadingIssue | None = None
    #: Scaled setpoint, when the spec has one and it is fresh.
    setpoint: float | None = None


def resolve_reading(
    spec: ParameterSpec,
    frames: Mapping[str, pd.DataFrame | None],
    now: datetime,
    *,
    stale_after: timedelta = STALE_AFTER,
) -> ParameterReading:
    """Resolve one parameter from the per-measurement status frames.

    ``frames[measurement] is None`` (or a missing key) means that measurement's
    fetch failed.
    """
    if not spec.has_source:
        return ParameterReading(spec, issue="no_source")

    frame = frames.get(spec.measurement)
    if frame is None:
        return ParameterReading(spec, issue="fetch_failed")

    def fresh(field: str | None) -> tuple[float, pd.Timestamp] | ReadingIssue:
        point = latest_finite_point(frame, field)
        if point is None:
            return "no_data"
        raw, ts = point
        if now - ts.to_pydatetime() > stale_after:
            return "stale"
        return raw * spec.scale, ts

    if spec.components:
        parts = [fresh(f) for f in spec.components]
        missing = next((p for p in parts if not isinstance(p, tuple)), None)
        if missing is not None:  # never show a partial sum/average
            return ParameterReading(spec, issue=missing)
        total = sum(v for v, _ in parts)
        return ParameterReading(
            spec,
            value=total / len(parts) if spec.aggregate == "mean" else total,
            timestamp=min(ts for _, ts in parts),  # as old as its oldest input
        )

    sp = fresh(spec.setpoint_field) if spec.setpoint_field else None
    setpoint = sp[0] if isinstance(sp, tuple) else None

    actual = fresh(spec.field)
    if not isinstance(actual, tuple):
        return ParameterReading(spec, issue=actual, setpoint=setpoint)
    value, ts = actual
    return ParameterReading(spec, value=value, timestamp=ts, setpoint=setpoint)


@dataclass(frozen=True)
class PlantStatus:
    """Telemetry-availability indicator (not a furnace safety/process state)."""

    level: Literal["live", "partial", "offline"]
    label: str
    detail: str
    coverage: float
    age: timedelta | None


def compute_plant_status(
    readings: Sequence[ParameterReading], now: datetime
) -> PlantStatus:
    """Classify telemetry as live / partial / offline.

    * live    — newest sample ≤ 5 min old and ≥ 70 % of source-backed parameters
      have a value
    * partial — newest sample ≤ 15 min old, but older or incomplete
    * offline — nothing recent
    """
    backed = [r for r in readings if r.spec.has_source]
    with_value = [r for r in backed if r.value is not None]
    coverage = len(with_value) / len(backed) if backed else 0.0
    detail = f"{len(with_value)} of {len(backed)} mapped parameters reporting"

    stamps = [r.timestamp for r in with_value if r.timestamp is not None]
    if not stamps:
        return PlantStatus("offline", "Offline", detail, coverage, None)

    age = max(now - max(stamps).to_pydatetime(), timedelta(0))
    if age > STALE_AFTER:
        return PlantStatus("offline", "Offline", detail, coverage, age)
    if age <= LIVE_MAX_AGE and coverage >= LIVE_MIN_COVERAGE:
        return PlantStatus("live", "Live", detail, coverage, age)
    return PlantStatus("partial", "Partial", detail, coverage, age)


@dataclass(frozen=True)
class StatusSnapshot:
    """Everything the dashboard needs for one render."""

    readings: tuple[ParameterReading, ...]
    status: PlantStatus
    #: Timestamp of the newest sample behind any displayed value (IST).
    last_updated: pd.Timestamp | None
    #: Measurements whose fetch raised; shown as a quiet caption, no details.
    failed_measurements: tuple[str, ...]


def build_status_snapshot(
    frames: Mapping[str, pd.DataFrame | None],
    now: datetime,
    specs: Sequence[ParameterSpec] = PARAMETERS,
) -> StatusSnapshot:
    """Resolve every parameter and the plant status from fetched frames."""
    readings = tuple(resolve_reading(spec, frames, now) for spec in specs)
    stamps = [r.timestamp for r in readings if r.timestamp is not None]
    failed = tuple(m for m in measurements_for(specs) if frames.get(m) is None)
    return StatusSnapshot(
        readings=readings,
        status=compute_plant_status(readings, now),
        last_updated=max(stamps) if stamps else None,
        failed_measurements=failed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Formatting (pure)
# ══════════════════════════════════════════════════════════════════════════════


def format_value(value: float | None, decimals: int) -> str:
    """Format a number, or return exactly :data:`NOT_AVAILABLE` if it is missing."""
    if value is None or not math.isfinite(value):
        return NOT_AVAILABLE
    if round(value, decimals) == 0:
        value = 0.0  # avoid "-0"
    return f"{value:,.{decimals}f}"


def format_reading(reading: ParameterReading) -> str:
    """Return the dashboard value text (no unit) for one reading.

    Setpoint/actual parameters render as ``<setpoint> / <actual>``; each side is
    ``Not available`` on its own, and the text is plain ``Not available`` only
    when both are missing.
    """
    spec = reading.spec
    actual = format_value(reading.value, spec.decimals)
    if not spec.setpoint_field or not has_display_value(reading):
        return actual
    return f"{format_value(reading.setpoint, spec.decimals)} / {actual}"


def has_display_value(reading: ParameterReading) -> bool:
    """True when the dashboard shows at least one number for this reading."""
    return reading.value is not None or reading.setpoint is not None


def format_ist(ts: pd.Timestamp | datetime | None, *, with_date: bool = True) -> str:
    """Format a timestamp in IST, or ``—`` when absent."""
    if ts is None:
        return "—"
    local = _to_ist(pd.Timestamp(ts))
    pattern = "%d %b %Y, %H:%M:%S IST" if with_date else "%H:%M:%S IST"
    return local.strftime(pattern)


# ══════════════════════════════════════════════════════════════════════════════
# Time ranges
# ══════════════════════════════════════════════════════════════════════════════


class RangeError(ValueError):
    """A requested trend range is invalid; the message is safe to show."""


def floor_to_minute(moment: datetime) -> datetime:
    """Drop seconds/microseconds so repeated reruns share a cache key."""
    return moment.replace(second=0, microsecond=0)


def resolve_fixed_range(interval: str, now: datetime) -> tuple[datetime, datetime]:
    """Return ``(start_utc, end_utc)`` for a fixed interval label such as ``"4h"``."""
    if interval not in FIXED_INTERVALS:
        raise RangeError(f"Unknown interval: {interval!r}")
    end = floor_to_minute(now.astimezone(timezone.utc))
    return end - FIXED_INTERVALS[interval], end


def resolve_custom_range(
    start_local: datetime, end_local: datetime, now: datetime
) -> tuple[datetime, datetime]:
    """Validate a custom IST range and return ``(start_utc, end_utc)``.

    Naive datetimes are interpreted as IST.  An end in the future is clamped to
    ``now`` so the request never asks for data that cannot exist.

    Raises:
        RangeError: start is not before end, or the range exceeds 90 days.
    """

    def as_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=IST)
        return moment.astimezone(timezone.utc)

    start = as_utc(start_local)
    end = min(as_utc(end_local), floor_to_minute(now.astimezone(timezone.utc)))

    if start >= end:
        raise RangeError("Start must be earlier than end (and not in the future).")
    if end - start > MAX_CUSTOM_RANGE:
        raise RangeError(
            f"Range is too large; choose at most {MAX_CUSTOM_RANGE.days} days."
        )
    return start, end


def choose_window(duration: timedelta) -> str:
    """Pick the aggregation window that keeps a range of ``duration`` responsive."""
    for longest, window in _WINDOW_BY_DURATION:
        if duration <= longest:
            return window
    return _WIDEST_WINDOW


def default_custom_range(now: datetime) -> tuple[datetime, datetime]:
    """Return the default custom range, the previous 8 hours, as IST datetimes."""
    end = floor_to_minute(now.astimezone(IST))
    return end - timedelta(hours=8), end


# ══════════════════════════════════════════════════════════════════════════════
# Fetching (uncached) — all live data goes through fetch_online_df
# ══════════════════════════════════════════════════════════════════════════════


def fetch_status_frame(
    measurement: str,
    time_range: str = STATUS_LOOKBACK,
    window_by: str = STATUS_WINDOW,
) -> pd.DataFrame:
    """Fetch one measurement's recent window with canonical field names."""
    return fetch_online_df(
        selected_measurements=[measurement],
        time_range=time_range,
        request_type="windowed-average",
        window_by=window_by,
        column_naming="field",
    )


def fetch_trend_frame(
    measurement: str,
    fields: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> pd.DataFrame:
    """Fetch one measurement over an explicit UTC range; keep only ``fields``.

    Returns an empty frame when none of the fields is in the response.
    """
    frame = fetch_online_df(
        selected_measurements=[measurement],
        time_range=_IGNORED_TIME_RANGE,
        request_type="windowed-average",
        window_by=window_by,
        start_time_override=start_utc,
        end_time_override=end_utc,
        column_naming="field",
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    present = [f for f in fields if f in frame.columns]
    return frame[present] if present else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Cached wrappers — keys carry measurement, range and window explicitly
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_status_frame(
    measurement: str, time_range: str, window_by: str
) -> pd.DataFrame:
    """Cached :func:`fetch_status_frame` (failures raise and are not cached)."""
    return fetch_status_frame(measurement, time_range, window_by)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_trend_frame(
    measurement: str,
    fields: tuple[str, ...],
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
) -> pd.DataFrame:
    """Cached :func:`fetch_trend_frame` (failures raise and are not cached)."""
    return fetch_trend_frame(measurement, fields, start_utc, end_utc, window_by)


def clear_furnace_status_caches() -> None:
    """Clear only this page's caches; unrelated application caches are untouched."""
    cached_status_frame.clear()
    cached_trend_frame.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

StatusFetcher = Callable[[str, str, str], pd.DataFrame]


def collect_status_frames(
    specs: Sequence[ParameterSpec] = PARAMETERS,
    fetch: StatusFetcher = cached_status_frame,
) -> dict[str, pd.DataFrame | None]:
    """Fetch each source measurement once; a failure only voids that measurement.

    Exceptions are logged server-side and never surfaced to the UI.
    """
    frames: dict[str, pd.DataFrame | None] = {}
    for measurement in measurements_for(specs):
        try:
            frames[measurement] = fetch(measurement, STATUS_LOOKBACK, STATUS_WINDOW)
        except Exception:  # noqa: BLE001 - one bad source must not blank the page
            log.exception("Furnace Status: fetch failed for %s", measurement)
            frames[measurement] = None
    return frames


def load_status_snapshot(
    now: datetime | None = None,
    specs: Sequence[ParameterSpec] = PARAMETERS,
    fetch: StatusFetcher = cached_status_frame,
) -> StatusSnapshot:
    """Fetch (cached) and resolve the full dashboard snapshot."""
    now = now or datetime.now(timezone.utc)
    return build_status_snapshot(collect_status_frames(specs, fetch), now, specs)


# ── Trend ─────────────────────────────────────────────────────────────────────

TrendStatus = Literal["ok", "no_source", "no_data", "error"]

TrendFetcher = Callable[[str, tuple[str, ...], datetime, datetime, str], pd.DataFrame]


@dataclass(frozen=True)
class TrendData:
    """Result of loading one parameter's trend."""

    spec: ParameterSpec
    status: TrendStatus
    #: Scaled numeric series on an IST index; only set when ``status == "ok"``.
    series: pd.Series | None
    #: Operator-safe explanation when ``status != "ok"``.
    message: str = ""
    #: Scaled setpoint series on the same index, when the spec has one.
    setpoint: pd.Series | None = None


def _scaled_series(frame: pd.DataFrame, field: str, spec: ParameterSpec) -> pd.Series:
    """Field as a scaled float series on an IST index, gaps (NaN) preserved."""
    series = pd.to_numeric(frame[field], errors="coerce").astype("float64")
    series = series.where(np.isfinite(series)).sort_index() * spec.scale
    index = pd.DatetimeIndex(series.index)
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    series.index = index.rename("time (IST)")
    return series


def with_derived_column(frame: pd.DataFrame, spec: ParameterSpec) -> pd.DataFrame:
    """Add ``spec.field`` as the per-row sum (or mean) of ``spec.components``.

    A row is NaN unless every component is present (no partial results); if a
    component column is absent altogether the frame is returned unchanged, so
    the parameter reads as no data.
    """
    if not spec.components or frame is None or frame.empty:
        return frame
    if not set(spec.components) <= set(frame.columns):
        return frame
    parts = frame[list(spec.components)].apply(pd.to_numeric, errors="coerce")
    parts = parts.where(np.isfinite(parts))
    total = parts.sum(axis=1, min_count=len(spec.components))
    if spec.aggregate == "mean":
        total = total / len(spec.components)
    return frame.assign(**{spec.field: total})


def load_trend(
    spec: ParameterSpec,
    start_utc: datetime,
    end_utc: datetime,
    window_by: str,
    fetch: TrendFetcher = cached_trend_frame,
) -> TrendData:
    """Load and scale one parameter's trend (and setpoint) over a UTC range."""
    if not spec.has_source:
        return TrendData(spec, "no_source", None, spec.unavailable_reason)

    try:
        frame = fetch(
            spec.measurement, spec.source_fields, start_utc, end_utc, window_by
        )
    except Exception:  # noqa: BLE001 - never expose connection details
        log.exception(
            "Furnace Status: trend fetch failed for %s.%s", spec.measurement, spec.field
        )
        return TrendData(
            spec, "error", None, "The data source could not be reached. Try Refresh."
        )

    frame = with_derived_column(frame, spec)
    if finite_series(frame, spec.field) is None:
        return TrendData(
            spec, "no_data", None, "No data was returned for the selected interval."
        )

    actual = _scaled_series(frame, spec.field, spec).rename(spec.title)
    setpoint = None
    if finite_series(frame, spec.setpoint_field) is not None:
        setpoint = _scaled_series(frame, spec.setpoint_field, spec).rename("Setpoint")
    return TrendData(spec, "ok", actual, setpoint=setpoint)


@dataclass(frozen=True)
class TrendStats:
    current: float
    current_at: pd.Timestamp
    minimum: float
    maximum: float
    mean: float


def compute_trend_stats(series: pd.Series | None) -> TrendStats | None:
    """Current / min / max / mean over the finite points, or ``None`` if none."""
    if series is None:
        return None
    finite = series[np.isfinite(series)]
    if finite.empty:
        return None
    return TrendStats(
        current=float(finite.iloc[-1]),
        current_at=_to_ist(finite.index[-1]),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        mean=float(finite.mean()),
    )
