"""Pure data-fetching helpers for the FurnaceMind agent tools.

No Streamlit import. All functions are safe to call from FastAPI, CLI scripts,
and background schedulers.

Public API
----------
to_ist_index              Convert a DataFrame index to IST-aware.
parse_iso8601_utc         Parse an ISO-8601 string to a UTC-aware datetime.
resolve_online_window     Choose an averaging window given a lookback duration.
normalize_time_range      Normalise compact / natural-language lookback strings.
fetch_online_df_for_agent Fetch InfluxDB telemetry; returns a DataFrame.
fetch_offline_df_for_agent Fetch Postgres offline report; returns (df, source_detail).
merge_dfs                 Align offline frames onto an online timestamp spine.
concat_dfs                Temporal union of multiple DataFrames (tz-normalised).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from furnace_data.influx.online import fetch_online_df
from furnace_data.influx.query import TIMEDELTAS
from furnace_data.neon_db.offline import (
    fetch_offline_data as _fetch_neon_table_df,
    fetch_offline_report as _fetch_neon_report_df,
    resolve_neon_table_name,
)

_ALL_MEASUREMENT_GROUPS = [
    "process_params",
    "cooling_water",
    "heatload_delta_t",
    "delta_t",
    "temperature_profile",
    "miscellaneous",
]

_CADENCE_DEFAULTS: dict[str, str] = {
    "HM_SLAG": "1h",
    "CHARGE": "1h",
    "RAW_MATERIAL_COMPOSITION": "8h",
    "RM_COMPOSITION": "8h",
    "DPR": "1d",
    "BURDEN_DISTRIBUTION": "1d",
    "HOPPER_MANAGEMENT": "1d",
}

_SKIP_RESAMPLE_REPORT_TYPES = {"RM_COMPOSITION", "BURDEN_DISTRIBUTION", "HOPPER_MANAGEMENT"}


# Time helpers

def to_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    """Localise/convert a DataFrame's DatetimeIndex to IST (Asia/Kolkata)."""
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    df.index = df.index.tz_convert("Asia/Kolkata")
    df.index.name = "time (IST)"
    return df


def parse_iso8601_utc(s: str) -> datetime:
    """Parse an ISO-8601 string to a UTC-aware :class:`datetime`."""
    dt = pd.to_datetime(s, utc=True)
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    raise ValueError(f"Invalid datetime: {s}")


def resolve_online_window(*, lookback: timedelta, window: Optional[str]) -> str:
    """Return averaging window string given a lookback duration.

    If *window* is explicitly provided it is returned as-is; otherwise the
    policy is: > 1 day -> ``"1 hour"``, else -> ``"15 minutes"``.
    """
    if isinstance(window, str) and window.strip():
        return window.strip()
    return "1 hour" if lookback > timedelta(days=1) else "15 minutes"


def normalize_time_range(user_time_range: str) -> str:
    """Normalise compact / natural-language lookback strings to TIMEDELTAS keys.

    Accepts ``"8h"``, ``"2d"``, ``"30m"``, ``"1 week"``, ``"last 8 hours"`` etc.
    Extends the shared TIMEDELTAS registry on the fly for new values.
    Falls back to ``"last 8 hours"`` when the string is unrecognised.
    """
    tr = (user_time_range or "").strip().lower()
    if not tr:
        return "last 8 hours"
    if tr in TIMEDELTAS:
        return tr

    _patterns = [
        (r"^(\d+)\s*(m|min|mins|minute|minutes)$", "minutes"),
        (r"^(\d+)\s*(h|hr|hrs|hour|hours)$", "hours"),
        (r"^(\d+)\s*(d|day|days)$", "days"),
        (r"^(\d+)\s*(w|week|weeks)$", "weeks"),
        (r"last\s+(\d+)\s*(minute|minutes|min|mins)\b", "minutes"),
        (r"last\s+(\d+)\s*(hour|hours|hr|hrs|h)\b", "hours"),
        (r"last\s+(\d+)\s*(day|days|d)\b", "days"),
        (r"last\s+(\d+)\s*(week|weeks|w)\b", "weeks"),
    ]
    for pat, unit in _patterns:
        m = re.search(pat, tr)
        if m:
            n = int(m.group(1))
            singular = unit.rstrip("s")
            key = f"last {n} {singular}s" if n != 1 else f"last 1 {singular}"
            td = {"minutes": timedelta(minutes=n), "hours": timedelta(hours=n),
                  "days": timedelta(days=n), "weeks": timedelta(weeks=n)}[unit]
            TIMEDELTAS.setdefault(key, td)
            return key

    return "last 8 hours"


# Online fetch

def fetch_online_df_for_agent(
    *,
    lookback: Optional[str] = None,
    window: Optional[str] = None,
    measurement_groups: Optional[list[str]] = None,
    start_time_utc: Optional[str] = None,
    end_time_utc: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch InfluxDB telemetry and return ``(df_ist, time_range_label)``.

    Pass either *lookback* (e.g. ``"8h"``) or *start_time_utc* / *end_time_utc*
    for an exact window - never both.

    Returns
    -------
    df : pd.DataFrame
        IST-indexed DataFrame of the requested measurements.
    time_range_label : str
        Human-readable description of the fetched window (for metadata).
    """
    selected = measurement_groups or _ALL_MEASUREMENT_GROUPS
    _now = datetime.now(timezone.utc)

    if start_time_utc:
        start_dt = parse_iso8601_utc(start_time_utc)
        end_dt = parse_iso8601_utc(end_time_utc) if end_time_utc else _now
        if start_dt > _now:
            raise ValueError(
                f"start_time_utc {start_dt.isoformat()} is in the future "
                f"(current UTC: {_now.strftime('%Y-%m-%dT%H:%M:%SZ')}). "
                "No online data exists for future dates."
            )
        if end_dt > _now:
            end_dt = _now
        duration = end_dt - start_dt
        window_final = resolve_online_window(lookback=duration, window=window)
        df = fetch_online_df(
            selected_measurements=selected,
            time_range="last 8 hours",  # unused when overrides are provided
            window_by=window_final,
            start_time_override=start_dt,
            end_time_override=end_dt,
        )
        label = f"{start_time_utc} -> {end_time_utc or 'now'}"
    else:
        normalized = normalize_time_range(lookback or "last 8 hours")
        lookback_td = TIMEDELTAS.get(normalized)
        if not isinstance(lookback_td, timedelta):
            lookback_td = timedelta(hours=8)
        window_final = resolve_online_window(lookback=lookback_td, window=window)
        df = fetch_online_df(
            selected_measurements=selected,
            time_range=normalized,
            window_by=window_final,
        )
        label = normalized

    return to_ist_index(df), label


# Offline fetch

def fetch_offline_df_for_agent(
    *,
    report_type: str,
    table_name: Optional[str] = None,
    start: datetime,
    end: datetime,
    cadence: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch a Postgres offline report and return ``(df_ist, source_detail)``.

    The DataFrame index is converted to IST. Resampling is applied for
    HM_SLAG / CHARGE / DPR report types; composition and distribution types
    are returned as-is.

    Parameters
    ----------
    report_type :
        Canonical report type (e.g. ``"HM_SLAG"``, ``"RM_COMPOSITION"``).
    table_name :
        Optional explicit table override (alias or schema-qualified name).
    start / end :
        UTC-aware datetime window.
    cadence :
        Resampling cadence (``"1h"``, ``"8h"``, ``"1d"``).
        Defaults per report type when *None*.
    """
    cadence_final = cadence or _CADENCE_DEFAULTS.get(report_type, "1h")

    if table_name:
        resolved_table = resolve_neon_table_name(table_name)
        df = _fetch_neon_table_df(table_name=resolved_table, time_range=(start, end))
        source_detail = resolved_table
    else:
        df = _fetch_neon_report_df(report_type=report_type, time_range=(start, end))
        source_detail = report_type

    df = to_ist_index(df)

    skip_resample = report_type in _SKIP_RESAMPLE_REPORT_TYPES or bool(table_name)
    if (
        not skip_resample
        and df is not None
        and not df.empty
        and isinstance(df.index, pd.DatetimeIndex)
    ):
        df = df.resample(cadence_final).mean(numeric_only=True).dropna(how="all")

    return df, source_detail


# Merge / concat

def merge_dfs(
    online_df: pd.DataFrame,
    offline_dfs: list[pd.DataFrame],
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Align offline frames onto the online timestamp spine and join.

    Parameters
    ----------
    online_df :
        High-frequency online DataFrame (IST index).
    offline_dfs :
        One or more lower-cadence offline DataFrames to align.
    fill_method :
        ``"ffill"`` forward-fills offline values to online timestamps;
        ``"none"`` performs a left join without filling.
    """
    online_df = to_ist_index(online_df)

    if not offline_dfs:
        raise ValueError("No offline DataFrames provided to merge.")

    offline_combined = offline_dfs[0]
    for part in offline_dfs[1:]:
        offline_combined = offline_combined.join(to_ist_index(part), how="outer")
    offline_combined = to_ist_index(offline_combined)

    if fill_method == "ffill":
        offline_aligned = offline_combined.sort_index().reindex(
            online_df.index, method="ffill"
        )
    else:
        offline_aligned = offline_combined

    return online_df.join(offline_aligned, how="left")


def concat_dfs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames vertically with timezone normalisation.

    Mixed tz-aware (UTC from online) and tz-naive (IST from static ML) frames
    are normalised to UTC before concatenation. Duplicate timestamps keep the
    last frame's value (later / online data wins on overlap).
    """
    if not frames:
        raise ValueError("No DataFrames provided to concat.")

    has_aware = any(
        isinstance(f.index, pd.DatetimeIndex) and f.index.tz is not None
        for f in frames
    )
    has_naive = any(
        isinstance(f.index, pd.DatetimeIndex) and f.index.tz is None
        for f in frames
    )

    if has_aware and has_naive:
        normalised: list[pd.DataFrame] = []
        for f in frames:
            if isinstance(f.index, pd.DatetimeIndex):
                f = f.copy()
                if f.index.tz is None:
                    f.index = f.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
                else:
                    f.index = f.index.tz_convert("UTC")
            normalised.append(f)
        frames = normalised

    combined = pd.concat(frames, axis=0, join="outer")
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined
