"""Day-windowed data fetchers for the Material Balance page.

The static ML dataset stores dataframe aliases from the cleaned ML export.
This module converts those aliases to the logical mapping names defined in
``setting_ds_dv.yml -> rename_dict``. Material Balance specific choices,
such as which logical columns are daily tonnages, live in
``material_balance.yml``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Tuple

import pandas as pd
import pytz

from furnace_data.dataset.static_csv import load_static_dataset
from furnace_data.config import load_config
from furnace_data.influx.base import BaseDataFetcher
from furnace_data.offline import fetch_offline_report

log = logging.getLogger("root")

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Config-backed column resolution
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _settings_config() -> dict[str, Any]:
    return load_config("setting_ds_dv.yml")


@lru_cache(maxsize=1)
def _material_balance_config() -> dict[str, Any]:
    return load_config("material_balance.yml")


@lru_cache(maxsize=1)
def _rename_dict() -> dict[str, str]:
    rename_dict = _settings_config().get("rename_dict", {}) or {}
    if not isinstance(rename_dict, dict):
        raise TypeError("setting_ds_dv.yml rename_dict must be a mapping.")
    return {str(key): str(value) for key, value in rename_dict.items()}


def _is_preferred_mapping_name(candidate: str, current: str) -> bool:
    """Prefer snake-case logical keys when duplicate dataframe aliases exist."""
    candidate_score = int(candidate == candidate.lower()) + int(" " not in candidate)
    current_score = int(current == current.lower()) + int(" " not in current)
    return candidate_score > current_score


@lru_cache(maxsize=1)
def _dataset_alias_to_mapping_name() -> dict[str, str]:
    """Build dataframe alias -> logical mapping-name conversion from config."""
    out: dict[str, str] = {}
    for mapping_name, dataframe_alias in _rename_dict().items():
        existing = out.get(dataframe_alias)
        if existing is None or _is_preferred_mapping_name(mapping_name, existing):
            out[dataframe_alias] = mapping_name
    return out


def _static_dataset_config() -> dict[str, Any]:
    config = _material_balance_config().get("static_dataset", {}) or {}
    if not isinstance(config, dict):
        raise TypeError("material_balance.yml static_dataset must be a mapping.")
    return config


def _configured_mapping_names(section: str) -> tuple[str, ...]:
    values = _static_dataset_config().get(section)
    if values is None:
        raise KeyError(f"Missing material_balance.yml static_dataset.{section}.")
    if not isinstance(values, list):
        raise TypeError(f"material_balance.yml static_dataset.{section} must be a list.")

    names = tuple(str(value) for value in values)
    missing = [name for name in names if name not in _rename_dict()]
    if missing:
        raise KeyError(
            f"Unknown mapping names in material_balance.yml static_dataset.{section}: "
            f"{missing}"
        )
    return names


@lru_cache(maxsize=1)
def _schema_mass_cols() -> set[str]:
    """Logical columns that must be summed over a day, not averaged."""
    return set(_configured_mapping_names("mass_alias_keys"))


@lru_cache(maxsize=1)
def _process_fields() -> tuple[str, ...]:
    return _configured_mapping_names("process_alias_keys")


@lru_cache(maxsize=1)
def _online_influx_to_mapping_name() -> dict[str, str]:
    mapping = _static_dataset_config().get("online_influx_alias_keys", {}) or {}
    if not isinstance(mapping, Mapping):
        raise TypeError(
            "material_balance.yml static_dataset.online_influx_alias_keys "
            "must be a mapping."
        )
    missing = [str(value) for value in mapping.values() if str(value) not in _rename_dict()]
    if missing:
        raise KeyError(
            "Unknown mapping names in material_balance.yml "
            f"static_dataset.online_influx_alias_keys: {missing}"
        )
    return {str(key): str(value) for key, value in mapping.items()}


# ---------------------------------------------------------------------------
# IST to UTC conversion helper
# ---------------------------------------------------------------------------


def get_day_window_utc(day: date) -> Tuple[datetime, datetime]:
    """Return ``(start_utc, end_utc)`` for one IST calendar day."""
    start_local = IST.localize(datetime(day.year, day.month, day.day, 0, 0, 0))
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Static ML dataset loader
# ---------------------------------------------------------------------------


def _load_static_dataset() -> pd.DataFrame:
    """Load, timestamp-parse, and rename the full static ML dataset."""
    df = load_static_dataset()
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.rename(columns=_dataset_alias_to_mapping_name(), inplace=True)
    return df


def _window_from_csv(
    df: pd.DataFrame,
    day: date,
    lag_hours: int = 0,
) -> pd.DataFrame:
    """Slice rows for one calendar day, with optional whole-day lag."""
    target_date = day - timedelta(hours=lag_hours)
    mask = df["timestamp"].dt.date == target_date
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Public static dataset fetchers
# ---------------------------------------------------------------------------


def fetch_static_rm_for_day(day: date, lag_hours: int = 0) -> pd.DataFrame:
    """Return one row with RM composition averages and daily tonnage sums."""
    df = _load_static_dataset()
    if df.empty:
        return pd.DataFrame()

    window = _window_from_csv(df, day, lag_hours)
    if window.empty:
        return pd.DataFrame()

    mass_cols = _schema_mass_cols()
    result: Dict[str, float] = {}
    for col in window.columns:
        if col == "timestamp":
            continue
        series = pd.to_numeric(window[col], errors="coerce")
        value = series.sum(skipna=True) if col in mass_cols else series.mean(skipna=True)
        result[col] = float(value) if pd.notna(value) else float("nan")

    out = pd.DataFrame([result])
    out.attrs["n_rows"] = len(window)
    out.attrs["lag_hours"] = lag_hours
    return out


def fetch_static_hm_slag_for_day(day: date) -> pd.DataFrame:
    """Return one row with HM and slag chemistry averaged over the day."""
    df = _load_static_dataset()
    if df.empty:
        return pd.DataFrame()

    window = _window_from_csv(df, day)
    if window.empty:
        return pd.DataFrame()

    avg = window.mean(numeric_only=True).to_frame().T
    avg.attrs["n_rows"] = len(window)
    return avg


def fetch_static_online_for_day(day: date, lag_hours: int = 0) -> Dict[str, float]:
    """Return blast/process-param averages from the static dataset."""
    df = _load_static_dataset()
    if df.empty:
        return {}

    window = _window_from_csv(df, day, lag_hours)
    if window.empty:
        return {}

    out: Dict[str, float] = {}
    for field in _process_fields():
        if field in window.columns:
            value = pd.to_numeric(window[field], errors="coerce").mean(skipna=True)
            out[field] = float(value) if pd.notna(value) else 0.0
        else:
            out[field] = 0.0
    return out


def get_csv_date_range() -> Tuple[date | None, date | None]:
    """Return the earliest and latest IST dates available in the static dataset."""
    df = _load_static_dataset()
    if df.empty or "timestamp" not in df.columns:
        return None, None
    ts = df["timestamp"].dropna()
    if ts.empty:
        return None, None
    dates = ts.dt.date
    return dates.min(), dates.max()


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def clear_day_caches(day: date) -> None:
    """Invalidate every per-day cached fetch for a given IST date."""
    for fn in (
        fetch_static_rm_for_day,
        fetch_static_hm_slag_for_day,
        fetch_static_online_for_day,
        fetch_rm_for_day,
        fetch_hm_slag_for_day,
        fetch_dpr_for_day,
        fetch_online_aggregates_for_day,
    ):
        try:
            fn.clear()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Offline database fetchers and online telemetry fetcher
# ---------------------------------------------------------------------------


def fetch_rm_for_day(day: date) -> pd.DataFrame:
    """Fetch raw material composition for one IST day from the offline database."""
    start_utc, end_utc = get_day_window_utc(day)
    try:
        df = fetch_offline_report("RM_COMPOSITION", (start_utc, end_utc))
    except Exception as exc:
        log.warning("fetch_rm_for_day(%s) failed: %s", day, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    avg = df.mean(numeric_only=True).to_frame().T
    avg.attrs["n_shifts"] = int(len(df))
    return avg


def fetch_hm_slag_for_day(day: date) -> pd.DataFrame:
    """Fetch hot-metal and slag chemistry for one IST day from the offline database."""
    start_utc, end_utc = get_day_window_utc(day)
    try:
        df = fetch_offline_report("HM_SLAG", (start_utc, end_utc))
    except Exception as exc:
        log.warning("fetch_hm_slag_for_day(%s) failed: %s", day, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    avg = df.mean(numeric_only=True).to_frame().T
    avg.attrs["n_rows"] = int(len(df))
    return avg


def fetch_dpr_for_day(day: date) -> pd.DataFrame:
    """Fetch daily production report row(s) from the offline database."""
    start_utc, end_utc = get_day_window_utc(day)
    try:
        df = fetch_offline_report("DPR", (start_utc, end_utc))
    except Exception as exc:
        log.warning("fetch_dpr_for_day(%s) failed: %s", day, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    return df


def fetch_online_aggregates_for_day(day: date) -> Dict[str, float]:
    """Fetch online process-param day averages from the live telemetry bucket."""
    start_utc, end_utc = get_day_window_utc(day)
    try:
        fetcher = BaseDataFetcher(
            variable_tag="process_params",
            database="bf2_evonith_raw",
            token="INFLUX_ONLINE_TOKEN",
        )
        df = fetcher.fetch_averaged_data(
            recent_data_of="over selected range",
            start_time=start_utc,
            end_time=end_utc,
            request_type="windowed-average",
            window_by="1h",
        )
    except Exception as exc:
        log.warning("fetch_online_aggregates_for_day(%s) failed: %s", day, exc)
        return {}

    if df is None or len(df) == 0:
        return {}

    df = df.rename(columns=_online_influx_to_mapping_name())
    out: Dict[str, float] = {}
    for field in _process_fields():
        if field in df.columns:
            try:
                value = pd.to_numeric(df[field], errors="coerce").mean()
                out[field] = float(value) if pd.notna(value) else 0.0
            except Exception:  # noqa: BLE001
                out[field] = 0.0
        else:
            out[field] = 0.0
    return out
