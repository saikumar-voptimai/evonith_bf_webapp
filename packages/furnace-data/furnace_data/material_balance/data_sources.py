"""Day-windowed data fetchers for the Material Balance page and API.

The static ML dataset stores dataframe aliases from the cleaned ML export.
This module converts those aliases to the logical mapping names defined in
``setting_ds_dv.yml -> rename_dict``. Material Balance specific choices,
such as which logical columns are daily tonnages, live in
``material_balance.yml``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import pytz

from furnace_data.config import load_config
from furnace_data.dataset.static_csv import get_static_dataset_path, load_static_dataset
from furnace_data.influx.base import BaseDataFetcher
from furnace_data.offline import fetch_offline_report

log = logging.getLogger("root")

IST = pytz.timezone("Asia/Kolkata")
STATIC_DATASET_ID = "static_ml_dataset"
WINDOW_POLICY_VERSION = "hourly_shift_v1"


@dataclass(frozen=True)
class MaterialBalanceWindow:
    """Resolved local and UTC window for one Material Balance slice."""

    local_start: datetime
    local_end: datetime
    utc_start: datetime
    utc_end: datetime


@dataclass(frozen=True)
class StaticDatasetSnapshot:
    """Immutable dataset snapshot used by one calculation."""

    dataset_id: str
    version: str
    frame: pd.DataFrame
    minimum_date: date | None
    maximum_date: date | None
    row_count: int


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
# IST to UTC conversion helpers
# ---------------------------------------------------------------------------


def _local_window(day: date, lag_hours: int = 0) -> MaterialBalanceWindow:
    start_local = IST.localize(datetime(day.year, day.month, day.day, 0, 0, 0))
    end_local = start_local + timedelta(days=1)
    if lag_hours:
        start_local = start_local - timedelta(hours=int(lag_hours))
        end_local = end_local - timedelta(hours=int(lag_hours))
    return MaterialBalanceWindow(
        local_start=start_local,
        local_end=end_local,
        utc_start=start_local.astimezone(timezone.utc),
        utc_end=end_local.astimezone(timezone.utc),
    )


def get_day_window_utc(day: date) -> Tuple[datetime, datetime]:
    """Return ``(start_utc, end_utc)`` for one IST calendar day."""
    window = _local_window(day)
    return window.utc_start, window.utc_end


def resolve_material_balance_windows(
    day: date,
    *,
    rm_lag_hours: int = 0,
    blast_lag_hours: int = 0,
) -> tuple[MaterialBalanceWindow, MaterialBalanceWindow, MaterialBalanceWindow]:
    """Return output, raw-material and blast windows for one IST day."""

    return (
        _local_window(day, 0),
        _local_window(day, int(rm_lag_hours)),
        _local_window(day, int(blast_lag_hours)),
    )


# ---------------------------------------------------------------------------
# Static ML dataset loader and snapshot helpers
# ---------------------------------------------------------------------------


def _resolve_existing_static_dataset_path() -> Path | None:
    """Return an existing static dataset CSV path without building it."""

    csv_path = get_static_dataset_path()
    if csv_path.exists():
        return csv_path
    try:
        from furnace_data.dataset import static_csv as static_csv_module

        fallback = static_csv_module._legacy_static_dataset_path()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        fallback = None
    if fallback is not None and fallback.exists():
        return fallback
    return None


def _dataset_file_version(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@lru_cache(maxsize=2)
def _load_static_dataset_from_path(path_text: str, version: str) -> pd.DataFrame:
    """Load a known CSV snapshot; *version* is part of the cache key."""

    _ = version
    df = load_static_dataset(Path(path_text))
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.rename(columns=_dataset_alias_to_mapping_name(), inplace=True)
    return df


@lru_cache(maxsize=2)
def _load_static_dataset() -> pd.DataFrame:
    """Load, timestamp-parse, and rename the full static ML dataset."""
    path = _resolve_existing_static_dataset_path()
    if path is None:
        return pd.DataFrame()
    return _load_static_dataset_from_path(str(path), _dataset_file_version(path))


def _window_from_csv(
    df: pd.DataFrame,
    day: date,
    lag_hours: int = 0,
) -> pd.DataFrame:
    """Slice rows for one IST day with a true hour-shifted input window."""
    _, lagged_window, _ = resolve_material_balance_windows(
        day,
        rm_lag_hours=int(lag_hours),
        blast_lag_hours=0,
    )
    return _window_from_snapshot_frame(df, lagged_window)


def _window_from_snapshot_frame(df: pd.DataFrame, window: MaterialBalanceWindow) -> pd.DataFrame:
    start = window.local_start.replace(tzinfo=None)
    end = window.local_end.replace(tzinfo=None)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    mask = (ts >= start) & (ts < end)
    return df.loc[mask].copy()


def get_static_dataset_metadata() -> dict[str, Any]:
    """Return public metadata for the canonical dataset without rebuilding it."""

    path = _resolve_existing_static_dataset_path()
    if path is None:
        return {
            "dataset_id": STATIC_DATASET_ID,
            "version": None,
            "status": "missing",
            "available_date_range": {"minimum": None, "maximum": None},
            "row_count": 0,
        }

    version = _dataset_file_version(path)
    try:
        index_frame = pd.read_csv(path, usecols=[0], low_memory=False)
        ts = pd.to_datetime(index_frame.iloc[:, 0], errors="coerce").dropna()
    except Exception as exc:  # noqa: BLE001
        log.warning("Static dataset metadata read failed: %s", exc)
        return {
            "dataset_id": STATIC_DATASET_ID,
            "version": version,
            "status": "not_ready",
            "available_date_range": {"minimum": None, "maximum": None},
            "row_count": 0,
        }

    dates = ts.dt.date if not ts.empty else pd.Series(dtype=object)
    minimum = dates.min() if not dates.empty else None
    maximum = dates.max() if not dates.empty else None
    return {
        "dataset_id": STATIC_DATASET_ID,
        "version": version,
        "status": "ready" if minimum and maximum else "missing",
        "available_date_range": {"minimum": minimum, "maximum": maximum},
        "row_count": int(len(index_frame)),
    }


def load_static_dataset_snapshot() -> StaticDatasetSnapshot:
    """Load the canonical dataset once for a calculation without rebuilding it."""

    metadata = get_static_dataset_metadata()
    version = metadata.get("version")
    if metadata.get("status") != "ready" or not version:
        raise FileNotFoundError("Static ML dataset is not available.")
    path = _resolve_existing_static_dataset_path()
    if path is None:
        raise FileNotFoundError("Static ML dataset is not available.")
    frame = _load_static_dataset_from_path(str(path), str(version)).copy()
    range_info = metadata.get("available_date_range") or {}
    return StaticDatasetSnapshot(
        dataset_id=STATIC_DATASET_ID,
        version=str(version),
        frame=frame,
        minimum_date=range_info.get("minimum"),
        maximum_date=range_info.get("maximum"),
        row_count=int(metadata.get("row_count") or len(frame)),
    )


def aggregate_rm_from_snapshot(
    snapshot: StaticDatasetSnapshot,
    window: MaterialBalanceWindow,
) -> pd.DataFrame:
    """Return one row with RM composition averages and mass sums."""

    source = _window_from_snapshot_frame(snapshot.frame, window)
    if source.empty:
        return pd.DataFrame()
    mass_cols = _schema_mass_cols()
    result: Dict[str, float] = {}
    for col in source.columns:
        if col == "timestamp":
            continue
        series = pd.to_numeric(source[col], errors="coerce")
        value = series.sum(skipna=True) if col in mass_cols else series.mean(skipna=True)
        result[col] = float(value) if pd.notna(value) else float("nan")
    out = pd.DataFrame([result])
    out.attrs["n_rows"] = len(source)
    return out


def aggregate_hm_slag_from_snapshot(
    snapshot: StaticDatasetSnapshot,
    window: MaterialBalanceWindow,
) -> pd.DataFrame:
    """Return one row with HM and slag chemistry averages."""

    source = _window_from_snapshot_frame(snapshot.frame, window)
    if source.empty:
        return pd.DataFrame()
    avg = source.mean(numeric_only=True).to_frame().T
    avg.attrs["n_rows"] = len(source)
    return avg


def aggregate_online_from_snapshot(
    snapshot: StaticDatasetSnapshot,
    window: MaterialBalanceWindow,
) -> Dict[str, float]:
    """Return blast/process averages from a snapshot."""

    source = _window_from_snapshot_frame(snapshot.frame, window)
    if source.empty:
        return {}
    out: Dict[str, float] = {}
    for field in _process_fields():
        if field in source.columns:
            value = pd.to_numeric(source[field], errors="coerce").mean(skipna=True)
            out[field] = float(value) if pd.notna(value) else 0.0
        else:
            out[field] = 0.0
    return out


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
    metadata = get_static_dataset_metadata()
    range_info = metadata.get("available_date_range") or {}
    return range_info.get("minimum"), range_info.get("maximum")


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def clear_day_caches(day: date) -> None:
    """Invalidate Material Balance dataset/config caches."""
    _ = day
    for fn in (
        _load_static_dataset,
        _load_static_dataset_from_path,
        _settings_config,
        _material_balance_config,
        _rename_dict,
        _dataset_alias_to_mapping_name,
        _schema_mass_cols,
        _process_fields,
        _online_influx_to_mapping_name,
    ):
        try:
            fn.cache_clear()
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


def fetch_dpr_for_window(window: MaterialBalanceWindow) -> pd.DataFrame:
    """Fetch daily production report rows for a resolved UTC window."""
    try:
        df = fetch_offline_report("DPR", (window.utc_start, window.utc_end))
    except Exception as exc:
        log.warning(
            "fetch_dpr_for_window(%s, %s) failed: %s",
            window.utc_start,
            window.utc_end,
            exc,
        )
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def fetch_dpr_for_day(day: date) -> pd.DataFrame:
    """Fetch daily production report row(s) from the offline database."""
    output_window, _, _ = resolve_material_balance_windows(day)
    df = fetch_dpr_for_window(output_window)
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