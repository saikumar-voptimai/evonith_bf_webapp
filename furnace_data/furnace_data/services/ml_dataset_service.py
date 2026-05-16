"""Backend-safe static ML dataset loader and helpers."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import inspect

from furnace_data.config import load_config
from furnace_data.neon_db.offline import fetch_offline_data
from furnace_data.relational.engine import build_relational_engine

log = logging.getLogger(__name__)

_STATIC_ML_TABLE = "historical_static_ml_dataset"
_STATIC_ML_SCHEMA = "offline_feed"
_RAW_BURDEN_COLUMN_RE = re.compile(
    r"^(?:coke|noncoke|non_coke)__p\d+_(?:angles|rings)$",
    re.IGNORECASE,
)


def get_static_dataset_path(data_rel_path: str | None = None) -> Path:
    """Resolve the legacy local static dataset path inside the app repo."""
    rel_path = data_rel_path or load_config("setting_ds_dv.yml")["DATA"]
    return (Path(__file__).resolve().parents[3] / rel_path).resolve()


def normalise_index(df: pd.DataFrame, *, assume_naive_utc: bool) -> pd.DataFrame:
    """Normalize a static dataset dataframe to an IST-naive DatetimeIndex."""
    if df.empty:
        return df

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for candidate in ("time", "date_time", "timestamp"):
            if candidate in out.columns:
                out[candidate] = pd.to_datetime(out[candidate], errors="coerce", utc=True)
                out = out.set_index(candidate)
                break

    if isinstance(out.index, pd.DatetimeIndex):
        local_tz = (
            load_config("setting_ds_dv.yml")
            .get("ml_dataset", {})
            .get("local_tz", "Asia/Kolkata")
        )
        if out.index.tz is None:
            if assume_naive_utc:
                out.index = out.index.tz_localize("UTC")
                out.index = out.index.tz_convert(ZoneInfo(local_tz)).tz_localize(None)
        else:
            out.index = out.index.tz_convert(ZoneInfo(local_tz)).tz_localize(None)
        out.index.name = "time"

    return out.sort_index()


def rename_columns_for_app(df: pd.DataFrame) -> pd.DataFrame:
    """Apply configured dataset display-column names."""
    rename_dict = load_config("setting_ds_dv.yml").get("rename_dict", {}) or {}
    if not isinstance(rename_dict, dict):
        raise TypeError("setting_ds_dv.yml rename_dict must be a mapping.")
    return df.rename(columns={str(k): str(v) for k, v in rename_dict.items()})


def available_static_dataset_columns() -> set[str]:
    """Return columns available in the static ML database table."""
    engine = build_relational_engine()
    try:
        return {
            str(col["name"])
            for col in inspect(engine).get_columns(
                _STATIC_ML_TABLE,
                schema=_STATIC_ML_SCHEMA,
            )
        }
    finally:
        engine.dispose()


def configured_static_dataset_columns() -> list[str]:
    """Return configured source columns for static ML fetches."""
    cleaning = load_config("setting_ds_dv.yml").get("cleaning", {}) or {}
    if not isinstance(cleaning, dict):
        raise TypeError("setting_ds_dv.yml cleaning section must be a mapping.")

    names: list[str] = []
    for group in (cleaning.get("column_groups", {}) or {}).values():
        names.extend(str(name) for name in (group or []))

    extra = cleaning.get("extra_keep_columns", {}) or {}
    names.extend(str(name) for name in (extra.get("alias_keys", []) or []))
    names.extend(str(name) for name in (extra.get("calc_alias_keys", []) or []))
    return list(dict.fromkeys(names))


def static_dataset_fetch_columns() -> list[str]:
    """Return configured static columns that exist in the database table."""
    available_columns = available_static_dataset_columns()
    candidates = [
        col
        for col in configured_static_dataset_columns()
        if col in available_columns and not _RAW_BURDEN_COLUMN_RE.match(col)
    ]

    if not candidates:
        raise RuntimeError("No configured static ML dataset columns exist in the database.")

    return candidates


def fetch_static_dataset_from_database(sort_index: bool = True) -> pd.DataFrame:
    """Fetch the full static ML dataset from the configured database."""
    try:
        df = fetch_offline_data(
            _STATIC_ML_TABLE,
            "full",
            query_type="raw",
            columns=static_dataset_fetch_columns(),
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load static ML dataset: {exc}") from exc

    if df.empty:
        raise RuntimeError("Static ML dataset returned 0 rows.")

    log.info("Loaded static ML dataset from database table %s", _STATIC_ML_TABLE)
    df = normalise_index(df, assume_naive_utc=True)
    df = rename_columns_for_app(df)
    return df.sort_index() if sort_index else df


def load_static_dataset(
    path: str | Path | None = None,
    *,
    index_col: int | None = 0,
    parse_dates: bool = True,
    low_memory: bool = False,
    sort_index: bool = True,
) -> pd.DataFrame:
    """Load the static ML dataset from local CSV or the database."""
    if path is None:
        csv_path = get_static_dataset_path()
    else:
        csv_path = Path(path)
        if not csv_path.is_absolute():
            csv_path = (Path(__file__).resolve().parents[3] / csv_path).resolve()

    if not csv_path.exists():
        return fetch_static_dataset_from_database(sort_index=sort_index)

    df = pd.read_csv(
        csv_path,
        index_col=index_col,
        parse_dates=parse_dates,
        low_memory=low_memory,
    )
    df = normalise_index(df, assume_naive_utc=False)
    df = rename_columns_for_app(df)
    return df.sort_index() if sort_index else df
