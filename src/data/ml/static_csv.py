"""Shared helpers for the webapp's static furnace ML dataset."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import inspect

from config.config_loader import load_config
from furnace_data.offline import fetch_offline_data
from furnace_data.relational.engine import build_relational_engine

log = logging.getLogger(__name__)

_STATIC_ML_TABLE = "historical_static_ml_dataset"
_STATIC_ML_SCHEMA = "offline_feed"
_RAW_BURDEN_COLUMN_RE = re.compile(
    r"^(?:coke|noncoke|non_coke)__p\d+_(?:angles|rings)$",
    re.IGNORECASE,
)


def get_static_dataset_path(data_rel_path: str | None = None) -> Path:
    """Resolve the legacy static dataset path inside the webapp repo.

    The current app can refresh this file from the static ML database table.
    This helper remains
    for older call sites that still pass a path-shaped argument around.
    """
    rel_path = data_rel_path or load_config("setting_ds_dv.yml")["DATA"]
    return (Path(__file__).resolve().parents[3] / rel_path).resolve()


def _normalise_index(df: pd.DataFrame, *, assume_naive_utc: bool) -> pd.DataFrame:
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


def _rename_columns_for_app(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = load_config("setting_ds_dv.yml").get("rename_dict", {}) or {}
    if not isinstance(rename_dict, dict):
        raise TypeError("setting_ds_dv.yml rename_dict must be a mapping.")
    return df.rename(columns={str(k): str(v) for k, v in rename_dict.items()})


def _available_static_dataset_columns() -> set[str]:
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


def _configured_static_dataset_columns() -> list[str]:
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


def _static_dataset_fetch_columns() -> list[str]:
    available_columns = _available_static_dataset_columns()
    candidates = [
        col
        for col in _configured_static_dataset_columns()
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
            columns=_static_dataset_fetch_columns(),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load static ML dataset: {exc}") from exc

    if df.empty:
        raise RuntimeError("Static ML dataset returned 0 rows.")

    log.info("Loaded static ML dataset from database table %s", _STATIC_ML_TABLE)
    df = _normalise_index(df, assume_naive_utc=True)
    df = _rename_columns_for_app(df)
    return df.sort_index() if sort_index else df


@st.cache_data(ttl=3600, show_spinner=False)
def load_static_dataset(
    path: str | Path | None = None,
    *,
    index_col: int | None = 0,
    parse_dates: bool = True,
    low_memory: bool = False,
    sort_index: bool = True,
) -> pd.DataFrame:
    """Load the static furnace ML dataset.

    The legacy CSV-oriented arguments are accepted for compatibility, but the
    local rotating cache is preferred when present. If no local copy exists,
    the full static ML table is fetched from the configured database.
    """
    if path is None:
        csv_path = get_static_dataset_path()
    else:
        csv_path = Path(path)
        if not csv_path.is_absolute():
            csv_path = (Path(__file__).resolve().parents[3] / csv_path).resolve()

    if not csv_path.exists():
        from data.ml.static_dataset_manager import StaticDatasetManager

        manager = StaticDatasetManager(csv_path)
        df = manager.update_static()
        manager.save(df)
        return df.sort_index() if sort_index else df

    df = pd.read_csv(
        csv_path,
        index_col=index_col,
        parse_dates=parse_dates,
        low_memory=low_memory,
    )
    df = _normalise_index(df, assume_naive_utc=False)
    df = _rename_columns_for_app(df)
    return df.sort_index() if sort_index else df


def update_cutoff_date(new_date: date) -> None:
    """Advance ``cutoff_date`` in ``setting_ds_dv.yml`` to *new_date*.

    Called after a successful extend/override so that
    ``DatasetFetcher._fetch_full_range`` uses the static table for the
    newly written period on subsequent interactive fetches.
    """
    yml_path = (Path(__file__).resolve().parents[2] / "config" / "setting_ds_dv.yml")
    text = yml_path.read_text(encoding="utf-8")
    updated = re.sub(
        r'(cutoff_date:\s*")[^"]+(")',
        rf'\g<1>{new_date.isoformat()}\g<2>',
        text,
    )
    if updated == text:
        log.warning("update_cutoff_date: pattern not found in setting_ds_dv.yml — skipped.")
        return
    yml_path.write_text(updated, encoding="utf-8")
    log.info("cutoff_date advanced to %s in setting_ds_dv.yml.", new_date.isoformat())
