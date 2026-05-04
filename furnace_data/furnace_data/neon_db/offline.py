"""Offline Neon/PostgreSQL table fetcher.

This module mirrors the simple DataFrame contract of
``furnace_data.influx.offline.fetch_offline_data`` while querying whitelisted
relational tables. It intentionally keeps Influx-compatible rollback separate:
callers choose the backend at their boundary and this module only handles Neon.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple, Union

import pandas as pd
import yaml
from sqlalchemy import text

from furnace_data.influx.query import TIMEDELTAS
from furnace_data.relational.engine import build_relational_engine

_SCHEMA_PATH = Path(__file__).with_name("neon_tables.yml")
_SCHEMA = yaml.safe_load(_SCHEMA_PATH.read_text(encoding="utf-8")) or {}

_TABLES: dict[str, dict] = _SCHEMA.get("tables", {})
_TIME_COLUMNS: dict[str, str] = {
    table: meta["time_column"] for table, meta in _TABLES.items()
}

NEON_OFFLINE_TABLES: dict[str, set[str]] = {
    table: set(meta["columns"]) for table, meta in _TABLES.items()
}

NEON_OFFLINE_REPORT_MAP: dict[str, list[str]] = {
    report_type: list(tables)
    for report_type, tables in (_SCHEMA.get("report_map", {}) or {}).items()
}

_NON_AVERAGED_COLUMNS: set[str] = set(_SCHEMA.get("non_averaged_columns", []))


def list_neon_offline_tables() -> dict[str, object]:
    """Return a JSON-serialisable snapshot of the Neon offline whitelist."""
    return {
        "report_map": {k: list(v) for k, v in NEON_OFFLINE_REPORT_MAP.items()},
        "tables": {
            table: {
                "time_column": _TIME_COLUMNS[table],
                "supports_aggregation": bool(meta.get("supports_aggregation", True)),
                "columns": sorted(NEON_OFFLINE_TABLES[table]),
            }
            for table, meta in _TABLES.items()
        },
    }


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalise_columns(table_name: str, columns: Iterable[str] | None) -> list[str]:
    allowed = NEON_OFFLINE_TABLES[table_name]
    if columns is None:
        return sorted(allowed)

    selected = list(columns)
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(f"Unknown column(s) for {table_name}: {unknown}")
    return selected


def _numeric_average_columns(selected_columns: Iterable[str]) -> list[str]:
    return [col for col in selected_columns if col not in _NON_AVERAGED_COLUMNS]


def _resolve_range(time_range: Union[str, Tuple]) -> tuple[pd.Timestamp, pd.Timestamp]:
    now = pd.Timestamp(datetime.now(timezone.utc))
    if isinstance(time_range, str) and time_range.strip().lower() == "full":
        return pd.Timestamp("2023-01-01T00:00:00Z"), now

    if isinstance(time_range, str):
        delta = TIMEDELTAS.get(time_range.strip().lower())
        if delta is None:
            raise ValueError(f"Unknown time_range preset: {time_range!r}")
        return now - delta, now

    if isinstance(time_range, tuple) and len(time_range) == 2:
        return (
            pd.to_datetime(time_range[0], utc=True),
            pd.to_datetime(time_range[1], utc=True),
        )

    raise ValueError(
        "time_range must be a preset string, 'full', or a (start, end) tuple."
    )


def _build_query(
    table_name: str,
    selected_columns: list[str],
    query_type: str,
    window: str | None,
) -> tuple[str, dict[str, object]]:
    time_col = _TIME_COLUMNS[table_name]
    table_sql = _quote_identifier(table_name)
    time_sql = _quote_identifier(time_col)
    query_type = query_type.strip().lower()

    if query_type in {"ts", "raw"}:
        cols = ", ".join(_quote_identifier(col) for col in selected_columns)
        return (
            f"SELECT {cols} FROM {table_sql} "
            f"WHERE {time_sql} >= :start_time AND {time_sql} <= :end_time "
            f"ORDER BY {time_sql}",
            {},
        )

    if not _TABLES[table_name].get("supports_aggregation", True):
        raise ValueError(f"{table_name} does not support SQL aggregation fetches.")

    avg_columns = _numeric_average_columns(selected_columns)
    if not avg_columns:
        raise ValueError(f"No numeric columns selected for averaging {table_name}.")

    avg_sql = ", ".join(
        f"AVG({_quote_identifier(col)}) AS {_quote_identifier(col)}"
        for col in avg_columns
    )

    if query_type in {"windowed-average", "hourly-average"}:
        window = window or "1 hour"
        return (
            "SELECT date_bin(CAST(:window AS interval), "
            f"{time_sql}, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS {time_sql}, "
            f"{avg_sql} FROM {table_sql} "
            f"WHERE {time_sql} >= :start_time AND {time_sql} <= :end_time "
            "GROUP BY 1 ORDER BY 1",
            {"window": window},
        )

    if query_type == "average":
        return (
            f"SELECT :start_time AS {time_sql}, {avg_sql} FROM {table_sql} "
            f"WHERE {time_sql} >= :start_time AND {time_sql} <= :end_time",
            {},
        )

    raise ValueError(
        "query_type must be 'ts', 'raw', 'average', or 'windowed-average'."
    )


def fetch_offline_data(
    table_name: str,
    time_range: Union[str, Tuple],
    query_type: str = "ts",
    window: str | None = None,
    columns: Iterable[str] | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Fetch offline data from one whitelisted Neon/PostgreSQL table."""
    if table_name not in NEON_OFFLINE_TABLES:
        raise ValueError(
            f"Unknown Neon offline table '{table_name}'. "
            f"Valid: {sorted(NEON_OFFLINE_TABLES)}"
        )

    start_time, end_time = _resolve_range(time_range)
    selected_columns = _normalise_columns(table_name, columns)
    time_col = _TIME_COLUMNS[table_name]
    if time_col not in selected_columns:
        selected_columns.insert(0, time_col)

    query, extra_params = _build_query(table_name, selected_columns, query_type, window)
    params = {
        "start_time": start_time.to_pydatetime(),
        "end_time": end_time.to_pydatetime(),
        **extra_params,
    }

    engine = build_relational_engine(database_url)
    try:
        df = pd.read_sql_query(text(query), engine, params=params)
    finally:
        engine.dispose()

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col)
        df.index.name = "time"

    return df.sort_index()


def fetch_offline_report(
    report_type: str,
    time_range: Union[str, Tuple],
    query_type: str = "ts",
    window: str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Fetch a logical offline report from one or more Neon tables."""
    tables = NEON_OFFLINE_REPORT_MAP.get(report_type)
    if not tables:
        raise ValueError(
            f"Unknown Neon offline report '{report_type}'. "
            f"Valid: {sorted(NEON_OFFLINE_REPORT_MAP)}"
        )

    frames: list[pd.DataFrame] = []
    for table in tables:
        df = fetch_offline_data(
            table_name=table,
            time_range=time_range,
            query_type=query_type,
            window=window,
            database_url=database_url,
        )
        if df is None or df.empty:
            continue
        df = df.copy()
        df.insert(0, "source_table", table)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=0, sort=False).sort_index()
