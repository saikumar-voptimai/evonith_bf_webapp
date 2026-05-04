"""Offline Neon/PostgreSQL table fetcher.

This mirrors the public shape of :mod:`furnace_data.influx.offline`, but queries
the relational offline tables whose schemas differ from the Influx measurements.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple, Union

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..influx.query import TIMEDELTAS

_schema = yaml.safe_load((Path(__file__).with_name("neon_tables.yml")).read_text())

_TIME_COLUMNS: dict[str, str] = {
    table: meta["time_column"] for table, meta in _schema["tables"].items()
}

NEON_OFFLINE_TABLES: dict[str, set[str]] = {
    table: set(meta["columns"]) for table, meta in _schema["tables"].items()
}

NEON_OFFLINE_REPORT_MAP: dict[str, str] = _schema["report_map"]

_NON_AVERAGED_COLUMNS: set[str] = set(_schema["non_averaged_columns"])


def _resolve_database_url(database_url: str | None = None) -> str:
    resolved = database_url or os.getenv("NEON_DATABASE_DEV_URL")
    if not resolved:
        raise ValueError("Missing NEON_DATABASE_DEV_URL environment variable.")
    return resolved


def _build_engine(database_url: str | None = None) -> Engine:
    return create_engine(_resolve_database_url(database_url), future=True, pool_pre_ping=True)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalise_columns(table_name: str, columns: Iterable[str] | None) -> list[str]:
    allowed = NEON_OFFLINE_TABLES[table_name]
    if columns is None:
        return sorted(allowed)
    cols = list(columns)
    unknown = sorted(set(cols) - allowed)
    if unknown:
        raise ValueError(f"Unknown column(s) for {table_name}: {unknown}")
    return cols


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
        return pd.to_datetime(time_range[0], utc=True), pd.to_datetime(time_range[1], utc=True)
    raise ValueError("time_range must be a preset string, 'full', or a (start, end) tuple.")


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

    avg_columns = _numeric_average_columns(selected_columns)
    if not avg_columns:
        raise ValueError(f"No numeric columns selected for averaging {table_name}.")

    if query_type in {"windowed-average", "hourly-average"}:
        window = window or "1 hour"
        avg_sql = ", ".join(
            f"AVG({_quote_identifier(col)}) AS {_quote_identifier(col)}"
            for col in avg_columns
        )
        return (
            "SELECT date_bin(CAST(:window AS interval), "
            f"{time_sql}, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS {time_sql}, "
            f"{avg_sql} FROM {table_sql} "
            f"WHERE {time_sql} >= :start_time AND {time_sql} <= :end_time "
            f"GROUP BY 1 ORDER BY 1",
            {"window": window},
        )

    if query_type == "average":
        avg_sql = ", ".join(
            f"AVG({_quote_identifier(col)}) AS {_quote_identifier(col)}"
            for col in avg_columns
        )
        return (
            f"SELECT :start_time AS {time_sql}, {avg_sql} FROM {table_sql} "
            f"WHERE {time_sql} >= :start_time AND {time_sql} <= :end_time",
            {},
        )

    raise ValueError("query_type must be 'ts', 'raw', 'average', or 'windowed-average'.")


def fetch_offline_data(
    table_name: str,
    time_range: Union[str, Tuple],
    query_type: str = "ts",
    window: str | None = None,
    columns: Iterable[str] | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Fetch offline data from a whitelisted Neon/PostgreSQL table.

    Args:
        table_name: One of :data:`NEON_OFFLINE_TABLES`.
        time_range: Preset string, ``"full"``, or ``(start, end)``.
        query_type: ``"ts"``/``"raw"`` for raw rows, ``"windowed-average"``
            for hourly/custom buckets, or ``"average"`` for one aggregate row.
        window: PostgreSQL interval string for ``"windowed-average"``. Examples:
            ``"1 hour"``, ``"15 minutes"``, ``"8 hours"``.
        columns: Optional column subset. Unknown columns are rejected.
        database_url: Optional SQLAlchemy URL override.

    Returns:
        Time-indexed :class:`pandas.DataFrame` with a UTC-aware index.
    """
    if table_name not in NEON_OFFLINE_TABLES:
        raise ValueError(
            f"Unknown Neon offline table '{table_name}'. Valid: {sorted(NEON_OFFLINE_TABLES)}"
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

    engine = _build_engine(database_url)
    try:
        df = pd.read_sql_query(text(query), engine, params=params)
    finally:
        engine.dispose()

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col)
        df.index.name = "time"

    return df.sort_index()
