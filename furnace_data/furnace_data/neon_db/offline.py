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
    """Resolve the Neon database URL from the argument or NEON_DATABASE_DEV_URL env var.

    Args:
         - database_url: str | None - Explicit SQLAlchemy connection URL; falls back to the env var if None.

    Returns:
         - str - Validated PostgreSQL connection URL.
    """
    resolved = database_url or os.getenv("NEON_DATABASE_DEV_URL")
    if not resolved:
        raise ValueError("Missing NEON_DATABASE_DEV_URL environment variable.")
    return resolved


def _build_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine with pre-ping health checking.

    Args:
         - database_url: str | None - SQLAlchemy connection URL; resolved via _resolve_database_url if None.

    Returns:
         - Engine - Configured SQLAlchemy engine.
    """
    return create_engine(
        _resolve_database_url(database_url), future=True, pool_pre_ping=True
    )


def _quote_identifier(identifier: str) -> str:
    """Wrap a PostgreSQL identifier in double quotes, escaping any internal double quotes.

    Args:
         - identifier: str - Column or table name to quote.

    Returns:
         - str - Double-quoted identifier safe for embedding in raw SQL.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _normalise_columns(table_name: str, columns: Iterable[str] | None) -> list[str]:
    """Validate and return the column list for a given Neon table.

    Args:
         - table_name: str - Neon table whose allowed columns are checked against.
         - columns: Iterable[str] | None - Requested columns; defaults to all allowed columns when None.

    Returns:
         - list[str] - Validated, ordered column list.
    """
    allowed = NEON_OFFLINE_TABLES[table_name]
    if columns is None:
        return sorted(allowed)
    cols = list(columns)
    unknown = sorted(set(cols) - allowed)
    if unknown:
        raise ValueError(f"Unknown column(s) for {table_name}: {unknown}")
    return cols


def _numeric_average_columns(selected_columns: Iterable[str]) -> list[str]:
    """Filter out non-numeric (categorical/timestamp) columns unsuitable for SQL AVG.

    Args:
         - selected_columns: Iterable[str] - Full set of columns selected for the query.

    Returns:
         - list[str] - Subset of columns eligible for SQL AVG aggregation.
    """
    return [col for col in selected_columns if col not in _NON_AVERAGED_COLUMNS]


def _resolve_range(time_range: Union[str, Tuple]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a time range argument to a concrete UTC (start, end) timestamp pair.

    Args:
         - time_range: Union[str, Tuple] - Preset string (e.g. "last 1 week"), "full" for all available data, or a (start, end) datetime/Timestamp tuple.

    Returns:
         - tuple[pd.Timestamp, pd.Timestamp] - UTC-aware (start, end) pair.
    """
    now = pd.Timestamp(datetime.now(timezone.utc))
    if isinstance(time_range, str) and time_range.strip().lower() == "full":
        return pd.Timestamp("2023-01-01T00:00:00Z"), now
    if isinstance(time_range, str):
        delta = TIMEDELTAS.get(time_range.strip().lower())
        if delta is None:
            raise ValueError(f"Unknown time_range preset: {time_range!r}")
        return now - delta, now
    if isinstance(time_range, tuple) and len(time_range) == 2:
        return pd.to_datetime(time_range[0], utc=True), pd.to_datetime(
            time_range[1], utc=True
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
    """Build a parameterised SQL SELECT statement for the requested query type.

    Args:
         - table_name: str - Target Neon table name.
         - selected_columns: list[str] - Columns to include in the SELECT clause.
         - query_type: str - One of "ts"/"raw" for raw rows, "windowed-average"/"hourly-average" for time-bucketed averages, or "average" for a single aggregate row.
         - window: str | None - PostgreSQL interval string (e.g. "1 hour") used for windowed-average queries.

    Returns:
         - tuple[str, dict] - (SQL string with :param placeholders, extra bind params dict).
    """
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
    """Fetch offline data from a whitelisted Neon/PostgreSQL table.

    Args:
         - table_name: str - One of the keys in NEON_OFFLINE_TABLES.
         - time_range: Union[str, Tuple] - Preset string (e.g. "last 1 week"), "full" for all data, or a (start, end) datetime/Timestamp tuple.
         - query_type: str - "ts"/"raw" for raw rows, "windowed-average" for time-bucketed averages, or "average" for a single aggregate row.
         - window: str | None - PostgreSQL interval string for "windowed-average" queries (e.g. "1 hour", "15 minutes").
         - columns: Iterable[str] | None - Column subset to fetch; defaults to all columns when None.
         - database_url: str | None - SQLAlchemy URL override; falls back to NEON_DATABASE_DEV_URL env var.

    Returns:
         - pd.DataFrame - Time-indexed DataFrame with UTC-aware index, sorted ascending.
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
