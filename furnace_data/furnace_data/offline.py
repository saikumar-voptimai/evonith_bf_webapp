"""Offline PostgreSQL table fetcher.

This module exposes a simple DataFrame contract for whitelisted offline tables.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Tuple, Union

import pandas as pd
import yaml
from sqlalchemy import text

from furnace_data.relational.engine import build_relational_engine

_SCHEMA_PATH = Path(__file__).with_name("offline_tables.yml")
_SCHEMA = yaml.safe_load(_SCHEMA_PATH.read_text(encoding="utf-8")) or {}

_TABLES: dict[str, dict] = _SCHEMA.get("tables", {})
_TABLE_ALIASES: dict[str, str] = _SCHEMA.get("table_aliases", {}) or {}
_TIME_COLUMNS: dict[str, str | None] = {
    table: meta.get("time_column") for table, meta in _TABLES.items()
}

OFFLINE_TABLES: dict[str, set[str] | None] = {
    table: None if meta.get("allow_all_columns") else set(meta.get("columns", []))
    for table, meta in _TABLES.items()
}

OFFLINE_REPORT_MAP: dict[str, list[str]] = {
    report_type: list(tables)
    for report_type, tables in (_SCHEMA.get("report_map", {}) or {}).items()
}

_NON_AVERAGED_COLUMNS: set[str] = set(_SCHEMA.get("non_averaged_columns", []))

TIMEDELTAS = {
    "last 1 minute": timedelta(minutes=1),
    "last 5 minutes": timedelta(minutes=5),
    "last 10 minutes": timedelta(minutes=10),
    "last 15 minutes": timedelta(minutes=15),
    "last 30 minutes": timedelta(minutes=30),
    "last 1 hour": timedelta(hours=1),
    "last 3 hours": timedelta(hours=3),
    "last 6 hours": timedelta(hours=6),
    "last 8 hours": timedelta(hours=8),
    "last 12 hours": timedelta(hours=12),
    "last 1 day": timedelta(days=1),
    "last 3 days": timedelta(days=3),
    "last 1 week": timedelta(weeks=1),
    "last 2 weeks": timedelta(weeks=2),
    "last 1 month": timedelta(days=30),
    "last 2 months": timedelta(days=60),
    "last 3 months": timedelta(days=90),
    "last 6 months": timedelta(days=180),
    "last 1 year": timedelta(days=365),
}


def resolve_offline_table_name(table_name: str) -> str:
    """Resolve a public/legacy table name to the canonical schema-qualified name."""
    if table_name in _TABLES:
        return table_name
    resolved = _TABLE_ALIASES.get(table_name)
    if resolved in _TABLES:
        return resolved
    raise ValueError(
        f"Unknown offline table '{table_name}'. "
        f"Valid: {sorted(_TABLES)}; aliases: {sorted(_TABLE_ALIASES)}"
    )


def list_offline_tables() -> dict[str, object]:
    """Return a JSON-serialisable snapshot of the offline table whitelist."""
    return {
        "report_map": {k: list(v) for k, v in OFFLINE_REPORT_MAP.items()},
        "aliases": dict(sorted(_TABLE_ALIASES.items())),
        "tables": {
            table: {
                "time_column": _TIME_COLUMNS[table],
                "supports_aggregation": bool(meta.get("supports_aggregation", True)),
                "columns": (
                    "*"
                    if OFFLINE_TABLES[table] is None
                    else sorted(OFFLINE_TABLES[table] or [])
                ),
            }
            for table, meta in _TABLES.items()
        },
    }


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table_sql(table_name: str) -> str:
    meta = _TABLES[table_name]
    schema = meta.get("schema")
    table = meta.get("table")
    if schema and table:
        return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
    return ".".join(_quote_identifier(part) for part in table_name.split("."))


def _normalise_columns(table_name: str, columns: Iterable[str] | None) -> list[str]:
    allowed = OFFLINE_TABLES[table_name]
    if allowed is None:
        return list(columns or ["*"])

    if columns is None:
        return sorted(allowed)

    selected = list(columns)
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(f"Unknown column(s) for {table_name}: {unknown}")
    return selected


def _numeric_average_columns(selected_columns: Iterable[str]) -> list[str]:
    return [
        col
        for col in selected_columns
        if col != "*" and col not in _NON_AVERAGED_COLUMNS
    ]


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
    table_sql = _table_sql(table_name)
    query_type = query_type.strip().lower()

    if query_type not in {"ts", "raw", "average", "windowed-average", "hourly-average"}:
        raise ValueError(
            "query_type must be 'ts', 'raw', 'average', or 'windowed-average'."
        )

    if time_col is None:
        if query_type not in {"ts", "raw"}:
            raise ValueError(f"{table_name} does not support SQL aggregation fetches.")
        cols = (
            "*"
            if selected_columns == ["*"]
            else ", ".join(_quote_identifier(col) for col in selected_columns)
        )
        return (f"SELECT {cols} FROM {table_sql}", {})

    time_sql = _quote_identifier(time_col)

    if query_type in {"ts", "raw"}:
        cols = (
            "*"
            if selected_columns == ["*"]
            else ", ".join(_quote_identifier(col) for col in selected_columns)
        )
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
    """Fetch offline data from one whitelisted PostgreSQL table."""
    table_name = resolve_offline_table_name(table_name)

    start_time, end_time = _resolve_range(time_range)
    selected_columns = _normalise_columns(table_name, columns)
    time_col = _TIME_COLUMNS[table_name]
    if time_col and selected_columns != ["*"] and time_col not in selected_columns:
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

    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col)
        df.index.name = "time"

    return df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df


def get_offline_table_bounds(
    table_name: str,
    database_url: str | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    """Return min/max timestamps and row count for one whitelisted table."""
    table_name = resolve_offline_table_name(table_name)
    time_col = _TIME_COLUMNS[table_name]
    if time_col is None:
        return None, None, 0

    query = (
        f"SELECT MIN({_quote_identifier(time_col)}) AS start_time, "
        f"MAX({_quote_identifier(time_col)}) AS end_time, "
        f"COUNT(*) AS row_count FROM {_table_sql(table_name)}"
    )
    engine = build_relational_engine(database_url)
    try:
        df = pd.read_sql_query(text(query), engine)
    finally:
        engine.dispose()

    if df.empty:
        return None, None, 0

    row = df.iloc[0]
    start = pd.to_datetime(row["start_time"], utc=True) if pd.notna(row["start_time"]) else None
    end = pd.to_datetime(row["end_time"], utc=True) if pd.notna(row["end_time"]) else None
    return start, end, int(row["row_count"] or 0)


def get_offline_report_bounds(
    report_type: str,
    database_url: str | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    """Return combined temporal bounds for a logical offline report."""
    tables = OFFLINE_REPORT_MAP.get(report_type)
    if not tables:
        raise ValueError(
            f"Unknown offline report '{report_type}'. "
            f"Valid: {sorted(OFFLINE_REPORT_MAP)}"
        )

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    row_count = 0
    for table in tables:
        start, end, count = get_offline_table_bounds(table, database_url)
        row_count += count
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)

    return (
        min(starts) if starts else None,
        max(ends) if ends else None,
        row_count,
    )


def fetch_offline_report(
    report_type: str,
    time_range: Union[str, Tuple],
    query_type: str = "ts",
    window: str | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Fetch a logical offline report from one or more whitelisted tables."""
    tables = OFFLINE_REPORT_MAP.get(report_type)
    if not tables:
        raise ValueError(
            f"Unknown offline report '{report_type}'. "
            f"Valid: {sorted(OFFLINE_REPORT_MAP)}"
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

    combined = pd.concat(frames, axis=0, sort=False)
    if isinstance(combined.index, pd.DatetimeIndex):
        return combined.sort_index()
    return combined


__all__ = [
    "OFFLINE_REPORT_MAP",
    "OFFLINE_TABLES",
    "TIMEDELTAS",
    "fetch_offline_data",
    "fetch_offline_report",
    "get_offline_report_bounds",
    "get_offline_table_bounds",
    "list_offline_tables",
    "resolve_offline_table_name",
]
