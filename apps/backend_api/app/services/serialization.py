"""JSON-safe DataFrame serialization helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from apps.backend_api.app.api.v1.schemas.data import DataColumnInfo


def _utc_timestamp(value: datetime | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars into JSON-native values without NaN."""
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return _utc_timestamp(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Decimal):
        return float(value)
    return value


def _column_dtype(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "string"


def _label_for_column(column: object) -> str:
    return str(column).replace("_", " ").strip().title() or str(column)


def dataframe_to_preview(
    df: pd.DataFrame,
    *,
    limit: int,
    offset: int = 0,
    include_index: bool = False,
) -> tuple[list[DataColumnInfo], list[dict[str, Any]], int, bool]:
    """Serialize a bounded DataFrame preview with canonical column metadata."""
    source = df.copy()
    row_count = len(source)
    offset = max(offset, 0)
    limit = max(limit, 0)

    if include_index:
        if isinstance(source.index, pd.DatetimeIndex):
            index = pd.to_datetime(source.index, errors="coerce", utc=True)
            source.index = index
            source.index.name = source.index.name or "time"
        elif source.index.name is None:
            source.index.name = "index"
        source = source.reset_index()

    window = source.iloc[offset : offset + limit]
    truncated = row_count > offset + len(window)
    columns = [
        DataColumnInfo(
            id=str(column),
            label=_label_for_column(column),
            dtype=_column_dtype(source[column]),
        )
        for column in source.columns
    ]
    rows = [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in window.to_dict(orient="records")
    ]
    return columns, rows, row_count, truncated
