"""JSON-safe DataFrame serialization helpers."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from app.api.v1.schemas.data import DataColumnInfo


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Decimal):
        return float(value)
    return value


def dataframe_to_preview(
    df: pd.DataFrame,
    *,
    limit: int,
    offset: int = 0,
    include_index: bool = False,
) -> tuple[list[DataColumnInfo], list[dict[str, Any]], int, bool]:
    """Serialize a DataFrame into columns, rows, total count, and truncated flag."""
    source = df.copy()
    row_count = len(source)
    offset = max(offset, 0)
    limit = max(limit, 0)

    if include_index:
        source = source.reset_index()

    window = source.iloc[offset : offset + limit]
    truncated = row_count > offset + len(window)
    columns = [
        DataColumnInfo(name=str(column), dtype=str(dtype))
        for column, dtype in window.dtypes.items()
    ]
    rows = [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in window.to_dict(orient="records")
    ]
    return columns, rows, row_count, truncated
