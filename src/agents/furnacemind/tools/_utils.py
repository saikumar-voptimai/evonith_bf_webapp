"""Shared helpers for FurnaceMind tool adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

log = logging.getLogger(__name__)


def summarize_df(df: pd.DataFrame, *, dataset_id: str, title: str) -> str:
    """Return a concise text summary of a dataframe for LLM consumption."""
    if df is None or df.empty:
        return f"{title}: No data found."
    preview = df.head(2).to_string() if len(df) else "<empty>"
    return (
        f"{title}: dataset_id={dataset_id}\n"
        f"Shape: {df.shape}\n"
        f"Columns ({len(df.columns)}): {list(df.columns)}\n\n"
        f"Preview:\n{preview}"
    )


def log_tool_error(*, tool_name: str, params: Dict[str, Any], error: str) -> None:
    """Log a tool failure without mutating the source tree."""
    log.warning("FurnaceMind tool failed: %s params=%s error=%s", tool_name, params, error)
