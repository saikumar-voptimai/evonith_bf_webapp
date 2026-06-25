"""Data fetching utilities for AI Copilot.

``fetch_recent_online`` — pulls windowed-average data from InfluxDB.
``df_packet``           — converts a DataFrame to a compact markdown string
                          suitable for embedding in an LLM prompt.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data.fetch_presets import ONLINE_MEASUREMENT_LABELS
from furnace_data.influx.online import fetch_online_df

MEASUREMENT_LABELS = ONLINE_MEASUREMENT_LABELS


@st.cache_data(show_spinner=False, ttl=600)
def fetch_recent_online(
    tr: str = "last 8 hours",
    window_by: str = "15 minutes",
) -> pd.DataFrame:
    """Fetch windowed-average telemetry from InfluxDB.

    Parameters
    ----------
    tr:       InfluxDB time-range string, e.g. ``"last 8 hours"``.
    window_by: Aggregation window, e.g. ``"15 minutes"`` or ``"1 hour"``.
    """
    return fetch_online_df(
        selected_measurements=list(MEASUREMENT_LABELS.keys()),
        time_range=tr,
        window_by=window_by,
        column_naming="field",
    )


def df_packet(df: pd.DataFrame, max_rows: int = 160) -> str:
    """Convert *df* to a compact markdown string for LLM prompt injection.

    Downsamples to at most *max_rows* rows to keep prompt size manageable.
    """
    if df.empty:
        return "_No data in the selected window._"

    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].astype(float).round(4)

    if len(d) > max_rows:
        step = max(1, len(d) // max_rows)
        d = d.iloc[::step]

    parts = [
        f"Rows: {len(df)} | Columns: {len(df.columns)}",
        d.reset_index(names="timestamp").to_markdown(index=False),
        "\n**Summary Stats:**",
        df.describe().round(3).to_markdown(),
    ]
    return "\n\n".join(parts)
