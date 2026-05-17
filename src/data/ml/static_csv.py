"""Shared helpers for the webapp's static furnace ML dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from furnace_data.services import ml_dataset_service


def fetch_static_dataset_from_database(sort_index: bool = True) -> pd.DataFrame:
    """Fetch the full static ML dataset from the configured database."""
    return ml_dataset_service.fetch_static_dataset_from_database(sort_index=sort_index)


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

    The legacy CSV-oriented arguments are accepted for call-site compatibility,
    but data is loaded from the active database view.
    """
    return ml_dataset_service.load_static_dataset(
        path,
        index_col=index_col,
        parse_dates=parse_dates,
        low_memory=low_memory,
        sort_index=sort_index,
    )
