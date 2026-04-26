"""Shared helpers for the webapp's static furnace dataset CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config.config_loader import load_config


def get_static_dataset_path(data_rel_path: str | None = None) -> Path:
    """Resolve the canonical static furnace dataset path inside the webapp repo."""
    rel_path = data_rel_path or load_config("setting_ds_dv.yml")["DATA"]
    return (Path(__file__).resolve().parents[3] / rel_path).resolve()


@st.cache_data(ttl=3600, show_spinner=False)
def load_static_dataset(
    path: str | Path | None = None,
    *,
    index_col: int | None = 0,
    parse_dates: bool = True,
    low_memory: bool = False,
    sort_index: bool = True,
) -> pd.DataFrame:
    """Load the static furnace dataset with the common parsing defaults.

    Results are cached for up to one hour (``ttl=3600``).  The background
    :mod:`utils.dataset_refresher` calls ``load_static_dataset.clear()``
    immediately after writing a new CSV so pages pick up fresh data on their
    next re-run without waiting for the TTL to expire.
    """
    if path is None:
        csv_path = get_static_dataset_path()
    else:
        csv_path = Path(path)
        if not csv_path.is_absolute():
            csv_path = (Path(__file__).resolve().parents[3] / csv_path).resolve()
    df = pd.read_csv(
        csv_path,
        index_col=index_col,
        parse_dates=parse_dates,
        low_memory=low_memory,
    )
    return df.sort_index() if sort_index else df
