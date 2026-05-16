"""Streamlit artifact store for FurnaceMind tools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


class StreamlitArtifactStore:
    """Artifact store backed by ``st.session_state``."""

    def new_dataset_id(self, prefix: str) -> str:
        return new_dataset_id(prefix)

    def save_dataset(self, *, dataset_id: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        save_dataset(dataset_id=dataset_id, df=df, meta=meta)

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return get_dataset(dataset_id)

    def get_all_datasets(self) -> Dict[str, Any]:
        return get_all_datasets()

    def get_active_df(self) -> Optional[pd.DataFrame]:
        return get_active_df()

    def get_ml_cache(self) -> Optional[pd.DataFrame]:
        return get_ml_cache()

    def set_ml_cache(self, df: pd.DataFrame) -> None:
        set_ml_cache(df)

    def save_figure(self, fig: Any, code: str) -> None:
        save_figure(fig, code)

    def append_plot_error(self, error: str) -> None:
        append_plot_error(error)


def ensure_dataset_store() -> Dict[str, Any]:
    """Return the ``fm_datasets`` session dict, creating it when missing."""
    if "fm_datasets" not in st.session_state or not isinstance(
        st.session_state.get("fm_datasets"), dict
    ):
        st.session_state["fm_datasets"] = {}
    return st.session_state["fm_datasets"]


def new_dataset_id(prefix: str) -> str:
    """Generate a session-scoped dataset id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    counter = st.session_state.get("fm_dataset_counter", 0) + 1
    st.session_state["fm_dataset_counter"] = counter
    return f"{prefix}_{ts}_{counter}"


def save_dataset(*, dataset_id: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    """Save a dataframe artifact and mark it active."""
    store = ensure_dataset_store()
    store[dataset_id] = {"df": df, "meta": meta}
    st.session_state.fm_df = df
    st.session_state.fm_df_meta = meta


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Return a stored dataframe artifact by id."""
    return ensure_dataset_store().get(dataset_id)


def get_all_datasets() -> Dict[str, Any]:
    """Return all stored dataframe artifacts."""
    return ensure_dataset_store()


def get_active_df() -> Optional[pd.DataFrame]:
    """Return the active dataframe artifact."""
    return st.session_state.get("fm_df")


def get_ml_cache() -> Optional[pd.DataFrame]:
    """Return the cached static ML dataframe."""
    return st.session_state.get("fm_ml_df_cache")


def set_ml_cache(df: pd.DataFrame) -> None:
    """Cache the static ML dataframe for this Streamlit session."""
    st.session_state["fm_ml_df_cache"] = df


def save_figure(fig: Any, code: str) -> None:
    """Save the latest generated figure."""
    st.session_state.fm_fig = fig
    st.session_state.last_plot_code = code


def append_plot_error(error: str) -> None:
    """Record the latest plotting error."""
    st.session_state.last_plot_error = error
