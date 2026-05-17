"""Streamlit adapter for backend-managed static dataset refresh status."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from data import dataset_refresh_api
from data.ml.static_csv import load_static_dataset


def sync_static_dataset_status(
    *,
    auto_enqueue: bool = True,
    rm_choice: str = "Full",
    cache_keys_to_clear: Iterable[str] = (),
    key_prefix: str = "static_dataset",
) -> dict:
    """Display backend refresh state and clear local caches on active-version change."""

    actor = str(st.session_state.get("auth_user_id") or st.session_state.get("auth_user") or "")
    status = dataset_refresh_api.get_static_status(
        auto_enqueue=auto_enqueue,
        rm_choice=rm_choice,
        triggered_by=actor or None,
    )

    version_key = f"_{key_prefix}_version"
    latest_version = status.get("latest_version_id")
    if latest_version and st.session_state.get(version_key) != latest_version:
        for cache_key in cache_keys_to_clear:
            st.session_state.pop(cache_key, None)
        load_static_dataset.clear()
        st.session_state[version_key] = latest_version

    _render_status(status)

    return status


def _render_status(status: dict) -> None:
    state = status.get("state")
    message = status.get("message") or ""
    last_refresh_at = status.get("last_refresh_at")
    if state == "refreshing":
        st.sidebar.info(message or "Dataset refresh is running. Using last active version.")
    elif state == "stale":
        st.sidebar.warning(message or "Dataset is stale. Using last active version.")
    elif state == "failed":
        st.sidebar.warning(status.get("error_message") or message or "Dataset refresh status failed.")
    elif last_refresh_at:
        st.sidebar.caption(f"Static dataset current: {last_refresh_at}")
