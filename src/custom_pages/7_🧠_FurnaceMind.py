"""FurnaceMind Streamlit page.

This file is intentionally kept small.
Each major tab is implemented in `FurnaceMind.ui.furnacemind_sections`.
"""

from __future__ import annotations

import streamlit as st

from agents.cooperate import render_ai_cooperate
from config.config_loader import load_config
from utils.dataset_refresher import get_version as _ds_get_version, maybe_refresh as _ds_maybe_refresh
from memory.structured_store import StructuredStore
from memory.vector_store import QdrantVectorStore
from ui.furnacemind_sections import (
    FIELD_LABELS,
    select_nav_tab,
)
from ui.layout import render_page_header
from ui.live_ops import render_live_operations, render_furnace_intelligence
from ui.reports import render_reports
from ui.styles import apply_styles


def main() -> None:
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    render_page_header()
    apply_styles()

    # ── dataset auto-refresh ───────────────────────────────────────────────
    _config = load_config("setting_ds_dv.yml")
    if _ds_maybe_refresh(_config):
        st.sidebar.caption("⏳ Refreshing dataset in background…")

    # Clear the session-cached ML dataset when a new version has landed so
    # FurnaceMind tools pick up the fresh CSV on the next agent invocation.
    _current_ds_version = _ds_get_version()
    if st.session_state.get("_ds_version") != _current_ds_version:
        st.session_state.pop("fm_ml_df_cache", None)
        st.session_state["_ds_version"] = _current_ds_version

    structured_store = StructuredStore()
    vector_store = QdrantVectorStore()

    st.sidebar.title("FurnaceMind")

    app_mode = select_nav_tab()

    if app_mode == "🤖 AI Co-Operate":
        render_ai_cooperate(field_labels=FIELD_LABELS)
        return

    if app_mode == "📊 Reports":
        render_reports(vector_store=vector_store)
        return

    if app_mode == "📡 Live Operations":
        render_live_operations(
            structured_store=structured_store, vector_store=vector_store
        )
        return

    if app_mode == "🧠 Furnace Intelligence":
        render_furnace_intelligence(structured_store=structured_store)
        return


if __name__ == "__main__":
    main()