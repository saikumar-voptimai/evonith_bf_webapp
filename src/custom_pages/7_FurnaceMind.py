"""FurnaceMind page — AI co-pilot for blast furnace operations.

Two tabs are available:
  AI Co-Operate — conversational agent with tool-calling and skill buttons.
  Reports       — live and saved shift handover reports.
"""

from __future__ import annotations

import streamlit as st

from agents.furnacemind.ai_cooperate_page import render_ai_cooperate
from config.config_loader import load_config
from ui.furnacemind_sections import select_nav_tab
from ui.furnacemind.reports import render_reports
from ui.styles import apply_styles
from utils.dataset_refresher import (
    get_version as _ds_get_version,
    maybe_refresh as _ds_maybe_refresh,
)


def main() -> None:
    """Render the FurnaceMind page."""
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    apply_styles()

    st.title("FurnaceMind")
    st.caption("Blast Furnace Operational Intelligence")

    # ── ML dataset auto-refresh ────────────────────────────────────────────
    _config = load_config("setting_ds_dv.yml")
    if _ds_maybe_refresh(_config):
        st.sidebar.caption("⏳ Refreshing dataset in background…")

    _current_ds_version = _ds_get_version()
    if st.session_state.get("_ds_version") != _current_ds_version:
        st.session_state.pop("fm_ml_df_cache", None)
        st.session_state["_ds_version"] = _current_ds_version

    app_mode = select_nav_tab()

    if app_mode == "🤖 AI Co-Operate":
        render_ai_cooperate(field_labels={})
        return

    if app_mode == "📊 Reports":
        render_reports()
        return


if __name__ == "__main__":
    main()
