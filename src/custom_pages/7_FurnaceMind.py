"""FurnaceMind page — AI co-pilot for blast furnace operations.

Two tabs are available:
  AI Co-Operate — conversational agent with tool-calling and skill buttons.
  Reports       — live and saved shift handover reports.
"""

from __future__ import annotations

import streamlit as st

from agents.furnacemind.ai_cooperate_page import render_ai_cooperate
from ui.furnacemind_sections import select_nav_tab
from ui.furnacemind.reports import render_reports
from ui.styles import apply_styles
from utils.dataset_refresh_status import sync_static_dataset_status


def main() -> None:
    """Render the FurnaceMind page."""
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    apply_styles()

    st.title("FurnaceMind")
    st.caption("Blast Furnace Operational Intelligence")

    # Dataset status is checked by the backend; no Streamlit background refresh.
    sync_static_dataset_status(
        cache_keys_to_clear=("fm_ml_df_cache",),
        key_prefix="furnacemind_static_dataset",
    )

    app_mode = select_nav_tab()

    if app_mode == "🤖 AI Co-Operate":
        render_ai_cooperate(field_labels={})
        return

    if app_mode == "📊 Reports":
        render_reports()
        return


if __name__ == "__main__":
    main()
