"""FurnaceMind page - AI co-pilot for blast furnace operations."""

from __future__ import annotations

import streamlit as st

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.config.config_loader import load_config
from apps.frontend_streamlit.ui.furnacemind_sections import select_nav_tab
from apps.frontend_streamlit.ui.styles import apply_styles
from apps.frontend_streamlit.utils.dataset_refresher import (
    get_version as _ds_get_version,
    maybe_refresh as _ds_maybe_refresh,
)


def _refresh_ml_dataset_cache() -> None:
    config = load_config("setting_ds_dv.yml")
    if _ds_maybe_refresh(config):
        st.sidebar.caption("Refreshing dataset in background...")

    current_ds_version = _ds_get_version()
    if st.session_state.get("_ds_version") != current_ds_version:
        st.session_state.pop("fm_ml_df_cache", None)
        st.session_state["_ds_version"] = current_ds_version


def main() -> None:
    """Render the FurnaceMind page."""
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    apply_styles()

    st.title("FurnaceMind")
    st.caption("Blast Furnace Operational Intelligence")

    backend_enabled = is_backend_api_enabled("furnacemind")
    st.caption(f"FurnaceMind mode: {'Backend API' if backend_enabled else 'Direct'}")

    if not backend_enabled:
        _refresh_ml_dataset_cache()

    app_mode = select_nav_tab()

    if app_mode == "🤖 AI Co-Operate":
        if backend_enabled:
            from apps.frontend_streamlit.agents.furnacemind.api_cooperate_page import render_ai_cooperate_api

            render_ai_cooperate_api(field_labels={})
        else:
            from apps.frontend_streamlit.agents.furnacemind.ai_cooperate_page import render_ai_cooperate

            render_ai_cooperate(field_labels={})
        return

    if app_mode == "📊 Reports":
        if backend_enabled:
            from apps.frontend_streamlit.ui.furnacemind.reports import render_reports_api

            render_reports_api()
        else:
            from apps.frontend_streamlit.ui.furnacemind.reports import render_reports

            render_reports()
        return


main()