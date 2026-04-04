"""FurnaceMind Streamlit page.

This file is intentionally kept small.
Each major tab is implemented in `FurnaceMind.ui.furnacemind_sections`.
"""

from __future__ import annotations

import streamlit as st

from FurnaceMind.ui.layout import render_page_header
from FurnaceMind.ui.styles import apply_styles

from FurnaceMind.ui.furnacemind_sections import (
    select_nav_tab,
    render_ai_cooperate,
    render_reports,
    render_live_operations,
    FIELD_LABELS,
)


def main() -> None:
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    render_page_header()
    apply_styles()

    from FurnaceMind.memory.vector_store import QdrantVectorStore
    vector_store = QdrantVectorStore()
    st.session_state.vector_store = vector_store

    st.sidebar.title("FurnaceMind")

    app_mode = select_nav_tab()

    if app_mode == "🤖 AI Co-Operate":
        render_ai_cooperate(field_labels=FIELD_LABELS)
        return

    if app_mode == "📊 Reports":
        render_reports(vector_store=vector_store)
        return

    if app_mode == "📡 Live Operations":
        render_live_operations()
        return




if __name__ == "__main__":
    main()