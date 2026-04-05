"""FurnaceMind Streamlit page.

This file is intentionally kept small.
Each major tab is implemented in `FurnaceMind.ui.furnacemind_sections`.
"""

from __future__ import annotations

import streamlit as st

from FurnaceMind.ui.layout import render_page_header
from FurnaceMind.ui.styles import apply_styles

from FurnaceMind.memory.structured_store import StructuredStore
from FurnaceMind.memory.vector_store import QdrantVectorStore

from FurnaceMind.ui.furnacemind_sections import (
    select_nav_tab,
    FIELD_LABELS,
)

from FurnaceMind.ui.cooperate import render_ai_cooperate
from FurnaceMind.ui.reports   import render_reports
from FurnaceMind.ui.live_ops  import render_live_operations


def main() -> None:
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    render_page_header()
    apply_styles()

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
        render_live_operations(structured_store=structured_store, vector_store=vector_store)
        return

    if app_mode == "🧠 Furnace Intelligence":
        render_furnace_intelligence(structured_store=structured_store)
        return


if __name__ == "__main__":
    main()