"""Shared Streamlit page layout helpers (header, sidebar skeleton).

Import and call these functions at the top of each page module to
apply a consistent page title, layout, and sidebar structure.
"""

# ui/layout.py
# Purpose: Streamlit page layout definitions

import streamlit as st


def render_page_header() -> None:
    """Configure ``st.set_page_config`` and render the dashboard page title."""
    st.set_page_config(
        page_title="FurnaceMind",
        layout="wide",
    )
    st.title("🔥 FurnaceMind – Blast Furnace Operational Intelligence")
    st.markdown("Context-aware furnace monitoring and learning system.")


def render_sidebar() -> None:
    """Render the standard sidebar header and help text."""
    st.sidebar.header("Controls")
    st.sidebar.info("Select time ranges and reports to explore furnace behavior.")
