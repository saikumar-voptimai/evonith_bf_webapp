# FurnaceMind/ui/layout.py
# Purpose: Streamlit page layout definitions
# Fixed: Removed st.set_page_config() — it's called in main() only
#        (Streamlit allows only one call per session)

import streamlit as st


def render_page_header():
    """Render the main page header. Page config is set in main()."""
    st.title("🔥 FurnaceMind – Blast Furnace Operational Intelligence")
    st.markdown(
        "Context-aware furnace monitoring and learning system."
    )


def render_sidebar():
    st.sidebar.header("Controls")
    st.sidebar.info(
        "Select time ranges and reports to explore furnace behavior."
    )