# ui/layout.py
# Purpose: Streamlit page layout definitions

import streamlit as st


def render_page_header():
    st.set_page_config(
        page_title="FurnaceMind",
        layout="wide",
    )
    st.title("🔥 FurnaceMind – Blast Furnace Operational Intelligence")
    st.markdown(
        "Context-aware furnace monitoring and learning system."
    )


def render_sidebar():
    st.sidebar.header("Controls")
    st.sidebar.info(
        "Select time ranges and reports to explore furnace behavior."
    )