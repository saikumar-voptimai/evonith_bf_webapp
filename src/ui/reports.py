"""render_reports — Historical Reports tab."""

from __future__ import annotations

from datetime import date

import streamlit as st

from ui.components import show_report
from utils.window_helpers import build_day_window_id, fetch_from_qdrant


def render_reports(*, vector_store) -> None:
    """Render the Historical Reports tab.

    Parameters
    ----------
    vector_store:
        QdrantVectorStore instance used to fetch report payloads by window_id.
    """
    st.header("📊 Historical Reports")

    report_level = st.sidebar.radio("Report Type", ["Shift", "Day", "Week", "Bi-week"])
    fetch_report = False

    if report_level == "Shift":
        selected_date = st.sidebar.date_input("Select date", date.today())
        shift_label = st.sidebar.selectbox("Select shift", ["A", "B", "C"])
        fetch_report = st.sidebar.button("Fetch Report")

    elif report_level == "Day":
        selected_date = st.sidebar.date_input("Select date", date.today())
        fetch_report = st.sidebar.button("Fetch Report")

    elif report_level == "Week":
        selected_week = st.sidebar.text_input("Week window_id (YYYY-MM-DD/YYYY-MM-DD)")
        fetch_report = st.sidebar.button("Fetch Report")

    elif report_level == "Bi-week":
        selected_biweek = st.sidebar.text_input(
            "Bi-week window_id (YYYY-MM-DD/YYYY-MM-DD)"
        )
        fetch_report = st.sidebar.button("Fetch Report")

    if not fetch_report:
        return

    if report_level == "Shift":
        window_id = f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"
    elif report_level == "Day":
        window_id = build_day_window_id(selected_date)
    elif report_level == "Week":
        window_id = f"week_{selected_week}"
    else:
        window_id = f"bi_week_{selected_biweek}"

    payload = fetch_from_qdrant(vector_store, window_id)
    if payload:
        display_window_id = payload.get("window_id", window_id)
        summary_text = payload.get("summary_text") or payload.get("summary")

        if summary_text:
            show_report(f"📄 Report ({display_window_id})", summary_text)
        else:
            st.warning("Report found, but summary text is missing.")
    else:
        st.warning("No report found.")