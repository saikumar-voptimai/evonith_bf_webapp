from __future__ import annotations

from datetime import date
import streamlit as st

from ui.components import show_report
from utils.window_helpers import build_day_window_id, fetch_from_qdrant


def render_reports(*, vector_store) -> None:
    """Renders the Historical Reports page, allowing users to select report type and parameters, and fetches the corresponding report summary from the Qdrant vector store.
    Args:
       vector_store: The Qdrant vector store instance to fetch reports from.
    returns: None
    results:
        Displays a form for selecting report type and parameters, and shows the fetched report summary.
    """

    st.header(" Historical Reports")
    st.markdown(
        """
    <style>
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background-color: transparent !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # report_level = st.sidebar.radio("Report Type", ["Shift", "Day", "Week", "Bi-week"])
    report_level = st.sidebar.text("Report Type (Shift)")

    #  FORM START (key change)
    with st.sidebar.form(key="report_form"):

        if report_level == "Shift":
            selected_date = st.date_input("Select date", date.today())
            shift_label = st.selectbox("Select shift", ["A", "B", "C"])

        elif report_level == "Day":
            selected_date = st.date_input("Select date", date.today())

        elif report_level == "Week":
            selected_week = st.text_input("Week window_id (YYYY-MM-DD/YYYY-MM-DD)")

        elif report_level == "Bi-week":
            selected_biweek = st.text_input("Bi-week window_id (YYYY-MM-DD/YYYY-MM-DD)")

        #  Submit button INSIDE form
        fetch_report = st.form_submit_button("Fetch Report")

    #  Do nothing until button clicked
    if not fetch_report:
        return

    #  Build window_id AFTER submit
    if report_level == "Shift":
        window_id = f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"

    elif report_level == "Day":
        window_id = build_day_window_id(selected_date)

    elif report_level == "Week":
        window_id = f"week_{selected_week}"

    else:
        window_id = f"bi_week_{selected_biweek}"

    #  Fetch data ONLY after submit
    payload = fetch_from_qdrant(vector_store, window_id)

    if payload:
        report_text = payload["summary_text"]

        show_report(f"📄 Report ({window_id})", report_text)

        # # ✅ DOWNLOAD BUTTON
        # st.download_button(
        #     label="⬇️ Download Report",
        #     data=report_text,
        #     file_name=f"{window_id}.pdf",
        #     mime="application/pdf"
        # )
    else:
        st.warning("No report found.")