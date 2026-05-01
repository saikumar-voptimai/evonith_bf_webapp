from __future__ import annotations

from datetime import date

import streamlit as st

from ui.components import show_report
from utils.window_helpers import build_day_window_id, fetch_from_qdrant


def _run_live_shift_report(d: date, label: str, *, status_box) -> str:
    """Fetch data, build metrics, run LLM analysis — no agent loop."""
    from agents.llm.llm_client import OpenRouterClient
    from reports.shift_report import ShiftReportService

    status_box.status("Fetching shift data…", expanded=False)
    _, markdown = ShiftReportService(llm_client=OpenRouterClient()).generate(d, label)
    return markdown


def render_reports(*, vector_store) -> None:
    """Renders the Reports page with Qdrant (historical) or Live (on-the-fly LLM) modes."""

    st.header("📊 Reports")
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

    st.sidebar.text("Report Type (Shift)")

    # ── Source selector ──────────────────────────────────────────────────────
    source = st.sidebar.radio(
        "Source",
        ["Saved", "Live"],
        horizontal=True,
        help=(
            "**Saved** — fetch a pre-stored shift summary.\n\n"
            "**Live** — generate the report on-the-fly from raw InfluxDB data."
        ),
    )

    # ── Report form ──────────────────────────────────────────────────────────
    with st.sidebar.form(key="report_form"):
        selected_date = st.date_input("Select date", date.today())
        shift_label = st.selectbox("Select shift", ["A", "B", "C"])
        btn_label = "Generate Report" if source == "Live" else "Fetch Report"
        fetch_report = st.form_submit_button(btn_label)

    if not fetch_report:
        return

    window_id = f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"

    # ── DB-Saved mode ──────────────────────────────────────────────────────────
    if source == "Saved":
        payload = fetch_from_qdrant(vector_store, window_id)
        if payload:
            show_report(f"📄 Report ({window_id})", payload["summary_text"])
        else:
            st.warning(
                f"No report found in DB-Saved for **{window_id}**. "
                "Try the **Live** source to generate one on-the-fly."
            )
        return

    # ── Live mode ────────────────────────────────────────────────────────────
    cache_key = f"live_report_{window_id}"

    # Show cached result immediately (avoid re-generating on widget interaction)
    if cache_key in st.session_state:
        st.info(f"Showing cached live report for **{window_id}**. Re-submit to refresh.")
        show_report(f"⚡ Live Report ({window_id})", st.session_state[cache_key])
        return

    status_box = st.empty()

    with st.spinner(
        f"Generating live report for **{window_id}** — fetching InfluxDB data and calling LLM…"
    ):
        report_text = _run_live_shift_report(
            selected_date, shift_label, status_box=status_box
        )

    status_box.empty()
    st.session_state[cache_key] = report_text
    show_report(f"⚡ Live Report ({window_id})", report_text)
