from __future__ import annotations

from datetime import date

import streamlit as st

from ui.components import show_report
from utils.window_helpers import fetch_from_qdrant


def _run_live_shift_report(
    shift_date: date,
    shift_label: str,
    *,
    include_analysis: bool,
    status_box,
) -> str:
    """Fetch live shift data and render markdown report text."""
    from agents.llm.llm_client import OpenRouterClient
    from reports.shift_report import ShiftReportService

    status_box.status("Fetching shift data...", expanded=False)
    _, markdown = ShiftReportService(llm_client=OpenRouterClient()).generate(
        shift_date,
        shift_label,
        include_analysis=include_analysis,
    )
    return markdown


def render_reports() -> None:
    """Render the Reports tab with saved and live shift report options."""
    st.header("Reports")

    with st.form(key="report_form_main"):
        controls_left, controls_right = st.columns(2)

        with controls_left:
            report_type = st.selectbox("Report Type", ["Shift"], index=0)
            source = st.radio(
                "Source",
                ["Saved", "Live"],
                horizontal=True,
                help=(
                    "Saved: fetch a pre-stored shift summary.\n\n"
                    "Live: generate report on-the-fly from raw data."
                ),
            )
            selected_date = st.date_input("Date", date.today())

        with controls_right:
            shift_label = st.selectbox("Shift", ["A", "B", "C"])
            agentic_analysis = st.toggle(
                "Agentic analysis",
                value=False,
                help="When enabled, compare current shift against previous shift.",
            )

            st.caption(
                f"Selected: {report_type} report | {source} source | "
                f"{selected_date:%Y-%m-%d} Shift {shift_label}"
            )

        generate_report = st.form_submit_button(
            "Generate Report",
            use_container_width=True,
        )

    if not generate_report:
        return

    window_id = f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"

    if source == "Saved":
        from agents.memory.vector_store import QdrantVectorStore

        vector_store = QdrantVectorStore()
        payload = fetch_from_qdrant(vector_store, window_id)
        if payload:
            show_report(f"Report ({window_id})", payload["summary_text"])
        else:
            st.warning(
                f"No saved report found for {window_id}. "
                "Try Live source to generate one."
            )
        return

    analysis_mode = "agentic" if agentic_analysis else "plain"
    cache_key = f"live_report_{window_id}_{analysis_mode}"

    if cache_key in st.session_state:
        st.info(
            "Showing cached live report for "
            f"{window_id} ({analysis_mode}). Re-submit to refresh."
        )
        show_report(f"Live Report ({window_id})", st.session_state[cache_key])
        return

    status_box = st.empty()
    spinner_message = (
        f"Generating live report for {window_id} "
        "from source data" + (" and LLM analysis..." if agentic_analysis else "...")
    )

    with st.spinner(spinner_message):
        report_text = _run_live_shift_report(
            selected_date,
            shift_label,
            include_analysis=agentic_analysis,
            status_box=status_box,
        )

    status_box.empty()
    st.session_state[cache_key] = report_text
    show_report(f"Live Report ({window_id})", report_text)
