from __future__ import annotations

from datetime import date

import streamlit as st

from reports.rendering import ReportDocument, document_to_markdown
from reports.shift_report.renderer import document_from_markdown
from ui.components import show_report_document
from utils.window_helpers import fetch_from_qdrant

_LAST_REPORT_KEY = "furnacemind_last_report"


def _remember_report(
    title: str,
    summary_text: str,
    document: ReportDocument | None = None,
) -> None:
    st.session_state[_LAST_REPORT_KEY] = {
        "title": title,
        "summary_text": summary_text,
        "document": document,
    }


def _show_last_report() -> bool:
    report = st.session_state.get(_LAST_REPORT_KEY)
    if not report:
        return False

    document = report.get("document") or document_from_markdown(
        report["title"],
        report["summary_text"],
    )
    show_report_document(document)
    return True


def _run_live_shift_report(
    shift_date: date,
    shift_label: str,
    *,
    include_analysis: bool,
    status_box,
) -> tuple[ReportDocument, str]:
    """Fetch live shift data and render markdown report text."""
    from agents.llm.llm_client import OpenRouterClient
    from reports.shift_report import ShiftReportService

    status_box.status("Fetching shift data...", expanded=False)
    _, document = ShiftReportService(llm_client=OpenRouterClient()).generate_document(
        shift_date,
        shift_label,
        include_analysis=include_analysis,
    )
    return document, document_to_markdown(document)


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
        _show_last_report()
        return

    window_id = f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"

    if source == "Saved":
        from agents.memory.vector_store import QdrantVectorStore

        vector_store = QdrantVectorStore()
        payload = fetch_from_qdrant(vector_store, window_id)
        if payload:
            title = f"Report ({window_id})"
            document = document_from_markdown(title, payload["summary_text"])
            _remember_report(title, payload["summary_text"], document)
            show_report_document(document)
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
        title = f"Live Report ({window_id})"
        cached = st.session_state[cache_key]
        if isinstance(cached, dict):
            document = cached["document"]
            markdown = cached["summary_text"]
        else:
            markdown = cached
            document = document_from_markdown(title, markdown)
        _remember_report(title, markdown, document)
        show_report_document(document)
        return

    status_box = st.empty()
    spinner_message = f"Generating live report for {window_id} from source data"
    spinner_message += " and LLM analysis..." if agentic_analysis else "..."

    with st.spinner(spinner_message):
        document, report_text = _run_live_shift_report(
            selected_date,
            shift_label,
            include_analysis=agentic_analysis,
            status_box=status_box,
        )

    status_box.empty()
    st.session_state[cache_key] = {
        "document": document,
        "summary_text": report_text,
    }
    title = f"Live Report ({window_id})"
    _remember_report(title, report_text, document)
    show_report_document(document)
