"""Feedback and shared ticket board page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.page_registry import get_feedback_page_options
from data.tickets import TicketCreateRequest, TicketCriticality, TicketService, TicketStatus
from utils.feedback_page import (
    build_attachment_payloads,
    load_feedback_css,
    render_board,
    render_management_panel,
    render_overview_kpis,
)

MAX_UPLOAD_FILES = 5
MAX_UPLOAD_SIZE_MB = 5

CRITICALITY_OPTIONS = [
    TicketCriticality.LOW.value,
    TicketCriticality.MEDIUM.value,
    TicketCriticality.HIGH.value,
    TicketCriticality.CRITICAL.value,
]
STATUS_OPTIONS = [
    TicketStatus.OPEN.value,
    TicketStatus.IN_PROGRESS.value,
    TicketStatus.RESOLVED.value,
    TicketStatus.DEPENDENCY_CONFLICT.value,
    TicketStatus.CLOSED.value,
]


@st.cache_resource(show_spinner=False)
def get_ticket_service() -> TicketService:
    """Return a cached ticket service instance for the page session."""
    return TicketService()


def _render_form(ticket_service: TicketService) -> None:
    """Render the feedback submission form."""
    st.markdown(
        "<div class='feedback-section-title'>New Feedback</div>",
        unsafe_allow_html=True,
    )

    page_options = get_feedback_page_options()
    auth_user = str(st.session_state.get("auth_user", "")).strip()

    with st.form("feedback_create_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            page_name = st.selectbox("Page with issue", options=page_options)
        with col2:
            st.text_input("User name", value=auth_user, disabled=True)

        col3, col4 = st.columns([1, 2])
        with col3:
            criticality = st.selectbox(
                "Issue criticality",
                options=CRITICALITY_OPTIONS,
                format_func=TicketService.criticality_label,
                index=1,
            )
        with col4:
            ideal_closure = st.text_input(
                "Ideal closure",
                placeholder="Example: Need this fixed before next shift handover.",
                max_chars=240,
            )

        description = st.text_area(
            "Issue description",
            placeholder="Describe what went wrong, where, and how to reproduce it.",
            height=140,
            max_chars=2000,
        )
        uploaded_files = st.file_uploader(
            "Screenshots (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help=(
                f"Up to {MAX_UPLOAD_FILES} files, max {MAX_UPLOAD_SIZE_MB} MB each. "
                "Supported: png, jpg, jpeg, webp."
            ),
        )
        if uploaded_files:
            st.caption("Preview")
            preview_cols = st.columns(3)
            for idx, uploaded_file in enumerate(uploaded_files):
                with preview_cols[idx % 3]:
                    st.image(
                        uploaded_file,
                        caption=uploaded_file.name,
                        use_container_width=True,
                    )

        submitted = st.form_submit_button("Submit Ticket", type="primary")

    if not submitted:
        return

    try:
        created_ticket = ticket_service.create_ticket(
            TicketCreateRequest(
                page_name=page_name,
                reported_by=auth_user,
                criticality=criticality,
                description=description,
                ideal_closure_text=ideal_closure,
                created_by=auth_user,
            ),
            attachments=build_attachment_payloads(uploaded_files or []),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success(f"Ticket {created_ticket.ticket_code} created successfully.")
    st.rerun()


def main() -> None:
    """Render the Feedback page."""
    if "auth_user" not in st.session_state:
        st.warning("Please login to access this page.")
        st.stop()

    load_feedback_css(Path(__file__))
    ticket_service = get_ticket_service()
    all_tickets = ticket_service.list_tickets()

    st.markdown("<div class='feedback-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='feedback-hero'>
            <h2>Feedback & Support Desk</h2>
            <p>
                Raise platform issues, track progress, and collaborate through a shared,
                role-aware ticket board.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.52, 0.48], gap="large")
    with left:
        _render_form(ticket_service=ticket_service)
    with right:
        render_overview_kpis(all_tickets)
        render_management_panel(
            ticket_service=ticket_service,
            tickets=all_tickets,
            status_options=STATUS_OPTIONS,
        )

    st.divider()
    render_board(
        ticket_service=ticket_service,
        status_options=STATUS_OPTIONS,
        criticality_options=CRITICALITY_OPTIONS,
    )
    st.markdown("</div>", unsafe_allow_html=True)


main()
