"""Feedback and shared ticket board page."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from config.page_registry import get_feedback_page_options
from data.tickets import (
    TicketCreateRequest,
    TicketCriticality,
    TicketQueryFilter,
    TicketService,
    TicketStatus,
    TicketStatusUpdateRequest,
)
from utils.session import is_admin, is_supervisor

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


def _load_css() -> None:
    """Load scoped CSS for the feedback page."""
    css_path = Path(__file__).resolve().parents[1] / "assets" / "css" / "feedback_style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _to_utc_datetime_floor(selected: date | datetime | None) -> datetime | None:
    """Convert a date-like value to inclusive UTC start timestamp."""
    if selected is None:
        return None
    if isinstance(selected, datetime):
        return selected if selected.tzinfo else selected.replace(tzinfo=timezone.utc)
    return datetime.combine(selected, datetime.min.time(), tzinfo=timezone.utc)


def _to_utc_datetime_ceiling(selected: date | datetime | None) -> datetime | None:
    """Convert a date-like value to inclusive UTC end timestamp."""
    if selected is None:
        return None
    if isinstance(selected, datetime):
        return selected if selected.tzinfo else selected.replace(tzinfo=timezone.utc)
    return datetime.combine(selected, datetime.max.time(), tzinfo=timezone.utc)


def _render_chip(label: str, css_class: str) -> str:
    """Build HTML for a status/criticality chip."""
    return f"<span class='feedback-chip {css_class}'>{label}</span>"


def _render_board(ticket_service: TicketService) -> None:
    """Render ticket board filters and ticket cards."""
    st.markdown("<div class='feedback-section-title'>Shared Ticket Board</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='feedback-note'>All logged-in users can view tickets. "
        "Status updates are restricted to admin and supervisor roles.</div>",
        unsafe_allow_html=True,
    )

    all_tickets = ticket_service.list_tickets()
    reporter_options = sorted({ticket.reported_by for ticket in all_tickets})
    page_options = sorted({ticket.page_name for ticket in all_tickets})

    today = date.today()
    default_from = today - timedelta(days=30)

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_statuses = st.multiselect(
                "Status",
                options=STATUS_OPTIONS,
                format_func=TicketService.status_label,
            )
            selected_pages = st.multiselect(
                "Page",
                options=page_options,
            )
        with col2:
            selected_criticalities = st.multiselect(
                "Criticality",
                options=CRITICALITY_OPTIONS,
                format_func=TicketService.criticality_label,
            )
            selected_reporters = st.multiselect(
                "Reporter",
                options=reporter_options,
            )
        with col3:
            keyword = st.text_input("Keyword", placeholder="code, issue text, user...")
            date_range = st.date_input(
                "Created date range",
                value=(default_from, today),
            )

    date_from: date | None = None
    date_to: date | None = None
    if isinstance(date_range, tuple):
        if len(date_range) == 2:
            date_from = date_range[0]
            date_to = date_range[1]
    elif isinstance(date_range, date):
        date_from = date_range
        date_to = date_range

    tickets = ticket_service.list_tickets(
        TicketQueryFilter(
            statuses=selected_statuses or None,
            criticalities=selected_criticalities or None,
            page_names=selected_pages or None,
            reported_bys=selected_reporters or None,
            date_from=_to_utc_datetime_floor(date_from),
            date_to=_to_utc_datetime_ceiling(date_to),
            keyword=keyword or None,
        )
    )

    if not tickets:
        st.markdown(
            "<div class='feedback-board-empty'>No tickets found for the current filters.</div>",
            unsafe_allow_html=True,
        )
        return

    can_update_status = is_admin() or is_supervisor()
    actor = str(st.session_state.get("auth_user", "")).strip()
    actor_role = str(st.session_state.get("role", "")).strip().lower()

    for ticket in tickets:
        status_label = TicketService.status_label(ticket.status)
        criticality_label = TicketService.criticality_label(ticket.criticality)
        status_css_class = f"feedback-chip-status-{ticket.status.replace('_', '-')}"
        criticality_css_class = f"feedback-chip-criticality-{ticket.criticality}"

        st.markdown("<div class='feedback-board-card'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.45, 1.35, 1.2, 1.9])
        with c1:
            st.markdown(f"**`{ticket.ticket_code}`**")
        with c2:
            st.markdown(ticket.page_name)
        with c3:
            st.markdown(
                _render_chip(criticality_label.lower(), criticality_css_class),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                _render_chip(status_label, status_css_class),
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div class='feedback-meta'>Raised by <b>{ticket.reported_by}</b> "
            f"on {ticket.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Last updated by <b>{ticket.updated_by}</b> on "
            f"{ticket.updated_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Open {ticket.ticket_code} details", expanded=False):
            st.markdown("**Issue description**")
            st.write(ticket.description)
            st.markdown("**Ideal closure**")
            st.write(ticket.ideal_closure_text)

            if can_update_status:
                st.markdown("**Status update**")
                with st.form(f"status_update_form_{ticket.id}"):
                    next_status = st.selectbox(
                        "New status",
                        options=STATUS_OPTIONS,
                        index=STATUS_OPTIONS.index(ticket.status),
                        format_func=TicketService.status_label,
                    )
                    update_comment = st.text_area(
                        "Update comment",
                        placeholder="Optional progress note...",
                        max_chars=600,
                    )
                    submit_status = st.form_submit_button("Update Status")
                if submit_status:
                    try:
                        ticket_service.update_status(
                            TicketStatusUpdateRequest(
                                ticket_id=ticket.id,
                                new_status=next_status,
                                actor=actor,
                                actor_role=actor_role,
                                comment=update_comment or None,
                            )
                        )
                        st.success(f"{ticket.ticket_code} updated to {TicketService.status_label(next_status)}.")
                        st.rerun()
                    except PermissionError as exc:
                        st.error(str(exc))
                    except ValueError as exc:
                        st.error(str(exc))
            else:
                st.info("Status updates are available only for admin/supervisor roles.")

            events = ticket_service.list_events(ticket.id)
            if events:
                st.markdown("**Event history**")
                events_df = pd.DataFrame(
                    {
                        "Time": [
                            event.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                            for event in events
                        ],
                        "Type": [event.event_type for event in events],
                        "From": [
                            TicketService.status_label(event.old_status)
                            if event.old_status
                            else "-"
                            for event in events
                        ],
                        "To": [
                            TicketService.status_label(event.new_status)
                            if event.new_status
                            else "-"
                            for event in events
                        ],
                        "Actor": [event.actor for event in events],
                        "Comment": [event.comment or "-" for event in events],
                    }
                )
                st.dataframe(events_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)


def _render_form(ticket_service: TicketService) -> None:
    """Render the feedback submission form."""
    st.markdown("<div class='feedback-section-title'>Raise an Issue</div>", unsafe_allow_html=True)

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
            )
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

    _load_css()
    ticket_service = get_ticket_service()

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

    _render_form(ticket_service=ticket_service)
    st.divider()
    _render_board(ticket_service=ticket_service)
    st.markdown("</div>", unsafe_allow_html=True)


main()
