"""Tests for the feedback ticket service."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.tickets.models import TicketStatus
from src.data.tickets.service import (
    TicketCreateRequest,
    TicketQueryFilter,
    TicketService,
    TicketStatusUpdateRequest,
)


@pytest.fixture
def ticket_service() -> TicketService:
    """Create a ticket service bound to an in-memory SQLite database."""
    return TicketService(db_url="sqlite:///:memory:")


def test_create_ticket_persists_fields_and_code(ticket_service: TicketService) -> None:
    """Creating a ticket should persist values and emit a formatted code."""
    created = ticket_service.create_ticket(
        TicketCreateRequest(
            page_name="V-Sense",
            reported_by="operator_1",
            criticality="high",
            description="Prediction panel failed to render.",
            ideal_closure_text="Need it fixed before next shift.",
        )
    )

    assert created.id > 0
    assert created.ticket_code == "TKT-000001"
    assert created.status == TicketStatus.OPEN.value
    assert created.criticality == "high"
    assert created.reported_by == "operator_1"

    listed = ticket_service.list_tickets()
    assert len(listed) == 1
    assert listed[0].ticket_code == created.ticket_code


def test_status_update_creates_audit_event(ticket_service: TicketService) -> None:
    """Status update should change ticket status and append event history."""
    created = ticket_service.create_ticket(
        TicketCreateRequest(
            page_name="Data Explorer",
            reported_by="supervisor_1",
            criticality="medium",
            description="CSV export includes duplicate columns.",
            ideal_closure_text="Need clean export for reporting.",
        )
    )

    updated = ticket_service.update_status(
        TicketStatusUpdateRequest(
            ticket_id=created.id,
            new_status="in_progress",
            actor="admin_1",
            actor_role="admin",
            comment="Assigned to data platform team.",
        )
    )

    assert updated.status == TicketStatus.IN_PROGRESS.value

    events = ticket_service.list_events(created.id)
    assert len(events) == 2
    assert events[0].event_type == "status_update"
    assert events[0].old_status == TicketStatus.OPEN.value
    assert events[0].new_status == TicketStatus.IN_PROGRESS.value
    assert events[0].comment == "Assigned to data platform team."


def test_non_admin_or_supervisor_cannot_update_status(ticket_service: TicketService) -> None:
    """Only admin and supervisor roles may update ticket status."""
    created = ticket_service.create_ticket(
        TicketCreateRequest(
            page_name="Material Balance",
            reported_by="user_1",
            criticality="low",
            description="Minor display formatting issue.",
            ideal_closure_text="Fix in upcoming patch release.",
        )
    )

    with pytest.raises(PermissionError):
        ticket_service.update_status(
            TicketStatusUpdateRequest(
                ticket_id=created.id,
                new_status="resolved",
                actor="user_1",
                actor_role="user",
                comment="Trying to self-resolve.",
            )
        )


def test_list_filters_status_page_criticality_reporter_keyword(
    ticket_service: TicketService,
) -> None:
    """List filter should apply all common ticket search dimensions."""
    ticket_service.create_ticket(
        TicketCreateRequest(
            page_name="V-Board",
            reported_by="user_alpha",
            criticality="critical",
            description="Trend chart crashed with division error.",
            ideal_closure_text="Restore chart before morning review.",
        )
    )
    target = ticket_service.create_ticket(
        TicketCreateRequest(
            page_name="CoPilot",
            reported_by="user_beta",
            criticality="high",
            description="Copilot answer quality dropped post refresh.",
            ideal_closure_text="Need stable answers for operations support.",
        )
    )
    ticket_service.update_status(
        TicketStatusUpdateRequest(
            ticket_id=target.id,
            new_status="in_progress",
            actor="supervisor_1",
            actor_role="supervisor",
            comment="Prompt tuning in progress.",
        )
    )

    query = TicketQueryFilter(
        statuses=["in_progress"],
        criticalities=["high"],
        page_names=["CoPilot"],
        reported_bys=["user_beta"],
        date_from=date.today() - timedelta(days=1),
        date_to=date.today() + timedelta(days=1),
        keyword="quality dropped",
    )
    results = ticket_service.list_tickets(query_filter=query)

    assert len(results) == 1
    assert results[0].id == target.id
    assert results[0].status == "in_progress"
