"""Tests for the feedback ticket service."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from src.data.tickets.models import TicketStatus
from src.data.tickets.service import (
    TicketCreateRequest,
    TicketDeleteRequest,
    TicketImageUpload,
    TicketQueryFilter,
    TicketService,
    TicketStatusUpdateRequest,
)


def _make_local_test_dir() -> Path:
    """Create a writable local temp directory under repo storage."""
    root = Path.cwd() / "storage" / "feedback" / "_pytest_local"
    root.mkdir(parents=True, exist_ok=True)
    test_dir = root / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    return test_dir


@pytest.fixture
def ticket_service() -> TicketService:
    """Create a ticket service bound to an in-memory SQLite database."""
    return TicketService(db_url="sqlite:///:memory:")


def test_create_ticket_persists_fields_and_code(ticket_service: TicketService) -> None:
    """Creating a ticket should persist values and emit a formatted code."""
    reporter_user_id = uuid4()
    creator_user_id = uuid4()
    created = ticket_service.create_ticket(
        TicketCreateRequest(
            page_name="V-Sense",
            reported_by="operator_1",
            criticality="high",
            description="Prediction panel failed to render.",
            ideal_closure_text="Need it fixed before next shift.",
            created_by="operator_1",
            reported_by_user_id=reporter_user_id,
            created_by_user_id=creator_user_id,
        )
    )

    assert created.id > 0
    assert created.ticket_code == "TKT-000001"
    assert created.status == TicketStatus.OPEN.value
    assert created.criticality == "high"
    assert created.reported_by == "operator_1"
    assert created.updated_by == "operator_1"
    assert created.reported_by_user_id == str(reporter_user_id)
    assert created.updated_by_user_id == str(creator_user_id)

    listed = ticket_service.list_tickets()
    assert len(listed) == 1
    assert listed[0].ticket_code == created.ticket_code

    events = ticket_service.list_events(created.id)
    assert events[0].event_type == "created"
    assert events[0].actor_user_id == str(creator_user_id)


def test_status_update_creates_audit_event(ticket_service: TicketService) -> None:
    """Status update should change ticket status and append event history."""
    actor_user_id = uuid4()
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
            actor_user_id=actor_user_id,
            comment="Assigned to data platform team.",
        )
    )

    assert updated.status == TicketStatus.IN_PROGRESS.value
    assert updated.updated_by == "admin_1"
    assert updated.updated_by_user_id == str(actor_user_id)

    events = ticket_service.list_events(created.id)
    assert len(events) == 2
    assert events[0].event_type == "status_update"
    assert events[0].old_status == TicketStatus.OPEN.value
    assert events[0].new_status == TicketStatus.IN_PROGRESS.value
    assert events[0].comment == "Assigned to data platform team."
    assert events[0].actor_user_id == str(actor_user_id)

    listed = ticket_service.list_tickets()
    assert listed[0].updated_by == "admin_1"


def test_non_admin_or_supervisor_cannot_update_status(
    ticket_service: TicketService,
) -> None:
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
    events = ticket_service.list_events(created.id)
    assert events[0].actor == "user_1"
    assert events[0].actor_user_id is None

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


def test_create_ticket_with_attachments_persists_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ticket creation should persist screenshot metadata and bytes."""
    local_temp = _make_local_test_dir()
    try:
        monkeypatch.chdir(local_temp)
        service = TicketService(db_url="sqlite:///:memory:")
        user_id = uuid4()

        created = service.create_ticket(
            TicketCreateRequest(
                page_name="Feedback",
                reported_by="user_gamma",
                criticality="high",
                description="Issue with screenshot flow.",
                ideal_closure_text="Need attachments visible in board.",
                reported_by_user_id=user_id,
                created_by_user_id=user_id,
            ),
            attachments=[
                TicketImageUpload(filename="screen1.png", content=b"png-bytes-1"),
                TicketImageUpload(filename="screen2.jpg", content=b"jpg-bytes-2"),
            ],
        )

        images = service.list_ticket_images(created.id)
        assert len(images) == 2
        assert images[0].ticket_id == created.id
        assert images[0].original_filename in {"screen1.png", "screen2.jpg"}
        assert {image.uploaded_by_user_id for image in images} == {str(user_id)}

        contents = [service.get_ticket_image_content(image.id) for image in images]
        assert {item[0] for item in contents if item} == {b"png-bytes-1", b"jpg-bytes-2"}
        assert not (local_temp / "src" / "storage" / "feedback" / "images").exists()

        service.delete_ticket(
            TicketDeleteRequest(
                ticket_id=created.id,
                actor="admin_1",
                actor_role="admin",
            )
        )
    finally:
        shutil.rmtree(local_temp, ignore_errors=True)


def test_attachment_validation_enforces_count_and_size(
    ticket_service: TicketService,
) -> None:
    """Attachment guardrails should enforce file count and per-file size limits."""
    too_many_attachments = [
        TicketImageUpload(filename=f"img_{idx}.png", content=b"ok") for idx in range(6)
    ]
    with pytest.raises(ValueError, match="Maximum 5 screenshots"):
        ticket_service.create_ticket(
            TicketCreateRequest(
                page_name="CoPilot",
                reported_by="user_many",
                criticality="medium",
                description="Many attachments.",
                ideal_closure_text="Should reject above threshold.",
            ),
            attachments=too_many_attachments,
        )

    oversize_file = TicketImageUpload(
        filename="huge.webp",
        content=b"x" * (5 * 1024 * 1024 + 1),
    )
    with pytest.raises(ValueError, match="exceeds the 5 MB size limit"):
        ticket_service.create_ticket(
            TicketCreateRequest(
                page_name="Data Explorer",
                reported_by="user_large",
                criticality="high",
                description="Large screenshot.",
                ideal_closure_text="Need size guardrail.",
            ),
            attachments=[oversize_file],
        )


def test_delete_ticket_cleans_up_images_and_enforces_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delete should be role-guarded and remove linked screenshot rows."""
    local_temp = _make_local_test_dir()
    try:
        monkeypatch.chdir(local_temp)
        service = TicketService(db_url="sqlite:///:memory:")

        created = service.create_ticket(
            TicketCreateRequest(
                page_name="V-Board",
                reported_by="admin_1",
                criticality="critical",
                description="Need to test delete flow.",
                ideal_closure_text="Delete should cleanup file artifacts.",
            ),
            attachments=[
                TicketImageUpload(filename="delete_me.jpeg", content=b"image-bytes")
            ],
        )
        image = service.list_ticket_images(created.id)[0]
        assert service.get_ticket_image_content(image.id)[0] == b"image-bytes"

        with pytest.raises(PermissionError):
            service.delete_ticket(
                TicketDeleteRequest(
                    ticket_id=created.id,
                    actor="user_1",
                    actor_role="user",
                )
            )
        assert service.list_tickets()
        assert service.get_ticket_image_content(image.id)[0] == b"image-bytes"

        service.delete_ticket(
            TicketDeleteRequest(
                ticket_id=created.id,
                actor="admin_1",
                actor_role="admin",
            )
        )
        assert not service.list_tickets()
        assert not service.list_ticket_images(created.id)
        assert service.get_ticket_image_content(image.id) is None
    finally:
        shutil.rmtree(local_temp, ignore_errors=True)
