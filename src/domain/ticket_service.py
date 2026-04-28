"""Feedback ticket service for case management workflows."""

from __future__ import annotations

from data.db import Database
from ui.page_catalog import get_feedback_page_options

CRITICALITY_OPTIONS: tuple[str, ...] = ("low", "medium", "high", "critical")
STATUS_OPTIONS: tuple[str, ...] = (
    "open",
    "in-progress",
    "resolved",
    "closed",
    "dependency-conflict",
)
MANAGER_ROLES = {"admin", "supervisor"}


def can_manage_tickets(role: str | None) -> bool:
    """Return whether a role can update feedback ticket status."""
    return role in MANAGER_ROLES


class TicketService:
    """Business service for feedback tickets and status audit events."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def create_ticket(
        self,
        *,
        page: str,
        reporter_name: str,
        submitted_by: str,
        criticality: str,
        description: str,
        ideal_closure: str,
        ip_address: str = "",
    ) -> int:
        """Validate and create a new feedback ticket."""
        page = self._validate_page(page)
        reporter_name = self._required_text(reporter_name, "Reporter name", 120)
        submitted_by = self._required_text(submitted_by, "Submitted by", 120)
        criticality = self._validate_criticality(criticality)
        description = self._required_text(description, "Issue description", 4000)
        ideal_closure = self._required_text(ideal_closure, "Ideal closure", 2000)

        return self.db.create_feedback_ticket(
            page=page,
            reporter_name=reporter_name,
            submitted_by=submitted_by,
            criticality=criticality,
            description=description,
            ideal_closure=ideal_closure,
            ip_address=(ip_address or "").strip(),
        )

    def list_tickets(
        self,
        *,
        status: str | None = None,
        criticality: str | None = None,
        page: str | None = None,
    ) -> list[dict]:
        """List feedback tickets with optional exact-match filters."""
        if status is not None:
            status = self._validate_status(status)
        if criticality is not None:
            criticality = self._validate_criticality(criticality)
        if page is not None:
            page = self._validate_page(page)

        return self.db.list_feedback_tickets(
            status=status,
            criticality=criticality,
            page=page,
        )

    def update_status(
        self,
        *,
        ticket_id: int,
        status: str,
        actor: str,
        actor_role: str | None,
        comment: str = "",
        ip_address: str = "",
    ) -> bool:
        """Update a ticket status when the actor is allowed to manage cases."""
        if not can_manage_tickets(actor_role):
            raise PermissionError("Only admin and supervisor users can update tickets.")

        actor = self._required_text(actor, "Actor", 120)
        status = self._validate_status(status)
        ticket_id = int(ticket_id)

        return self.db.update_feedback_ticket_status(
            ticket_id=ticket_id,
            status=status,
            actor=actor,
            comment=(comment or "").strip()[:1000],
            ip_address=(ip_address or "").strip(),
        )

    def get_events(self, ticket_id: int) -> list[dict]:
        """Return audit events for a ticket."""
        return self.db.get_feedback_ticket_events(int(ticket_id))

    @staticmethod
    def _required_text(value: str, label: str, max_len: int) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError(f"{label} is required.")
        if len(cleaned) > max_len:
            raise ValueError(f"{label} must be {max_len} characters or fewer.")
        return cleaned

    @staticmethod
    def _validate_criticality(value: str) -> str:
        cleaned = (value or "").strip().lower()
        if cleaned not in CRITICALITY_OPTIONS:
            raise ValueError("Invalid criticality.")
        return cleaned

    @staticmethod
    def _validate_status(value: str) -> str:
        cleaned = (value or "").strip().lower()
        if cleaned not in STATUS_OPTIONS:
            raise ValueError("Invalid status.")
        return cleaned

    @staticmethod
    def _validate_page(value: str) -> str:
        cleaned = (value or "").strip()
        if cleaned not in get_feedback_page_options():
            raise ValueError("Invalid page selection.")
        return cleaned
