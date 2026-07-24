"""Gateway abstraction for the Streamlit Feedback page."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import uuid4

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services import feedback_api


class FeedbackGateway(Protocol):
    def get_config(self) -> Any: ...
    def get_summary(self, filters: dict[str, Any] | None = None) -> Any: ...
    def list_tickets(self, query: dict[str, Any] | None = None) -> Any: ...
    def create_ticket(self, request: dict[str, Any], *, idempotency_key: str) -> Any: ...
    def get_ticket(self, ticket_id: str) -> Any: ...
    def update_ticket(self, ticket_id: str, request: dict[str, Any]) -> Any: ...
    def transition_ticket(self, ticket_id: str, request: dict[str, Any], *, idempotency_key: str) -> Any: ...
    def delete_ticket(self, ticket_id: str, *, expected_version: int, idempotency_key: str) -> Any: ...
    def list_events(self, ticket_id: str, query: dict[str, Any] | None = None) -> Any: ...
    def list_comments(self, ticket_id: str, query: dict[str, Any] | None = None) -> Any: ...
    def add_comment(self, ticket_id: str, body: str, *, idempotency_key: str) -> Any: ...
    def list_attachments(self, ticket_id: str) -> Any: ...
    def upload_attachment(self, ticket_id: str, file: Any, *, idempotency_key: str) -> Any: ...
    def preview_attachment(self, attachment_id: str) -> bytes: ...
    def download_attachment(self, attachment_id: str) -> bytes: ...
    def delete_attachment(self, attachment_id: str, *, idempotency_key: str) -> Any: ...


class ApiFeedbackGateway:
    """HTTP-backed feedback gateway."""

    def __init__(self, *, token: str | None = None) -> None:
        self.token = token

    def get_config(self) -> Any:
        return feedback_api.get_feedback_config(token=self.token)

    def get_summary(self, filters: dict[str, Any] | None = None) -> Any:
        return feedback_api.get_summary(filters, token=self.token)

    def list_tickets(self, query: dict[str, Any] | None = None) -> Any:
        return feedback_api.list_tickets(query, token=self.token)

    def create_ticket(self, request: dict[str, Any], *, idempotency_key: str) -> Any:
        return feedback_api.create_ticket(request, token=self.token, idempotency_key=idempotency_key)

    def get_ticket(self, ticket_id: str) -> Any:
        return feedback_api.get_ticket(ticket_id, token=self.token)

    def update_ticket(self, ticket_id: str, request: dict[str, Any]) -> Any:
        return feedback_api.update_ticket(ticket_id, request, token=self.token)

    def transition_ticket(self, ticket_id: str, request: dict[str, Any], *, idempotency_key: str) -> Any:
        return feedback_api.transition_ticket(ticket_id, request, token=self.token, idempotency_key=idempotency_key)

    def delete_ticket(self, ticket_id: str, *, expected_version: int, idempotency_key: str) -> Any:
        return feedback_api.delete_ticket(ticket_id, expected_version=expected_version, token=self.token, idempotency_key=idempotency_key)

    def list_events(self, ticket_id: str, query: dict[str, Any] | None = None) -> Any:
        return feedback_api.list_events(ticket_id, query, token=self.token)

    def list_comments(self, ticket_id: str, query: dict[str, Any] | None = None) -> Any:
        return feedback_api.list_comments(ticket_id, query, token=self.token)

    def add_comment(self, ticket_id: str, body: str, *, idempotency_key: str) -> Any:
        return feedback_api.add_comment(ticket_id, body, token=self.token, idempotency_key=idempotency_key)

    def list_attachments(self, ticket_id: str) -> Any:
        return feedback_api.list_attachments(ticket_id, token=self.token)

    def upload_attachment(self, ticket_id: str, file: Any, *, idempotency_key: str) -> Any:
        return feedback_api.upload_attachment(
            ticket_id,
            filename=file.name,
            content=file.getvalue(),
            content_type=file.type or "application/octet-stream",
            token=self.token,
            idempotency_key=idempotency_key,
        )

    def preview_attachment(self, attachment_id: str) -> bytes:
        return feedback_api.preview_attachment(attachment_id, token=self.token)

    def download_attachment(self, attachment_id: str) -> bytes:
        return feedback_api.download_attachment(attachment_id, token=self.token)

    def delete_attachment(self, attachment_id: str, *, idempotency_key: str) -> Any:
        return feedback_api.delete_attachment(attachment_id, token=self.token, idempotency_key=idempotency_key)


class DirectFeedbackGateway:
    """Lazy adapter around the temporary direct Streamlit ticket implementation."""

    def __init__(self) -> None:
        self._service: Any | None = None

    @property
    def service(self) -> Any:
        if self._service is None:
            from apps.frontend_streamlit.data.tickets import TicketService
            self._service = TicketService()
        return self._service

    def get_config(self) -> Any:
        from apps.frontend_streamlit.config.page_registry import get_feedback_page_catalog
        return {
            "pages": get_feedback_page_catalog(),
            "statuses": [
                {"id": item, "label": item.replace("_", " ").title(), "allowed_next_status_ids": [], "terminal": item == "closed"}
                for item in ["open", "in_progress", "dependency_conflict", "resolved", "closed"]
            ],
            "priorities": [{"id": item, "label": item.title(), "rank": idx + 1} for idx, item in enumerate(["low", "medium", "high", "critical"])],
            "limits": {"max_attachment_mb": 5, "max_attachments_per_ticket": 5},
            "attachments": {"allowed_extensions": [".png", ".jpg", ".jpeg", ".webp"], "allowed_content_types": ["image/png", "image/jpeg", "image/webp"]},
            "capabilities": {"can_create": True, "can_view_all": True, "can_moderate": True, "can_delete_tickets": True, "can_delete_attachments": True},
        }

    def get_summary(self, filters: dict[str, Any] | None = None) -> Any:
        tickets = self.service.list_tickets()
        statuses: dict[str, int] = {}
        priorities: dict[str, int] = {}
        for ticket in tickets:
            statuses[ticket.status.value] = statuses.get(ticket.status.value, 0) + 1
            priorities[ticket.criticality.value] = priorities.get(ticket.criticality.value, 0) + 1
        return {
            "scope": "all",
            "total": len(tickets),
            "counts_by_status": [{"status_id": key, "count": value} for key, value in statuses.items()],
            "counts_by_priority": [{"priority_id": key, "count": value} for key, value in priorities.items()],
            "resolved_or_closed_count": sum(statuses.get(item, 0) for item in ("resolved", "closed")),
            "dependency_conflict_count": statuses.get("dependency_conflict", 0),
            "rejected_count": 0,
            "high_or_critical_count": sum(priorities.get(item, 0) for item in ("high", "critical")),
            "facets": {"pages": [], "reporters": []},
        }

    def list_tickets(self, query: dict[str, Any] | None = None) -> Any:
        return {"items": self.service.list_tickets(), "total": len(self.service.list_tickets()), "limit": 200, "offset": 0}

    def create_ticket(self, request: dict[str, Any], *, idempotency_key: str) -> Any:
        from apps.frontend_streamlit.data.tickets import TicketCreateRequest
        created = self.service.create_ticket(
            TicketCreateRequest(
                page_name=str(request.get("page") or request.get("page_id") or "Feedback"),
                reported_by=str(request.get("reported_by") or ""),
                criticality=str(request.get("priority") or "medium"),
                description=str(request.get("description") or ""),
                ideal_closure_text=str(request.get("ideal_closure") or ""),
                created_by=str(request.get("reported_by") or ""),
            ),
            attachments=[],
        )
        return created

    def get_ticket(self, ticket_id: str) -> Any:
        return self.service.get_ticket(ticket_id)

    def update_ticket(self, ticket_id: str, request: dict[str, Any]) -> Any:
        return self.service.update_ticket_status(ticket_id, request.get("status") or request.get("target_status_id"), request.get("note") or "")

    def transition_ticket(self, ticket_id: str, request: dict[str, Any], *, idempotency_key: str) -> Any:
        return self.update_ticket(ticket_id, request)

    def delete_ticket(self, ticket_id: str, *, expected_version: int, idempotency_key: str) -> Any:
        self.service.delete_ticket(ticket_id)
        return {"deleted": True, "ticket_id": ticket_id}

    def list_events(self, ticket_id: str, query: dict[str, Any] | None = None) -> Any:
        return {"items": [], "total": 0, "limit": 50, "offset": 0}

    def list_comments(self, ticket_id: str, query: dict[str, Any] | None = None) -> Any:
        return {"items": [], "total": 0, "limit": 50, "offset": 0}

    def add_comment(self, ticket_id: str, body: str, *, idempotency_key: str) -> Any:
        return {"id": f"direct_{uuid4().hex}", "ticket_id": ticket_id, "body": body}

    def list_attachments(self, ticket_id: str) -> Any:
        return {"items": [], "total": 0}

    def upload_attachment(self, ticket_id: str, file: Any, *, idempotency_key: str) -> Any:
        return {"id": f"direct_{uuid4().hex}", "ticket_id": ticket_id}

    def preview_attachment(self, attachment_id: str) -> bytes:
        return b""

    def download_attachment(self, attachment_id: str) -> bytes:
        return b""

    def delete_attachment(self, attachment_id: str, *, idempotency_key: str) -> Any:
        return {"deleted": True, "attachment_id": attachment_id}


def get_feedback_gateway(*, token: str | None = None) -> FeedbackGateway:
    """Return the configured feedback gateway without silent fallback."""
    if is_backend_api_enabled("feedback"):
        return ApiFeedbackGateway(token=token)
    return DirectFeedbackGateway()
