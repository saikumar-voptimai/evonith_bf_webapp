"""Business logic for backend-owned feedback tickets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.feedback_repository import (
    FeedbackAttachmentRecord,
    FeedbackCommentRecord,
    FeedbackRepository,
    FeedbackTicketRecord,
)
from apps.backend_api.app.services.feedback_attachment_service import (
    FeedbackAttachmentService,
    ParsedUpload,
)

log = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    "Data Explorer",
    "V-Board",
    "V-Sense",
    "AI CoPilot",
    "Material Balance",
    "FurnaceMind",
    "Feedback",
    "Blend Optimizer",
]


def _is_manager(user: dict[str, Any] | None) -> bool:
    role = str((user or {}).get("role") or "").strip().lower()
    return role in {"admin", "supervisor"}


def _user_id(user: dict[str, Any] | None) -> str | None:
    if not user:
        return None
    value = user.get("id")
    return str(value) if value else None


def _username(user: dict[str, Any] | None) -> str | None:
    if not user:
        return None
    value = user.get("username")
    return str(value) if value else None


class FeedbackService:
    """Feedback ticket business service."""

    def __init__(
        self,
        *,
        repository: FeedbackRepository | None = None,
        attachment_service: FeedbackAttachmentService | None = None,
        settings: BackendSettings | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = self.settings.feedback_database_url.strip() or None
        self.repository = repository or FeedbackRepository(
            database_url=database_url,
            ticket_prefix=self.settings.feedback_ticket_id_prefix,
        )
        self.attachment_service = attachment_service or FeedbackAttachmentService(
            repository=self.repository,
            settings=self.settings,
        )

    def ensure_storage(self) -> None:
        """Create backend feedback storage if needed."""
        self.repository.ensure_schema()

    def _require_user(self, current_user: dict[str, Any] | None) -> None:
        if self.settings.feedback_require_auth and not current_user:
            raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)

    def _can_access(self, ticket: FeedbackTicketRecord, user: dict[str, Any] | None) -> bool:
        if not self.settings.feedback_require_auth:
            return True
        if _is_manager(user):
            return True
        return bool(user and ticket.created_by and ticket.created_by == _user_id(user))

    def _require_access(self, ticket: FeedbackTicketRecord, user: dict[str, Any] | None) -> None:
        if not self._can_access(ticket, user):
            raise ApiError(
                "FEEDBACK_TICKET_FORBIDDEN",
                "You are not allowed to access this ticket.",
                status_code=403,
            )

    def _require_manager_or_owner_open(
        self, ticket: FeedbackTicketRecord, user: dict[str, Any] | None
    ) -> None:
        if _is_manager(user):
            return
        if not self._can_access(ticket, user) or ticket.status not in {"open", "in_progress"}:
            raise ApiError(
                "FEEDBACK_TICKET_FORBIDDEN",
                "You are not allowed to update this ticket.",
                status_code=403,
            )

    def _validate_status(self, status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized not in {item.lower() for item in self.settings.feedback_allowed_statuses}:
            raise ApiError(
                "FEEDBACK_TICKET_INVALID_STATUS",
                "Ticket status is not allowed.",
                status_code=422,
                details={"status": status},
            )
        return normalized

    def _validate_priority(self, priority: str) -> str:
        normalized = str(priority or "").strip().lower()
        if normalized not in {
            item.lower() for item in self.settings.feedback_allowed_priorities
        }:
            raise ApiError(
                "FEEDBACK_TICKET_INVALID_PRIORITY",
                "Ticket priority is not allowed.",
                status_code=422,
                details={"priority": priority},
            )
        return normalized

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        forbidden = {"password", "token", "secret", "authorization", "cookie"}
        output: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if any(marker in str(key).lower() for marker in forbidden):
                continue
            output[str(key)] = value
        return output

    @staticmethod
    def _closed_at_for_status(status: str) -> str | None:
        if status in {"closed", "rejected", "resolved"}:
            return datetime.now(timezone.utc).isoformat()
        return None

    def _ticket_response(self, ticket: FeedbackTicketRecord) -> dict[str, Any]:
        return {
            "id": ticket.id,
            "ticket_number": ticket.ticket_number,
            "title": ticket.title,
            "description": ticket.description,
            "category": ticket.category,
            "priority": ticket.priority,
            "status": ticket.status,
            "page": ticket.page,
            "tags": ticket.tags,
            "created_by": ticket.created_by,
            "created_by_username": ticket.created_by_username,
            "assigned_to": ticket.assigned_to,
            "created_at": ticket.created_at,
            "updated_at": ticket.updated_at,
            "closed_at": ticket.closed_at,
            "attachment_count": ticket.attachment_count,
            "comment_count": ticket.comment_count,
            "metadata": ticket.metadata,
        }

    @staticmethod
    def _comment_response(comment: FeedbackCommentRecord) -> dict[str, Any]:
        return {
            "id": comment.id,
            "ticket_id": comment.ticket_id,
            "body": comment.body,
            "created_by": comment.created_by,
            "created_by_username": comment.created_by_username,
            "created_at": comment.created_at,
        }

    @staticmethod
    def _attachment_response(attachment: FeedbackAttachmentRecord) -> dict[str, Any]:
        return {
            "id": attachment.id,
            "ticket_id": attachment.ticket_id,
            "filename": attachment.filename,
            "original_filename": attachment.original_filename,
            "content_type": attachment.content_type,
            "size_bytes": attachment.size_bytes,
            "created_by": attachment.created_by,
            "created_at": attachment.created_at,
            "download_url": f"/feedback/attachments/{attachment.id}/download",
        }

    def config(self) -> dict[str, Any]:
        """Return public feedback configuration."""
        return {
            "statuses": list(self.settings.feedback_allowed_statuses),
            "priorities": list(self.settings.feedback_allowed_priorities),
            "categories": DEFAULT_CATEGORIES,
            "max_attachment_mb": self.settings.feedback_max_attachment_mb,
            "allowed_attachment_types": list(
                self.settings.feedback_allowed_attachment_types
            ),
            "allowed_attachment_extensions": list(
                self.settings.feedback_allowed_attachment_extensions
            ),
            "max_attachments_per_ticket": self.settings.feedback_max_attachments_per_ticket,
        }

    def create_ticket(
        self,
        *,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
        request_id: str,
    ) -> dict[str, Any]:
        """Create a ticket."""
        self._require_user(current_user)
        priority = self._validate_priority(payload.get("priority") or "medium")
        status = self._validate_status(self.settings.feedback_default_status)
        ticket = self.repository.create_ticket(
            {
                "title": payload["title"],
                "description": payload["description"],
                "category": payload.get("category"),
                "priority": priority,
                "status": status,
                "page": payload.get("page"),
                "tags": payload.get("tags") or [],
                "metadata": self._safe_metadata(payload.get("metadata")),
                "created_by": _user_id(current_user),
                "created_by_username": _username(current_user)
                or payload.get("created_by_username"),
            }
        )
        log.info(
            "feedback_ticket_created request_id=%s ticket_id=%s ticket_number=%s user=%s",
            request_id,
            ticket.id,
            ticket.ticket_number,
            ticket.created_by_username,
        )
        return self._ticket_response(ticket)

    def list_tickets(
        self,
        *,
        filters: dict[str, Any],
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """List tickets visible to the current user."""
        self._require_user(current_user)
        query = dict(filters)
        if self.settings.feedback_require_auth and not _is_manager(current_user):
            query["created_by"] = _user_id(current_user)
        if query.get("status"):
            query["status"] = self._validate_status(query["status"])
        if query.get("priority"):
            query["priority"] = self._validate_priority(query["priority"])
        limit = min(200, max(1, int(query.get("limit") or 50)))
        offset = max(0, int(query.get("offset") or 0))
        query["limit"] = limit
        query["offset"] = offset
        tickets, total = self.repository.list_tickets(query)
        return {
            "items": [self._ticket_response(ticket) for ticket in tickets],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_ticket(self, ticket_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        """Return a ticket if visible."""
        self._require_user(current_user)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError(
                "FEEDBACK_TICKET_NOT_FOUND",
                "Feedback ticket not found.",
                status_code=404,
            )
        self._require_access(ticket, current_user)
        return self._ticket_response(ticket)

    def update_ticket(
        self,
        *,
        ticket_id: str,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Update a ticket."""
        self._require_user(current_user)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", 404)
        self._require_manager_or_owner_open(ticket, current_user)

        changes: dict[str, Any] = {}
        manager_only_fields = {"assigned_to", "resolution_notes"}
        for key in ("title", "description", "category", "assigned_to", "resolution_notes"):
            if key in payload and payload[key] is not None:
                if (
                    key in manager_only_fields
                    and self.settings.feedback_require_auth
                    and not _is_manager(current_user)
                ):
                    raise ApiError(
                        "FEEDBACK_TICKET_FORBIDDEN",
                        "Only feedback managers can update this field.",
                        status_code=403,
                    )
                changes[key] = payload[key]
        if payload.get("priority") is not None:
            changes["priority"] = self._validate_priority(payload["priority"])
        if payload.get("status") is not None:
            if self.settings.feedback_require_auth and not _is_manager(current_user):
                raise ApiError(
                    "FEEDBACK_TICKET_FORBIDDEN",
                    "Only feedback managers can update ticket status.",
                    status_code=403,
                )
            status = self._validate_status(payload["status"])
            changes["status"] = status
            changes["closed_at"] = self._closed_at_for_status(status)
        if payload.get("tags") is not None:
            changes["tags_json"] = json.dumps(payload.get("tags") or [], sort_keys=True)
        if payload.get("metadata") is not None:
            changes["metadata_json"] = json.dumps(
                self._safe_metadata(payload.get("metadata")),
                sort_keys=True,
            )

        updated = self.repository.update_ticket(ticket.id, changes)
        if updated is None:
            raise ApiError("FEEDBACK_TICKET_UPDATE_FAILED", "Ticket update failed.", 500)
        return self._ticket_response(updated)

    def close_ticket(
        self,
        *,
        ticket_id: str,
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Close a ticket."""
        return self.update_ticket(
            ticket_id=ticket_id,
            payload={"status": "closed"},
            current_user=current_user,
        )

    def reopen_ticket(
        self,
        *,
        ticket_id: str,
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Reopen a ticket."""
        return self.update_ticket(
            ticket_id=ticket_id,
            payload={"status": self.settings.feedback_default_status},
            current_user=current_user,
        )

    def add_comment(
        self,
        *,
        ticket_id: str,
        body: str,
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Add a ticket comment."""
        self._require_user(current_user)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", 404)
        self._require_access(ticket, current_user)
        comment = self.repository.add_comment(
            {
                "ticket_id": ticket.id,
                "body": body,
                "created_by": _user_id(current_user),
                "created_by_username": _username(current_user),
            }
        )
        return self._comment_response(comment)

    def list_comments(
        self,
        *,
        ticket_id: str,
        current_user: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """List ticket comments."""
        self._require_user(current_user)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", 404)
        self._require_access(ticket, current_user)
        return [
            self._comment_response(comment)
            for comment in self.repository.list_comments(ticket.id)
        ]

    def list_attachments(
        self,
        *,
        ticket_id: str,
        current_user: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """List ticket attachments."""
        self._require_user(current_user)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", 404)
        self._require_access(ticket, current_user)
        return [
            self._attachment_response(attachment)
            for attachment in self.repository.list_attachments(ticket.id)
        ]

    def add_attachment(
        self,
        *,
        ticket_id: str,
        upload: ParsedUpload,
        current_user: dict[str, Any] | None,
        request_id: str,
    ) -> dict[str, Any]:
        """Add one attachment to a ticket."""
        self._require_user(current_user)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", 404)
        self._require_manager_or_owner_open(ticket, current_user)
        attachment = self.attachment_service.store_attachment(
            ticket_id=ticket.id,
            upload=upload,
            current_user=current_user,
            request_id=request_id,
        )
        return self._attachment_response(attachment)

    def get_attachment_for_download(
        self,
        attachment_id: str,
        current_user: dict[str, Any] | None,
    ) -> tuple[FeedbackAttachmentRecord, Any]:
        """Return attachment metadata and safe file path."""
        self._require_user(current_user)
        attachment = self.repository.get_attachment(attachment_id)
        if attachment is None:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_NOT_FOUND",
                "Feedback attachment not found.",
                status_code=404,
            )
        ticket = self.repository.get_ticket(attachment.ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", 404)
        self._require_access(ticket, current_user)
        return attachment, self.attachment_service.resolve_download_path(attachment)

    def delete_attachment(
        self,
        *,
        attachment_id: str,
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Delete an attachment. Only managers can delete attachments."""
        self._require_user(current_user)
        attachment = self.repository.get_attachment(attachment_id)
        if attachment is None:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_NOT_FOUND",
                "Feedback attachment not found.",
                status_code=404,
            )
        ticket = self.repository.get_ticket(attachment.ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", 404)
        if not _is_manager(current_user):
            raise ApiError(
                "FEEDBACK_TICKET_FORBIDDEN",
                "Only feedback managers can delete attachments.",
                status_code=403,
            )
        self.attachment_service.delete_attachment_file(attachment)
        self.repository.delete_attachment_metadata(attachment.id)
        return {"deleted": True}
