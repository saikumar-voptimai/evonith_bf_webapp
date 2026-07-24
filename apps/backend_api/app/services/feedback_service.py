
"""Business logic for backend-owned feedback tickets."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import json
import logging
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.rbac import permissions_for_role
from apps.backend_api.app.repositories.feedback_repository import (
    FeedbackAttachmentRecord,
    FeedbackCommentRecord,
    FeedbackEventRecord,
    FeedbackRepository,
    FeedbackTicketRecord,
    utc_now,
)
from apps.backend_api.app.services.feedback_attachment_service import (
    FeedbackAttachmentService,
    ParsedUpload,
)
from furnace_data.app_catalog import APP_PAGE_BY_ID, APP_PAGES, canonical_page_id, page_label

log = logging.getLogger(__name__)

CATALOG_VERSION = "feedback-catalog-v1"
WORKFLOW_VERSION = "feedback-workflow-v1"
DISPLAY_TIMEZONE = "Asia/Kolkata"
MAX_LIST_PAGE_SIZE = 100

STATUS_WORKFLOW: dict[str, dict[str, Any]] = {
    "open": {"label": "Open", "terminal": False, "next": ["in_progress", "dependency_conflict", "resolved", "rejected", "closed"]},
    "in_progress": {"label": "In Progress", "terminal": False, "next": ["dependency_conflict", "resolved", "rejected", "closed", "open"]},
    "dependency_conflict": {"label": "Dependency Conflict", "terminal": False, "next": ["in_progress", "resolved", "rejected", "closed", "open"]},
    "resolved": {"label": "Resolved", "terminal": False, "next": ["closed", "open", "in_progress"]},
    "rejected": {"label": "Rejected", "terminal": True, "next": ["open", "closed"]},
    "closed": {"label": "Closed", "terminal": True, "next": ["open"]},
}
PRIORITY_RANKS = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SAFE_CLIENT_CONTEXT_KEYS = {"frontend", "frontend_version"}


def _user_id(user: dict[str, Any] | None) -> str | None:
    if not user:
        return None
    value = user.get("id")
    return str(value) if value is not None and str(value).strip() else None


def _username(user: dict[str, Any] | None) -> str | None:
    if not user:
        return None
    value = user.get("username")
    return str(value) if value is not None and str(value).strip() else None


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _key_hash(idempotency_key: str) -> str:
    return hashlib.sha256(idempotency_key.strip().encode("utf-8")).hexdigest()


def _as_list(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


class FeedbackService:
    """Feedback ticket business service."""

    def __init__(
        self,
        *,
        repository: FeedbackRepository | None = None,
        attachment_service: FeedbackAttachmentService | None = None,
        settings: BackendSettings | None = None,
        audit_service: Any | None = None,
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
        self.audit_service = audit_service

    def ensure_storage(self) -> None:
        """Create backend feedback storage if needed."""
        self.repository.ensure_schema()

    def _permissions(self, current_user: dict[str, Any] | None) -> set[str]:
        issued = {str(item) for item in (current_user or {}).get("permissions") or []}
        if issued:
            return issued
        return set(permissions_for_role((current_user or {}).get("role")))

    def _has_permission(self, current_user: dict[str, Any] | None, permission: str) -> bool:
        if not self.settings.feedback_require_auth:
            return True
        return permission in self._permissions(current_user)

    def _require_permission(self, current_user: dict[str, Any] | None, permission: str) -> None:
        if not self.settings.feedback_require_auth:
            return
        if not current_user:
            raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)
        if not self._has_permission(current_user, permission):
            raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)

    def _can_read_any(self, current_user: dict[str, Any] | None) -> bool:
        return self._has_permission(current_user, "feedback:read:any") or self._has_permission(current_user, "feedback:moderate")

    def _can_access(self, ticket: FeedbackTicketRecord, current_user: dict[str, Any] | None) -> bool:
        if not self.settings.feedback_require_auth:
            return True
        if self._can_read_any(current_user):
            return True
        return bool(ticket.created_by and ticket.created_by == _user_id(current_user))

    def _require_access(self, ticket: FeedbackTicketRecord, current_user: dict[str, Any] | None) -> None:
        if ticket.deleted_at is not None:
            raise ApiError("FEEDBACK_TICKET_DELETED", "Feedback ticket was deleted.", status_code=410)
        if not self._can_access(ticket, current_user):
            raise ApiError("FEEDBACK_TICKET_FORBIDDEN", "You are not allowed to access this ticket.", status_code=403)

    def _require_mutable_owner_or_moderator(self, ticket: FeedbackTicketRecord, current_user: dict[str, Any] | None) -> None:
        if self._has_permission(current_user, "feedback:moderate"):
            return
        if not self._can_access(ticket, current_user) or ticket.status not in {"open", "in_progress"}:
            raise ApiError("FEEDBACK_TICKET_FORBIDDEN", "You are not allowed to update this ticket.", status_code=403)

    def _require_idempotency(self, idempotency_key: str | None) -> str:
        clean = str(idempotency_key or "").strip()
        if not clean:
            raise ApiError("FEEDBACK_IDEMPOTENCY_REQUIRED", "Idempotency-Key is required.", status_code=400)
        if len(clean) > 255:
            raise ApiError("FEEDBACK_IDEMPOTENCY_REQUIRED", "Idempotency-Key is too long.", status_code=400)
        return clean

    def _idempotency_owner(self, current_user: dict[str, Any] | None) -> str:
        return _user_id(current_user) or "anonymous"

    def _find_idempotency(self, *, operation: str, idempotency_key: str, current_user: dict[str, Any] | None, fingerprint: str):
        record = self.repository.find_idempotency_record(
            owner_key=self._idempotency_owner(current_user),
            operation=operation,
            key_hash=_key_hash(idempotency_key),
        )
        if record is None:
            return None
        if record.request_fingerprint != fingerprint:
            raise ApiError(
                "FEEDBACK_IDEMPOTENCY_CONFLICT",
                "Idempotency-Key was already used for a different request.",
                status_code=409,
            )
        return record

    def _store_idempotency(self, *, operation: str, idempotency_key: str, current_user: dict[str, Any] | None, fingerprint: str, resource_type: str, resource_id: str) -> None:
        self.repository.store_idempotency_record(
            owner_key=self._idempotency_owner(current_user),
            operation=operation,
            key_hash=_key_hash(idempotency_key),
            request_fingerprint=fingerprint,
            resource_type=resource_type,
            resource_id=resource_id,
        )

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        forbidden = {"password", "token", "secret", "authorization", "cookie"}
        output: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if any(marker in str(key).lower() for marker in forbidden):
                continue
            if str(key) in {"reported_by", "created_by", "updated_by", "actor_role"}:
                continue
            output[str(key)] = value
        return output

    @staticmethod
    def _safe_client_context(context: dict[str, Any] | None) -> dict[str, Any]:
        return {str(key): value for key, value in (context or {}).items() if str(key) in SAFE_CLIENT_CONTEXT_KEYS}

    def _allowed_status_ids(self) -> list[str]:
        allowed = [str(item).strip().lower() for item in self.settings.feedback_allowed_statuses]
        merged = []
        for status in [*allowed, "dependency_conflict", "rejected"]:
            if status in STATUS_WORKFLOW and status not in merged:
                merged.append(status)
        return merged

    def _status_record(self, status_id: str) -> dict[str, Any]:
        status = STATUS_WORKFLOW.get(status_id, {"label": status_id.replace("_", " ").title(), "terminal": False, "next": []})
        allowed_ids = set(self._allowed_status_ids())
        return {
            "id": status_id,
            "label": str(status["label"]),
            "terminal": bool(status["terminal"]),
            "allowed_next_status_ids": [item for item in status["next"] if item in allowed_ids],
        }

    def _priority_record(self, priority_id: str) -> dict[str, Any]:
        return {
            "id": priority_id,
            "label": priority_id.replace("_", " ").title(),
            "rank": PRIORITY_RANKS.get(priority_id, 0),
        }

    def _validate_status(self, status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized not in self._allowed_status_ids():
            raise ApiError(
                "FEEDBACK_TICKET_INVALID_STATUS",
                "Ticket status is not allowed.",
                status_code=422,
                details={"status": status},
            )
        return normalized

    def _validate_priority(self, priority: str) -> str:
        normalized = str(priority or "").strip().lower()
        allowed = {str(item).lower() for item in self.settings.feedback_allowed_priorities}
        if normalized not in allowed:
            raise ApiError(
                "FEEDBACK_PRIORITY_INVALID",
                "Ticket priority is not allowed.",
                status_code=422,
                details={"priority": priority},
            )
        return normalized

    @staticmethod
    def _validate_text(value: str | None, *, code: str, message: str, max_chars: int) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ApiError(code, message, status_code=422)
        if len(clean) > max_chars:
            raise ApiError(code, message, status_code=422, details={"max_chars": max_chars})
        return clean

    def _validate_page_id(self, value: str | None) -> str:
        page_id = canonical_page_id(value)
        if page_id is None or page_id not in APP_PAGE_BY_ID:
            raise ApiError(
                "FEEDBACK_PAGE_INVALID",
                "Feedback page is not allowed.",
                status_code=422,
                details={"page_id": value},
            )
        return page_id

    def _normalize_filter_datetime(self, value: Any, *, field: str) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, time.min, tzinfo=timezone.utc)
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ApiError("VALIDATION_ERROR", f"{field} must be an ISO datetime.", status_code=422) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ApiError("VALIDATION_ERROR", f"{field} must include a timezone offset.", status_code=422)
        return parsed.astimezone(timezone.utc).isoformat()

    def _allowed_actions(self, ticket: FeedbackTicketRecord, current_user: dict[str, Any] | None) -> list[str]:
        if ticket.deleted_at is not None or not self._can_access(ticket, current_user):
            return []
        actions = ["comment"] if self._has_permission(current_user, "feedback:comment") else []
        if ticket.status in {"open", "in_progress"} and self._has_permission(current_user, "feedback:attachments:write"):
            actions.append("upload_attachment")
        if self._has_permission(current_user, "feedback:moderate"):
            actions.append("transition")
        if self._has_permission(current_user, "feedback:delete"):
            actions.append("delete_ticket")
        if self._has_permission(current_user, "feedback:attachments:delete"):
            actions.append("delete_attachment")
        return actions

    def _ticket_response(self, ticket: FeedbackTicketRecord, current_user: dict[str, Any] | None = None) -> dict[str, Any]:
        page = {"id": ticket.page_id, "label": ticket.page or page_label(ticket.page_id)} if ticket.page_id else None
        return {
            "id": ticket.id,
            "ticket_number": ticket.ticket_number,
            "version": ticket.version,
            "title": ticket.title,
            "description": ticket.description,
            "ideal_closure": ticket.ideal_closure,
            "page": page,
            "priority": self._priority_record(ticket.priority),
            "status": self._status_record(ticket.status),
            "reported_by": {"user_id": ticket.created_by, "username": ticket.created_by_username},
            "updated_by": {"user_id": ticket.updated_by, "username": ticket.updated_by_username} if ticket.updated_by or ticket.updated_by_username else None,
            "assigned_to": ticket.assigned_to,
            "resolution_notes": ticket.resolution_notes,
            "created_at": ticket.created_at,
            "updated_at": ticket.updated_at,
            "last_activity_at": ticket.last_activity_at,
            "resolved_at": ticket.resolved_at,
            "closed_at": ticket.closed_at,
            "deleted_at": ticket.deleted_at,
            "attachment_count": ticket.attachment_count,
            "comment_count": ticket.comment_count,
            "event_count": ticket.event_count,
            "tags": ticket.tags,
            "allowed_actions": self._allowed_actions(ticket, current_user),
            "metadata": ticket.metadata,
            "page_id": ticket.page_id,
            "priority_id": ticket.priority,
            "status_id": ticket.status,
            "category": ticket.category,
            "created_by": ticket.created_by,
            "created_by_username": ticket.created_by_username,
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
    def _event_response(event: FeedbackEventRecord) -> dict[str, Any]:
        return {
            "id": event.id,
            "ticket_id": event.ticket_id,
            "event_type": event.event_type,
            "sequence": event.sequence,
            "actor": {"user_id": event.actor_user_id, "username": event.actor_username} if event.actor_user_id or event.actor_username else None,
            "old_status_id": event.old_status,
            "new_status_id": event.new_status,
            "note": event.note,
            "payload": event.payload,
            "created_at": event.created_at,
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
            "checksum_sha256": attachment.checksum_sha256,
            "storage_status": attachment.storage_status,
            "created_by": attachment.created_by,
            "created_at": attachment.created_at,
        }

    def _add_event(self, *, ticket_id: str, event_type: str, current_user: dict[str, Any] | None, old_status: str | None = None, new_status: str | None = None, note: str | None = None, payload: dict[str, Any] | None = None) -> None:
        self.repository.add_event(
            {
                "ticket_id": ticket_id,
                "event_type": event_type,
                "actor_user_id": _user_id(current_user),
                "actor_username": _username(current_user),
                "old_status": old_status,
                "new_status": new_status,
                "note": note,
                "payload": payload or {},
            }
        )

    def _audit(self, event_type: str, current_user: dict[str, Any] | None, request_id: str | None, resource_id: str, metadata: dict[str, Any] | None = None) -> None:
        if self.audit_service is None:
            return
        self.audit_service.record_event(
            {
                "request_id": request_id,
                "actor_user_id": _user_id(current_user),
                "actor_username": _username(current_user),
                "event_type": event_type,
                "resource_type": "feedback",
                "resource_id": resource_id,
                "action": event_type.rsplit(".", 1)[-1],
                "result": "success",
                "status_code": 200,
                "metadata": metadata or {},
            }
        )

    def config(self, current_user: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return public feedback configuration."""
        self._require_permission(current_user, "feedback:read")
        return {
            "catalog_version": CATALOG_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "display_timezone": DISPLAY_TIMEZONE,
            "pages": [{"id": page.id, "label": page.label} for page in APP_PAGES],
            "statuses": [self._status_record(status_id) for status_id in self._allowed_status_ids()],
            "priorities": [self._priority_record(priority) for priority in self.settings.feedback_allowed_priorities],
            "limits": {
                "title_max_chars": 240,
                "description_max_chars": 5000,
                "ideal_closure_max_chars": 1000,
                "comment_max_chars": 4000,
                "max_attachment_mb": self.settings.feedback_max_attachment_mb,
                "max_attachments_per_ticket": self.settings.feedback_max_attachments_per_ticket,
                "max_list_page_size": MAX_LIST_PAGE_SIZE,
            },
            "attachments": {
                "allowed_content_types": list(self.settings.feedback_allowed_attachment_types),
                "allowed_extensions": list(self.settings.feedback_allowed_attachment_extensions),
                "image_preview_available": True,
            },
            "capabilities": {
                "can_create": self._has_permission(current_user, "feedback:create"),
                "can_view_all": self._can_read_any(current_user),
                "can_moderate": self._has_permission(current_user, "feedback:moderate"),
                "can_delete_tickets": self._has_permission(current_user, "feedback:delete"),
                "can_delete_attachments": self._has_permission(current_user, "feedback:attachments:delete"),
            },
            "etag": f"{CATALOG_VERSION}:{WORKFLOW_VERSION}",
        }

    def _scope_filters(self, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        query = dict(filters)
        if self.settings.feedback_require_auth and not self._can_read_any(current_user):
            if query.get("reporter_user_id") or query.get("created_by"):
                requested = str(query.get("reporter_user_id") or query.get("created_by"))
                if requested != str(_user_id(current_user)):
                    raise ApiError("FEEDBACK_TICKET_FORBIDDEN", "Reporter filter is not allowed.", status_code=403)
            query["created_by"] = _user_id(current_user)
        return query

    def _normalize_list_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        query = dict(filters)
        statuses = [_ for _ in (_as_list(query.get("status"))) if _]
        priorities = [_ for _ in (_as_list(query.get("priority"))) if _]
        page_ids = [_ for _ in (_as_list(query.get("page_id"))) if _]
        if statuses:
            query["status"] = [self._validate_status(item) for item in statuses]
        if priorities:
            query["priority"] = [self._validate_priority(item) for item in priorities]
        if page_ids:
            query["page_id"] = [self._validate_page_id(item) for item in page_ids]
        if query.get("created_from"):
            query["created_from"] = self._normalize_filter_datetime(query["created_from"], field="created_from")
        if query.get("created_to"):
            query["created_to"] = self._normalize_filter_datetime(query["created_to"], field="created_to")
        return query

    def summary(self, *, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        """Return aggregate counts over the authorized ticket scope."""
        self._require_permission(current_user, "feedback:read")
        query = self._scope_filters(self._normalize_list_filters(filters), current_user)
        include_reporters = self._can_read_any(current_user)
        aggregate = self.repository.summary(query, include_reporters=include_reporters)
        return {
            "scope": "all" if include_reporters else "own",
            "total": aggregate["total"],
            "counts_by_status": [
                {"status_id": status_id, "count": count}
                for status_id, count in sorted(aggregate["counts_by_status"].items())
            ],
            "counts_by_priority": [
                {"priority_id": priority_id, "count": count}
                for priority_id, count in sorted(aggregate["counts_by_priority"].items())
            ],
            "resolved_or_closed_count": aggregate["resolved_or_closed_count"],
            "dependency_conflict_count": aggregate["dependency_conflict_count"],
            "rejected_count": aggregate["rejected_count"],
            "high_or_critical_count": aggregate["high_or_critical_count"],
            "facets": aggregate["facets"],
            "as_of": utc_now(),
        }

    def list_tickets(self, *, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        """List tickets visible to the current user."""
        self._require_permission(current_user, "feedback:read")
        query = self._scope_filters(self._normalize_list_filters(filters), current_user)
        limit = min(MAX_LIST_PAGE_SIZE, max(1, int(query.get("limit") or 50)))
        offset = max(0, int(query.get("offset") or 0))
        query["limit"] = limit
        query["offset"] = offset
        tickets, total = self.repository.list_tickets(query)
        next_offset = offset + limit if offset + limit < total else None
        return {
            "items": [self._ticket_response(ticket, current_user) for ticket in tickets],
            "total": total,
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset,
        }

    def create_ticket(self, *, payload: dict[str, Any], current_user: dict[str, Any] | None, request_id: str, idempotency_key: str | None = None) -> dict[str, Any]:
        """Create a ticket."""
        self._require_permission(current_user, "feedback:create")
        idempotency_key = self._require_idempotency(idempotency_key)
        page_id = self._validate_page_id(payload.get("page_id") or payload.get("page") or payload.get("category"))
        description = self._validate_text(payload.get("description"), code="VALIDATION_ERROR", message="Description is required.", max_chars=5000)
        ideal = payload.get("ideal_closure") or (payload.get("metadata") or {}).get("ideal_closure_text")
        ideal_closure = self._validate_text(ideal, code="VALIDATION_ERROR", message="Ideal closure is required.", max_chars=1000)
        priority = self._validate_priority(payload.get("priority") or "medium")
        title = str(payload.get("title") or f"{page_label(page_id)} feedback").strip()[:240]
        tags = [str(tag).strip()[:40] for tag in payload.get("tags") or [] if str(tag).strip()][:20]
        client_context = self._safe_client_context(payload.get("client_context"))
        fingerprint_payload = {
            "page_id": page_id,
            "title": title,
            "description": description,
            "ideal_closure": ideal_closure,
            "priority": priority,
            "tags": tags,
            "client_context": client_context,
        }
        fingerprint = _json_hash(fingerprint_payload)
        prior = self._find_idempotency(operation="create-ticket", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint)
        if prior is not None:
            existing = self.repository.get_ticket(prior.resource_id)
            if existing is not None:
                return self._ticket_response(existing, current_user)
        status = self._validate_status(self.settings.feedback_default_status)
        ticket = self.repository.create_ticket(
            {
                "title": title,
                "description": description,
                "ideal_closure": ideal_closure,
                "category": page_label(page_id),
                "priority": priority,
                "status": status,
                "page_id": page_id,
                "page": page_label(page_id),
                "tags": tags,
                "metadata": {"client_context": client_context} if client_context else {},
                "created_by": _user_id(current_user),
                "created_by_username": _username(current_user),
            }
        )
        self._add_event(ticket_id=ticket.id, event_type="ticket_created", current_user=current_user, new_status=status, payload={"ticket_number": ticket.ticket_number})
        self._store_idempotency(operation="create-ticket", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint, resource_type="ticket", resource_id=ticket.id)
        self._audit("feedback.ticket.created", current_user, request_id, ticket.id, {"ticket_number": ticket.ticket_number})
        log.info("feedback_ticket_created request_id=%s ticket_id=%s ticket_number=%s", request_id, ticket.id, ticket.ticket_number)
        refreshed = self.repository.get_ticket(ticket.id) or ticket
        return self._ticket_response(refreshed, current_user)

    def get_ticket(self, ticket_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        """Return a ticket if visible."""
        self._require_permission(current_user, "feedback:read")
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            deleted = self.repository.get_ticket(ticket_id, include_deleted=True)
            if deleted is not None:
                raise ApiError("FEEDBACK_TICKET_DELETED", "Feedback ticket was deleted.", status_code=410)
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        return self._ticket_response(ticket, current_user)

    def _check_version(self, ticket: FeedbackTicketRecord, expected_version: int | None) -> int | None:
        if expected_version is None:
            return None
        if int(expected_version) != int(ticket.version):
            raise ApiError(
                "FEEDBACK_TICKET_VERSION_CONFLICT",
                "Another user changed this ticket. Reload it before saving.",
                status_code=409,
                details={"current_version": ticket.version, "expected_version": expected_version},
            )
        return int(expected_version)

    def update_ticket(self, *, ticket_id: str, payload: dict[str, Any], current_user: dict[str, Any] | None, request_id: str | None = None) -> dict[str, Any]:
        """Update non-lifecycle ticket fields."""
        self._require_permission(current_user, "feedback:read")
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        self._require_mutable_owner_or_moderator(ticket, current_user)
        expected_version = self._check_version(ticket, payload.get("expected_version"))
        if payload.get("status") is not None:
            return self.transition_ticket(
                ticket_id=ticket.id,
                payload={
                    "target_status_id": payload["status"],
                    "expected_version": expected_version or ticket.version,
                    "note": payload.get("resolution_notes"),
                    "resolution_notes": payload.get("resolution_notes"),
                },
                current_user=current_user,
                request_id=request_id,
                idempotency_key=f"patch-transition-{ticket.id}-{expected_version or ticket.version}-{payload['status']}",
                internal_idempotency=True,
            )
        changes: dict[str, Any] = {
            "updated_by": _user_id(current_user),
            "updated_by_username": _username(current_user),
        }
        manager_only_fields = {"assigned_to", "resolution_notes"}
        for key in ("title", "description", "ideal_closure", "assigned_to", "resolution_notes"):
            if key in payload and payload[key] is not None:
                if key in manager_only_fields and self.settings.feedback_require_auth and not self._has_permission(current_user, "feedback:moderate"):
                    raise ApiError("FEEDBACK_TICKET_FORBIDDEN", "Only feedback managers can update this field.", status_code=403)
                changes[key] = str(payload[key]).strip()
        if payload.get("page_id") is not None:
            page_id = self._validate_page_id(payload["page_id"])
            changes["page_id"] = page_id
            changes["page"] = page_label(page_id)
            changes["category"] = page_label(page_id)
        if payload.get("priority") is not None:
            changes["priority"] = self._validate_priority(payload["priority"])
        if payload.get("tags") is not None:
            changes["tags_json"] = json.dumps([str(tag).strip() for tag in payload.get("tags") or [] if str(tag).strip()], sort_keys=True)
        if payload.get("metadata") is not None:
            changes["metadata_json"] = json.dumps(self._safe_metadata(payload.get("metadata")), sort_keys=True)
        updated = self.repository.update_ticket(ticket.id, changes, expected_version=expected_version)
        if updated is None:
            raise ApiError("FEEDBACK_TICKET_VERSION_CONFLICT", "Another user changed this ticket. Reload it before saving.", status_code=409)
        self._add_event(ticket_id=updated.id, event_type="ticket_updated", current_user=current_user, payload={"fields": sorted(changes)})
        self._audit("feedback.ticket.updated", current_user, request_id, updated.id, {"fields": sorted(changes)})
        refreshed = self.repository.get_ticket(updated.id) or updated
        return self._ticket_response(refreshed, current_user)

    def transition_ticket(
        self,
        *,
        ticket_id: str,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        internal_idempotency: bool = False,
    ) -> dict[str, Any]:
        """Transition ticket status through the workflow endpoint."""
        self._require_permission(current_user, "feedback:moderate")
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        target = self._validate_status(payload.get("target_status_id") or payload.get("status"))
        expected_version = self._check_version(ticket, payload.get("expected_version")) or ticket.version
        if target != ticket.status and target not in self._status_record(ticket.status)["allowed_next_status_ids"]:
            raise ApiError(
                "FEEDBACK_TICKET_INVALID_TRANSITION",
                "Ticket status transition is not allowed.",
                status_code=422,
                details={"from": ticket.status, "to": target},
            )
        idempotency_key = self._require_idempotency(idempotency_key)
        fingerprint = _json_hash({"ticket_id": ticket.id, "target": target, "version": expected_version, "note": payload.get("note"), "resolution_notes": payload.get("resolution_notes")})
        if not internal_idempotency:
            prior = self._find_idempotency(operation="transition-ticket", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint)
            if prior is not None:
                existing = self.repository.get_ticket(prior.resource_id)
                if existing is not None:
                    return self._ticket_response(existing, current_user)
        now = utc_now().isoformat()
        changes = {
            "status": target,
            "updated_by": _user_id(current_user),
            "updated_by_username": _username(current_user),
            "resolution_notes": payload.get("resolution_notes") or ticket.resolution_notes,
            "resolved_at": now if target in {"resolved", "rejected"} else None if target in {"open", "in_progress"} else ticket.resolved_at.isoformat() if ticket.resolved_at else None,
            "closed_at": now if target == "closed" else None if target in {"open", "in_progress"} else ticket.closed_at.isoformat() if ticket.closed_at else None,
        }
        updated = self.repository.update_ticket(ticket.id, changes, expected_version=expected_version)
        if updated is None:
            raise ApiError("FEEDBACK_TICKET_VERSION_CONFLICT", "Another user changed this ticket. Reload it before saving.", status_code=409)
        self._add_event(ticket_id=updated.id, event_type="ticket_transitioned", current_user=current_user, old_status=ticket.status, new_status=target, note=payload.get("note"), payload={"resolution_notes_present": bool(payload.get("resolution_notes"))})
        if not internal_idempotency:
            self._store_idempotency(operation="transition-ticket", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint, resource_type="ticket", resource_id=updated.id)
        self._audit("feedback.ticket.transitioned", current_user, request_id, updated.id, {"from": ticket.status, "to": target})
        refreshed = self.repository.get_ticket(updated.id) or updated
        return self._ticket_response(refreshed, current_user)

    def close_ticket(self, *, ticket_id: str, current_user: dict[str, Any] | None, request_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        return self.transition_ticket(ticket_id=ticket.id, payload={"target_status_id": "closed", "expected_version": ticket.version}, current_user=current_user, request_id=request_id, idempotency_key=idempotency_key or f"close-{ticket.id}-{ticket.version}")

    def reopen_ticket(self, *, ticket_id: str, current_user: dict[str, Any] | None, request_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        return self.transition_ticket(ticket_id=ticket.id, payload={"target_status_id": self.settings.feedback_default_status, "expected_version": ticket.version}, current_user=current_user, request_id=request_id, idempotency_key=idempotency_key or f"reopen-{ticket.id}-{ticket.version}")

    def add_comment(self, *, ticket_id: str, body: str, current_user: dict[str, Any] | None, request_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        """Add a ticket comment."""
        self._require_permission(current_user, "feedback:comment")
        idempotency_key = self._require_idempotency(idempotency_key)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        clean = str(body or "").strip()
        if not clean:
            raise ApiError("FEEDBACK_COMMENT_EMPTY", "Comment body is required.", status_code=422)
        if len(clean) > 4000:
            raise ApiError("VALIDATION_ERROR", "Comment is too long.", status_code=422)
        fingerprint = _json_hash({"ticket_id": ticket.id, "body": clean})
        prior = self._find_idempotency(operation="create-comment", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint)
        if prior is not None:
            comments, _ = self.repository.list_comments(ticket.id, limit=500, offset=0)
            for comment in comments:
                if comment.id == prior.resource_id:
                    return self._comment_response(comment)
        comment = self.repository.add_comment({"ticket_id": ticket.id, "body": clean, "created_by": _user_id(current_user), "created_by_username": _username(current_user)})
        self._add_event(ticket_id=ticket.id, event_type="comment_created", current_user=current_user, payload={"comment_id": comment.id})
        self._store_idempotency(operation="create-comment", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint, resource_type="comment", resource_id=comment.id)
        self._audit("feedback.comment.created", current_user, request_id, ticket.id, {"comment_id": comment.id})
        return self._comment_response(comment)

    def list_comments(self, *, ticket_id: str, current_user: dict[str, Any] | None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List ticket comments."""
        self._require_permission(current_user, "feedback:read")
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        bounded_limit = min(MAX_LIST_PAGE_SIZE, max(1, int(limit)))
        bounded_offset = max(0, int(offset))
        comments, total = self.repository.list_comments(ticket.id, limit=bounded_limit, offset=bounded_offset)
        next_offset = bounded_offset + bounded_limit if bounded_offset + bounded_limit < total else None
        return {"items": [self._comment_response(comment) for comment in comments], "total": total, "limit": bounded_limit, "offset": bounded_offset, "next_offset": next_offset}

    def list_events(self, *, ticket_id: str, current_user: dict[str, Any] | None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List durable lifecycle events for a ticket."""
        self._require_permission(current_user, "feedback:read")
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        bounded_limit = min(MAX_LIST_PAGE_SIZE, max(1, int(limit)))
        bounded_offset = max(0, int(offset))
        events, total = self.repository.list_events(ticket.id, limit=bounded_limit, offset=bounded_offset)
        next_offset = bounded_offset + bounded_limit if bounded_offset + bounded_limit < total else None
        return {"items": [self._event_response(event) for event in events], "total": total, "limit": bounded_limit, "offset": bounded_offset, "next_offset": next_offset}

    def list_attachments(self, *, ticket_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        """List ticket attachment metadata."""
        self._require_permission(current_user, "feedback:read")
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        attachments = self.repository.list_attachments(ticket.id)
        return {"items": [self._attachment_response(attachment) for attachment in attachments], "total": len(attachments)}

    def add_attachment(self, *, ticket_id: str, upload: ParsedUpload, current_user: dict[str, Any] | None, request_id: str, idempotency_key: str | None = None) -> dict[str, Any]:
        """Add one attachment to a ticket."""
        self._require_permission(current_user, "feedback:attachments:write")
        idempotency_key = self._require_idempotency(idempotency_key)
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_mutable_owner_or_moderator(ticket, current_user)
        fingerprint = _json_hash({"ticket_id": ticket.id, "filename": upload.filename, "content_type": upload.content_type, "checksum": hashlib.sha256(upload.content).hexdigest()})
        prior = self._find_idempotency(operation="upload-attachment", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint)
        if prior is not None:
            existing = self.repository.get_attachment(prior.resource_id)
            if existing is not None:
                return self._attachment_response(existing)
        attachment = self.attachment_service.store_attachment(ticket_id=ticket.id, upload=upload, current_user=current_user, request_id=request_id)
        self._add_event(ticket_id=ticket.id, event_type="attachment_uploaded", current_user=current_user, payload={"attachment_id": attachment.id, "filename": attachment.original_filename})
        self._store_idempotency(operation="upload-attachment", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint, resource_type="attachment", resource_id=attachment.id)
        self._audit("feedback.attachment.uploaded", current_user, request_id, ticket.id, {"attachment_id": attachment.id})
        return self._attachment_response(attachment)

    def get_attachment_for_download(self, attachment_id: str, current_user: dict[str, Any] | None) -> tuple[FeedbackAttachmentRecord, Any]:
        """Return attachment metadata and safe file path."""
        self._require_permission(current_user, "feedback:read")
        attachment = self.repository.get_attachment(attachment_id)
        if attachment is None:
            raise ApiError("FEEDBACK_ATTACHMENT_NOT_FOUND", "Feedback attachment not found.", status_code=404)
        ticket = self.repository.get_ticket(attachment.ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        return attachment, self.attachment_service.resolve_download_path(attachment)

    def get_attachment_preview(self, attachment_id: str, current_user: dict[str, Any] | None) -> tuple[bytes, str, FeedbackAttachmentRecord]:
        """Return bounded image preview bytes and metadata."""
        self._require_permission(current_user, "feedback:read")
        attachment = self.repository.get_attachment(attachment_id)
        if attachment is None:
            raise ApiError("FEEDBACK_ATTACHMENT_NOT_FOUND", "Feedback attachment not found.", status_code=404)
        ticket = self.repository.get_ticket(attachment.ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        data, content_type = self.attachment_service.build_image_preview(attachment)
        return data, content_type, attachment

    def delete_attachment(self, *, attachment_id: str, current_user: dict[str, Any] | None, request_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        """Delete an attachment. Only feedback managers can delete attachments."""
        self._require_permission(current_user, "feedback:attachments:delete")
        idempotency_key = self._require_idempotency(idempotency_key)
        fingerprint = _json_hash({"attachment_id": attachment_id})
        prior = self._find_idempotency(operation="delete-attachment", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint)
        if prior is not None:
            return {"deleted": True, "attachment_id": prior.resource_id}
        attachment = self.repository.get_attachment(attachment_id)
        if attachment is None:
            raise ApiError("FEEDBACK_ATTACHMENT_NOT_FOUND", "Feedback attachment not found.", status_code=404)
        ticket = self.repository.get_ticket(attachment.ticket_id)
        if ticket is None:
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        deleted_record = self.repository.delete_attachment_metadata(attachment.id)
        if deleted_record is None:
            raise ApiError("FEEDBACK_ATTACHMENT_NOT_FOUND", "Feedback attachment not found.", status_code=404)
        try:
            self.attachment_service.delete_attachment_file(deleted_record)
        except ApiError:
            log.warning("feedback attachment file cleanup failed attachment_id=%s", deleted_record.id, exc_info=True)
        self._add_event(ticket_id=ticket.id, event_type="attachment_deleted", current_user=current_user, payload={"attachment_id": deleted_record.id})
        self._store_idempotency(operation="delete-attachment", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint, resource_type="attachment", resource_id=deleted_record.id)
        self._audit("feedback.attachment.deleted", current_user, request_id, ticket.id, {"attachment_id": deleted_record.id})
        return {"deleted": True, "attachment_id": deleted_record.id}

    def delete_ticket(self, *, ticket_id: str, expected_version: int, current_user: dict[str, Any] | None, request_id: str | None = None, idempotency_key: str | None = None, reason: str | None = None) -> dict[str, Any]:
        """Soft-delete one ticket."""
        self._require_permission(current_user, "feedback:delete")
        idempotency_key = self._require_idempotency(idempotency_key)
        fingerprint = _json_hash({"ticket_id": ticket_id, "expected_version": expected_version, "reason": reason})
        prior = self._find_idempotency(operation="delete-ticket", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint)
        if prior is not None:
            deleted = self.repository.get_ticket(prior.resource_id, include_deleted=True)
            return {"deleted": True, "ticket_id": prior.resource_id, "version": deleted.version if deleted else None}
        ticket = self.repository.get_ticket(ticket_id)
        if ticket is None:
            deleted = self.repository.get_ticket(ticket_id, include_deleted=True)
            if deleted is not None:
                raise ApiError("FEEDBACK_TICKET_DELETED", "Feedback ticket was deleted.", status_code=410)
            raise ApiError("FEEDBACK_TICKET_NOT_FOUND", "Feedback ticket not found.", status_code=404)
        self._require_access(ticket, current_user)
        self._check_version(ticket, expected_version)
        deleted = self.repository.soft_delete_ticket(ticket.id, expected_version=expected_version, actor_user_id=_user_id(current_user), actor_username=_username(current_user), reason=reason)
        if deleted is None:
            raise ApiError("FEEDBACK_TICKET_VERSION_CONFLICT", "Another user changed this ticket. Reload it before deleting.", status_code=409)
        self._add_event(ticket_id=deleted.id, event_type="ticket_deleted", current_user=current_user, payload={"reason_present": bool(reason)})
        self._store_idempotency(operation="delete-ticket", idempotency_key=idempotency_key, current_user=current_user, fingerprint=fingerprint, resource_type="ticket", resource_id=deleted.id)
        self._audit("feedback.ticket.deleted", current_user, request_id, deleted.id, {})
        return {"deleted": True, "ticket_id": deleted.id, "version": deleted.version}
