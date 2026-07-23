"""Best-effort audit logging service."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from starlette.requests import Request

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.repositories.audit_repository import AuditEventRecord, AuditRepository
from apps.backend_api.app.services.redaction_service import redact_dict

log = logging.getLogger(__name__)


class AuditService:
    """Create redacted audit events without breaking business flows."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        repository: AuditRepository | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = self.settings.audit_database_url.strip() or None
        self.repository = repository or AuditRepository(database_url=database_url)

    def ensure_storage(self) -> None:
        if self.settings.audit_log_enabled:
            self.repository.ensure_schema()

    def record_event(self, payload: dict[str, Any]) -> AuditEventRecord | None:
        if not self.settings.audit_log_enabled:
            return None
        try:
            clean = dict(payload)
            clean["metadata"] = redact_dict(clean.get("metadata") or {})
            return self.repository.insert_event(clean)
        except Exception as exc:  # noqa: BLE001
            log.warning("Audit event could not be stored: %s", exc)
            return None

    def record_request(
        self,
        request: Request,
        *,
        status_code: int,
        error_code: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        event_type = self._event_type_for_request(request, status_code)
        if event_type is None:
            return
        user = getattr(request.state, "current_user", None) or {}
        self.record_event(
            {
                "request_id": getattr(request.state, "request_id", None),
                "actor_user_id": user.get("id"),
                "actor_username": user.get("username"),
                "event_type": event_type,
                "resource_type": self._resource_type(request.url.path),
                "resource_id": None,
                "action": request.method.lower(),
                "result": "success" if status_code < 400 else "failure",
                "status_code": status_code,
                "error_code": error_code,
                "ip_hash": self._client_hash(request),
                "metadata": {
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms or 0.0, 3),
                },
            }
        )

    def list_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
    ) -> dict[str, Any]:
        records, total = self.repository.list_events(
            limit=limit,
            offset=offset,
            event_type=event_type,
            actor_user_id=actor_user_id,
        )
        return {
            "items": [self.response(record) for record in records],
            "total": total,
            "limit": min(500, max(1, int(limit))),
            "offset": max(0, int(offset)),
        }

    def cleanup_retention(self) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.audit_retention_days)
        deleted = self.repository.cleanup_before(cutoff.isoformat())
        return {"deleted": deleted, "cutoff": cutoff.isoformat()}

    @staticmethod
    def response(record: AuditEventRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "timestamp": record.timestamp,
            "request_id": record.request_id,
            "actor_user_id": record.actor_user_id,
            "actor_username": record.actor_username,
            "event_type": record.event_type,
            "resource_type": record.resource_type,
            "resource_id": record.resource_id,
            "action": record.action,
            "result": record.result,
            "status_code": record.status_code,
            "error_code": record.error_code,
            "ip_hash": record.ip_hash,
            "metadata": record.metadata,
            "created_at": record.created_at,
        }

    @staticmethod
    def _client_hash(request: Request) -> str | None:
        host = getattr(request.client, "host", None)
        if not host:
            return None
        return hashlib.sha256(str(host).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _resource_type(path: str) -> str:
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "v1":
            return parts[2]
        return parts[0] if parts else "api"

    @staticmethod
    def _event_type_for_request(request: Request, status_code: int) -> str | None:
        path = request.url.path
        method = request.method.upper()
        if path.endswith("/auth/login"):
            return "auth.login.success" if status_code < 400 else "auth.login.failed"
        if path.endswith("/auth/logout"):
            return "auth.logout"
        if path.startswith("/api/v1/ops/cleanup") and method == "POST":
            return "cleanup.executed"
        if path.startswith("/api/v1/ops") or path.startswith("/api/v1/status") or path.startswith("/api/v1/metrics"):
            return "ops.status.viewed" if method == "GET" else "ops.changed"
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        if path.startswith("/api/v1/admin/hopper-mappings"):
            if method == "DELETE":
                return "plant.hopper_history.deleted"
            if method in {"PUT", "POST", "PATCH"}:
                return "plant.hopper_mapping.updated"
        if path.startswith("/api/v1/admin/burden-distribution"):
            if method == "DELETE":
                return "plant.burden_history.deleted"
            if method in {"PUT", "POST", "PATCH"}:
                return "plant.burden_distribution.updated"
        if path.startswith("/api/v1/admin/users") and method in {"POST", "PATCH"}:
            return "admin.user.updated" if "/deactivate" in path or "/activate" in path or method == "PATCH" else "admin.user.created"
        if path.startswith("/api/v1/feedback") and method == "POST":
            return "feedback.ticket.created"
        if path.startswith("/api/v1/material-balance"):
            return "compute.material_balance.run"
        if path.startswith("/api/v1/recommendations"):
            return "compute.recommendations.run"
        if path.startswith("/api/v1/blend-optimizer"):
            return "compute.blend_optimizer.run"
        if path.startswith("/api/v1/copilot"):
            return "copilot.analysis.run"
        if path.startswith("/api/v1/furnacemind/conversations") and path.endswith("/runs"):
            return "furnacemind.run.created"
        if path.startswith("/api/v1/furnacemind/conversations"):
            return "furnacemind.conversation.created"
        if path.startswith("/api/v1/furnacemind/documents"):
            return "furnacemind.document.uploaded"
        return "api.state_changed"

