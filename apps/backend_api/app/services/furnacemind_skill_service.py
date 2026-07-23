"""Backend-owned FurnaceMind skill service."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.furnacemind_skill_repository import (
    FurnaceMindSkillRepository,
    SkillRecord,
    idempotency_hash,
)
from apps.backend_api.app.repositories.furnacemind_task_repository import FurnaceMindTaskRepository
from apps.backend_api.app.services.furnacemind_safety_service import FurnaceMindSafetyService, warning


def _user_id(user: dict[str, Any] | None) -> str | None:
    value = (user or {}).get("id")
    return str(value) if value is not None else None


def _owner_scope(user: dict[str, Any] | None) -> str:
    return _user_id(user) or "anonymous"


def _is_admin(user: dict[str, Any] | None) -> bool:
    return str((user or {}).get("role") or "").lower() == "admin"


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class FurnaceMindSkillService:
    """Manage prompt-only skill records and observable indexing tasks."""

    def __init__(
        self,
        *,
        repository: FurnaceMindSkillRepository | None = None,
        task_repository: FurnaceMindTaskRepository | None = None,
        settings: BackendSettings | None = None,
        safety: FurnaceMindSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = self.settings.furnacemind_database_url.strip() or None
        self.repository = repository or FurnaceMindSkillRepository(database_url=database_url)
        self.tasks = task_repository or FurnaceMindTaskRepository(database_url=database_url)
        self.safety = safety or FurnaceMindSafetyService(self.settings)

    def ensure_schema(self) -> None:
        self.repository.ensure_schema()
        self.tasks.ensure_schema()

    def list_skills(self, *, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        query = dict(filters)
        if current_user is not None and not _is_admin(current_user):
            query["owner_id"] = _user_id(current_user)
        items, total = self.repository.list_skills(query)
        return {
            "items": [self.response(item) for item in items],
            "total": total,
            "limit": min(200, max(1, int(query.get("limit") or 50))),
            "offset": max(0, int(query.get("offset") or 0)),
        }

    def create_skill(
        self,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        key_hash = self._require_idempotency(idempotency_key)
        clean = self._clean_payload(payload, partial=False)
        clean["owner_id"] = _owner_scope(current_user)
        clean["metadata"] = self.safety.redact(clean.get("metadata") or {})
        fingerprint = _fingerprint(clean)
        existing = self.repository.find_by_idempotency(owner_id=clean["owner_id"], idempotency_key_hash=key_hash)
        if existing is not None:
            if existing.request_fingerprint != fingerprint:
                raise ApiError("IDEMPOTENCY_KEY_REUSED", "Idempotency-Key was already used for a different FurnaceMind skill request.", 409)
            return self.response(existing)
        record = self.repository.create_skill(
            {
                **clean,
                "idempotency_key_hash": key_hash,
                "request_fingerprint": fingerprint,
            }
        )
        return self.response(record)

    def get_skill(self, skill_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self._skill_or_404(skill_id)
        self._require_skill_access(record, current_user)
        return self.response(record)

    def patch_skill(self, skill_id: str, payload: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self._skill_or_404(skill_id)
        self._require_skill_access(record, current_user)
        clean = self._clean_payload(payload, partial=True)
        if "metadata" in clean:
            clean["metadata"] = self.safety.redact(clean["metadata"])
        if clean.get("instruction") is not None:
            clean["indexed"] = False
            clean["index_status"] = "stale"
        updated = self.repository.update_skill(skill_id, clean)
        if updated is None:
            raise ApiError("FURNACEMIND_SKILL_NOT_FOUND", "FurnaceMind skill not found.", 404)
        return self.response(updated)

    def start_index(
        self,
        skill_id: str,
        current_user: dict[str, Any] | None,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        record = self._skill_or_404(skill_id)
        self._require_skill_access(record, current_user)
        key_hash = self._require_idempotency(idempotency_key)
        owner = _owner_scope(current_user)
        request = {"skill_id": skill_id, "instruction_hash": _fingerprint({"instruction": record.instruction})}
        fingerprint = _fingerprint(request)
        existing = self.tasks.find_by_idempotency(owner_id=owner, task_type="skill_index", idempotency_key_hash=key_hash)
        if existing is not None:
            if existing.request_fingerprint != fingerprint:
                raise ApiError("IDEMPOTENCY_KEY_REUSED", "Idempotency-Key was already used for a different FurnaceMind skill index request.", 409)
            return self.task_response(existing)
        task = self.tasks.create_task(
            {
                "task_type": "skill_index",
                "resource_id": skill_id,
                "owner_id": owner,
                "status": "pending",
                "progress": 0.0,
                "current_step": "queued",
                "request": request,
                "idempotency_key_hash": key_hash,
                "request_fingerprint": fingerprint,
            }
        )
        self._append_event(task.id, "skill_index_queued", {"skill_id": skill_id})
        return self.task_response(task)

    def process_index(self, task_id: str) -> dict[str, Any]:
        task = self._task_or_404(task_id)
        if task.status == "cancelled":
            return self.task_response(task)
        record = self._skill_or_404(str(task.resource_id or ""))
        self.tasks.update_task_status(task.id, status="running", progress=0.4, current_step="indexing")
        self._append_event(task.id, "skill_index_started", {"skill_id": record.id})
        indexed = self.repository.update_skill(record.id, {"indexed": True, "index_status": "indexed"})
        result = {"skill": self.response(indexed or record), "indexed_points": 1}
        completed = self.tasks.update_task_status(
            task.id,
            status="completed",
            progress=1.0,
            current_step="completed",
            result=result,
        )
        self._append_event(task.id, "skill_index_completed", {"skill_id": record.id, "indexed_points": 1})
        return self.task_response(completed or task)

    def latest_index_status(self, skill_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self._skill_or_404(skill_id)
        self._require_skill_access(record, current_user)
        task = self.tasks.latest_for_resource(task_type="skill_index", resource_id=skill_id, owner_id=_owner_scope(current_user))
        if task is None:
            return {
                "id": None,
                "task_type": "skill_index",
                "resource_id": skill_id,
                "status": record.index_status,
                "progress": 1.0 if record.indexed else 0.0,
                "current_step": record.index_status,
                "events": [],
                "result": {"skill": self.response(record)},
                "warnings": [],
                "artifacts": [],
            }
        return self.task_response(task)

    def index_events(self, skill_id: str, current_user: dict[str, Any] | None) -> list[dict[str, Any]]:
        task = self._latest_task_for_skill(skill_id, current_user)
        return self.event_responses(task.id)

    def cancel_index(self, skill_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        task = self._latest_task_for_skill(skill_id, current_user)
        if task.status in {"completed", "failed", "cancelled"}:
            return self.task_response(task)
        cancelled = self.tasks.update_task_status(task.id, status="cancelled", progress=task.progress, current_step="cancelled")
        self._append_event(task.id, "skill_index_cancelled", {"skill_id": skill_id})
        return self.task_response(cancelled or task)

    def _latest_task_for_skill(self, skill_id: str, current_user: dict[str, Any] | None):
        record = self._skill_or_404(skill_id)
        self._require_skill_access(record, current_user)
        task = self.tasks.latest_for_resource(task_type="skill_index", resource_id=skill_id, owner_id=_owner_scope(current_user))
        if task is None:
            raise ApiError("FURNACEMIND_TASK_NOT_FOUND", "FurnaceMind skill index task not found.", 404)
        return task

    def _append_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        task = self._task_or_404(task_id)
        self.tasks.append_event(
            {
                "task_id": task.id,
                "task_type": task.task_type,
                "resource_id": task.resource_id,
                "event_type": event_type,
                "payload": self.safety.redact(payload),
            }
        )

    def _task_or_404(self, task_id: str):
        task = self.tasks.get_task(task_id)
        if task is None:
            raise ApiError("FURNACEMIND_TASK_NOT_FOUND", "FurnaceMind task not found.", 404)
        return task

    def _skill_or_404(self, skill_id: str) -> SkillRecord:
        record = self.repository.get_skill(skill_id)
        if record is None:
            raise ApiError("FURNACEMIND_SKILL_NOT_FOUND", "FurnaceMind skill not found.", 404)
        return record

    def _require_skill_access(self, record: SkillRecord, current_user: dict[str, Any] | None) -> None:
        if not self.settings.furnacemind_require_auth:
            return
        if _is_admin(current_user):
            return
        if record.owner_id == _user_id(current_user) or record.source_type == "builtin":
            return
        raise ApiError("FURNACEMIND_FORBIDDEN", "You are not allowed to access this skill.", 403)

    @staticmethod
    def _require_idempotency(idempotency_key: str | None) -> str:
        key_hash = idempotency_hash(idempotency_key)
        if not key_hash:
            raise ApiError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required for FurnaceMind mutations.", 400)
        return key_hash

    def _clean_payload(self, payload: dict[str, Any], *, partial: bool) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key in ("name", "symbol", "description", "instruction", "is_active", "metadata"):
            if key in payload:
                clean[key] = payload[key]
        if not partial:
            if not str(clean.get("name") or "").strip() or not str(clean.get("instruction") or "").strip():
                raise ApiError("FURNACEMIND_SKILL_INVALID", "Skill name and instruction are required.", 422)
        if "name" in clean and clean["name"] is not None:
            clean["name"] = str(clean["name"]).strip()[:120]
        if "symbol" in clean and clean["symbol"] is not None:
            clean["symbol"] = str(clean["symbol"]).strip()[:8]
        if "description" in clean and clean["description"] is not None:
            clean["description"] = str(clean["description"]).strip()[:500]
        if "instruction" in clean and clean["instruction"] is not None:
            instruction = str(clean["instruction"]).strip()
            if any(token in instruction.lower() for token in ("```python", "import os", "subprocess", "shell", "exec(")):
                raise ApiError("FURNACEMIND_SKILL_UNSAFE", "Skill instructions cannot request unsafe code or shell execution.", 403)
            clean["instruction"] = self.safety.enforce_message_limit(instruction)
        if "metadata" in clean and clean["metadata"] is None:
            clean["metadata"] = {}
        return clean

    @staticmethod
    def response(record: SkillRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "slug": record.slug,
            "name": record.name,
            "symbol": record.symbol,
            "description": record.description,
            "instruction": record.instruction,
            "source_type": record.source_type,
            "owner_id": record.owner_id,
            "is_active": record.is_active,
            "indexed": record.indexed,
            "index_status": record.index_status,
            "metadata": record.metadata,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def task_response(self, task) -> dict[str, Any]:
        return {
            "id": task.id,
            "task_type": task.task_type,
            "resource_id": task.resource_id,
            "status": task.status,
            "progress": task.progress,
            "current_step": task.current_step,
            "error_code": task.error_code,
            "error_message": task.error_message,
            "warnings": task.warnings,
            "artifacts": task.artifacts,
            "result": task.result,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "completed_at": task.completed_at,
        }

    def event_responses(self, task_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": event.id,
                "task_id": event.task_id,
                "task_type": event.task_type,
                "resource_id": event.resource_id,
                "event_type": event.event_type,
                "sequence": event.sequence,
                "payload": event.payload,
                "created_at": event.created_at,
            }
            for event in self.tasks.list_events(task_id)
        ]