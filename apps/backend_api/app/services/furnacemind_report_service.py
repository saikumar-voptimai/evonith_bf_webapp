"""Backend-owned FurnaceMind report generation service."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.furnacemind_task_repository import (
    FurnaceMindTaskRepository,
    idempotency_hash,
)
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.furnacemind_safety_service import FurnaceMindSafetyService, warning


def _user_id(user: dict[str, Any] | None) -> str | None:
    value = (user or {}).get("id")
    return str(value) if value is not None else None


def _owner_scope(user: dict[str, Any] | None) -> str:
    return _user_id(user) or "anonymous"


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _iso_datetime(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _iso_datetime(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_iso_datetime(item) for item in value]
    if isinstance(value, tuple):
        return [_iso_datetime(item) for item in value]
    return value


class FurnaceMindReportService:
    """Create observable shift/day report tasks and structured artifacts."""

    def __init__(
        self,
        *,
        task_repository: FurnaceMindTaskRepository | None = None,
        settings: BackendSettings | None = None,
        safety: FurnaceMindSafetyService | None = None,
        artifact_service: ComputeArtifactService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = self.settings.furnacemind_database_url.strip() or None
        self.tasks = task_repository or FurnaceMindTaskRepository(database_url=database_url)
        self.safety = safety or FurnaceMindSafetyService(self.settings)
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)

    def ensure_schema(self) -> None:
        self.tasks.ensure_schema()

    def config(self) -> dict[str, Any]:
        return {
            "report_types": ["Shift", "Day"],
            "shift_labels": ["A", "B", "C"],
            "default_include_analysis": False,
            "artifact_ttl_hours": self.settings.furnacemind_artifact_ttl_hours,
        }

    def create_report(
        self,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        key_hash = self._require_idempotency(idempotency_key)
        request = self._normalize_request(payload)
        owner = _owner_scope(current_user)
        fingerprint = _fingerprint(request)
        existing = self.tasks.find_by_idempotency(owner_id=owner, task_type="report", idempotency_key_hash=key_hash)
        if existing is not None:
            if existing.request_fingerprint != fingerprint:
                raise ApiError("IDEMPOTENCY_KEY_REUSED", "Idempotency-Key was already used for a different FurnaceMind report request.", 409)
            return self.report_response(existing)
        report_id = f"fmrp_{uuid4().hex}"
        task = self.tasks.create_task(
            {
                "task_type": "report",
                "resource_id": report_id,
                "owner_id": owner,
                "status": "pending",
                "progress": 0.0,
                "current_step": "queued",
                "request": request,
                "idempotency_key_hash": key_hash,
                "request_fingerprint": fingerprint,
            }
        )
        self._append_event(task.id, "report_queued", {"report_id": report_id})
        return self.report_response(task)

    def process_report(self, task_id: str) -> dict[str, Any]:
        task = self._task_or_404(task_id)
        if task.status == "cancelled":
            return self.report_response(task)
        warnings: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        self.tasks.update_task_status(task.id, status="running", progress=0.15, current_step="fetching")
        self._append_event(task.id, "report_started", {"report_id": task.resource_id})
        request = dict(task.request)
        try:
            document = self._generate_live_document(request)
        except Exception as exc:  # pragma: no cover - exercised when plant data is unavailable
            warnings.append(
                warning(
                    "FURNACEMIND_REPORT_SOURCE_UNAVAILABLE",
                    "Live report data source is unavailable; returned an empty typed report shell.",
                    {"error": str(exc)[:200]},
                )
            )
            document = self._fallback_document(request)
        document_payload = self._document_payload(document)
        markdown = self._document_markdown(document)
        result = {
            "id": task.resource_id,
            "report_type": request["report_type"],
            "selected_date": request["selected_date"],
            "shift_label": request.get("shift_label"),
            "include_analysis": request["include_analysis"],
            "document": document_payload,
            "markdown": markdown,
            "warnings": warnings,
        }
        metadata = self.artifacts.create_json_artifact(
            workflow="furnacemind",
            filename_prefix="furnacemind_report",
            payload={
                "report": result,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            ttl_hours=self.settings.furnacemind_artifact_ttl_hours,
        )
        artifacts.append(self.artifacts.response(metadata, "/furnacemind"))
        completed = self.tasks.update_task_status(
            task.id,
            status="completed",
            progress=1.0,
            current_step="completed",
            result=result,
            warnings=warnings,
            artifacts=artifacts,
        )
        self._append_event(task.id, "report_completed", {"report_id": task.resource_id, "artifact_count": len(artifacts)})
        return self.report_response(completed or task)

    def list_reports(self, *, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        query = dict(filters)
        query["task_type"] = "report"
        if current_user is not None:
            query["owner_id"] = _owner_scope(current_user)
        items, total = self.tasks.list_tasks(query)
        return {
            "items": [self.report_response(item) for item in items],
            "total": total,
            "limit": min(200, max(1, int(query.get("limit") or 50))),
            "offset": max(0, int(query.get("offset") or 0)),
        }

    def get_report(self, report_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        task = self._report_task_or_404(report_id, current_user)
        return self.report_response(task)

    def report_events(self, report_id: str, current_user: dict[str, Any] | None) -> list[dict[str, Any]]:
        task = self._report_task_or_404(report_id, current_user)
        return self.event_responses(task.id)

    def cancel_report(self, report_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        task = self._report_task_or_404(report_id, current_user)
        if task.status in {"completed", "failed", "cancelled"}:
            return self.report_response(task)
        cancelled = self.tasks.update_task_status(task.id, status="cancelled", progress=task.progress, current_step="cancelled")
        self._append_event(task.id, "report_cancelled", {"report_id": report_id})
        return self.report_response(cancelled or task)

    def _report_task_or_404(self, report_id: str, current_user: dict[str, Any] | None):
        task = self.tasks.latest_for_resource(task_type="report", resource_id=report_id, owner_id=_owner_scope(current_user))
        if task is None:
            raise ApiError("FURNACEMIND_REPORT_NOT_FOUND", "FurnaceMind report not found.", 404)
        return task

    def _task_or_404(self, task_id: str):
        task = self.tasks.get_task(task_id)
        if task is None:
            raise ApiError("FURNACEMIND_TASK_NOT_FOUND", "FurnaceMind task not found.", 404)
        return task

    def _normalize_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        report_type = str(payload.get("report_type") or "Shift").strip().title()
        if report_type not in {"Shift", "Day"}:
            raise ApiError("FURNACEMIND_REPORT_INVALID", "Report type must be Shift or Day.", 422)
        selected_date = date.fromisoformat(str(payload.get("selected_date") or date.today().isoformat()))
        shift_label = payload.get("shift_label")
        if report_type == "Shift":
            shift_label = str(shift_label or "A").strip().upper()
            if shift_label not in {"A", "B", "C"}:
                raise ApiError("FURNACEMIND_REPORT_INVALID", "Shift label must be A, B, or C.", 422)
        else:
            shift_label = None
        return {
            "report_type": report_type,
            "selected_date": selected_date.isoformat(),
            "shift_label": shift_label,
            "include_analysis": bool(payload.get("include_analysis", False)),
        }

    def _generate_live_document(self, request: dict[str, Any]):
        from apps.frontend_streamlit.reports.furnace_report import ShiftReportService

        selected_date = date.fromisoformat(request["selected_date"])
        _, document = ShiftReportService(llm_client=None).generate_document(
            selected_date,
            request.get("shift_label"),
            report_type=request["report_type"],
            include_analysis=bool(request.get("include_analysis")),
        )
        return document

    @staticmethod
    def _fallback_document(request: dict[str, Any]):
        from apps.frontend_streamlit.reports.rendering import ReportDocument, ReportNote

        title = (
            f"Live Report ({request['selected_date']}_SHIFT_{request.get('shift_label') or 'A'})"
            if request["report_type"] == "Shift"
            else f"Live Day Report ({request['selected_date']})"
        )
        return ReportDocument(
            title=title,
            pre_blocks=("Live report data was unavailable for this backend runtime.",),
            notes=(ReportNote("No source rows were returned.", placement="right", kind="note"),),
            generated_at_ist=datetime.now(timezone.utc),
        )

    @staticmethod
    def _document_payload(document: Any) -> dict[str, Any]:
        return _iso_datetime(asdict(document))

    @staticmethod
    def _document_markdown(document: Any) -> str:
        from apps.frontend_streamlit.reports.rendering import document_to_markdown

        return document_to_markdown(document)

    @staticmethod
    def _require_idempotency(idempotency_key: str | None) -> str:
        key_hash = idempotency_hash(idempotency_key)
        if not key_hash:
            raise ApiError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required for FurnaceMind mutations.", 400)
        return key_hash

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

    def report_response(self, task) -> dict[str, Any]:
        result = dict(task.result or {})
        return {
            "id": task.resource_id,
            "task_id": task.id,
            "status": task.status,
            "progress": task.progress,
            "current_step": task.current_step,
            "report_type": result.get("report_type") or task.request.get("report_type"),
            "selected_date": result.get("selected_date") or task.request.get("selected_date"),
            "shift_label": result.get("shift_label") if result else task.request.get("shift_label"),
            "include_analysis": result.get("include_analysis") if result else task.request.get("include_analysis"),
            "document": result.get("document"),
            "markdown": result.get("markdown"),
            "error_code": task.error_code,
            "error_message": task.error_message,
            "warnings": task.warnings or result.get("warnings") or [],
            "artifacts": task.artifacts,
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