"""Backend-owned FurnaceMind orchestration service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.furnacemind_conversation_repository import (
    ConversationRecord,
    FurnaceMindConversationRepository,
    MessageFeedbackRecord,
    MessageRecord,
)
from apps.backend_api.app.repositories.furnacemind_document_repository import DocumentRecord, FurnaceMindDocumentRepository
from apps.backend_api.app.repositories.furnacemind_run_repository import FurnaceMindRunRepository, RunRecord, idempotency_hash
from apps.backend_api.app.repositories.furnacemind_task_repository import FurnaceMindTaskRepository
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.furnacemind_document_service import FurnaceMindDocumentService
from apps.backend_api.app.services.furnacemind_event_service import FurnaceMindEventService
from apps.backend_api.app.services.furnacemind_llm_service import FurnaceMindLLMService
from apps.backend_api.app.services.furnacemind_memory_service import FurnaceMindMemoryService
from apps.backend_api.app.services.furnacemind_prompt_service import FurnaceMindPromptService
from apps.backend_api.app.services.furnacemind_report_service import FurnaceMindReportService
from apps.backend_api.app.services.furnacemind_retrieval_service import FurnaceMindRetrievalService
from apps.backend_api.app.services.furnacemind_safety_service import FurnaceMindSafetyService, warning
from apps.backend_api.app.services.furnacemind_skill_service import FurnaceMindSkillService
from apps.backend_api.app.services.furnacemind_tool_executor import FurnaceMindToolExecutor
from apps.backend_api.app.services.furnacemind_tool_registry import FurnaceMindToolRegistry

log = logging.getLogger(__name__)
_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
_TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}

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

class FurnaceMindService:
    """Coordinate FurnaceMind conversations, runs, docs, memory, LLM, and tools."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        conversation_repository: FurnaceMindConversationRepository | None = None,
        run_repository: FurnaceMindRunRepository | None = None,
        document_repository: FurnaceMindDocumentRepository | None = None,
        task_repository: FurnaceMindTaskRepository | None = None,
        safety_service: FurnaceMindSafetyService | None = None,
        memory_service: FurnaceMindMemoryService | None = None,
        llm_service: FurnaceMindLLMService | None = None,
        prompt_service: FurnaceMindPromptService | None = None,
        artifact_service: ComputeArtifactService | None = None,
        skill_service: FurnaceMindSkillService | None = None,
        report_service: FurnaceMindReportService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = self.settings.furnacemind_database_url.strip() or None
        self.conversations = conversation_repository or FurnaceMindConversationRepository(database_url=database_url)
        self.runs = run_repository or FurnaceMindRunRepository(database_url=database_url)
        self.documents = document_repository or FurnaceMindDocumentRepository(database_url=database_url)
        self.tasks = task_repository or FurnaceMindTaskRepository(database_url=database_url)
        self.safety = safety_service or FurnaceMindSafetyService(self.settings)
        self.memory = memory_service or FurnaceMindMemoryService(self.settings, safety=self.safety)
        self.llm = llm_service or FurnaceMindLLMService(self.settings)
        self.prompts = prompt_service or FurnaceMindPromptService(settings=self.settings, safety=self.safety)
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)
        self.document_service = FurnaceMindDocumentService(
            repository=self.documents,
            settings=self.settings,
            safety=self.safety,
        )
        self.skill_service = skill_service or FurnaceMindSkillService(
            task_repository=self.tasks,
            settings=self.settings,
            safety=self.safety,
        )
        self.report_service = report_service or FurnaceMindReportService(
            task_repository=self.tasks,
            settings=self.settings,
            safety=self.safety,
            artifact_service=self.artifacts,
        )
        self.retrieval = FurnaceMindRetrievalService(
            conversation_repository=self.conversations,
            document_repository=self.documents,
            memory_service=self.memory,
            settings=self.settings,
            safety=self.safety,
        )
        self.tool_registry = FurnaceMindToolRegistry(self.settings)
        self.tool_executor = FurnaceMindToolExecutor(
            registry=self.tool_registry,
            settings=self.settings,
            safety=self.safety,
        )
        self.events = FurnaceMindEventService(
            repository=self.runs,
            settings=self.settings,
            safety=self.safety,
        )

    def ensure_storage(self) -> None:
        self.conversations.ensure_schema()
        self.runs.ensure_schema()
        self.documents.ensure_schema()
        self.tasks.ensure_schema()
        self.skill_service.ensure_schema()
        self.report_service.ensure_schema()

    def get_config(self) -> dict[str, Any]:
        warnings = []
        if not self.settings.furnacemind_llm_enabled:
            warnings.append(warning("FURNACEMIND_LLM_DISABLED", "FurnaceMind LLM calls are disabled by default."))
        if not self.settings.furnacemind_memory_enabled:
            warnings.append(warning("FURNACEMIND_MEMORY_DISABLED", "FurnaceMind vector memory is disabled by default."))
        if not self.settings.furnacemind_tools_enabled:
            warnings.append(warning("FURNACEMIND_TOOLS_DISABLED", "FurnaceMind tools are disabled by default."))
        return {
            "llm_enabled": self.settings.furnacemind_llm_enabled,
            "provider_configured": self.llm.is_available(),
            "provider_name": self.settings.furnacemind_provider or None,
            "model_name": self.settings.furnacemind_model or None,
            "memory_enabled": self.settings.furnacemind_memory_enabled,
            "vector_backend": self.settings.furnacemind_vector_backend or None,
            "qdrant_configured": bool(self.settings.furnacemind_qdrant_url and os.getenv(self.settings.furnacemind_qdrant_api_key_env, "")),
            "embeddings_enabled": self.settings.furnacemind_embeddings_enabled,
            "documents_enabled": self.settings.furnacemind_documents_enabled,
            "skills_enabled": True,
            "reports_enabled": True,
            "tools_enabled": self.settings.furnacemind_tools_enabled,
            "streaming_enabled": self.settings.furnacemind_streaming_enabled,
            "polling_fallback_enabled": self.settings.furnacemind_polling_fallback_enabled,
            "code_execution_enabled": self.settings.furnacemind_enable_code_execution,
            "shell_execution_enabled": self.settings.furnacemind_enable_shell_execution,
            "max_message_chars": self.settings.furnacemind_max_message_chars,
            "max_prompt_chars": self.settings.furnacemind_max_prompt_chars,
            "max_history_messages": self.settings.furnacemind_max_history_messages,
            "max_context_docs": self.settings.furnacemind_max_context_docs,
            "allowed_document_types": list(self.settings.furnacemind_allowed_document_types),
            "allowed_document_extensions": list(self.settings.furnacemind_allowed_document_extensions),
            "allowed_tools": self.tool_registry.allowed_tool_names(),
            "warnings": warnings,
        }

    def create_conversation(self, payload: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        self._require_user(current_user)
        title = payload.get("title") or "New FurnaceMind conversation"
        record = self.conversations.create_conversation(
            {
                "title": str(title)[:120],
                "owner_id": _user_id(current_user),
                "metadata": self.safety.redact(payload.get("metadata") or {}),
            }
        )
        log.info("furnacemind.conversation.created conversation_id=%s owner_id=%s", record.id, record.owner_id)
        return self.conversation_response(record)

    def list_conversations(self, *, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        self._require_user(current_user)
        query = dict(filters)
        if self.settings.furnacemind_require_auth and not _is_admin(current_user):
            query["owner_id"] = _user_id(current_user)
        items, total = self.conversations.list_conversations(query)
        limit = min(200, max(1, int(query.get("limit") or 50)))
        offset = max(0, int(query.get("offset") or 0))
        return {
            "items": [self.conversation_response(item) for item in items],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_conversation(self, conversation_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self._conversation_or_404(conversation_id)
        self._require_conversation_access(record, current_user)
        return self.conversation_response(record)

    def update_conversation(
        self,
        conversation_id: str,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        record = self._conversation_or_404(conversation_id)
        self._require_conversation_access(record, current_user)
        changes: dict[str, Any] = {}
        if "title" in payload and payload.get("title") is not None:
            changes["title"] = str(payload["title"])[:120]
        if "archived" in payload and payload.get("archived") is not None:
            changes["archived"] = int(bool(payload["archived"]))
        if payload.get("metadata") is not None:
            changes["metadata_json"] = self._json_safe(payload["metadata"])
        updated = self.conversations.update_conversation(record.id, changes)
        if updated is None:
            raise ApiError("FURNACEMIND_CONVERSATION_NOT_FOUND", "FurnaceMind conversation not found.", 404)
        return self.conversation_response(updated)

    def archive_conversation(self, conversation_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        return self.update_conversation(conversation_id, {"archived": True}, current_user)

    def list_messages(self, conversation_id: str, current_user: dict[str, Any] | None) -> list[dict[str, Any]]:
        record = self._conversation_or_404(conversation_id)
        self._require_conversation_access(record, current_user)
        return [self.message_response(item) for item in self.conversations.list_messages(record.id)]

    def create_user_message(
        self,
        conversation_id: str,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        record = self._conversation_or_404(conversation_id)
        self._require_conversation_access(record, current_user)
        content = self.safety.enforce_message_limit(payload.get("content") or "")
        message = self.conversations.create_message(
            {
                "conversation_id": record.id,
                "role": "user",
                "content": self.safety.redact(content),
                "status": "complete",
                "parent_message_id": payload.get("parent_message_id"),
                "metadata": {
                    "document_ids": payload.get("document_ids") or [],
                    "tool_mode": payload.get("tool_mode") or "auto",
                },
            }
        )
        return self.message_response(message)

    def create_run(
        self,
        conversation_id: str,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
        *,
        idempotency_key: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        record = self._conversation_or_404(conversation_id)
        self._require_conversation_access(record, current_user)
        message = self.safety.enforce_message_limit(payload.get("message") or "")
        options = payload.get("options") or {}
        self._block_unsafe_run_options(options)
        request_payload = {
            "conversation_id": record.id,
            "message": message,
            "document_ids": [str(item) for item in (payload.get("document_ids") or [])],
            "tool_mode": str(payload.get("tool_mode") or "auto").lower(),
            "allow_llm": payload.get("allow_llm"),
            "stream": bool(payload.get("stream", False)),
            "options": self.safety.redact(options),
        }
        key_hash = self._require_idempotency(idempotency_key)
        owner = _owner_scope(current_user)
        fingerprint = _fingerprint(request_payload)
        existing = self.runs.find_run_for_idempotency(owner_id=owner, idempotency_key_hash=key_hash)
        if existing is not None:
            if existing.request_fingerprint != fingerprint:
                raise ApiError("IDEMPOTENCY_KEY_REUSED", "Idempotency-Key was already used for a different FurnaceMind run request.", 409)
            return self.run_response(existing)
        user_message = self.conversations.create_message(
            {
                "conversation_id": record.id,
                "role": "user",
                "content": self.safety.redact(message),
                "status": "complete",
                "metadata": {"document_ids": request_payload["document_ids"]},
            }
        )
        run = self.runs.create_run(
            {
                "conversation_id": record.id,
                "owner_id": owner,
                "status": "pending",
                "user_message_id": user_message.id,
                "progress": 0.0,
                "current_step": "queued",
                "request": request_payload,
                "idempotency_key_hash": key_hash,
                "request_fingerprint": fingerprint,
            }
        )
        self.events.append(run_id=run.id, conversation_id=record.id, event_type="run_created", payload={"request_id": request_id})
        self.events.append(run_id=run.id, conversation_id=record.id, event_type="message_created", payload={"message_id": user_message.id})
        return self.run_response(run)

    def process_run(
        self,
        run_id: str,
        current_user: dict[str, Any] | None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        run = self._run_or_404(run_id)
        if run.status in _TERMINAL_RUN_STATUSES:
            return self.run_response(run)
        conversation = self._conversation_or_404(run.conversation_id)
        self._require_conversation_access(conversation, current_user)
        user_message = self.conversations.get_message(run.user_message_id or "")
        if user_message is None:
            failed = self.runs.update_run_status(
                run.id,
                status="failed",
                progress=1.0,
                current_step="failed",
                error_code="FURNACEMIND_MESSAGE_NOT_FOUND",
                error_message="FurnaceMind run message not found.",
            )
            return self.run_response(failed or run)
        try:
            return self._process_run(run, user_message, run.request, current_user, request_id=request_id)
        except ApiError:
            return self.run_response(self._run_or_404(run.id))

    def _process_run(
        self,
        run: RunRecord,
        user_message: MessageRecord,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
        *,
        request_id: str | None,
    ) -> dict[str, Any]:
        warnings: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        try:
            waiting = self._documents_waiting_for_index(payload.get("document_ids") or [], current_user)
            if waiting:
                waiting_warning = warning(
                    "FURNACEMIND_DOCUMENTS_NOT_READY",
                    "One or more selected documents must be indexed before this run can continue.",
                    {"document_ids": waiting},
                )
                updated = self.runs.update_run_status(
                    run.id,
                    status="waiting_for_documents",
                    progress=0.05,
                    current_step="waiting_for_documents",
                    warnings=[waiting_warning],
                )
                self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="run_waiting_for_documents", payload={"document_ids": waiting})
                return self.run_response(updated or run)

            self.runs.update_run_status(run.id, status="running", progress=0.15, current_step="retrieval")
            self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="run_started", payload={})
            self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="retrieval_started", payload={})
            context = self.retrieval.build_context(
                conversation_id=run.conversation_id,
                message=user_message.content,
                document_ids=list(payload.get("document_ids") or []),
                owner_id=_user_id(current_user),
            )
            warnings.extend(context.get("warnings", []))
            self.events.append(
                run_id=run.id,
                conversation_id=run.conversation_id,
                event_type="retrieval_completed",
                payload={"evidence_count": len(context.get("evidence") or [])},
            )

            tool_results: list[dict[str, Any]] = []
            tool_mode = str(payload.get("tool_mode") or "auto").lower()
            if tool_mode != "none" and not self.settings.furnacemind_tools_enabled:
                warnings.append(warning("FURNACEMIND_TOOLS_DISABLED", "FurnaceMind tools are disabled."))
            elif tool_mode != "none":
                self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="tool_policy_applied", payload={"allowed_tools": self.tool_registry.allowed_tool_names()})
            prompt, prompt_warnings = self.prompts.build_prompt(
                message=user_message.content,
                context=context,
                tool_results=tool_results,
            )
            warnings.extend(prompt_warnings)
            answer, llm_used, provider_name, model_name = self._answer(
                prompt=prompt,
                message=user_message.content,
                context=context,
                warnings=warnings,
                allow_llm=payload.get("allow_llm"),
                options=payload.get("options") or {},
                request_id=request_id,
            )
            if llm_used:
                self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="llm_completed", payload={"provider": provider_name, "model": model_name})
            if (payload.get("options") or {}).get("export"):
                metadata = self.artifacts.create_json_artifact(
                    workflow="furnacemind",
                    filename_prefix="furnacemind_run",
                    payload={
                        "conversation_id": run.conversation_id,
                        "run_id": run.id,
                        "answer": answer,
                        "evidence": context.get("evidence", []),
                        "warnings": warnings,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ttl_hours=self.settings.furnacemind_artifact_ttl_hours,
                    owner_user_id=_owner_scope(current_user),
                    calculation_id=run.id,
                )
                artifacts.append(self.artifacts.response(metadata, "/furnacemind"))
                self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="artifact_created", payload={"artifact_id": metadata.artifact_id})

            assistant = self.conversations.create_message(
                {
                    "conversation_id": run.conversation_id,
                    "role": "assistant",
                    "content": answer[: self.settings.furnacemind_max_response_chars],
                    "status": "complete",
                    "run_id": run.id,
                    "metadata": {
                        "llm_used": llm_used,
                        "provider_name": provider_name,
                        "model_name": model_name,
                        "evidence": context.get("evidence", []),
                        "tool_results": tool_results,
                    },
                    "artifacts": artifacts,
                    "warnings": warnings,
                }
            )
            completed = self.runs.update_run_status(
                run.id,
                status="completed",
                progress=1.0,
                current_step="completed",
                assistant_message_id=assistant.id,
                warnings=warnings,
                artifacts=artifacts,
            )
            self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="run_completed", payload={"assistant_message_id": assistant.id})
            log.info("furnacemind.run.completed run_id=%s conversation_id=%s warnings=%s", run.id, run.conversation_id, len(warnings))
            return self.run_response(completed or run)
        except ApiError as exc:
            self.runs.update_run_status(
                run.id,
                status="failed",
                progress=1.0,
                current_step="failed",
                error_code=exc.code,
                error_message=exc.message,
                warnings=warnings,
            )
            self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="run_failed", payload={"error_code": exc.code})
            raise
        except Exception as exc:
            self.runs.update_run_status(
                run.id,
                status="failed",
                progress=1.0,
                current_step="failed",
                error_code="FURNACEMIND_RUN_FAILED",
                error_message="FurnaceMind run failed.",
                warnings=warnings,
            )
            self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="run_failed", payload={"error_code": "FURNACEMIND_RUN_FAILED"})
            raise ApiError("FURNACEMIND_RUN_FAILED", "FurnaceMind run failed.", 500) from exc

    def _answer(
        self,
        *,
        prompt: str,
        message: str,
        context: dict[str, Any],
        warnings: list[dict[str, Any]],
        allow_llm: bool | None,
        options: dict[str, Any],
        request_id: str | None,
    ) -> tuple[str, bool, str | None, str | None]:
        if allow_llm is True:
            self.events  # keep attribute access for type checkers
            llm_result = self.llm.generate(prompt, request_id=request_id, options=options)
            return llm_result.text, True, llm_result.provider_name, llm_result.model_name
        if allow_llm is None and self.settings.furnacemind_llm_enabled and not self.llm.is_available():
            warnings.append(warning("FURNACEMIND_LLM_PROVIDER_NOT_CONFIGURED", "LLM is enabled but no provider is available."))
        evidence_count = len(context.get("evidence") or [])
        memory_note = "Memory is enabled." if self.settings.furnacemind_memory_enabled else "Memory is disabled."
        answer = (
            "FurnaceMind non-LLM response.\n\n"
            f"Message: {message}\n\n"
            f"Retrieved evidence items: {evidence_count}.\n"
            f"{memory_note}\n"
            f"Warnings: {len(warnings)}."
        )
        return answer, False, None, None

    def get_run_status(self, run_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        run = self._run_or_404(run_id)
        conversation = self._conversation_or_404(run.conversation_id)
        self._require_conversation_access(conversation, current_user)
        result_message = self.conversations.get_message(run.assistant_message_id) if run.assistant_message_id else None
        return {
            "id": run.id,
            "conversation_id": run.conversation_id,
            "status": run.status,
            "progress": run.progress,
            "current_step": run.current_step,
            "events": self.events.list_events(run.id),
            "result_message": self.message_response(result_message) if result_message else None,
            "error_code": run.error_code,
            "error_message": run.error_message,
            "artifacts": run.artifacts,
        }

    def list_run_events(self, run_id: str, current_user: dict[str, Any] | None) -> list[dict[str, Any]]:
        run = self._run_or_404(run_id)
        conversation = self._conversation_or_404(run.conversation_id)
        self._require_conversation_access(conversation, current_user)
        return self.events.list_events(run.id)

    def cancel_run(self, run_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        run = self._run_or_404(run_id)
        conversation = self._conversation_or_404(run.conversation_id)
        self._require_conversation_access(conversation, current_user)
        if run.status in _TERMINAL_RUN_STATUSES:
            return self.get_run_status(run_id, current_user)
        cancelled = self.runs.update_run_status(run.id, status="cancelled", progress=run.progress, current_step="cancelled")
        self.events.append(run_id=run.id, conversation_id=run.conversation_id, event_type="run_cancelled", payload={})
        return self.get_run_status((cancelled or run).id, current_user)
    def list_tools(self) -> list[dict[str, Any]]:
        return self.tool_registry.list_tools()

    def start_document_index(
        self,
        document_id: str,
        current_user: dict[str, Any] | None,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        record = self._document_or_404(document_id)
        self.document_service.require_document_access(record, current_user)
        key_hash = self._require_idempotency(idempotency_key)
        owner = _owner_scope(current_user)
        request = {"document_id": document_id, "chunk_count": record.chunk_count, "size_bytes": record.size_bytes}
        fingerprint = _fingerprint(request)
        existing = self.tasks.find_by_idempotency(owner_id=owner, task_type="document_index", idempotency_key_hash=key_hash)
        if existing is not None:
            if existing.request_fingerprint != fingerprint:
                raise ApiError("IDEMPOTENCY_KEY_REUSED", "Idempotency-Key was already used for a different FurnaceMind document index request.", 409)
            return self.task_response(existing)
        task = self.tasks.create_task(
            {
                "task_type": "document_index",
                "resource_id": document_id,
                "owner_id": owner,
                "status": "pending",
                "progress": 0.0,
                "current_step": "queued",
                "request": request,
                "idempotency_key_hash": key_hash,
                "request_fingerprint": fingerprint,
            }
        )
        self._append_task_event(task.id, "document_index_queued", {"document_id": document_id})
        return self.task_response(task)

    def process_document_index(self, task_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        task = self._task_or_404(task_id)
        if task.status in _TERMINAL_TASK_STATUSES:
            return self.task_response(task)
        try:
            self.tasks.update_task_status(task.id, status="running", progress=0.35, current_step="indexing")
            self._append_task_event(task.id, "document_index_started", {"document_id": task.resource_id})
            result = self.document_service.index_document(
                str(task.resource_id or ""),
                current_user=current_user,
                memory_service=self.memory,
            )
            completed = self.tasks.update_task_status(
                task.id,
                status="completed",
                progress=1.0,
                current_step="completed",
                result={"document": result},
                warnings=result.get("warnings", []),
            )
            self._append_task_event(task.id, "document_index_completed", {"document_id": task.resource_id, "indexed": result.get("indexed")})
            return self.task_response(completed or task)
        except ApiError as exc:
            failed = self.tasks.update_task_status(
                task.id,
                status="failed",
                progress=1.0,
                current_step="failed",
                error_code=exc.code,
                error_message=exc.message,
            )
            self._append_task_event(task.id, "document_index_failed", {"error_code": exc.code})
            return self.task_response(failed or task)

    def document_index_status(self, document_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self._document_or_404(document_id)
        self.document_service.require_document_access(record, current_user)
        task = self.tasks.latest_for_resource(task_type="document_index", resource_id=document_id, owner_id=_owner_scope(current_user))
        if task is None:
            return {
                "id": None,
                "task_type": "document_index",
                "resource_id": document_id,
                "status": "completed" if record.indexed else "not_started",
                "progress": 1.0 if record.indexed else 0.0,
                "current_step": "indexed" if record.indexed else record.status,
                "error_code": None,
                "error_message": None,
                "warnings": [],
                "artifacts": [],
                "result": {"document": self.document_service.response(record)},
                "created_at": None,
                "updated_at": None,
                "completed_at": None,
            }
        return self.task_response(task)

    def document_index_events(self, document_id: str, current_user: dict[str, Any] | None) -> list[dict[str, Any]]:
        task = self._latest_document_task(document_id, current_user)
        return self.task_event_responses(task.id)

    def cancel_document_index(self, document_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        task = self._latest_document_task(document_id, current_user)
        if task.status in _TERMINAL_TASK_STATUSES:
            return self.task_response(task)
        cancelled = self.tasks.update_task_status(task.id, status="cancelled", progress=task.progress, current_step="cancelled")
        self._append_task_event(task.id, "document_index_cancelled", {"document_id": document_id})
        return self.task_response(cancelled or task)
    def submit_message_feedback(
        self,
        message_id: str,
        payload: dict[str, Any],
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        message = self.conversations.get_message(message_id)
        if message is None:
            raise ApiError("FURNACEMIND_MESSAGE_NOT_FOUND", "FurnaceMind message not found.", 404)
        conversation = self._conversation_or_404(message.conversation_id)
        self._require_conversation_access(conversation, current_user)
        record = self.conversations.store_message_feedback(
            {
                "message_id": message.id,
                "conversation_id": message.conversation_id,
                "owner_id": _user_id(current_user),
                "rating": payload.get("rating"),
                "helpful": payload.get("helpful"),
                "comment": self.safety.redact(payload.get("comment")),
                "tags": payload.get("tags") or [],
            }
        )
        return self.feedback_response(record)

    def list_feedback(self, *, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        self._require_user(current_user)
        query = dict(filters)
        if self.settings.furnacemind_require_auth and not _is_admin(current_user):
            query["owner_id"] = _user_id(current_user)
        items, total = self.conversations.list_message_feedback(query)
        return {
            "items": [self.feedback_response(item) for item in items],
            "total": total,
            "limit": min(200, max(1, int(query.get("limit") or 50))),
            "offset": max(0, int(query.get("offset") or 0)),
        }

    def _documents_waiting_for_index(self, document_ids: list[str], current_user: dict[str, Any] | None) -> list[str]:
        waiting: list[str] = []
        for document_id in document_ids:
            record = self._document_or_404(str(document_id))
            self.document_service.require_document_access(record, current_user)
            if not record.indexed:
                waiting.append(record.id)
        return waiting

    def _latest_document_task(self, document_id: str, current_user: dict[str, Any] | None):
        record = self._document_or_404(document_id)
        self.document_service.require_document_access(record, current_user)
        task = self.tasks.latest_for_resource(task_type="document_index", resource_id=document_id, owner_id=_owner_scope(current_user))
        if task is None:
            raise ApiError("FURNACEMIND_TASK_NOT_FOUND", "FurnaceMind document index task not found.", 404)
        return task

    def _append_task_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
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

    def _document_or_404(self, document_id: str) -> DocumentRecord:
        record = self.documents.get_document(document_id)
        if record is None:
            raise ApiError("FURNACEMIND_DOCUMENT_NOT_FOUND", "FurnaceMind document not found.", 404)
        return record
    def _require_user(self, current_user: dict[str, Any] | None) -> None:
        if self.settings.furnacemind_require_auth and not current_user:
            raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)

    def _conversation_or_404(self, conversation_id: str) -> ConversationRecord:
        record = self.conversations.get_conversation(conversation_id)
        if record is None:
            raise ApiError("FURNACEMIND_CONVERSATION_NOT_FOUND", "FurnaceMind conversation not found.", 404)
        return record

    def _run_or_404(self, run_id: str) -> RunRecord:
        record = self.runs.get_run(run_id)
        if record is None:
            raise ApiError("FURNACEMIND_RUN_NOT_FOUND", "FurnaceMind run not found.", 404)
        return record

    def _require_conversation_access(self, conversation: ConversationRecord, current_user: dict[str, Any] | None) -> None:
        self._require_user(current_user)
        if not self.settings.furnacemind_require_auth:
            return
        if _is_admin(current_user):
            return
        if conversation.owner_id and conversation.owner_id == _user_id(current_user):
            return
        raise ApiError("FURNACEMIND_FORBIDDEN", "You are not allowed to access this conversation.", 403)


    @staticmethod
    def _require_idempotency(idempotency_key: str | None) -> str:
        key_hash = idempotency_hash(idempotency_key)
        if not key_hash:
            raise ApiError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required for FurnaceMind mutations.", 400)
        return key_hash

    def _block_unsafe_run_options(self, options: dict[str, Any]) -> None:
        self.safety.block_unsafe_options(options)
        blocked_keys = {
            "tool_calls",
            "raw_sql",
            "sql",
            "raw_flux",
            "flux",
            "python",
            "shell",
            "provider_settings",
            "api_key",
            "system_prompt",
        }
        for key in blocked_keys:
            if key in options:
                raise ApiError(
                    "FURNACEMIND_UNSAFE_INPUT",
                    "FurnaceMind API does not accept client-supplied tools, code, provider settings, SQL, Flux, or system prompts.",
                    status_code=403,
                )
    def run_response(self, run: RunRecord) -> dict[str, Any]:
        return {
            "id": run.id,
            "conversation_id": run.conversation_id,
            "status": run.status,
            "user_message_id": run.user_message_id,
            "assistant_message_id": run.assistant_message_id,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "completed_at": run.completed_at,
            "error_code": run.error_code,
            "error_message": run.error_message,
            "warnings": run.warnings,
            "artifacts": run.artifacts,
        }

    @staticmethod
    def conversation_response(record: ConversationRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "title": record.title,
            "owner_id": record.owner_id,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "archived": record.archived,
            "message_count": record.message_count,
            "last_message_at": record.last_message_at,
            "metadata": record.metadata,
        }

    @staticmethod
    def message_response(record: MessageRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "id": record.id,
            "conversation_id": record.conversation_id,
            "role": record.role,
            "content": record.content,
            "status": record.status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "run_id": record.run_id,
            "parent_message_id": record.parent_message_id,
            "metadata": record.metadata,
            "artifacts": record.artifacts,
            "warnings": record.warnings,
        }


    @staticmethod
    def feedback_response(record: MessageFeedbackRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "message_id": record.message_id,
            "conversation_id": record.conversation_id,
            "owner_id": record.owner_id,
            "rating": record.rating,
            "helpful": record.helpful,
            "comment": record.comment,
            "tags": record.tags,
            "created_at": record.created_at,
        }

    @staticmethod
    def task_response(task) -> dict[str, Any]:
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

    def task_event_responses(self, task_id: str) -> list[dict[str, Any]]:
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
    def _json_safe(self, value: Any) -> str:
        import json

        return json.dumps(self.safety.redact(value), sort_keys=True, default=str)
