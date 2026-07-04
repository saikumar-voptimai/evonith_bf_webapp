"""FurnaceMind retrieval-context builder."""

from __future__ import annotations

from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.repositories.furnacemind_conversation_repository import FurnaceMindConversationRepository
from app.repositories.furnacemind_document_repository import FurnaceMindDocumentRepository
from app.services.furnacemind_memory_service import FurnaceMindMemoryService
from app.services.furnacemind_safety_service import FurnaceMindSafetyService, warning


class FurnaceMindRetrievalService:
    """Build bounded, redacted context from history, documents, and memory."""

    def __init__(
        self,
        *,
        conversation_repository: FurnaceMindConversationRepository,
        document_repository: FurnaceMindDocumentRepository,
        memory_service: FurnaceMindMemoryService,
        settings: BackendSettings | None = None,
        safety: FurnaceMindSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.conversations = conversation_repository
        self.documents = document_repository
        self.memory = memory_service
        self.safety = safety or FurnaceMindSafetyService(self.settings)

    def build_context(
        self,
        *,
        conversation_id: str,
        message: str,
        document_ids: list[str],
        owner_id: str | None,
    ) -> dict[str, Any]:
        warnings: list[dict[str, Any]] = []
        history_records = self.conversations.list_messages(
            conversation_id,
            limit=self.settings.furnacemind_max_history_messages * 2,
        )
        history = [
            {"role": item.role, "content": item.content, "created_at": item.created_at}
            for item in history_records
            if item.role in {"user", "assistant"}
        ]
        history, history_warnings = self.safety.cap_history(history)
        warnings.extend(history_warnings)

        docs: list[dict[str, Any]] = []
        for document_id in document_ids:
            record = self.documents.get_document(document_id)
            if record is None:
                warnings.append(warning("FURNACEMIND_DOCUMENT_NOT_FOUND", "Selected document was not found.", {"document_id": document_id}))
                continue
            if owner_id is not None and record.owner_id not in {None, owner_id}:
                warnings.append(warning("FURNACEMIND_FORBIDDEN", "Selected document is not visible.", {"document_id": document_id}))
                continue
            chunks = self.documents.list_chunks(record.id, limit=3)
            docs.append(
                {
                    "document_id": record.id,
                    "filename": record.filename,
                    "chunks": [chunk.content for chunk in chunks],
                }
            )
        docs, doc_warnings = self.safety.cap_context_docs(docs)
        warnings.extend(doc_warnings)

        memory_results: list[dict[str, Any]] = []
        try:
            memory_results = self.memory.search(
                query=message,
                owner_id=owner_id,
                top_k=self.settings.furnacemind_vector_top_k,
            )
        except Exception as exc:
            warnings.append(warning(getattr(exc, "code", "FURNACEMIND_MEMORY_UNAVAILABLE"), str(getattr(exc, "message", exc))))
        if not self.memory.is_enabled():
            warnings.extend(self.memory.status_warnings())

        context_text = self._context_text(history=history, docs=docs, memory_results=memory_results)
        context_text, char_warnings = self.safety.cap_context_chars(context_text)
        warnings.extend(char_warnings)
        evidence = [
            {
                "id": item["document_id"],
                "title": item.get("filename") or item["document_id"],
                "type": "document",
                "metadata": {"chunk_count": len(item.get("chunks") or [])},
            }
            for item in docs
        ] + [
            {
                "id": f"memory_{index}",
                "title": str(item.get("metadata", {}).get("filename") or item.get("document_id")),
                "type": "memory",
                "metadata": {"score": item.get("score"), "document_id": item.get("document_id")},
            }
            for index, item in enumerate(memory_results, start=1)
        ]
        return self.safety.redact(
            {
                "history": history,
                "documents": docs,
                "memory_results": memory_results,
                "context_text": context_text,
                "evidence": evidence,
                "warnings": warnings,
            }
        )

    @staticmethod
    def _context_text(
        *,
        history: list[dict[str, Any]],
        docs: list[dict[str, Any]],
        memory_results: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        if history:
            parts.append("Conversation history:")
            parts.extend(f"{item['role']}: {item['content']}" for item in history)
        for doc in docs:
            parts.append(f"Document {doc['filename']}:")
            parts.extend(str(chunk) for chunk in doc.get("chunks") or [])
        if memory_results:
            parts.append("Memory search results:")
            parts.extend(str(item.get("content") or "") for item in memory_results)
        return "\n\n".join(parts)
