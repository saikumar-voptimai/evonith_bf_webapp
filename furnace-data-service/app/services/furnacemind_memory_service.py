"""Optional FurnaceMind memory/vector-search service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from app.services.furnacemind_safety_service import FurnaceMindSafetyService, warning
from app.services.optional_dependency_service import require_optional_module


@dataclass(frozen=True)
class MemorySearchResult:
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any]


class FurnaceMindMemoryService:
    """Memory interface with disabled, fake, and lazy Qdrant modes."""

    def __init__(
        self,
        settings: BackendSettings | None = None,
        *,
        safety: FurnaceMindSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.safety = safety or FurnaceMindSafetyService(self.settings)
        self._fake_index: dict[str, list[dict[str, Any]]] = {}
        self._qdrant_client: Any | None = None

    def is_enabled(self) -> bool:
        return bool(self.settings.furnacemind_memory_enabled)

    def status_warnings(self) -> list[dict[str, Any]]:
        if not self.is_enabled():
            return [warning("FURNACEMIND_MEMORY_DISABLED", "FurnaceMind memory is disabled.")]
        if self.settings.furnacemind_vector_backend == "qdrant" and not self.settings.furnacemind_qdrant_url:
            return [warning("FURNACEMIND_MEMORY_UNAVAILABLE", "Qdrant URL is not configured.")]
        if not self.settings.furnacemind_embeddings_enabled and self.settings.furnacemind_vector_backend != "fake":
            return [warning("FURNACEMIND_EMBEDDINGS_DISABLED", "FurnaceMind embeddings are disabled.")]
        return []

    def index_document_chunks(
        self,
        *,
        document_id: str,
        chunks: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.is_enabled():
            return {
                "indexed": False,
                "chunk_count": 0,
                "warnings": self.status_warnings(),
            }
        backend = self.settings.furnacemind_vector_backend
        if backend == "fake":
            self._fake_index[document_id] = [
                {
                    "content": self.safety.redact(chunk),
                    "metadata": self.safety.redact(metadata),
                }
                for chunk in chunks
            ]
            return {"indexed": True, "chunk_count": len(chunks), "warnings": []}
        if backend == "qdrant":
            if not self.settings.furnacemind_qdrant_url:
                raise ApiError(
                    "FURNACEMIND_MEMORY_UNAVAILABLE",
                    "Qdrant is enabled but no Qdrant URL is configured.",
                    status_code=503,
                )
            if not self.settings.furnacemind_embeddings_enabled:
                raise ApiError(
                    "FURNACEMIND_EMBEDDINGS_DISABLED",
                    "Embeddings must be enabled before Qdrant indexing.",
                    status_code=409,
                )
            self._lazy_qdrant_client()
            raise ApiError(
                "FURNACEMIND_VECTOR_BACKEND_UNAVAILABLE",
                "Qdrant indexing is not implemented in this Phase 9 runtime.",
                status_code=503,
            )
        raise ApiError(
            "FURNACEMIND_VECTOR_BACKEND_UNAVAILABLE",
            "Configured FurnaceMind vector backend is unavailable.",
            status_code=503,
        )

    def search(
        self,
        *,
        query: str,
        owner_id: str | int | None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.is_enabled():
            return []
        backend = self.settings.furnacemind_vector_backend
        limit = min(self.settings.furnacemind_vector_top_k, max(1, int(top_k or self.settings.furnacemind_vector_top_k)))
        if backend == "fake":
            tokens = {token.lower() for token in str(query or "").split() if token.strip()}
            results: list[MemorySearchResult] = []
            for document_id, chunks in self._fake_index.items():
                for item in chunks:
                    content = str(item.get("content") or "")
                    metadata = dict(item.get("metadata") or {})
                    if owner_id is not None and str(metadata.get("owner_id")) not in {"", str(owner_id)}:
                        continue
                    content_tokens = {token.lower().strip(".,:;") for token in content.split()}
                    overlap = len(tokens & content_tokens)
                    score = overlap / max(len(tokens), 1)
                    if overlap or not tokens:
                        results.append(MemorySearchResult(document_id, content, score, metadata))
            results.sort(key=lambda item: item.score, reverse=True)
            return [
                {
                    "document_id": item.document_id,
                    "content": item.content,
                    "score": item.score,
                    "metadata": self.safety.redact(item.metadata),
                }
                for item in results[:limit]
            ]
        if backend == "qdrant":
            if not self.settings.furnacemind_qdrant_url:
                raise ApiError("FURNACEMIND_MEMORY_UNAVAILABLE", "Qdrant URL is not configured.", 503)
            if not self.settings.furnacemind_embeddings_enabled:
                raise ApiError("FURNACEMIND_EMBEDDINGS_DISABLED", "Embeddings are disabled.", 409)
            self._lazy_qdrant_client()
            raise ApiError("FURNACEMIND_VECTOR_SEARCH_FAILED", "Qdrant search is unavailable.", 503)
        raise ApiError("FURNACEMIND_VECTOR_BACKEND_UNAVAILABLE", "Vector backend is unavailable.", 503)

    def delete_document(self, document_id: str) -> None:
        self._fake_index.pop(document_id, None)

    def _lazy_qdrant_client(self) -> Any:
        if self._qdrant_client is not None:
            return self._qdrant_client
        QdrantClient = require_optional_module("qdrant_client", "backend-vector").QdrantClient
        self._qdrant_client = QdrantClient(
            url=self.settings.furnacemind_qdrant_url,
            api_key=None,
            timeout=self.settings.furnacemind_vector_timeout_seconds,
        )
        return self._qdrant_client
