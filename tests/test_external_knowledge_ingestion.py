from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import httpx

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

# Other tests may install lightweight module doubles during collection. These
# tests need the real ingestion module and web-ingestion module.
sys.modules.pop("agents.multimodal.ingestion", None)
sys.modules.pop("agents.furnacemind.web_ingestion", None)

from agents.furnacemind import web_ingestion as web_ingestion_module  # noqa: E402
from agents.furnacemind.web_ingestion import (  # noqa: E402
    JinaReaderProvider,
    ingest_external_knowledge_url,
)
from utils.settings import WebScrapeConfig  # noqa: E402


class _FakeResponse:
    """Small response double for Jina Reader provider tests."""

    def __init__(
        self,
        text: str = "",
        *,
        headers: dict[str, str] | None = None,
        status_error: Exception | None = None,
    ):
        self.text = text
        self.headers = headers or {}
        self.status_error = status_error

    def raise_for_status(self) -> None:
        """Raise a configured HTTP-status error, if one was supplied."""
        if self.status_error:
            raise self.status_error


class _FakeClient:
    """Context-manager HTTP client that replays configured responses/errors."""

    def __init__(self, factory: "_FakeClientFactory", *, timeout: float):
        self.factory = factory
        self.timeout = timeout

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def get(self, url: str, *, headers: dict | None = None) -> _FakeResponse:
        """Record request details and return or raise the next configured item."""
        self.factory.calls.append(
            {
                "url": url,
                "headers": headers or {},
                "timeout": self.timeout,
            }
        )
        item = self.factory.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClientFactory:
    """Factory object matching the ``httpx.Client`` constructor shape."""

    def __init__(self, items: list[_FakeResponse | Exception]):
        self.items = list(items)
        self.calls: list[dict] = []

    def __call__(self, *, timeout: float) -> _FakeClient:
        return _FakeClient(self, timeout=timeout)


class _FakeEmbedding:
    """Deterministic text embedding client for ingestion tests."""

    dimension = 3

    def embed_text(self, text: str, *, input_type: str | None = None) -> list[float]:
        assert text.strip()
        assert input_type == "document"
        return [0.1, 0.2, 0.3]


class _FakeQdrantClient:
    """Qdrant client stand-in recording upsert payloads."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict] = []

    def upsert(self, **kwargs) -> None:
        self.upsert_calls.append(kwargs)


class _FakeDocumentRepository:
    """SQL repository stand-in recording document and raw-byte writes."""

    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.store_calls: list[dict] = []
        self.documents: list[SimpleNamespace] = []

    def list_documents(self, *, user_id: str | None = None, active_only: bool = True):
        return list(self.documents)

    def create_document(self, **kwargs) -> object:
        self.create_calls.append(kwargs)
        document = SimpleNamespace(
            document_id="sql-doc-1",
            metadata_json={
                "document_id": kwargs["metadata"]["document_id"],
                "qdrant_point_ids": kwargs["qdrant_point_ids"],
            },
        )
        self.documents.append(document)
        return document

    def store_document_file(self, **kwargs) -> None:
        self.store_calls.append(kwargs)


class _FakeChunkRepository:
    """SQL chunk repository stand-in recording chunk writes."""

    def __init__(self) -> None:
        self.create_calls: list[dict] = []

    def create_chunks(self, **kwargs) -> int:
        self.create_calls.append(kwargs)
        return len(kwargs["parts"])


def _config(**overrides) -> WebScrapeConfig:  # noqa: ANN003
    """Return a Jina Reader config suitable for unit tests."""
    values = {
        "provider": "jina_reader",
        "api_key": None,
        "endpoint": "https://r.jina.ai",
        "timeout_seconds": 3.0,
        "max_retries": 0,
        "max_chars": 1000,
    }
    values.update(overrides)
    return WebScrapeConfig(**values)


def test_jina_reader_provider_returns_normalized_markdown_page() -> None:
    """Jina Reader output should become clean source text with metadata."""
    factory = _FakeClientFactory(
        [
            _FakeResponse(
                "Title: Blast Furnace Coke Quality\n\n"
                "URL Source: https://example.com/bf-coke\n\n"
                "Markdown Content:\n# Coke quality\n\n"
                "CSR and CRI affect blast furnace fuel demand."
            )
        ]
    )
    provider = JinaReaderProvider(_config(), client_factory=factory)

    response = provider.fetch("https://example.com/bf-coke")

    assert response.error is None
    assert response.page is not None
    assert response.page.title == "Blast Furnace Coke Quality"
    assert "CSR and CRI" in response.page.content
    assert "URL Source:" not in response.page.content
    assert response.page.provider == "jina_reader"
    assert response.page.sha256
    assert factory.calls[0]["url"] == "https://r.jina.ai/https://example.com/bf-coke"
    assert factory.calls[0]["headers"]["X-Return-Format"] == "markdown"
    assert "Authorization" not in factory.calls[0]["headers"]


def test_jina_reader_provider_rejects_non_http_urls() -> None:
    """Only absolute public web URLs should be accepted for scraping."""
    provider = JinaReaderProvider(_config())

    response = provider.fetch("file:///etc/passwd")

    assert response.page is None
    assert "http/https" in response.error


def test_jina_reader_provider_retries_timeout_and_returns_error(monkeypatch) -> None:
    """Timeouts should retry within the configured budget and then degrade."""
    monkeypatch.setattr(web_ingestion_module.time, "sleep", lambda _: None)
    factory = _FakeClientFactory(
        [
            httpx.TimeoutException("slow page"),
            httpx.TimeoutException("still slow"),
        ]
    )
    provider = JinaReaderProvider(_config(max_retries=1), client_factory=factory)

    response = provider.fetch("https://example.com/slow")

    assert len(factory.calls) == 2
    assert response.page is None
    assert "timed out" in response.error


def test_jina_reader_provider_adds_authorization_header() -> None:
    """Configured Jina API keys should be sent as bearer tokens."""
    factory = _FakeClientFactory(
        [
            _FakeResponse(
                "Title: BF Article\n\nMarkdown Content:\nBlast furnace operating notes."
            )
        ]
    )
    provider = JinaReaderProvider(
        _config(api_key="jina-test-key"), client_factory=factory
    )

    response = provider.fetch("https://example.com/bf")

    assert response.ok
    assert factory.calls[0]["headers"]["Authorization"] == "Bearer jina-test-key"


def test_ingest_external_knowledge_url_writes_shared_knowledge_metadata() -> None:
    """Scraped pages should reuse MRAG SQL, chunk, and Qdrant persistence."""
    provider = JinaReaderProvider(
        _config(),
        client_factory=_FakeClientFactory(
            [
                _FakeResponse(
                    "Title: Cast House SOP\n\n"
                    "URL Source: https://example.com/cast-house-sop\n\n"
                    "Markdown Content:\n"
                    + "Runner dryness and taphole checks are required "
                    "before tapping.\n" * 40,
                )
            ]
        ),
    )
    qdrant_client = _FakeQdrantClient()
    store = SimpleNamespace(
        client=qdrant_client,
        collection_name="furnacemind_knowledge",
        embedding_dim=3,
    )
    document_repository = _FakeDocumentRepository()
    chunk_repository = _FakeChunkRepository()

    result = ingest_external_knowledge_url(
        "https://example.com/cast-house-sop",
        knowledge_store=store,
        embedding_client=_FakeEmbedding(),
        user_id="user-1",
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        provider=provider,
    )

    assert result.status == "indexed"
    assert result.chunk_count > 0
    assert result.sql_document_id == "sql-doc-1"
    assert result.qdrant_collection == "furnacemind_knowledge"
    assert result.filename.endswith(".md")

    point = qdrant_client.upsert_calls[0]["points"][0]
    assert point.payload["source_type"] == "web_scrape"
    assert point.payload["source_url"] == "https://example.com/cast-house-sop"
    assert point.payload["title"] == "Cast House SOP"

    create_call = document_repository.create_calls[0]
    assert create_call["metadata"]["source_type"] == "web_scrape"
    assert create_call["metadata"]["source_url"] == "https://example.com/cast-house-sop"
    assert create_call["metadata"]["title"] == "Cast House SOP"

    store_call = document_repository.store_calls[0]
    assert store_call["content_type"] == "text/markdown; charset=utf-8"
    assert b"Runner dryness" in store_call["file_bytes"]
    assert chunk_repository.create_calls


def test_ingest_external_knowledge_url_does_not_raise_on_empty_page() -> None:
    """Empty Jina Reader output should return an unavailable ingestion result."""
    provider = JinaReaderProvider(
        _config(),
        client_factory=_FakeClientFactory([_FakeResponse("")]),
    )
    store = SimpleNamespace(
        client=_FakeQdrantClient(),
        collection_name="furnacemind_knowledge",
        embedding_dim=3,
    )

    result = ingest_external_knowledge_url(
        "https://example.com/empty",
        knowledge_store=store,
        embedding_client=_FakeEmbedding(),
        provider=provider,
    )

    assert result.status == "unavailable"
    assert "empty content" in result.error


def test_ingest_external_knowledge_url_skips_existing_source_url() -> None:
    """Existing source URLs should not be re-embedded into Qdrant."""
    provider = JinaReaderProvider(
        _config(),
        client_factory=_FakeClientFactory(
            [
                _FakeResponse(
                    "Title: Cast House SOP\n\n"
                    "URL Source: https://example.com/cast-house-sop\n\n"
                    "Markdown Content:\nRunner dryness and taphole checks are required "
                    "before tapping.",
                )
            ]
        ),
    )
    qdrant_client = _FakeQdrantClient()
    store = SimpleNamespace(
        client=qdrant_client,
        collection_name="furnacemind_knowledge",
        embedding_dim=3,
    )
    document_repository = _FakeDocumentRepository()
    document_repository.documents.append(
        SimpleNamespace(
            document_id="sql-existing",
            filename="web_existing.md",
            qdrant_collection="furnacemind_knowledge",
            metadata_json={
                "document_id": "doc-existing",
                "source_url": "https://example.com/cast-house-sop",
                "chunk_count": 2,
            },
            sha256=None,
        )
    )

    result = ingest_external_knowledge_url(
        "https://example.com/cast-house-sop",
        knowledge_store=store,
        embedding_client=_FakeEmbedding(),
        document_repository=document_repository,
        provider=provider,
    )

    assert result.status == "already_indexed"
    assert result.sql_document_id == "sql-existing"
    assert result.document_id == "doc-existing"
    assert result.chunk_count == 2
    assert qdrant_client.upsert_calls == []
    assert document_repository.create_calls == []
