"""Ingest approved public web pages into shared FurnaceMind knowledge.

This module is deliberately separate from the chat-time ``web_search`` tool.
Search returns current web results without storing anything. External knowledge
ingestion is an admin/background workflow: it sends an approved public URL to
Jina Reader, stores the returned Markdown bytes on ``memory_documents`` when SQL
repositories are available, chunks the text, embeds it, and writes the chunks
into the shared MRAG knowledge collection.

Brave is used only for search. Jina Reader is used for URL reading/scraping so
messy public HTML is converted into cleaner Markdown before it reaches the
existing knowledge ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import httpx

from agents.multimodal.ingestion import process_file
from utils.settings import WebScrapeConfig, settings

logger = logging.getLogger(__name__)
_FILENAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_INLINE_WS_RE = re.compile(r"[ \t\f\v]+")
_PROVIDER_NAME = "jina_reader"
_MARKDOWN_CONTENT_MARKER = "Markdown Content:"
_TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_URL_SOURCE_RE = re.compile(r"^URL Source:\s*.+$", re.IGNORECASE | re.MULTILINE)
_READER_HEADERS = {
    "Accept": "text/plain, text/markdown;q=0.9, */*;q=0.8",
    "User-Agent": "FurnaceMind-JinaReader/1.0",
    "X-Return-Format": "markdown",
}


@dataclass(frozen=True)
class ScrapedWebPage:
    """Clean Markdown extracted from one approved public URL.

    Attributes:
        url: Original source URL requested for scraping.
        title: Best-effort page title parsed from Jina Reader output or URL.
        content: Markdown content retained for MRAG ingestion.
        provider: Scraper implementation name, currently ``"jina_reader"``.
        fetched_at: UTC timestamp when the page was read.
        sha256: SHA-256 digest of the retained Markdown content.
        content_type: MIME-like type used when storing the source bytes in SQL.
    """

    url: str
    title: str
    content: str
    provider: str
    fetched_at: datetime
    sha256: str
    content_type: str = "text/markdown; charset=utf-8"


@dataclass(frozen=True)
class WebScrapeResponse:
    """Normalized reader response for chat tools and background jobs.

    ``page`` is populated only for successful reads. ``error`` is populated for
    invalid URLs, empty Reader output, timeouts, HTTP failures, and parser
    failures. Callers can report missing-data notes without crashing the
    conversation or ingestion job.
    """

    url: str
    provider: str
    page: ScrapedWebPage | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Return True when Jina Reader produced usable page content."""
        return self.page is not None and not self.error


@dataclass(frozen=True)
class ExternalKnowledgeIngestionResult:
    """Outcome of storing one read web page in shared knowledge.

    Attributes:
        url: Source URL requested for ingestion.
        status: ``"indexed"``, ``"already_indexed"``, ``"no_content"``, or
            ``"unavailable"``.
        document_id: Stable MRAG document id from generated chunks, if indexed.
        sql_document_id: SQL ``memory_documents`` row id, when SQL persistence ran.
        filename: Generated Markdown filename used in citations and SQL metadata.
        chunk_count: Number of Qdrant chunks written.
        qdrant_collection: Knowledge collection used for vector storage.
        error: Optional missing-data or provider failure message.
    """

    url: str
    status: str
    document_id: str | None = None
    sql_document_id: str | None = None
    filename: str | None = None
    chunk_count: int = 0
    qdrant_collection: str | None = None
    error: str | None = None


class JinaReaderProvider:
    """Jina Reader client for approved public web pages.

    The provider accepts only absolute ``http`` and ``https`` source URLs, calls
    the configured Jina Reader endpoint, and converts the returned text into the
    stable ``ScrapedWebPage`` object consumed by the existing MRAG ingestion
    pipeline. Network and provider failures are converted to
    ``WebScrapeResponse(error=...)`` so chat and ingestion jobs can continue
    safely.
    """

    def __init__(
        self,
        config: WebScrapeConfig,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        """Create a provider from runtime config and an injectable HTTP client."""
        self.config = config
        self.client_factory = client_factory
        self.provider = _PROVIDER_NAME

    def fetch(self, url: str) -> WebScrapeResponse:
        """Read and normalize one public web page with Jina Reader.

        Args:
            url: Absolute ``http`` or ``https`` source URL approved for ingestion.

        Returns:
            A successful ``WebScrapeResponse`` with ``page`` populated, or an
            unavailable response with ``error`` explaining why no content was
            indexed.
        """
        clean_url = _normalize_source_url(url)
        if not clean_url:
            return WebScrapeResponse(
                url=str(url or "").strip(),
                provider=self.provider,
                error="Only absolute http/https URLs can be read by Jina Reader.",
            )
        logger.info(
            "web_scrape_request",
            extra={"provider": self.provider, "url": clean_url},
        )
        response = self._request_with_retries(clean_url)
        if response.error:
            logger.warning(
                "web_scrape_failed",
                extra={
                    "provider": self.provider,
                    "url": clean_url,
                    "error": response.error,
                },
            )
            return response
        logger.info(
            "web_scrape_response",
            extra={
                "provider": self.provider,
                "url": clean_url,
                "chars": len(response.page.content) if response.page else 0,
            },
        )
        return response

    def _request_with_retries(self, url: str) -> WebScrapeResponse:
        """Execute one Jina Reader request with bounded retry/backoff."""
        last_error = "Jina Reader request failed."
        attempts = max(0, self.config.max_retries) + 1
        reader_url = _reader_request_url(self.config.endpoint, url)
        headers = _reader_headers(self.config.api_key)
        for attempt in range(attempts):
            try:
                with self.client_factory(timeout=self.config.timeout_seconds) as client:
                    response = client.get(reader_url, headers=headers)
                    response.raise_for_status()
                    page = _parse_jina_reader_response(
                        url=url,
                        raw_text=response.text,
                        provider=self.provider,
                        max_chars=self.config.max_chars,
                    )
                    return WebScrapeResponse(url=url, provider=self.provider, page=page)
            except ValueError as exc:
                last_error = str(exc)
            except httpx.TimeoutException as exc:
                last_error = f"Jina Reader request timed out: {exc}"
            except httpx.HTTPStatusError as exc:
                status_code = getattr(exc.response, "status_code", "unknown")
                last_error = f"Jina Reader returned HTTP {status_code}."
            except httpx.RequestError as exc:
                last_error = f"Jina Reader request failed: {exc}"
            except Exception as exc:
                last_error = f"Jina Reader request failed: {type(exc).__name__}: {exc}"

            if attempt < attempts - 1:
                time.sleep(min(0.5 * (2**attempt), 3.0))

        return WebScrapeResponse(url=url, provider=self.provider, error=last_error)


def build_web_scrape_provider(
    config: WebScrapeConfig | None = None,
) -> JinaReaderProvider:
    """Build the configured external knowledge reader provider.

    Raises:
        ValueError: If ``WEB_SCRAPE_PROVIDER`` names an unsupported provider.
    """
    cfg = config or settings.web_scrape
    if cfg.provider != _PROVIDER_NAME:
        raise ValueError(f"Unsupported WEB_SCRAPE_PROVIDER: {cfg.provider}")
    return JinaReaderProvider(cfg)


def scrape_web_page(url: str) -> WebScrapeResponse:
    """Read one URL through Jina Reader without storing it."""
    try:
        provider = build_web_scrape_provider()
        return provider.fetch(url)
    except Exception as exc:
        return WebScrapeResponse(
            url=str(url or "").strip(),
            provider=_PROVIDER_NAME,
            error=f"Jina Reader unavailable: {type(exc).__name__}: {exc}",
        )


def ingest_external_knowledge_url(
    url: str,
    *,
    knowledge_store: Any,
    embedding_client: Any,
    user_id: str | None = None,
    document_repository: Any | None = None,
    chunk_repository: Any | None = None,
    provider: JinaReaderProvider | None = None,
) -> ExternalKnowledgeIngestionResult:
    """Read an approved URL and index it into shared MRAG knowledge.

    ``user_id`` is recorded as the uploader/audit owner when SQL persistence is
    available. Retrieval stays shared because FurnaceMind knowledge search uses
    active document ids and does not require user-scoped Qdrant filtering.
    """
    scrape_provider = provider or build_web_scrape_provider()
    response = scrape_provider.fetch(url)
    if response.error or response.page is None:
        return ExternalKnowledgeIngestionResult(
            url=response.url,
            status="unavailable",
            error=response.error or "No page content returned by Jina Reader.",
        )

    page = response.page
    existing_document = _find_existing_external_document(
        document_repository=document_repository,
        source_url=page.url,
        sha256=page.sha256,
    )
    if existing_document is not None:
        metadata = getattr(existing_document, "metadata_json", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        return ExternalKnowledgeIngestionResult(
            url=page.url,
            status="already_indexed",
            document_id=str(metadata.get("document_id") or "") or None,
            sql_document_id=str(getattr(existing_document, "document_id", "") or "")
            or None,
            filename=str(getattr(existing_document, "filename", "") or "") or None,
            chunk_count=int(metadata.get("chunk_count") or 0),
            qdrant_collection=str(
                getattr(existing_document, "qdrant_collection", "") or ""
            )
            or None,
            error="Source URL or content hash is already indexed in shared knowledge.",
        )

    filename = _scraped_filename(page)
    upload = _ScrapedUpload(
        page.content.encode("utf-8"),
        name=filename,
        content_type=page.content_type,
    )
    source_metadata = {
        "source_type": "web_scrape",
        "source_url": page.url,
        "title": page.title,
        "web_title": page.title,
        "provider": page.provider,
        "fetched_at": page.fetched_at.isoformat(),
        "sha256": page.sha256,
    }
    parts = process_file(
        upload,
        knowledge_store,
        embedding_client,
        user_id=user_id,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        source_metadata=source_metadata,
    )
    if not parts:
        return ExternalKnowledgeIngestionResult(
            url=page.url,
            status="no_content",
            filename=filename,
            error="Jina Reader output did not produce any indexable chunks.",
        )

    sql_document_id = _find_sql_document_id(
        document_repository=document_repository,
        user_id=user_id,
        mrag_document_id=parts[0].document_id,
    )
    return ExternalKnowledgeIngestionResult(
        url=page.url,
        status="indexed",
        document_id=parts[0].document_id,
        sql_document_id=sql_document_id,
        filename=filename,
        chunk_count=len(parts),
        qdrant_collection=getattr(knowledge_store, "collection_name", None),
    )


def ingest_external_knowledge_urls(
    urls: Iterable[str],
    *,
    knowledge_store: Any,
    embedding_client: Any,
    user_id: str | None = None,
    document_repository: Any | None = None,
    chunk_repository: Any | None = None,
    provider: JinaReaderProvider | None = None,
) -> list[ExternalKnowledgeIngestionResult]:
    """Index multiple approved URLs, returning one result per URL."""
    return [
        ingest_external_knowledge_url(
            url,
            knowledge_store=knowledge_store,
            embedding_client=embedding_client,
            user_id=user_id,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            provider=provider,
        )
        for url in urls
    ]


class _ScrapedUpload(BytesIO):
    """File-like wrapper that lets web Markdown reuse ``process_file`` unchanged."""

    def __init__(self, data: bytes, *, name: str, content_type: str) -> None:
        super().__init__(data)
        self.name = name
        self.type = content_type


def _normalize_source_url(url: str) -> str | None:
    """Return a clean absolute source URL, or ``None`` when unsupported."""
    clean_url = str(url or "").strip()
    parsed = urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return clean_url


def _reader_headers(api_key: str | None) -> dict[str, str]:
    """Return headers for a Jina Reader request without mutating defaults."""
    headers = dict(_READER_HEADERS)
    clean_key = str(api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"
    return headers


def _reader_request_url(endpoint: str, source_url: str) -> str:
    """Return the Jina Reader URL for one source URL."""
    base = str(endpoint or "https://r.jina.ai").strip().rstrip("/")
    return f"{base}/{source_url}"


def _parse_jina_reader_response(
    *,
    url: str,
    raw_text: str,
    provider: str,
    max_chars: int,
) -> ScrapedWebPage:
    """Convert Jina Reader Markdown output into stored source content."""
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Jina Reader returned empty content.")

    title = _extract_reader_title(text) or _fallback_title(url)
    body = _extract_reader_body(text)
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars].rstrip()
    if not body:
        raise ValueError("Jina Reader returned no usable page content.")

    content = f"# {title}\n\nSource URL: {url}\n\n{body}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return ScrapedWebPage(
        url=url,
        title=title,
        content=content,
        provider=provider,
        fetched_at=datetime.now(timezone.utc),
        sha256=digest,
    )


def _extract_reader_title(text: str) -> str | None:
    """Return a title from Jina metadata or the first Markdown heading."""
    metadata_match = _TITLE_RE.search(text)
    if metadata_match:
        title = _normalize_inline_text(metadata_match.group(1))
        if title:
            return title
    heading_match = _HEADING_RE.search(text)
    if heading_match:
        title = _normalize_inline_text(heading_match.group(1))
        if title:
            return title
    return None


def _extract_reader_body(text: str) -> str:
    """Return page Markdown from Jina Reader output without metadata wrappers."""
    _, marker, remainder = text.partition(_MARKDOWN_CONTENT_MARKER)
    body = remainder if marker else text
    body = _TITLE_RE.sub("", body)
    body = _URL_SOURCE_RE.sub("", body)
    body = body.replace(_MARKDOWN_CONTENT_MARKER, "")
    return _normalize_markdown(body)


def _normalize_markdown(value: str) -> str:
    """Trim noisy whitespace while preserving readable Markdown line breaks."""
    normalized_lines: list[str] = []
    blank_pending = False
    for raw_line in str(value or "").replace("\xa0", " ").splitlines():
        line = _INLINE_WS_RE.sub(" ", raw_line).rstrip()
        if not line.strip():
            blank_pending = bool(normalized_lines)
            continue
        if blank_pending:
            normalized_lines.append("")
            blank_pending = False
        normalized_lines.append(line.strip())
    return "\n".join(normalized_lines).strip()


def _normalize_inline_text(value: str) -> str:
    """Collapse a short title-like value into a single readable line."""
    return " ".join(str(value or "").split()).strip(" #")


def _fallback_title(url: str) -> str:
    """Build a readable title from the URL host/path when Reader has no title."""
    parsed = urlparse(url)
    label = parsed.netloc or url
    path = parsed.path.strip("/").rsplit("/", 1)[-1]
    if path:
        label = f"{label} {path}"
    return label.replace("-", " ").replace("_", " ").strip() or "web source"


def _scraped_filename(page: ScrapedWebPage) -> str:
    """Return a deterministic Markdown filename for a read web page."""
    slug_source = page.title or _fallback_title(page.url)
    slug = _FILENAME_SAFE_RE.sub("_", slug_source).strip("._-").lower()
    slug = slug[:80] or "web_source"
    return f"web_{page.sha256[:12]}_{slug}.md"


def _find_existing_external_document(
    *,
    document_repository: Any | None,
    source_url: str,
    sha256: str,
) -> Any | None:
    """Return an active SQL document matching a source URL or content digest."""
    if document_repository is None or not hasattr(
        document_repository, "list_documents"
    ):
        return None
    try:
        documents = document_repository.list_documents(user_id=None, active_only=True)
    except TypeError:
        try:
            documents = document_repository.list_documents(user_id=None)
        except Exception:
            return None
    except Exception:
        return None

    normalized_url = _normalize_source_url(source_url) or str(source_url or "").strip()
    for document in documents:
        metadata = getattr(document, "metadata_json", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata_url = _normalize_source_url(str(metadata.get("source_url") or ""))
        metadata_sha = str(metadata.get("sha256") or "").strip().lower()
        row_sha = str(getattr(document, "sha256", "") or "").strip().lower()
        if metadata_url and metadata_url == normalized_url:
            return document
        if sha256 and sha256.lower() in {metadata_sha, row_sha}:
            return document
    return None


def _find_sql_document_id(
    *,
    document_repository: Any | None,
    user_id: str | None,
    mrag_document_id: str,
) -> str | None:
    """Best-effort lookup of the SQL row created for an ingested page."""
    if document_repository is None or not hasattr(
        document_repository, "list_documents"
    ):
        return None
    try:
        for document in document_repository.list_documents(user_id=user_id):
            metadata = getattr(document, "metadata_json", None)
            if isinstance(metadata, dict) and str(metadata.get("document_id")) == str(
                mrag_document_id
            ):
                return str(getattr(document, "document_id", "") or "") or None
    except Exception:
        return None
    return None
