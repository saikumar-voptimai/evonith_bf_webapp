"""Provider-neutral web search support for FurnaceMind tools.

The FurnaceMind agent exposes a single ``web_search`` tool to the model. This
module keeps the provider-specific HTTP details behind a small interface so the
current Brave Search integration can be replaced later without changing prompt
routing or the tool registry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from utils.settings import WebSearchConfig, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    """One normalized web search result returned to FurnaceMind.

    Attributes:
        title: Human-readable result title.
        url: Canonical result URL used for source citation.
        snippet: Short provider summary or page excerpt.
    """

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class WebSearchResponse:
    """Normalized response from any web search provider.

    Attributes:
        query: Original search query.
        provider: Provider name, such as ``"brave"``.
        results: Ordered normalized search results.
        error: Optional provider/configuration error. When present, the tool
            should return a graceful unavailable message instead of raising.
    """

    query: str
    provider: str
    results: list[WebSearchResult]
    error: str | None = None

    def to_tool_text(self) -> str:
        """Render search results in a stable format for LLM tool consumption."""
        lines = [
            "WEB_SEARCH_RESULTS",
            f"Provider: {self.provider}",
            f"Query: {self.query}",
        ]
        if self.error:
            lines.extend(
                [
                    "Status: unavailable",
                    "Missing-data notes:",
                    f"- {self.error}",
                ]
            )
            return "\n".join(lines)

        if not self.results:
            lines.extend(
                [
                    "Status: no_results",
                    "Results: none",
                    "Missing-data notes:",
                    "- The search provider returned no results for this query.",
                ]
            )
            return "\n".join(lines)

        lines.extend(["Status: ok", "Results:"])
        for index, result in enumerate(self.results, start=1):
            lines.extend(
                [
                    f"{index}. {result.title}",
                    f"   URL: {result.url}",
                    f"   Snippet: {result.snippet}",
                ]
            )
        lines.extend(
            [
                "Citation guidance:",
                "- Use the URL fields above when citing web-derived claims.",
            ]
        )
        return "\n".join(lines)


class BraveSearchProvider:
    """HTTP client for Brave Search's web search API.

    The provider returns normalized ``WebSearchResponse`` objects and never
    raises provider/network failures to callers. Retries are intentionally small
    and bounded by ``WebSearchConfig.max_retries`` so a slow search service does
    not block the FurnaceMind conversation for too long.
    """

    def __init__(
        self,
        config: WebSearchConfig,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        """Create a Brave provider from runtime configuration.

        Args:
            config: Search settings loaded from environment variables.
            client_factory: Injectable HTTP client factory used by tests.
        """
        self.config = config
        self.client_factory = client_factory

    def search(self, *, query: str, limit: int | None = None) -> WebSearchResponse:
        """Search Brave and return normalized results.

        Args:
            query: Search query text.
            limit: Optional per-call result cap. The effective value is clamped
                to the configured maximum.

        Returns:
            ``WebSearchResponse`` containing results, no-result status, or a
            graceful unavailable error.
        """
        clean_query = str(query or "").strip()
        effective_limit = _effective_limit(limit, self.config.max_results)
        if not clean_query:
            return WebSearchResponse(
                query=clean_query,
                provider=self.config.provider,
                results=[],
                error="Search query is empty.",
            )
        if not self.config.api_key:
            return WebSearchResponse(
                query=clean_query,
                provider=self.config.provider,
                results=[],
                error="BRAVE_SEARCH_API_KEY or WEB_SEARCH_API_KEY is not configured.",
            )

        logger.info(
            "web_search_request",
            extra={
                "provider": self.config.provider,
                "query": clean_query,
                "limit": effective_limit,
            },
        )
        response = self._request_with_retries(clean_query, effective_limit)
        if response.error:
            logger.warning(
                "web_search_failed",
                extra={
                    "provider": self.config.provider,
                    "query": clean_query,
                    "error": response.error,
                },
            )
            return response
        logger.info(
            "web_search_response",
            extra={
                "provider": self.config.provider,
                "query": clean_query,
                "result_count": len(response.results),
            },
        )
        return response

    def _request_with_retries(self, query: str, limit: int) -> WebSearchResponse:
        """Execute one Brave request with bounded retry/backoff."""
        last_error = "Search provider failed."
        attempts = max(0, self.config.max_retries) + 1
        for attempt in range(attempts):
            try:
                with self.client_factory(timeout=self.config.timeout_seconds) as client:
                    response = client.get(
                        self.config.endpoint,
                        headers={
                            "Accept": "application/json",
                            "X-Subscription-Token": self.config.api_key or "",
                        },
                        params={
                            "q": query,
                            "count": limit,
                            "text_decorations": "false",
                        },
                    )
                    response.raise_for_status()
                    return WebSearchResponse(
                        query=query,
                        provider=self.config.provider,
                        results=_parse_brave_results(response.json(), limit),
                    )
            except httpx.TimeoutException as exc:
                last_error = f"Search request timed out: {exc}"
            except httpx.HTTPStatusError as exc:
                status_code = getattr(exc.response, "status_code", "unknown")
                last_error = f"Search provider returned HTTP {status_code}."
            except httpx.RequestError as exc:
                last_error = f"Search request failed: {exc}"
            except Exception as exc:  # provider payload or client-factory errors
                last_error = f"Search provider failed: {type(exc).__name__}: {exc}"

            if attempt < attempts - 1:
                time.sleep(min(0.25 * (2**attempt), 2.0))

        return WebSearchResponse(
            query=query,
            provider=self.config.provider,
            results=[],
            error=last_error,
        )


def _effective_limit(limit: int | None, configured_limit: int) -> int:
    """Return a safe result limit for one search request."""
    try:
        requested = int(limit) if limit is not None else int(configured_limit)
    except (TypeError, ValueError):
        requested = int(configured_limit)
    return max(1, min(requested, int(configured_limit), 10))


def _clean_text(value: Any) -> str:
    """Convert provider text fields into single-line strings."""
    return " ".join(str(value or "").split())


def _parse_brave_results(payload: dict[str, Any], limit: int) -> list[WebSearchResult]:
    """Normalize Brave Search JSON into FurnaceMind search results."""
    raw_results = (payload.get("web") or {}).get("results") or []
    results: list[WebSearchResult] = []
    for raw in raw_results:
        title = _clean_text(raw.get("title"))
        url = _clean_text(raw.get("url"))
        snippet = _clean_text(raw.get("description") or raw.get("snippet"))
        if not title or not url:
            continue
        results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break
    return results


def build_web_search_provider(
    config: WebSearchConfig | None = None,
) -> BraveSearchProvider:
    """Build the configured web search provider.

    Raises:
        ValueError: If ``WEB_SEARCH_PROVIDER`` names an unsupported provider.
    """
    cfg = config or settings.web_search
    if cfg.provider != "brave":
        raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER: {cfg.provider}")
    return BraveSearchProvider(cfg)


def search_web(query: str, *, limit: int | None = None) -> str:
    """Run web search through the configured provider and render tool output."""
    try:
        provider = build_web_search_provider()
        response = provider.search(query=query, limit=limit)
    except Exception as exc:
        cfg = settings.web_search
        response = WebSearchResponse(
            query=str(query or "").strip(),
            provider=cfg.provider,
            results=[],
            error=f"Search provider unavailable: {type(exc).__name__}: {exc}",
        )
        logger.warning(
            "web_search_failed",
            extra={
                "provider": cfg.provider,
                "query": str(query or "").strip(),
                "error": response.error,
            },
        )
    return response.to_tool_text()
