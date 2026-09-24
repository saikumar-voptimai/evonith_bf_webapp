"""Provider-neutral web search support for FurnaceMind tools.

FurnaceMind exposes one ``web_search`` tool to the model. This module keeps
provider-specific authentication, request formats, and response parsing behind
adapters that all return the same normalized result type. Changing providers
therefore requires configuration only; prompts and tool dispatch stay stable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable, Protocol
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

import httpx

from utils.settings import (
    WebSearchConfig,
    normalize_web_search_provider_name,
    settings,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    """One normalized result returned by any configured search provider.

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
        provider: Canonical provider name, such as ``"brave"`` or ``"tavily"``.
        results: Ordered normalized search results.
        error: Optional provider/configuration error. Provider failures are
            represented here instead of escaping into the conversation flow.
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


class WebSearchProvider(Protocol):
    """Contract implemented by every FurnaceMind web-search adapter."""

    provider_name: str

    def configuration_error(self) -> str | None:
        """Return why this provider is unavailable, or ``None`` when ready."""

    def search(self, *, query: str, limit: int | None = None) -> WebSearchResponse:
        """Search the provider and return normalized, source-citable results."""


WebSearchProviderFactory = Callable[[WebSearchConfig], WebSearchProvider]


class BaseWebSearchAdapter:
    """Share validation, logging, retries, and graceful failures across adapters.

    Subclasses only define provider metadata and implement ``_perform_search``.
    This keeps operational behavior consistent when the configured provider is
    changed and prevents network or payload failures from crashing FurnaceMind.
    """

    provider_name = ""
    requires_api_key = True
    api_key_environment = "WEB_SEARCH_API_KEY"
    endpoint_environment = "WEB_SEARCH_ENDPOINT"

    def __init__(
        self,
        config: WebSearchConfig,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        """Create an adapter from runtime configuration and an HTTP client."""
        self.config = config
        self.client_factory = client_factory

    def configuration_error(self) -> str | None:
        """Return a provider-specific setup error when required values are absent."""
        if not str(self.config.endpoint or "").strip():
            return (
                f"The configured '{self.provider_name}' web-search provider is "
                "missing its endpoint. Set "
                f"{self.endpoint_environment} or WEB_SEARCH_ENDPOINT."
            )
        if self.requires_api_key and not self.config.api_key:
            return (
                f"The configured '{self.provider_name}' web-search provider is "
                "missing its API key. Set "
                f"{self.api_key_environment} or WEB_SEARCH_API_KEY."
            )
        return None

    def search(self, *, query: str, limit: int | None = None) -> WebSearchResponse:
        """Search the provider using common validation and failure handling."""
        clean_query = str(query or "").strip()
        effective_limit = _effective_limit(limit, self.config.max_results)
        if not clean_query:
            return WebSearchResponse(
                query=clean_query,
                provider=self.provider_name,
                results=[],
                error="Search query is empty.",
            )

        configuration_error = self.configuration_error()
        if configuration_error:
            return WebSearchResponse(
                query=clean_query,
                provider=self.provider_name,
                results=[],
                error=configuration_error,
            )

        logger.info(
            "web_search_request",
            extra={
                "provider": self.provider_name,
                "query": clean_query,
                "limit": effective_limit,
            },
        )
        response = self._request_with_retries(clean_query, effective_limit)
        if response.error:
            logger.warning(
                "web_search_failed",
                extra={
                    "provider": self.provider_name,
                    "query": clean_query,
                    "error": response.error,
                },
            )
        else:
            logger.info(
                "web_search_response",
                extra={
                    "provider": self.provider_name,
                    "query": clean_query,
                    "result_count": len(response.results),
                },
            )
        return response

    def _request_with_retries(self, query: str, limit: int) -> WebSearchResponse:
        """Execute one provider request with configured retry and backoff limits."""
        last_error = "Search provider failed."
        attempts = max(0, self.config.max_retries) + 1
        for attempt in range(attempts):
            try:
                with self.client_factory(timeout=self.config.timeout_seconds) as client:
                    results = self._perform_search(client, query, limit)
                return WebSearchResponse(
                    query=query,
                    provider=self.provider_name,
                    results=results[:limit],
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
            provider=self.provider_name,
            results=[],
            error=last_error,
        )

    def _perform_search(
        self,
        client: httpx.Client,
        query: str,
        limit: int,
    ) -> list[WebSearchResult]:
        """Execute and normalize one provider request."""
        raise NotImplementedError


class BraveSearchAdapter(BaseWebSearchAdapter):
    """Adapt the Brave Search API to FurnaceMind's normalized contract."""

    provider_name = "brave"
    api_key_environment = "BRAVE_SEARCH_API_KEY"
    endpoint_environment = "BRAVE_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.get(
            self.config.endpoint,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.config.api_key or "",
            },
            params={"q": query, "count": limit, "text_decorations": "false"},
        )
        response.raise_for_status()
        return _parse_brave_results(response.json(), limit)


class TavilySearchAdapter(BaseWebSearchAdapter):
    """Adapt Tavily's search endpoint to FurnaceMind search results."""

    provider_name = "tavily"
    api_key_environment = "TAVILY_API_KEY"
    endpoint_environment = "TAVILY_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.post(
            self.config.endpoint,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key or ''}",
                "Content-Type": "application/json",
            },
            json={"query": query, "max_results": limit},
        )
        response.raise_for_status()
        return _normalize_result_rows(
            response.json().get("results") or [],
            limit,
            title_keys=("title",),
            url_keys=("url",),
            snippet_keys=("content", "snippet", "raw_content"),
        )


class ExaSearchAdapter(BaseWebSearchAdapter):
    """Adapt Exa search and highlight content to normalized results."""

    provider_name = "exa"
    api_key_environment = "EXA_API_KEY"
    endpoint_environment = "EXA_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.post(
            self.config.endpoint,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key or "",
            },
            json={
                "query": query,
                "numResults": limit,
                "contents": {"highlights": True},
            },
        )
        response.raise_for_status()
        return _normalize_result_rows(
            response.json().get("results") or [],
            limit,
            title_keys=("title",),
            url_keys=("url",),
            snippet_keys=("summary", "highlights", "text"),
        )


class ParallelSearchAdapter(BaseWebSearchAdapter):
    """Adapt Parallel Search API excerpts to normalized search results."""

    provider_name = "parallel"
    api_key_environment = "PARALLEL_API_KEY"
    endpoint_environment = "PARALLEL_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.post(
            self.config.endpoint,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key or "",
            },
            json={"objective": query, "search_queries": [query]},
        )
        response.raise_for_status()
        return _normalize_result_rows(
            response.json().get("results") or [],
            limit,
            title_keys=("title", "name"),
            url_keys=("url", "link"),
            snippet_keys=("excerpts", "snippet", "content"),
        )


class FirecrawlSearchAdapter(BaseWebSearchAdapter):
    """Adapt Firecrawl's web-search response without using its scrape flow."""

    provider_name = "firecrawl"
    api_key_environment = "FIRECRAWL_API_KEY"
    endpoint_environment = "FIRECRAWL_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.post(
            self.config.endpoint,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key or ''}",
                "Content-Type": "application/json",
            },
            json={"query": query, "limit": limit, "sources": ["web"]},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        rows = data.get("web") if isinstance(data, dict) else data
        return _normalize_result_rows(
            rows or [],
            limit,
            title_keys=("title",),
            url_keys=("url",),
            snippet_keys=("description", "snippet", "markdown"),
        )


class SerperSearchAdapter(BaseWebSearchAdapter):
    """Adapt Serper's Google organic search results to the common contract."""

    provider_name = "serper"
    api_key_environment = "SERPER_API_KEY"
    endpoint_environment = "SERPER_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.post(
            self.config.endpoint,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-API-KEY": self.config.api_key or "",
            },
            json={"q": query, "num": limit},
        )
        response.raise_for_status()
        return _normalize_result_rows(
            response.json().get("organic") or [],
            limit,
            title_keys=("title",),
            url_keys=("link", "url"),
            snippet_keys=("snippet", "description"),
        )


class DuckDuckGoSearchAdapter(BaseWebSearchAdapter):
    """Parse DuckDuckGo's public HTML results without requiring an API key.

    DuckDuckGo does not expose a supported full web-results JSON API. This
    adapter therefore targets the lightweight HTML endpoint and may require
    maintenance if that page structure changes.
    """

    provider_name = "duckduckgo"
    requires_api_key = False
    endpoint_environment = "DUCKDUCKGO_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.get(
            self.config.endpoint,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "FurnaceMind/1.0 web-search",
            },
            params={"q": query},
        )
        response.raise_for_status()
        parser = _DuckDuckGoHTMLResultParser()
        parser.feed(response.text)
        return parser.results[:limit]


class SearXNGSearchAdapter(BaseWebSearchAdapter):
    """Adapt a configured self-hosted SearXNG JSON search endpoint."""

    provider_name = "searxng"
    requires_api_key = False
    endpoint_environment = "SEARXNG_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.get(
            _endpoint_with_default_path(self.config.endpoint, "/search"),
            headers=_optional_bearer_headers(self.config.api_key),
            params={"q": query, "format": "json"},
        )
        response.raise_for_status()
        return _normalize_result_rows(
            response.json().get("results") or [],
            limit,
            title_keys=("title",),
            url_keys=("url",),
            snippet_keys=("content", "snippet"),
        )


class WhoogleSearchAdapter(BaseWebSearchAdapter):
    """Adapt a configured self-hosted Whoogle JSON search endpoint."""

    provider_name = "whoogle"
    requires_api_key = False
    endpoint_environment = "WHOOGLE_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.get(
            _endpoint_with_default_path(self.config.endpoint, "/search"),
            headers=_optional_bearer_headers(self.config.api_key),
            params={"q": query, "format": "json"},
        )
        response.raise_for_status()
        return _normalize_result_rows(
            response.json().get("results") or [],
            limit,
            title_keys=("text", "title"),
            url_keys=("href", "url"),
            snippet_keys=("description", "snippet"),
        )


class YaCySearchAdapter(BaseWebSearchAdapter):
    """Adapt a configured self-hosted YaCy JSON search endpoint."""

    provider_name = "yacy"
    requires_api_key = False
    endpoint_environment = "YACY_SEARCH_ENDPOINT"

    def _perform_search(
        self, client: httpx.Client, query: str, limit: int
    ) -> list[WebSearchResult]:
        response = client.get(
            _endpoint_with_default_path(self.config.endpoint, "/yacysearch.json"),
            headers=_optional_bearer_headers(self.config.api_key),
            params={"query": query, "maximumRecords": limit},
        )
        response.raise_for_status()
        return _normalize_result_rows(
            _extract_yacy_rows(response.json()),
            limit,
            title_keys=("title",),
            url_keys=("link", "url"),
            snippet_keys=("description", "snippet"),
        )


def _effective_limit(limit: int | None, configured_limit: int) -> int:
    """Return a safe result limit for one search request."""
    try:
        requested = int(limit) if limit is not None else int(configured_limit)
    except (TypeError, ValueError):
        requested = int(configured_limit)
    return max(1, min(requested, int(configured_limit), 10))


def _clean_text(value: Any) -> str:
    """Convert provider text fields, including lists, into one clean line."""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(_clean_text(item) for item in value if item is not None)
    return " ".join(str(value or "").split())


def _first_row_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty normalized value from candidate payload keys."""
    for key in keys:
        value = _clean_text(row.get(key))
        if value:
            return value
    return ""


def _normalize_result_rows(
    rows: list[dict[str, Any]],
    limit: int,
    *,
    title_keys: tuple[str, ...],
    url_keys: tuple[str, ...],
    snippet_keys: tuple[str, ...],
) -> list[WebSearchResult]:
    """Normalize provider rows using ordered candidate field names."""
    results: list[WebSearchResult] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = _first_row_value(row, title_keys)
        url = _first_row_value(row, url_keys)
        snippet = _first_row_value(row, snippet_keys)
        if not title or not url:
            continue
        results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break
    return results


def _parse_brave_results(payload: dict[str, Any], limit: int) -> list[WebSearchResult]:
    """Normalize Brave Search JSON into FurnaceMind search results."""
    return _normalize_result_rows(
        (payload.get("web") or {}).get("results") or [],
        limit,
        title_keys=("title",),
        url_keys=("url",),
        snippet_keys=("description", "snippet"),
    )


def _optional_bearer_headers(api_key: str | None) -> dict[str, str]:
    """Build JSON headers and add optional auth for protected self-hosted APIs."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _endpoint_with_default_path(endpoint: str, default_path: str) -> str:
    """Append an API path when a self-hosted endpoint only contains its origin."""
    parsed = urlsplit(str(endpoint or "").strip())
    if parsed.path not in ("", "/"):
        return endpoint
    return urlunsplit(
        (parsed.scheme, parsed.netloc, default_path, parsed.query, parsed.fragment)
    )


def _extract_yacy_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract YaCy's RSS-like result rows across supported JSON layouts."""
    direct_items = payload.get("items")
    if isinstance(direct_items, list):
        return direct_items

    channel = payload.get("channel")
    if isinstance(channel, dict) and isinstance(channel.get("items"), list):
        return channel["items"]

    rows: list[dict[str, Any]] = []
    for item in payload.get("channels") or []:
        if isinstance(item, dict) and isinstance(item.get("items"), list):
            rows.extend(item["items"])
    return rows


def _normalize_duckduckgo_url(url: str) -> str:
    """Resolve DuckDuckGo redirect links to their external destination URL."""
    candidate = str(url or "").strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    parsed = urlsplit(candidate)
    redirected = parse_qs(parsed.query).get("uddg")
    if redirected:
        return unquote(redirected[0])
    return candidate


class _DuckDuckGoHTMLResultParser(HTMLParser):
    """Extract result titles, links, and snippets from DuckDuckGo HTML output."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[WebSearchResult] = []
        self._capture: str | None = None
        self._capture_tag: str | None = None
        self._buffer: list[str] = []
        self._result_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start capturing recognized result title or snippet elements."""
        attributes = dict(attrs)
        classes = set(str(attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._capture = "title"
            self._capture_tag = tag
            self._buffer = []
            self._result_url = _normalize_duckduckgo_url(
                str(attributes.get("href") or "")
            )
        elif "result__snippet" in classes:
            self._capture = "snippet"
            self._capture_tag = tag
            self._buffer = []

    def handle_data(self, data: str) -> None:
        """Collect visible text while inside a recognized result element."""
        if self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Finish the active result field when its containing element closes."""
        if not self._capture or tag != self._capture_tag:
            return
        value = _clean_text(" ".join(self._buffer))
        if self._capture == "title" and value and self._result_url:
            self.results.append(
                WebSearchResult(title=value, url=self._result_url, snippet="")
            )
        elif self._capture == "snippet" and value and self.results:
            previous = self.results[-1]
            self.results[-1] = WebSearchResult(
                title=previous.title,
                url=previous.url,
                snippet=value,
            )
        self._capture = None
        self._capture_tag = None
        self._buffer = []


_WEB_SEARCH_PROVIDER_FACTORIES: dict[str, WebSearchProviderFactory] = {
    BraveSearchAdapter.provider_name: BraveSearchAdapter,
    TavilySearchAdapter.provider_name: TavilySearchAdapter,
    ExaSearchAdapter.provider_name: ExaSearchAdapter,
    ParallelSearchAdapter.provider_name: ParallelSearchAdapter,
    FirecrawlSearchAdapter.provider_name: FirecrawlSearchAdapter,
    SerperSearchAdapter.provider_name: SerperSearchAdapter,
    DuckDuckGoSearchAdapter.provider_name: DuckDuckGoSearchAdapter,
    SearXNGSearchAdapter.provider_name: SearXNGSearchAdapter,
    WhoogleSearchAdapter.provider_name: WhoogleSearchAdapter,
    YaCySearchAdapter.provider_name: YaCySearchAdapter,
}


def register_web_search_provider(
    provider_name: str,
    factory: WebSearchProviderFactory,
) -> None:
    """Register or replace an adapter factory under a normalized provider name."""
    normalized_name = normalize_web_search_provider_name(provider_name)
    if not normalized_name:
        raise ValueError("Web-search provider name cannot be empty.")
    _WEB_SEARCH_PROVIDER_FACTORIES[normalized_name] = factory


def build_web_search_provider(
    config: WebSearchConfig | None = None,
) -> WebSearchProvider:
    """Build the configured adapter or reject an unsupported provider name."""
    cfg = config or settings.web_search
    provider_name = normalize_web_search_provider_name(cfg.provider)
    factory = _WEB_SEARCH_PROVIDER_FACTORIES.get(provider_name)
    if factory is None:
        supported = ", ".join(sorted(_WEB_SEARCH_PROVIDER_FACTORIES))
        raise ValueError(
            f"Unsupported WEB_SEARCH_PROVIDER: {cfg.provider}. "
            f"Supported providers: {supported}."
        )
    return factory(cfg)


def web_search_configuration_error(
    config: WebSearchConfig | None = None,
) -> str | None:
    """Return why the configured search adapter cannot run, if applicable."""
    try:
        provider = build_web_search_provider(config)
    except Exception as exc:
        return f"Web search is unavailable: {exc}"
    return provider.configuration_error()


def search_web(query: str, *, limit: int | None = None) -> str:
    """Run the configured provider and render normalized tool output."""
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


# Preserve the original public class name used by older imports.
BraveSearchProvider = BraveSearchAdapter
