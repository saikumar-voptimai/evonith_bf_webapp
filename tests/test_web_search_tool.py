from __future__ import annotations

import sys
import types

import httpx
import pytest


def _install_plotly_stubs() -> None:
    """Install lightweight Plotly stubs before importing furnace_tools."""
    plotly = types.ModuleType("plotly")
    express = types.ModuleType("plotly.express")
    graph_objects = types.ModuleType("plotly.graph_objects")
    subplots = types.ModuleType("plotly.subplots")
    subplots.make_subplots = lambda *args, **kwargs: None
    plotly.express = express
    plotly.graph_objects = graph_objects
    plotly.subplots = subplots
    sys.modules.setdefault("plotly", plotly)
    sys.modules.setdefault("plotly.express", express)
    sys.modules.setdefault("plotly.graph_objects", graph_objects)
    sys.modules.setdefault("plotly.subplots", subplots)


def _install_langchain_tool_stub() -> None:
    """Install a no-op LangChain tool decorator for import-only tests."""
    langchain = types.ModuleType("langchain")
    tools = types.ModuleType("langchain.tools")

    def tool(func=None, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        if func is None:
            return lambda wrapped: wrapped
        return func

    tools.tool = tool
    langchain.tools = tools
    sys.modules.setdefault("langchain", langchain)
    sys.modules.setdefault("langchain.tools", tools)


_install_plotly_stubs()
_install_langchain_tool_stub()

from agents import furnace_tools  # noqa: E402
from agents.furnacemind import web_search as web_search_module  # noqa: E402
from agents.furnacemind.web_search import (  # noqa: E402
    BraveSearchAdapter,
    DuckDuckGoSearchAdapter,
    ExaSearchAdapter,
    FirecrawlSearchAdapter,
    ParallelSearchAdapter,
    SearXNGSearchAdapter,
    SerperSearchAdapter,
    TavilySearchAdapter,
    WebSearchResponse,
    WhoogleSearchAdapter,
    YaCySearchAdapter,
    build_web_search_provider,
    register_web_search_provider,
)
from utils.settings import Settings, WebSearchConfig  # noqa: E402


class _FakeResponse:
    """Small response double for JSON and HTML provider unit tests."""

    def __init__(
        self,
        payload: dict | None = None,
        *,
        text: str = "",
        status_error: Exception | None = None,
    ):
        self.payload = payload or {}
        self.text = text
        self.status_error = status_error

    def raise_for_status(self) -> None:
        """Raise a configured HTTP-status error, if one was supplied."""
        if self.status_error:
            raise self.status_error

    def json(self) -> dict:
        """Return the configured JSON payload."""
        return self.payload


class _FakeClient:
    """Context-manager HTTP client that replays configured responses/errors."""

    def __init__(self, factory: "_FakeClientFactory", *, timeout: float):
        self.factory = factory
        self.timeout = timeout

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def get(self, url: str, *, headers: dict, params: dict) -> _FakeResponse:
        """Record request details and return or raise the next configured item."""
        return self._send(
            method="GET",
            url=url,
            headers=headers,
            params=params,
        )

    def post(self, url: str, *, headers: dict, json: dict) -> _FakeResponse:
        """Record a JSON POST and return or raise the next configured item."""
        return self._send(
            method="POST",
            url=url,
            headers=headers,
            json=json,
        )

    def _send(self, *, method: str, url: str, headers: dict, **data) -> _FakeResponse:
        """Record common request details and replay one response or exception."""
        self.factory.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "timeout": self.timeout,
                **data,
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


def _config(**overrides) -> WebSearchConfig:  # noqa: ANN003
    """Return a Brave web-search config suitable for unit tests."""
    values = {
        "provider": "brave",
        "api_key": "test-token",
        "endpoint": "https://search.example.test",
        "max_results": 5,
        "timeout_seconds": 3.0,
        "max_retries": 0,
        "max_requests_per_session": 5,
    }
    values.update(overrides)
    return WebSearchConfig(**values)


def test_brave_provider_returns_standard_results() -> None:
    """Brave responses should normalize into source-citable results."""
    factory = _FakeClientFactory(
        [
            _FakeResponse(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Blast furnace reference",
                                "url": "https://example.com/bf",
                                "description": "Current public reference summary.",
                            }
                        ]
                    }
                }
            )
        ]
    )
    provider = BraveSearchAdapter(_config(), client_factory=factory)

    response = provider.search(query="latest blast furnace guidance", limit=3)

    assert response.error is None
    assert response.results[0].title == "Blast furnace reference"
    assert response.results[0].url == "https://example.com/bf"
    assert response.results[0].snippet == "Current public reference summary."
    assert factory.calls[0]["method"] == "GET"
    assert factory.calls[0]["headers"]["X-Subscription-Token"] == "test-token"
    assert factory.calls[0]["params"] == {
        "q": "latest blast furnace guidance",
        "count": 3,
        "text_decorations": "false",
    }


def test_brave_provider_empty_results_render_no_result_status() -> None:
    """Empty provider results should be explicit instead of raising."""
    factory = _FakeClientFactory([_FakeResponse({"web": {"results": []}})])
    provider = BraveSearchAdapter(_config(), client_factory=factory)

    response = provider.search(query="no matching result")

    assert response.error is None
    assert response.results == []
    assert "Status: no_results" in response.to_tool_text()


def test_brave_provider_missing_api_key_is_graceful() -> None:
    """A missing API key should make the optional tool unavailable, not crash."""
    provider = BraveSearchAdapter(_config(api_key=None))

    response = provider.search(query="current public info")

    assert response.results == []
    assert "BRAVE_SEARCH_API_KEY" in response.error
    assert "Status: unavailable" in response.to_tool_text()


def test_brave_provider_retries_timeout_and_returns_error(monkeypatch) -> None:
    """Timeouts should retry within the configured budget and then degrade."""
    monkeypatch.setattr(web_search_module.time, "sleep", lambda _: None)
    factory = _FakeClientFactory(
        [
            httpx.TimeoutException("slow provider"),
            httpx.TimeoutException("still slow"),
        ]
    )
    provider = BraveSearchAdapter(
        _config(max_retries=1),
        client_factory=factory,
    )

    response = provider.search(query="latest external standard")

    assert len(factory.calls) == 2
    assert response.results == []
    assert "timed out" in response.error


@pytest.mark.parametrize(
    ("adapter_class", "provider", "payload", "method", "expected_snippet"),
    [
        (
            TavilySearchAdapter,
            "tavily",
            {
                "results": [
                    {
                        "title": "Tavily result",
                        "url": "https://example.com/tavily",
                        "content": "Tavily summary",
                    }
                ]
            },
            "POST",
            "Tavily summary",
        ),
        (
            ExaSearchAdapter,
            "exa",
            {
                "results": [
                    {
                        "title": "Exa result",
                        "url": "https://example.com/exa",
                        "highlights": ["Exa", "highlight"],
                    }
                ]
            },
            "POST",
            "Exa highlight",
        ),
        (
            ParallelSearchAdapter,
            "parallel",
            {
                "results": [
                    {
                        "title": "Parallel result",
                        "url": "https://example.com/parallel",
                        "excerpts": ["Parallel excerpt"],
                    }
                ]
            },
            "POST",
            "Parallel excerpt",
        ),
        (
            FirecrawlSearchAdapter,
            "firecrawl",
            {
                "data": {
                    "web": [
                        {
                            "title": "Firecrawl result",
                            "url": "https://example.com/firecrawl",
                            "description": "Firecrawl summary",
                        }
                    ]
                }
            },
            "POST",
            "Firecrawl summary",
        ),
        (
            SerperSearchAdapter,
            "serper",
            {
                "organic": [
                    {
                        "title": "Serper result",
                        "link": "https://example.com/serper",
                        "snippet": "Serper summary",
                    }
                ]
            },
            "POST",
            "Serper summary",
        ),
        (
            SearXNGSearchAdapter,
            "searxng",
            {
                "results": [
                    {
                        "title": "SearXNG result",
                        "url": "https://example.com/searxng",
                        "content": "SearXNG summary",
                    }
                ]
            },
            "GET",
            "SearXNG summary",
        ),
        (
            WhoogleSearchAdapter,
            "whoogle",
            {
                "results": [
                    {
                        "text": "Whoogle result",
                        "href": "https://example.com/whoogle",
                        "description": "Whoogle summary",
                    }
                ]
            },
            "GET",
            "Whoogle summary",
        ),
        (
            YaCySearchAdapter,
            "yacy",
            {
                "channels": [
                    {
                        "items": [
                            {
                                "title": "YaCy result",
                                "link": "https://example.com/yacy",
                                "description": "YaCy summary",
                            }
                        ]
                    }
                ]
            },
            "GET",
            "YaCy summary",
        ),
    ],
)
def test_json_provider_adapters_normalize_results(
    adapter_class,
    provider: str,
    payload: dict,
    method: str,
    expected_snippet: str,
) -> None:
    """Every JSON provider should expose the same normalized result contract."""
    factory = _FakeClientFactory([_FakeResponse(payload)])
    adapter = adapter_class(
        _config(provider=provider, endpoint="https://search.example.test"),
        client_factory=factory,
    )

    response = adapter.search(query="current furnace reference", limit=2)

    assert response.error is None
    assert response.provider == provider
    assert len(response.results) == 1
    assert response.results[0].url.startswith("https://example.com/")
    assert response.results[0].snippet == expected_snippet
    assert factory.calls[0]["method"] == method


def test_duckduckgo_adapter_parses_html_and_resolves_redirect() -> None:
    """DuckDuckGo HTML results should become normal source-citable results."""
    html = """
    <div class="result">
      <a class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fddg">
        DuckDuckGo result
      </a>
      <a class="result__snippet">DuckDuckGo summary</a>
    </div>
    """
    factory = _FakeClientFactory([_FakeResponse(text=html)])
    adapter = DuckDuckGoSearchAdapter(
        _config(
            provider="duckduckgo",
            api_key=None,
            endpoint="https://html.duckduckgo.com/html/",
        ),
        client_factory=factory,
    )

    response = adapter.search(query="blast furnace reference")

    assert response.error is None
    assert response.results[0].title == "DuckDuckGo result"
    assert response.results[0].url == "https://example.com/ddg"
    assert response.results[0].snippet == "DuckDuckGo summary"


def test_self_hosted_provider_requires_endpoint_not_api_key() -> None:
    """Self-hosted adapters should explain missing endpoints without requiring keys."""
    adapter = SearXNGSearchAdapter(
        _config(provider="searxng", api_key=None, endpoint="")
    )

    response = adapter.search(query="blast furnace reference")

    assert response.results == []
    assert "SEARXNG_SEARCH_ENDPOINT" in response.error
    assert "API key" not in response.error


@pytest.mark.parametrize(
    ("alias", "expected_provider"),
    [
        ("serpent", "serper"),
        ("ddg", "duckduckgo"),
        ("searx", "searxng"),
        ("whoogle-search", "whoogle"),
    ],
)
def test_provider_aliases_build_canonical_adapters(
    alias: str, expected_provider: str
) -> None:
    """Common alternate names should resolve to canonical provider adapters."""
    provider = build_web_search_provider(
        _config(provider=alias, endpoint="https://search.example.test")
    )

    assert provider.provider_name == expected_provider


def test_provider_registry_accepts_an_alternate_adapter(monkeypatch) -> None:
    """A new provider should plug in without changing FurnaceMind tool code."""

    class ExampleSearchAdapter:
        provider_name = "example"

        def __init__(self, config: WebSearchConfig) -> None:
            self.config = config

        def configuration_error(self) -> str | None:
            return None

        def search(
            self,
            *,
            query: str,
            limit: int | None = None,
        ) -> WebSearchResponse:
            return WebSearchResponse(
                query=query,
                provider=self.provider_name,
                results=[],
            )

    monkeypatch.setattr(
        web_search_module,
        "_WEB_SEARCH_PROVIDER_FACTORIES",
        dict(web_search_module._WEB_SEARCH_PROVIDER_FACTORIES),
    )
    register_web_search_provider("Example", ExampleSearchAdapter)

    provider = build_web_search_provider(_config(provider="example"))
    response = provider.search(query="current furnace reference")

    assert provider.provider_name == "example"
    assert response.provider == "example"


def test_alternate_provider_uses_generic_environment_settings(monkeypatch) -> None:
    """Non-Brave providers must not inherit Brave endpoint or credentials."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "example")
    monkeypatch.setenv("WEB_SEARCH_ENDPOINT", "https://example.test/search")
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "example-token")
    monkeypatch.setenv("BRAVE_SEARCH_ENDPOINT", "https://brave.test/search")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-token")

    config = Settings._load_web_search_config()

    assert config.provider == "example"
    assert config.endpoint == "https://example.test/search"
    assert config.api_key == "example-token"


def test_tavily_provider_uses_specific_environment_settings(monkeypatch) -> None:
    """Provider-specific values should override generic search configuration."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("WEB_SEARCH_ENDPOINT", "https://generic.test/search")
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "generic-token")
    monkeypatch.setenv("TAVILY_SEARCH_ENDPOINT", "https://tavily.test/search")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-token")

    config = Settings._load_web_search_config()

    assert config.provider == "tavily"
    assert config.endpoint == "https://tavily.test/search"
    assert config.api_key == "tavily-token"


def test_serpent_environment_alias_selects_serper(monkeypatch) -> None:
    """The requested 'serpent' spelling should configure the Serper adapter."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "serpent")
    monkeypatch.setenv("SERPER_API_KEY", "serper-token")
    monkeypatch.delenv("WEB_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("SERPER_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("SERPER_ENDPOINT", raising=False)

    config = Settings._load_web_search_config()

    assert config.provider == "serper"
    assert config.endpoint == "https://google.serper.dev/search"
    assert config.api_key == "serper-token"


def test_web_search_tool_schema_and_dispatch(monkeypatch) -> None:
    """The FurnaceMind tool registry should expose and dispatch web_search."""
    monkeypatch.setattr(
        furnace_tools.st,
        "session_state",
        {"fm_web_search_enabled": True},
        raising=False,
    )
    monkeypatch.setattr(
        furnace_tools,
        "search_web",
        lambda query, *, limit=None: f"searched {query} limit={limit}",
    )

    tool_names = {
        schema["function"]["name"] for schema in furnace_tools.get_openai_tool_schemas()
    }
    result = furnace_tools.execute_openai_tool_call(
        name="web_search",
        arguments={"query": "latest coke CSR reference", "limit": 2},
    )

    assert "web_search" in tool_names
    assert result == "searched latest coke CSR reference limit=2"
    assert furnace_tools.st.session_state["fm_web_search_request_count"] == 1


def test_web_search_tool_respects_sidebar_toggle(monkeypatch) -> None:
    """Disabled live web search should not call the Brave provider."""
    monkeypatch.setattr(
        furnace_tools.st,
        "session_state",
        {"fm_web_search_enabled": False},
        raising=False,
    )
    monkeypatch.setattr(
        furnace_tools,
        "search_web",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("search_web called")),
    )

    result = furnace_tools.web_search(query="latest blast furnace news")

    assert "web_search disabled" in result


def test_web_search_tool_validation_returns_error(monkeypatch) -> None:
    """Invalid model-generated search arguments should return a tool error."""
    monkeypatch.setattr(furnace_tools, "_append_tool_error", lambda **_: None)

    result = furnace_tools.web_search(query="")

    assert result.startswith("web_search Error:")


def test_web_search_tool_stops_at_session_limit(monkeypatch) -> None:
    """Exhausted session quota should avoid another provider API call."""
    state = {
        "fm_web_search_enabled": True,
        "fm_web_search_request_count": 2,
    }
    monkeypatch.setattr(furnace_tools.st, "session_state", state, raising=False)
    monkeypatch.setattr(
        furnace_tools.settings,
        "web_search",
        types.SimpleNamespace(
            provider="example",
            max_requests_per_session=2,
        ),
    )
    monkeypatch.setattr(
        furnace_tools,
        "search_web",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("provider should not be called")
        ),
    )

    result = furnace_tools.web_search(query="latest furnace news")

    assert "Status: session_limit_reached" in result
    assert "Session requests used: 2/2" in result
    assert state["fm_web_search_request_count"] == 2


def test_web_scrape_ingest_tool_schema_and_dispatch(monkeypatch) -> None:
    """The tool registry should expose and dispatch approved URL ingestion."""
    monkeypatch.setattr(
        furnace_tools,
        "settings",
        types.SimpleNamespace(web_scrape=types.SimpleNamespace(ingest_enabled=True)),
    )
    state = {
        "knowledge_store": object(),
        "knowledge_embedding_client": object(),
        "fm_user_id": "user-1",
        "knowledge_document_repository": object(),
        "knowledge_chunk_repository": object(),
    }
    monkeypatch.setattr(furnace_tools.st, "session_state", state, raising=False)
    calls = {}

    def fake_ingest(url, **kwargs):  # noqa: ANN001, ANN202
        calls["url"] = url
        calls["kwargs"] = kwargs
        return types.SimpleNamespace(
            status="indexed",
            url=url,
            filename="web_source.md",
            document_id="doc-1",
            sql_document_id="sql-doc-1",
            chunk_count=3,
            qdrant_collection="furnacemind_knowledge",
            error=None,
        )

    monkeypatch.setattr(furnace_tools, "ingest_external_knowledge_url", fake_ingest)

    tool_names = {
        schema["function"]["name"] for schema in furnace_tools.get_openai_tool_schemas()
    }
    result = furnace_tools.execute_openai_tool_call(
        name="web_scrape_ingest",
        arguments={"url": "https://example.com/blast-furnace-sop"},
    )

    assert "web_scrape_ingest" in tool_names
    assert "Status: indexed" in result
    assert "Chunks indexed: 3" in result
    assert calls["url"] == "https://example.com/blast-furnace-sop"
    assert calls["kwargs"]["knowledge_store"] is state["knowledge_store"]
    assert calls["kwargs"]["embedding_client"] is state["knowledge_embedding_client"]
    assert calls["kwargs"]["user_id"] == "user-1"


def test_web_scrape_ingest_requires_initialized_knowledge_store(monkeypatch) -> None:
    """The scraper tool should fail gracefully before page setup is ready."""
    monkeypatch.setattr(
        furnace_tools,
        "settings",
        types.SimpleNamespace(web_scrape=types.SimpleNamespace(ingest_enabled=True)),
    )
    monkeypatch.setattr(
        furnace_tools.st,
        "session_state",
        {"knowledge_embedding_client": object()},
        raising=False,
    )

    result = furnace_tools.web_scrape_ingest(url="https://example.com/source")

    assert "Knowledge store is not initialized" in result


def test_web_scrape_ingest_rejects_non_http_url(monkeypatch) -> None:
    """The tool should reject local or non-web targets before scraping."""
    monkeypatch.setattr(furnace_tools, "_append_tool_error", lambda **_: None)

    result = furnace_tools.web_scrape_ingest(url="file:///tmp/source.md")

    assert result == "web_scrape_ingest Error: URL must be an absolute http/https URL."


def test_web_scrape_ingest_returns_provider_failure(monkeypatch) -> None:
    """Provider failures should become a normal tool message, not an exception."""
    monkeypatch.setattr(
        furnace_tools,
        "settings",
        types.SimpleNamespace(web_scrape=types.SimpleNamespace(ingest_enabled=True)),
    )
    monkeypatch.setattr(
        furnace_tools.st,
        "session_state",
        {
            "knowledge_store": object(),
            "knowledge_embedding_client": object(),
        },
        raising=False,
    )
    monkeypatch.setattr(
        furnace_tools,
        "ingest_external_knowledge_url",
        lambda url, **kwargs: types.SimpleNamespace(
            status="unavailable",
            url=url,
            filename=None,
            document_id=None,
            sql_document_id=None,
            chunk_count=0,
            qdrant_collection=None,
            error="Scrape request timed out.",
        ),
    )

    result = furnace_tools.web_scrape_ingest(url="https://example.com/slow")

    assert "Status: unavailable" in result
    assert "Scrape request timed out" in result


def test_web_scrape_ingest_disabled_by_config(monkeypatch) -> None:
    """Permanent URL ingestion should require an explicit config enable flag."""
    monkeypatch.setattr(
        furnace_tools,
        "settings",
        types.SimpleNamespace(web_scrape=types.SimpleNamespace(ingest_enabled=False)),
    )

    result = furnace_tools.web_scrape_ingest(url="https://example.com/source")

    assert "WEB_SCRAPE_INGEST_ENABLED=true" in result
