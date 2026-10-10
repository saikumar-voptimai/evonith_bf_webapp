"""Centralised configuration loader for all FurnaceMind services.

Reads environment variables (loaded from ``.env`` via python-dotenv) and
assembles typed :mod:`dataclasses` for LLM, embedding, Qdrant, anomaly
detection, and general app settings.

The module-level singleton :data:`settings` is the single source of truth
for runtime configuration; import it via::

    from utils.settings import settings
"""

# utils/settings.py
# Purpose: Centralized configuration for FurnaceMind application

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# ==========================================================
#  LLM CONFIGURATION
# ==========================================================


REASONING_LEVELS = ("Low", "Medium", "High")
DEFAULT_OPENROUTER_REASONING_MODEL = {
    "Low": "google/gemma-4-26b-a4b-it",
    "Medium": "openai/gpt-5.4-nano",
    "High": "google/gemini-3.1-flash-lite-preview",
}
_REASONING_LEVEL_ALIASES = {
    "low": "Low",
    "fast": "Low",
    "medium": "Medium",
    "high": "High",
    "slow": "High",
}


def normalize_openrouter_reasoning_level(value: str | None) -> str:
    """Return a supported OpenRouter reasoning profile name.

    User-facing controls and persisted sessions should only use ``Low``,
    ``Medium``, or ``High``. Missing or invalid values fall back to
    ``Medium``, which is the balanced profile required by the ticket.
    """
    normalized = str(value or "").strip().lower()
    return _REASONING_LEVEL_ALIASES.get(normalized, "Medium")


def _parse_int_env(name: str, default: int) -> int:
    """Return an integer environment value, falling back on invalid input."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _parse_float_env(name: str, default: float) -> float:
    """Return a float environment value, falling back on invalid input."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _parse_bool_env(name: str, default: bool) -> bool:
    """Return a boolean environment value, falling back on invalid input."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_first(*names: str) -> str | None:
    """Return the first configured environment variable from ``names``.

    New profile names use ``LOW`` and ``HIGH``. The old ``FAST`` and ``SLOW``
    variables are still accepted so existing deployments keep working until
    their environment files are renamed.
    """
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class OpenRouterReasoningProfile:
    """Model routing profile for one FurnaceMind reasoning level.

    Attributes:
        level: User-facing reasoning level label: ``Low``, ``Medium``, or
            ``High``.
        model_name: OpenRouter model used for this level.
        reasoning_effort: Optional OpenRouter reasoning effort. Empty values
            disable the reasoning request parameter.
    """

    level: str
    model_name: str
    reasoning_effort: str | None = None


@dataclass
class OpenRouterLLMConfig:
    """Connection and model settings for the OpenRouter LLM gateway.

    Attributes:
        api_key:    OpenRouter API key (``OPENROUTER_API_KEY`` env var).
        base_url:   OpenRouter API base URL.
        model_name: Fully-qualified model identifier (e.g. ``openai/gpt-4o-mini``).
        memory_compression_model_name: Model used for memory summary compression.
        reasoning_profiles: Per-level model and reasoning-effort routing config.
        default_reasoning_level: Profile used for missing/invalid user choices.
        max_tokens: Maximum completion tokens to request.
    """

    api_key: str | None
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "openai/gpt-4o-mini"
    memory_compression_model_name: str = "openai/gpt-4o-mini"
    reasoning_profiles: dict[str, OpenRouterReasoningProfile] = field(
        default_factory=dict
    )
    default_reasoning_level: str = "Medium"
    max_tokens: int = 800


@dataclass
class OpenAILLMConfig:
    """Connection and model settings for the OpenAI API.

    Attributes:
        api_key:    OpenAI API key (``OPENAI_API_KEY`` env var).
        base_url:   Optional custom base URL (e.g. for Azure OpenAI).
        model_name: Model identifier (e.g. ``gpt-4o-mini``).
        max_tokens: Maximum completion tokens to request.
        api_mode:   Which API surface to use - ``"responses"`` or
                    ``"chat_completions"``.
    """

    api_key: str | None
    base_url: str | None = None
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 800
    api_mode: str = "chat_completions"  # "responses" | "chat_completions"


@dataclass
class LLMSettings:
    """Top-level LLM provider selection and per-provider config.

    Attributes:
        provider:   Active provider - ``"openrouter"`` or ``"openai"``.
        openrouter: :class:`OpenRouterLLMConfig` for the OpenRouter gateway.
        openai:     :class:`OpenAILLMConfig` for the OpenAI API.
    """

    provider: str = "openrouter"
    openrouter: OpenRouterLLMConfig = field(
        default_factory=lambda: OpenRouterLLMConfig(api_key=None)
    )
    openai: OpenAILLMConfig = field(
        default_factory=lambda: OpenAILLMConfig(api_key=None)
    )


# ==========================================================
#  VECTOR DATABASE CONFIG
# ==========================================================


@dataclass
class QdrantConfig:
    """Connection settings for a single Qdrant collection.

    Attributes:
        url:            Qdrant endpoint URL (cloud or local).
        api_key:        API key (required for Qdrant Cloud).
        collection_name: Target collection name.
        embedding_dim:  Vector dimension expected by this collection.
        timeout:        HTTP request timeout in seconds.
    """

    url: str
    api_key: str | None
    collection_name: str
    embedding_dim: int
    timeout: int


# ==========================================================
#  APPLICATION CONFIG
# ==========================================================


@dataclass
class AppConfig:
    """General application runtime settings.

    Attributes:
        shift_hours:  Length of each production shift in hours (always 8).
        timezone:     Default timezone for display purposes.
        environment:  Deployment environment tag (``"dev"`` or ``"prod"``).
    """

    shift_hours: int = 8
    timezone: str = "UTC"
    environment: str = "dev"


@dataclass
class SemanticMemoryConfig:
    """SQL-backed, Qdrant-indexed long-term semantic memory settings.

    Attributes:
        enabled: Whether the semantic memory layer should be initialized.
        collection_name: Qdrant collection used for long-term memories.
        llm_model: OpenAI/OpenRouter-compatible model used for fact extraction.
        max_memories: Maximum memories to retrieve per chat turn.
        search_threshold: Optional minimum Qdrant vector score.
    """

    enabled: bool
    collection_name: str
    llm_model: str
    max_memories: int
    search_threshold: float | None


# ==========================================================
#  SETTINGS LOADER
# ==========================================================


_WEB_SEARCH_PROVIDER_ALIASES = {
    "ddg": "duckduckgo",
    "duck-duck-go": "duckduckgo",
    "duck_duck_go": "duckduckgo",
    "fire_crawl": "firecrawl",
    "searx": "searxng",
    "serpent": "serper",
    "whoogle-search": "whoogle",
    "whoogle_search": "whoogle",
}

_WEB_SEARCH_PROVIDER_CONFIG = {
    "brave": {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "endpoint_envs": ("BRAVE_SEARCH_ENDPOINT",),
        "api_key_envs": ("BRAVE_SEARCH_API_KEY",),
    },
    "tavily": {
        "endpoint": "https://api.tavily.com/search",
        "endpoint_envs": ("TAVILY_SEARCH_ENDPOINT", "TAVILY_ENDPOINT"),
        "api_key_envs": ("TAVILY_API_KEY", "TAVILY_SEARCH_API_KEY"),
    },
    "exa": {
        "endpoint": "https://api.exa.ai/search",
        "endpoint_envs": ("EXA_SEARCH_ENDPOINT", "EXA_ENDPOINT"),
        "api_key_envs": ("EXA_API_KEY", "EXA_SEARCH_API_KEY"),
    },
    "parallel": {
        "endpoint": "https://api.parallel.ai/v1/search",
        "endpoint_envs": ("PARALLEL_SEARCH_ENDPOINT", "PARALLEL_ENDPOINT"),
        "api_key_envs": ("PARALLEL_API_KEY", "PARALLEL_SEARCH_API_KEY"),
    },
    "firecrawl": {
        "endpoint": "https://api.firecrawl.dev/v2/search",
        "endpoint_envs": ("FIRECRAWL_SEARCH_ENDPOINT", "FIRECRAWL_ENDPOINT"),
        "api_key_envs": ("FIRECRAWL_API_KEY",),
    },
    "serper": {
        "endpoint": "https://google.serper.dev/search",
        "endpoint_envs": ("SERPER_SEARCH_ENDPOINT", "SERPER_ENDPOINT"),
        "api_key_envs": ("SERPER_API_KEY",),
    },
    "duckduckgo": {
        "endpoint": "https://html.duckduckgo.com/html/",
        "endpoint_envs": ("DUCKDUCKGO_SEARCH_ENDPOINT", "DUCKDUCKGO_ENDPOINT"),
        "api_key_envs": (),
    },
    "searxng": {
        "endpoint": "",
        "endpoint_envs": ("SEARXNG_SEARCH_ENDPOINT", "SEARXNG_ENDPOINT"),
        "api_key_envs": ("SEARXNG_API_KEY",),
    },
    "whoogle": {
        "endpoint": "",
        "endpoint_envs": ("WHOOGLE_SEARCH_ENDPOINT", "WHOOGLE_ENDPOINT"),
        "api_key_envs": ("WHOOGLE_API_KEY",),
    },
    "yacy": {
        "endpoint": "",
        "endpoint_envs": ("YACY_SEARCH_ENDPOINT", "YACY_ENDPOINT"),
        "api_key_envs": ("YACY_API_KEY",),
    },
}


def normalize_web_search_provider_name(provider: str | None) -> str:
    """Return the canonical registry name for a configured search provider."""
    normalized = str(provider or "").strip().lower()
    return _WEB_SEARCH_PROVIDER_ALIASES.get(normalized, normalized)


def _first_configured_environment_value(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty value from an ordered env-var list."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class WebSearchConfig:
    """Configuration for the provider-neutral FurnaceMind web search tool.

    Attributes:
        provider: Canonical search provider name. Supported adapters are Brave,
            Tavily, Exa, Parallel, Firecrawl, Serper, DuckDuckGo, SearXNG,
            Whoogle, and YaCy.
        api_key: Provider API key. The generic ``WEB_SEARCH_API_KEY`` is
            supported along with provider-specific environment variables.
        endpoint: Provider web-search endpoint. Hosted providers have defaults;
            self-hosted SearXNG, Whoogle, and YaCy deployments must configure it.
        max_results: Maximum results returned to the model per search.
        timeout_seconds: HTTP timeout for one provider request.
        max_retries: Number of retry attempts after the initial request fails.
        max_requests_per_session: Maximum logical searches allowed in one
            Streamlit browser session.
    """

    provider: str = "brave"
    api_key: str | None = None
    endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    max_results: int = 5
    timeout_seconds: float = 10.0
    max_retries: int = 2
    max_requests_per_session: int = 5


@dataclass(frozen=True)
class WebScrapeConfig:
    """Configuration for approved external-page ingestion.

    Search is a chat-time tool and does not persist anything. Scraping is a
    separate background/admin ingestion path that intentionally adds approved
    public pages to the shared FurnaceMind knowledge base. Jina Reader converts
    public pages into readable Markdown before the existing MRAG chunking and
    embedding pipeline stores them.

    Attributes:
        provider: Reader provider name. The first implementation supports
            ``"jina_reader"``.
        api_key: Optional Jina Reader API key.
        endpoint: Jina Reader endpoint used to fetch readable page content.
        timeout_seconds: HTTP timeout for one reader request.
        max_retries: Number of retry attempts after the initial request fails.
        max_chars: Maximum characters retained from one scraped page.
        ingest_enabled: Whether chat/tool-triggered ingestion is allowed.
    """

    provider: str = "jina_reader"
    api_key: str | None = None
    endpoint: str = "https://r.jina.ai"
    timeout_seconds: float = 20.0
    max_retries: int = 2
    max_chars: int = 200_000
    ingest_enabled: bool = False


class Settings:
    """Singleton configuration object for all FurnaceMind sub-systems.

    Reads environment variables on first instantiation and populates typed
    config objects for LLM, embedding, Qdrant (shift + knowledge stores),
    anomaly detection, and general app settings.

    Two Qdrant targets are supported:

    * ``qdrant_shift`` - 384-dim local embeddings, shift summary store.
    * ``qdrant_knowledge`` - 1024-dim cloud embeddings, Knowledge Hub.
    * ``qdrant_skills`` - 1024-dim cloud embeddings, skill retrieval index.

    The ``qdrant`` attribute is an alias for ``qdrant_shift`` for backward
    compatibility with existing call-sites.

    Attributes:
        llm:               :class:`LLMSettings` for the active LLM provider.
        qdrant_shift:      :class:`QdrantConfig` for the shift summary collection.
        qdrant_knowledge:  :class:`QdrantConfig` for the knowledge document collection.
        qdrant_skills:     :class:`QdrantConfig` for the FurnaceMind skill index.
        qdrant:            Alias for ``qdrant_shift`` (backward compatibility).
        app:               :class:`AppConfig` general runtime settings.
        memory_summary_message_window: Number of chat messages per memory summary.
        memory_summary_token_limit:    Maximum requested memory summary tokens.
        web_search:                    Provider, timeout, retry, and result-limit
                                       config for external web search.
        web_scrape:                    Provider, timeout, retry, and size-limit
                                       config for external knowledge ingestion.
    """

    def __init__(self) -> None:
        """Load all configuration sections from environment variables.

        Raises:
            ValueError: If required environment variables are missing or invalid.
        """

        self.llm = self._load_llm_settings()

        # Two Qdrant targets
        self.qdrant_shift = self._load_qdrant_config(
            collection_env="SHIFT_QDRANT_COLLECTION",
            dim_env="SHIFT_QDRANT_EMBED_DIM",
            default_collection="furnace_shift_summaries",
            default_dim=384,
            # backward compatible: accept old vars if SHIFT_* not set
            fallback_collection_env="QDRANT_COLLECTION",
            fallback_dim_env="QDRANT_EMBED_DIM",
        )

        self.qdrant_knowledge = self._load_qdrant_config(
            collection_env="KNOWLEDGE_QDRANT_COLLECTION",
            dim_env="KNOWLEDGE_QDRANT_EMBED_DIM",
            default_collection="furnacemind_knowledge",
            default_dim=1024,
            # do NOT fall back to QDRANT_COLLECTION by default here,
            # because that usually points to shift summaries and causes mixups.
            fallback_collection_env=None,
            fallback_dim_env=None,
        )

        self.qdrant_skills = self._load_qdrant_config(
            collection_env="SKILL_QDRANT_COLLECTION",
            dim_env="SKILL_QDRANT_EMBED_DIM",
            default_collection="furnacemind_skills",
            default_dim=1024,
            # Skills use the same cloud embedding family as MRAG knowledge.
            fallback_collection_env=None,
            fallback_dim_env="KNOWLEDGE_QDRANT_EMBED_DIM",
        )

        # Backward-compatible alias: existing code that uses settings.qdrant
        # will keep using the shift store unless you change those call-sites.
        self.qdrant = self.qdrant_shift

        self.app = AppConfig()
        self.memory_summary_message_window = int(
            os.getenv("MEMORY_SUMMARY_MESSAGE_WINDOW", 8)
        )
        self.memory_summary_token_limit = int(
            os.getenv("MEMORY_SUMMARY_TOKEN_LIMIT", 2000)
        )
        self.semantic_memory = self._load_semantic_memory_config()
        self.web_search = self._load_web_search_config()
        self.web_scrape = self._load_web_scrape_config()
        self._validate()

    # ------------------------------------------------------
    # LLM LOADER
    # ------------------------------------------------------
    @staticmethod
    def _load_llm_settings() -> LLMSettings:
        """
        Build LLM configuration from environment variables.

        This loader keeps the default chat model and the memory-compression
        model in the same OpenRouter configuration section. If
        ``MEMORY_COMPRESSION_MODEL`` is not set, memory summaries use the normal
        ``OPENROUTER_MODEL`` value so local setups continue to work.

        Returns:
             - return: LLMSettings - Populated LLM settings for OpenRouter and OpenAI.
        """
        provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        memory_compression_model = (
            os.getenv("MEMORY_COMPRESSION_MODEL", openrouter_model).strip()
            or openrouter_model
        )

        low_model = DEFAULT_OPENROUTER_REASONING_MODEL["Low"]
        medium_model = DEFAULT_OPENROUTER_REASONING_MODEL["Medium"]
        high_model = DEFAULT_OPENROUTER_REASONING_MODEL["High"]
        reasoning_profiles = {
            "Low": OpenRouterReasoningProfile(
                level="Low",
                model_name=(
                    _env_first("OPENROUTER_LOW_MODEL", "OPENROUTER_FAST_MODEL")
                    or low_model
                ).strip()
                or low_model,
                reasoning_effort=(
                    _env_first(
                        "OPENROUTER_LOW_REASONING_EFFORT",
                        "OPENROUTER_FAST_REASONING_EFFORT",
                    )
                    or ""
                ).strip()
                or None,
            ),
            "Medium": OpenRouterReasoningProfile(
                level="Medium",
                model_name=(
                    os.getenv("OPENROUTER_MEDIUM_MODEL", medium_model).strip()
                    or medium_model
                ),
                reasoning_effort=os.getenv(
                    "OPENROUTER_MEDIUM_REASONING_EFFORT", ""
                ).strip()
                or None,
            ),
            "High": OpenRouterReasoningProfile(
                level="High",
                model_name=(
                    _env_first("OPENROUTER_HIGH_MODEL", "OPENROUTER_SLOW_MODEL")
                    or high_model
                ).strip()
                or high_model,
                reasoning_effort=(
                    _env_first(
                        "OPENROUTER_HIGH_REASONING_EFFORT",
                        "OPENROUTER_SLOW_REASONING_EFFORT",
                    )
                    or ""
                ).strip()
                or None,
            ),
        }
        default_reasoning_level = normalize_openrouter_reasoning_level(
            os.getenv("OPENROUTER_DEFAULT_REASONING_LEVEL", "Medium")
        )

        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_api_mode = os.getenv("OPENAI_API_MODE", "responses").strip().lower()

        max_tokens = int(os.getenv("LLM_MAX_TOKENS", 800))

        return LLMSettings(
            provider=provider,
            openrouter=OpenRouterLLMConfig(
                api_key=openrouter_api_key,
                base_url=openrouter_base_url,
                model_name=openrouter_model,
                memory_compression_model_name=memory_compression_model,
                reasoning_profiles=reasoning_profiles,
                default_reasoning_level=default_reasoning_level,
                max_tokens=max_tokens,
            ),
            openai=OpenAILLMConfig(
                api_key=openai_api_key,
                base_url=openai_base_url if openai_base_url else None,
                model_name=openai_model,
                max_tokens=max_tokens,
                api_mode=openai_api_mode,
            ),
        )

    # ------------------------------------------------------
    # QDRANT LOADER (Reusable for Shift + Knowledge)
    # ------------------------------------------------------
    @staticmethod
    def _load_qdrant_config(
        *,
        collection_env: str,
        dim_env: str,
        default_collection: str,
        default_dim: int,
        fallback_collection_env: str | None,
        fallback_dim_env: str | None,
    ) -> QdrantConfig:
        """Build a :class:`QdrantConfig` for a single Qdrant collection.

        Supports two URL env vars (``QDRANT_ENDPOINT`` for cloud and
        ``QDRANT_URL`` for local) and optional fallback env vars for
        backward compatibility.

        Args:
            collection_env:          Env var name for the collection name.
            dim_env:                 Env var name for the embedding dimension.
            default_collection:      Default collection name if env var is unset.
            default_dim:             Default embedding dimension if env var is unset.
            fallback_collection_env: Optional legacy env var for the collection name.
            fallback_dim_env:        Optional legacy env var for the dimension.

        Returns:
            Populated :class:`QdrantConfig`.

        Raises:
            ValueError: If no URL is set, HTTPS is not used for cloud, or
                        API key is missing for Qdrant Cloud.
        """
        # support both names (your existing file checked QDRANT_ENDPOINT or QDRANT_URL)
        endpoint = os.getenv("QDRANT_ENDPOINT")
        local_url = os.getenv("QDRANT_URL")

        if not endpoint and not local_url:
            raise ValueError(
                "Either QDRANT_ENDPOINT (cloud) or QDRANT_URL (local) must be set."
            )

        effective_url = endpoint if endpoint else local_url

        # collection / dim: prefer specific env; optionally fallback to old global vars
        collection = os.getenv(collection_env)
        if not collection and fallback_collection_env:
            collection = os.getenv(fallback_collection_env)
        if not collection:
            collection = default_collection

        dim_val = os.getenv(dim_env)
        if (dim_val is None or str(dim_val).strip() == "") and fallback_dim_env:
            dim_val = os.getenv(fallback_dim_env)
        embedding_dim = int(dim_val) if dim_val not in (None, "") else int(default_dim)

        timeout = int(os.getenv("QDRANT_TIMEOUT", 30))

        # api key required for Qdrant Cloud in practice; keep same behavior
        api_key = os.getenv("QDRANT_API_KEY")
        if endpoint and not api_key:
            raise ValueError(
                "QDRANT_API_KEY must be set when using Qdrant Cloud (QDRANT_ENDPOINT)."
            )

        # basic safety
        if "cloud.qdrant.io" in (effective_url or "") and not str(
            effective_url
        ).startswith("https://"):
            raise ValueError("Qdrant Cloud endpoint must use HTTPS.")

        return QdrantConfig(
            url=effective_url,
            api_key=api_key,
            collection_name=collection,
            embedding_dim=embedding_dim,
            timeout=timeout,
        )

    @staticmethod
    def _load_semantic_memory_config() -> SemanticMemoryConfig:
        """Build semantic-memory settings from environment variables."""
        threshold_raw = os.getenv("SEMANTIC_MEMORY_SEARCH_THRESHOLD", "").strip()
        threshold = float(threshold_raw) if threshold_raw else None

        return SemanticMemoryConfig(
            enabled=os.getenv("SEMANTIC_MEMORY_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            collection_name=os.getenv(
                "SEMANTIC_MEMORY_QDRANT_COLLECTION",
                "furnacemind_long_term_memory",
            ),
            llm_model=os.getenv(
                "SEMANTIC_MEMORY_LLM_MODEL",
                os.getenv(
                    "MEMORY_COMPRESSION_MODEL",
                    os.getenv("OPENROUTER_MODEL", "gpt-4o-mini"),
                ),
            ),
            max_memories=int(
                os.getenv(
                    "SEMANTIC_MEMORY_MAX_MEMORIES",
                    5,
                )
            ),
            search_threshold=threshold,
        )

    @staticmethod
    def _load_web_search_config() -> WebSearchConfig:
        """Build web-search settings from environment variables.

        Provider-specific credentials and endpoints take precedence over the
        generic ``WEB_SEARCH_API_KEY`` and ``WEB_SEARCH_ENDPOINT`` values. A
        missing key or self-hosted endpoint is allowed at startup because web
        search is optional; the selected adapter reports a graceful unavailable
        message if the user later enables it.
        """
        configured_provider = os.getenv("WEB_SEARCH_PROVIDER", "brave")
        provider = normalize_web_search_provider_name(configured_provider) or "brave"
        generic_endpoint = os.getenv("WEB_SEARCH_ENDPOINT", "").strip()
        generic_api_key = (os.getenv("WEB_SEARCH_API_KEY") or "").strip() or None
        provider_config = _WEB_SEARCH_PROVIDER_CONFIG.get(provider, {})
        endpoint = (
            _first_configured_environment_value(
                provider_config.get("endpoint_envs", ())
            )
            or generic_endpoint
            or str(provider_config.get("endpoint", ""))
        ).strip()
        api_key = (
            _first_configured_environment_value(provider_config.get("api_key_envs", ()))
            or generic_api_key
        )
        max_results = _parse_int_env("WEB_SEARCH_MAX_RESULTS", 5)
        timeout_seconds = _parse_float_env("WEB_SEARCH_TIMEOUT_SECONDS", 10.0)
        max_retries = _parse_int_env("WEB_SEARCH_MAX_RETRIES", 2)
        max_requests_per_session = _parse_int_env(
            "WEB_SEARCH_MAX_REQUESTS_PER_SESSION", 5
        )

        return WebSearchConfig(
            provider=provider,
            api_key=api_key,
            endpoint=endpoint,
            max_results=max(1, min(max_results, 10)),
            timeout_seconds=max(1.0, timeout_seconds),
            max_retries=max(0, max_retries),
            max_requests_per_session=max(1, max_requests_per_session),
        )

    @staticmethod
    def _load_web_scrape_config() -> WebScrapeConfig:
        """Build external knowledge scrape settings from environment variables.

        Scrape ingestion uses Jina Reader to convert public pages into Markdown
        before storing them as shared FurnaceMind knowledge. The API key is
        optional so local testing can still return graceful unavailable messages
        instead of failing application startup.
        """
        provider = (
            os.getenv("WEB_SCRAPE_PROVIDER", "jina_reader").strip().lower()
            or "jina_reader"
        )
        endpoint = os.getenv(
            "JINA_READER_ENDPOINT",
            os.getenv("WEB_SCRAPE_ENDPOINT", "https://r.jina.ai"),
        ).strip()
        api_key = (
            os.getenv("JINA_READER_API_KEY")
            or os.getenv("JINA_API_KEY")
            or os.getenv("WEB_SCRAPE_API_KEY")
        )
        timeout_seconds = _parse_float_env("WEB_SCRAPE_TIMEOUT_SECONDS", 20.0)
        max_retries = _parse_int_env("WEB_SCRAPE_MAX_RETRIES", 2)
        max_chars = _parse_int_env("WEB_SCRAPE_MAX_CHARS", 200_000)
        ingest_enabled = _parse_bool_env("WEB_SCRAPE_INGEST_ENABLED", False)
        return WebScrapeConfig(
            provider=provider,
            api_key=api_key,
            endpoint=endpoint or "https://r.jina.ai",
            timeout_seconds=max(1.0, timeout_seconds),
            max_retries=max(0, max_retries),
            max_chars=max(1_000, max_chars),
            ingest_enabled=ingest_enabled,
        )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------
    def _validate(self) -> None:
        """Validate that required configuration values are set and consistent.

        Raises:
            ValueError: If required API keys are missing or config values are
                        outside the set of supported options.
        """
        # LLM validation
        if not self.llm.openrouter.api_key and not self.llm.openai.api_key:
            raise ValueError(
                "At least one of OPENROUTER_API_KEY or OPENAI_API_KEY must be set."
            )
        if self.llm.provider not in {"openrouter", "openai"}:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm.provider}")
        if self.llm.openai.api_mode not in {"responses", "chat_completions"}:
            raise ValueError(
                "OPENAI_API_MODE must be 'responses' or 'chat_completions'."
            )


# ==========================================================
#  SINGLETON INSTANCE
# ==========================================================

settings = Settings()
