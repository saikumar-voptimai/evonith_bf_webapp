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
# 🔹 LLM CONFIGURATION
# ==========================================================


REASONING_LEVELS = ("Low", "Medium", "High")
DEFAULT_OPENROUTER_REASONING_MODELS = {
    "Low": (
        "google/gemma-4-26b-a4b-it",
        "bytedance-seed/seed-2.0-mini",
        "google/gemma-4-31b-it",
    ),
    "Medium": (
        "openai/gpt-5.4-nano",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "stepfun/step-3.7-flash",
    ),
    "High": (
        "google/gemini-3.1-flash-lite-preview",
        "minimax/minimax-m3",
        "qwen/qwen3.7-plus",
    ),
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


def _csv_values(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated environment value into model names.

    Empty entries and surrounding whitespace are ignored so fallback model
    variables can be written naturally in ``.env`` files.
    """
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


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
        model_name: Primary OpenRouter model used for this level.
        reasoning_effort: Optional OpenRouter reasoning effort. Empty values
            disable the reasoning request parameter.
        fallback_model_names: Ordered fallback models that OpenRouter can try
            when the primary model is unavailable.
    """

    level: str
    model_name: str
    reasoning_effort: str | None = None
    fallback_model_names: tuple[str, ...] = ()


@dataclass
class OpenRouterLLMConfig:
    """Connection and model settings for the OpenRouter LLM gateway.

    Attributes:
        api_key:    OpenRouter API key (``OPENROUTER_API_KEY`` env var).
        base_url:   OpenRouter API base URL.
        model_name: Fully-qualified model identifier (e.g. ``openai/gpt-4o-mini``).
        memory_compression_model_name: Model used for memory summary compression.
        reasoning_profiles: Per-level model and reasoning-effort routing config.
        default_reasoning_level: Fallback profile for missing/invalid user choices.
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
        api_mode:   Which API surface to use — ``"responses"`` or
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
        provider:   Active provider — ``"openrouter"`` or ``"openai"``.
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
# 🔹 VECTOR DATABASE CONFIG
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
# 🔹 APPLICATION CONFIG
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
# 🔹 SETTINGS LOADER
# ==========================================================


class Settings:
    """Singleton configuration object for all FurnaceMind sub-systems.

    Reads environment variables on first instantiation and populates typed
    config objects for LLM, embedding, Qdrant (shift + knowledge stores),
    anomaly detection, and general app settings.

    Two Qdrant targets are supported:

    * ``qdrant_shift`` — 384-dim local embeddings, shift summary store.
    * ``qdrant_knowledge`` — 1024-dim cloud embeddings, Knowledge Hub.
    * ``qdrant_skills`` — 1024-dim cloud embeddings, skill retrieval index.

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

        default_fallback_models = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
        low_models = DEFAULT_OPENROUTER_REASONING_MODELS["Low"]
        medium_models = DEFAULT_OPENROUTER_REASONING_MODELS["Medium"]
        high_models = DEFAULT_OPENROUTER_REASONING_MODELS["High"]
        low_fallback_models = _env_first(
            "OPENROUTER_LOW_FALLBACK_MODELS",
            "OPENROUTER_FAST_FALLBACK_MODELS",
        )
        if low_fallback_models is None:
            low_fallback_models = default_fallback_models or ",".join(low_models[1:])
        high_fallback_models = _env_first(
            "OPENROUTER_HIGH_FALLBACK_MODELS",
            "OPENROUTER_SLOW_FALLBACK_MODELS",
        )
        if high_fallback_models is None:
            high_fallback_models = default_fallback_models or ",".join(high_models[1:])
        reasoning_profiles = {
            "Low": OpenRouterReasoningProfile(
                level="Low",
                model_name=(
                    _env_first("OPENROUTER_LOW_MODEL", "OPENROUTER_FAST_MODEL")
                    or low_models[0]
                ).strip()
                or low_models[0],
                reasoning_effort=(
                    _env_first(
                        "OPENROUTER_LOW_REASONING_EFFORT",
                        "OPENROUTER_FAST_REASONING_EFFORT",
                    )
                    or ""
                ).strip()
                or None,
                fallback_model_names=_csv_values(low_fallback_models),
            ),
            "Medium": OpenRouterReasoningProfile(
                level="Medium",
                model_name=(
                    os.getenv("OPENROUTER_MEDIUM_MODEL", medium_models[0]).strip()
                    or medium_models[0]
                ),
                reasoning_effort=os.getenv(
                    "OPENROUTER_MEDIUM_REASONING_EFFORT", ""
                ).strip()
                or None,
                fallback_model_names=_csv_values(
                    os.getenv(
                        "OPENROUTER_MEDIUM_FALLBACK_MODELS",
                        default_fallback_models or ",".join(medium_models[1:]),
                    )
                ),
            ),
            "High": OpenRouterReasoningProfile(
                level="High",
                model_name=(
                    _env_first("OPENROUTER_HIGH_MODEL", "OPENROUTER_SLOW_MODEL")
                    or high_models[0]
                ).strip()
                or high_models[0],
                reasoning_effort=(
                    _env_first(
                        "OPENROUTER_HIGH_REASONING_EFFORT",
                        "OPENROUTER_SLOW_REASONING_EFFORT",
                    )
                    or ""
                ).strip()
                or None,
                fallback_model_names=_csv_values(high_fallback_models),
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
# 🔹 SINGLETON INSTANCE
# ==========================================================

settings = Settings()
