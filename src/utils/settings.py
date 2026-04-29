"""Centralised configuration loader for all FurnaceMind services.

Reads environment variables (loaded from ``.env`` via python-dotenv) and
assembles typed :mod:`dataclasses` for LLM, embedding, Qdrant, anomaly
detection, and general app settings.

The module-level singleton :data:`settings` is the single source of truth
for runtime configuration; import it via::

    from FurnaceMind.utils.settings import settings
"""

# FurnaceMind/utils/settings.py
# Purpose: Centralized configuration for FurnaceMind application

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_FILE)


# ==========================================================
# 🔹 LLM CONFIGURATION
# ==========================================================


@dataclass
class OpenRouterLLMConfig:
    """Connection and model settings for the OpenRouter LLM gateway.

    Attributes:
        api_key:    OpenRouter API key (``OPENROUTER_API_KEY`` env var).
        base_url:   OpenRouter API base URL.
        model_name: Fully-qualified model identifier (e.g. ``openai/gpt-4o-mini``).
        ai_cooperate_fast_model_name: Fast AI Co-Operate model override.
        ai_cooperate_reasoning_model_name: Deep-reasoning AI Co-Operate model override.
        max_tokens: Maximum completion tokens to request.
    """

    api_key: str | None
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "openai/gpt-4o-mini"
    ai_cooperate_fast_model_name: str = "google/gemini-2.5-flash"
    ai_cooperate_reasoning_model_name: str = "openai/gpt-5.4"
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
# 🔹 EMBEDDING CONFIGURATION (DUAL SUPPORT)
# ==========================================================


@dataclass
class LocalEmbeddingConfig:
    """Configuration for the local (on-device) sentence-transformer embedder.

    Attributes:
        provider:   Embedding library — always ``"sentence_transformer"``.
        model_name: HuggingFace model path (e.g. ``all-MiniLM-L6-v2``).
        device:     Torch device string — ``"cpu"`` or ``"cuda"``.
        dimension:  Output embedding dimension (default 384).
    """

    provider: str
    model_name: str
    device: str
    dimension: int


@dataclass
class CloudEmbeddingConfig:
    """Configuration for the cloud embedding service (OpenAI / Voyage).

    Attributes:
        provider:   Service name — ``"openai"``, ``"voyage"``, or ``"openrouter"``.
        model_name: Model identifier (e.g. ``text-embedding-3-large``).
        api_key:    API key for the embedding service.
        dimension:  Output embedding dimension (default 1024).
    """

    provider: str
    model_name: str
    api_key: str | None
    dimension: int


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
# 🔹 ANOMALY CONFIGURATION
# ==========================================================


@dataclass
class AnomalyConfig:
    """Thresholds for the shift anomaly and stability index computations.

    Attributes:
        z_warn:     Z-score threshold that triggers a *warning* anomaly.
        z_critical: Z-score threshold that triggers a *critical* anomaly.
        delta_warn: Fractional delta threshold for early-drift detection.
    """

    z_warn: float = 2.0
    z_critical: float = 3.0
    delta_warn: float = 0.05


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

    The ``qdrant`` attribute is an alias for ``qdrant_shift`` for backward
    compatibility with existing call-sites.

    Attributes:
        llm:               :class:`LLMSettings` for the active LLM provider.
        embedding:         ``dict`` with ``"local"`` and ``"cloud"`` embedding configs.
        qdrant_shift:      :class:`QdrantConfig` for the shift summary collection.
        qdrant_knowledge:  :class:`QdrantConfig` for the knowledge document collection.
        qdrant:            Alias for ``qdrant_shift`` (backward compatibility).
        anomaly:           :class:`AnomalyConfig` detection thresholds.
        app:               :class:`AppConfig` general runtime settings.
    """

    def __init__(self) -> None:
        """Load all configuration sections from environment variables.

        Raises:
            ValueError: If required environment variables are missing or invalid.
        """

        self.llm = self._load_llm_settings()
        self.embedding = self._load_embedding_config()

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
            default_collection="knowledge_docs_voyage_1024",
            default_dim=1024,
            # do NOT fall back to QDRANT_COLLECTION by default here,
            # because that usually points to shift summaries and causes mixups.
            fallback_collection_env=None,
            fallback_dim_env=None,
        )

        # Backward-compatible alias: existing code that uses settings.qdrant
        # will keep using the shift store unless you change those call-sites.
        self.qdrant = self.qdrant_shift

        self.anomaly = AnomalyConfig()
        self.app = AppConfig()
        self._validate()

    # ------------------------------------------------------
    # LLM LOADER
    # ------------------------------------------------------
    @staticmethod
    def _load_llm_settings() -> LLMSettings:
        """Build :class:`LLMSettings` from environment variables.

        Returns:
            Populated :class:`LLMSettings` instance.
        """
        provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        ai_cooperate_fast_model = os.getenv(
            "AI_COOPERATE_FAST_MODEL",
            "google/gemini-2.5-flash",
        )
        ai_cooperate_reasoning_model = os.getenv(
            "AI_COOPERATE_REASONING_MODEL",
            "openai/gpt-5.4",
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
                ai_cooperate_fast_model_name=ai_cooperate_fast_model,
                ai_cooperate_reasoning_model_name=ai_cooperate_reasoning_model,
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
    # EMBEDDING LOADER (DUAL MODE)
    # ------------------------------------------------------
    @staticmethod
    def _load_embedding_config() -> dict:
        """Build a dual-entry embedding config dict from environment variables.

        Returns:
            Dict with keys ``"local"`` (:class:`LocalEmbeddingConfig`) and
            ``"cloud"`` (:class:`CloudEmbeddingConfig`).
        """
        # Local (Shift Reports)
        local_provider = os.getenv("LOCAL_EMBEDDING_PROVIDER", "sentence_transformer")
        local_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        local_device = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")
        local_dim = int(os.getenv("LOCAL_EMBEDDING_DIM", 384))

        # Cloud (Knowledge Hub)
        cloud_provider = os.getenv("CLOUD_EMBEDDING_PROVIDER", "openai")
        cloud_model = os.getenv("CLOUD_EMBEDDING_MODEL", "text-embedding-3-large")
        cloud_api_key = os.getenv("CLOUD_EMBEDDING_API_KEY")
        cloud_dim = int(os.getenv("CLOUD_EMBEDDING_DIM", 1024))

        return {
            "local": LocalEmbeddingConfig(
                provider=local_provider,
                model_name=local_model,
                device=local_device,
                dimension=local_dim,
            ),
            "cloud": CloudEmbeddingConfig(
                provider=cloud_provider,
                model_name=cloud_model,
                api_key=cloud_api_key,
                dimension=cloud_dim,
            ),
        }

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

        # Embedding validation
        if self.embedding["local"].provider != "sentence_transformer":
            raise ValueError("Unsupported local embedding provider.")
        if self.embedding["cloud"].provider not in {"openai", "openrouter", "voyage"}:
            raise ValueError("Unsupported cloud embedding provider.")

        # Dimension sanity (optional but helpful)
        if self.qdrant_shift.embedding_dim != self.embedding["local"].dimension:
            # don’t hard-fail; just warn via exception message if you want
            pass
        if self.qdrant_knowledge.embedding_dim != self.embedding["cloud"].dimension:
            pass


# ==========================================================
# 🔹 SINGLETON INSTANCE
# ==========================================================

settings = Settings()
