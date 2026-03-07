# FurnaceMind/utils/settings.py
# Purpose: Centralized configuration for FurnaceMind application
# Fixed: API keys use lazy loading (re-read from env on access)
#        to support key rotation without restart.

from __future__ import annotations

from dataclasses import dataclass, field
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ==========================================================
# LLM CONFIGURATION
# ==========================================================

@dataclass
class OpenRouterLLMConfig:
    _api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "openai/gpt-4o-mini"
    max_tokens: int = 800

    @property
    def api_key(self) -> str | None:
        """Lazy read — supports rotation without restart."""
        return os.getenv(self._api_key_env)


@dataclass
class OpenAILLMConfig:
    _api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 800
    api_mode: str = "chat_completions"

    @property
    def api_key(self) -> str | None:
        return os.getenv(self._api_key_env)


@dataclass
class LLMSettings:
    provider: str = "openrouter"
    openrouter: OpenRouterLLMConfig = field(default_factory=OpenRouterLLMConfig)
    openai: OpenAILLMConfig = field(default_factory=OpenAILLMConfig)


# ==========================================================
# EMBEDDING CONFIGURATION (DUAL SUPPORT)
# ==========================================================

@dataclass
class LocalEmbeddingConfig:
    provider: str
    model_name: str
    device: str
    dimension: int


@dataclass
class CloudEmbeddingConfig:
    provider: str
    model_name: str
    _api_key_env: str
    dimension: int

    @property
    def api_key(self) -> str | None:
        return os.getenv(self._api_key_env)


# ==========================================================
# VECTOR DATABASE CONFIG
# ==========================================================

@dataclass
class QdrantConfig:
    url: str
    _api_key_env: str | None
    collection_name: str
    embedding_dim: int
    timeout: int

    @property
    def api_key(self) -> str | None:
        if self._api_key_env:
            return os.getenv(self._api_key_env)
        return os.getenv("QDRANT_API_KEY")


# ==========================================================
# ANOMALY CONFIGURATION
# ==========================================================

@dataclass
class AnomalyConfig:
    z_warn: float = 2.0
    z_critical: float = 3.0
    delta_warn: float = 0.05


# ==========================================================
# APPLICATION CONFIG
# ==========================================================

@dataclass
class AppConfig:
    shift_hours: int = 8
    timezone: str = "UTC"
    environment: str = "dev"


# ==========================================================
# SETTINGS LOADER
# ==========================================================

class Settings:
    """
    Centralized settings with lazy API key loading.
    Supports dual embedding (local + cloud) and dual Qdrant collections.
    """

    def __init__(self):
        self.llm = self._load_llm_settings()
        self.embedding = self._load_embedding_config()

        self.qdrant_shift = self._load_qdrant_config(
            collection_env="SHIFT_QDRANT_COLLECTION",
            dim_env="SHIFT_QDRANT_EMBED_DIM",
            default_collection="furnace_shift_summaries",
            default_dim=384,
            fallback_collection_env="QDRANT_COLLECTION",
            fallback_dim_env="QDRANT_EMBED_DIM",
        )

        self.qdrant_knowledge = self._load_qdrant_config(
            collection_env="KNOWLEDGE_QDRANT_COLLECTION",
            dim_env="KNOWLEDGE_QDRANT_EMBED_DIM",
            default_collection="knowledge_docs_voyage_1024",
            default_dim=1024,
            fallback_collection_env=None,
            fallback_dim_env=None,
        )

        # Backward-compatible alias
        self.qdrant = self.qdrant_shift

        self.anomaly = AnomalyConfig()
        self.app = AppConfig()
        self._validate()

    # ------------------------------------------------------
    # LLM LOADER
    # ------------------------------------------------------
    @staticmethod
    def _load_llm_settings() -> LLMSettings:
        provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_api_mode = os.getenv("OPENAI_API_MODE", "responses").strip().lower()

        max_tokens = int(os.getenv("LLM_MAX_TOKENS", 800))

        return LLMSettings(
            provider=provider,
            openrouter=OpenRouterLLMConfig(
                base_url=openrouter_base_url,
                model_name=openrouter_model,
                max_tokens=max_tokens,
            ),
            openai=OpenAILLMConfig(
                base_url=openai_base_url if openai_base_url else None,
                model_name=openai_model,
                max_tokens=max_tokens,
                api_mode=openai_api_mode,
            ),
        )

    # ------------------------------------------------------
    # EMBEDDING LOADER
    # ------------------------------------------------------
    @staticmethod
    def _load_embedding_config():
        local_provider = os.getenv("LOCAL_EMBEDDING_PROVIDER", "sentence_transformer")
        local_model = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        local_device = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")
        local_dim = int(os.getenv("LOCAL_EMBEDDING_DIM", 384))

        cloud_provider = os.getenv("CLOUD_EMBEDDING_PROVIDER", "openai")
        cloud_model = os.getenv("CLOUD_EMBEDDING_MODEL", "text-embedding-3-large")
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
                _api_key_env="CLOUD_EMBEDDING_API_KEY",
                dimension=cloud_dim,
            ),
        }

    # ------------------------------------------------------
    # QDRANT LOADER
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

        endpoint = os.getenv("QDRANT_ENDPOINT")
        local_url = os.getenv("QDRANT_URL")

        if not endpoint and not local_url:
            raise ValueError("Either QDRANT_ENDPOINT (cloud) or QDRANT_URL (local) must be set.")

        effective_url = endpoint if endpoint else local_url

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

        api_key_env: str | None = "QDRANT_API_KEY"
        if endpoint and not os.getenv("QDRANT_API_KEY"):
            raise ValueError("QDRANT_API_KEY must be set when using Qdrant Cloud (QDRANT_ENDPOINT).")

        if "cloud.qdrant.io" in (effective_url or "") and not str(effective_url).startswith("https://"):
            raise ValueError("Qdrant Cloud endpoint must use HTTPS.")

        return QdrantConfig(
            url=effective_url,
            _api_key_env=api_key_env,
            collection_name=collection,
            embedding_dim=embedding_dim,
            timeout=timeout,
        )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------
    def _validate(self) -> None:
        if not self.llm.openrouter.api_key and not self.llm.openai.api_key:
            raise ValueError("At least one of OPENROUTER_API_KEY or OPENAI_API_KEY must be set.")
        if self.llm.provider not in {"openrouter", "openai"}:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm.provider}")
        if self.llm.openai.api_mode not in {"responses", "chat_completions"}:
            raise ValueError("OPENAI_API_MODE must be 'responses' or 'chat_completions'.")

        if self.embedding["local"].provider != "sentence_transformer":
            raise ValueError("Unsupported local embedding provider.")
        if self.embedding["cloud"].provider not in {"openai", "openrouter", "voyage"}:
            raise ValueError("Unsupported cloud embedding provider.")

        # Dimension sanity warnings
        if self.qdrant_shift.embedding_dim != self.embedding["local"].dimension:
            logger.warning(
                f"Shift Qdrant dim ({self.qdrant_shift.embedding_dim}) != "
                f"local embedding dim ({self.embedding['local'].dimension})"
            )
        if self.qdrant_knowledge.embedding_dim != self.embedding["cloud"].dimension:
            logger.warning(
                f"Knowledge Qdrant dim ({self.qdrant_knowledge.embedding_dim}) != "
                f"cloud embedding dim ({self.embedding['cloud'].dimension})"
            )


# ==========================================================
# SINGLETON INSTANCE
# ==========================================================

settings = Settings()