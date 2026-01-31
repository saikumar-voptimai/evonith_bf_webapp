# FurnaceMind/utils/settings.py
# Purpose: Centralized configuration for FurnaceMind application

from __future__ import annotations

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()



# LLM CONFIGURATION
@dataclass
class OpenRouterLLMConfig:
    api_key: str | None
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "openai/gpt-4o-mini"
    max_tokens: int = 800


@dataclass
class OpenAILLMConfig:
    api_key: str | None
    base_url: str | None = None
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 800
    api_mode: str = "chat_completions"  # "responses" | "chat_completions"


@dataclass
class LLMSettings:
    """
    LLM configuration supporting BOTH OpenRouter and OpenAI.
    """

    provider: str = "openrouter"

    openrouter: OpenRouterLLMConfig = field(
        default_factory=lambda: OpenRouterLLMConfig(api_key=None)
    )

    openai: OpenAILLMConfig = field(
        default_factory=lambda: OpenAILLMConfig(api_key=None)
    )



# EMBEDDING CONFIGURATION
@dataclass
class EmbeddingConfig:
    provider: str
    model_name: str
    device: str


# VECTOR DATABASE CONFIG
@dataclass
class QdrantConfig:
    url: str
    api_key: str | None
    collection_name: str
    embedding_dim: int
    timeout: int



# ANOMALY CONFIGURATION
@dataclass
class AnomalyConfig:
    z_warn: float = 2.0
    z_critical: float = 3.0
    delta_warn: float = 0.05



# APPLICATION CONFIG
@dataclass
class AppConfig:
    shift_hours: int = 8
    timezone: str = "UTC"
    environment: str = "dev"



# SETTINGS LOADER
class Settings:
    def __init__(self):
        self.llm = self._load_llm_settings()
        self.embedding = self._load_embedding_config()
        self.qdrant = self._load_qdrant_config()
        self.anomaly = AnomalyConfig()
        self.app = AppConfig()

        self._validate()


    # LOADERS
    @staticmethod
    def _load_llm_settings() -> LLMSettings:
        provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

        # OpenRouter
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        openrouter_model = os.getenv(
            "OPENROUTER_MODEL", "openai/gpt-4o-mini"
        )

        # OpenAI
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

    @staticmethod
    def _load_embedding_config() -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformer"),
            model_name=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        )

    @staticmethod
    def _load_qdrant_config() -> QdrantConfig:
        endpoint = os.getenv("QDRANT_ENDPOINT")
        local_url = os.getenv("QDRANT_URL")

        if not endpoint and not local_url:
            raise ValueError(
                "Either QDRANT_ENDPOINT (cloud) or QDRANT_URL (local) must be set."
            )

        if endpoint and not os.getenv("QDRANT_API_KEY"):
            raise ValueError(
                "QDRANT_API_KEY must be set when using Qdrant Cloud."
            )

        effective_url = endpoint if endpoint else local_url

        return QdrantConfig(
            url=effective_url,
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv(
                "QDRANT_COLLECTION", "furnace_shift_summaries"
            ),
            embedding_dim=int(os.getenv("QDRANT_EMBED_DIM", 384)),
            timeout=int(os.getenv("QDRANT_TIMEOUT", 30)),
        )


    # VALIDATION
    def _validate(self) -> None:
        if not self.llm.openrouter.api_key and not self.llm.openai.api_key:
            raise ValueError(
                "At least one of OPENROUTER_API_KEY or OPENAI_API_KEY must be set."
            )

        if self.llm.provider not in {"openrouter", "openai"}:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {self.llm.provider}"
            )

        if self.llm.openai.api_mode not in {"responses", "chat_completions"}:
            raise ValueError(
                "OPENAI_API_MODE must be 'responses' or 'chat_completions'."
            )

        if self.embedding.provider not in {
            "sentence_transformer",
            "openrouter",
        }:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding.provider}"
            )

        if (
            "cloud.qdrant.io" in self.qdrant.url
            and not self.qdrant.url.startswith("https://")
        ):
            raise ValueError(
                "Qdrant Cloud endpoint must use HTTPS."
            )


# Singleton
settings = Settings()
