"""Embedding configuration for local and cloud embedding clients.

Reads from environment variables (loaded from ``.env`` via python-dotenv).
Import directly from this module instead of ``utils.settings``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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
        model_name: Model identifier (e.g. ``voyage-multimodal-3.5``).
        api_key:    API key for the embedding service.
        dimension:  Output embedding dimension (default 1024).
    """

    provider: str
    model_name: str
    api_key: str | None
    dimension: int


def load_local_embedding_config() -> LocalEmbeddingConfig:
    """Build :class:`LocalEmbeddingConfig` from environment variables."""
    return LocalEmbeddingConfig(
        provider=os.getenv("LOCAL_EMBEDDING_PROVIDER", "sentence_transformer"),
        model_name=os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        device=os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu"),
        dimension=int(os.getenv("LOCAL_EMBEDDING_DIM", 384)),
    )


def load_cloud_embedding_config() -> CloudEmbeddingConfig:
    """Build :class:`CloudEmbeddingConfig` from environment variables."""
    return CloudEmbeddingConfig(
        provider=os.getenv("CLOUD_EMBEDDING_PROVIDER", "voyage"),
        model_name=os.getenv("CLOUD_EMBEDDING_MODEL", "voyage-multimodal-3.5"),
        api_key=os.getenv("CLOUD_EMBEDDING_API_KEY") or os.getenv("VOYAGE_API_KEY"),
        dimension=int(os.getenv("CLOUD_EMBEDDING_DIM", 1024)),
    )
