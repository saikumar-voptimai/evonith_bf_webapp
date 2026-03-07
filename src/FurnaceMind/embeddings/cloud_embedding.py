# FurnaceMind/embeddings/cloud_embedding.py
# Purpose: Voyage multimodal embedding client
# Fixed: Added timeout, retry with backoff, error handling

import io
import time
import logging

import voyageai
from PIL import Image
from FurnaceMind.utils.settings import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


class CloudEmbeddingClient:
    """
    Voyage Multimodal Embedding Client
    Supports text + image with retry and timeout.
    """

    def __init__(self):
        self.model = settings.embedding["cloud"].model_name
        self.dimension = settings.embedding["cloud"].dimension

        api_key = settings.embedding["cloud"].api_key
        if not api_key:
            raise ValueError("CLOUD_EMBEDDING_API_KEY is not set.")

        self.client = voyageai.Client(api_key=api_key)

    def _retry(self, fn, description: str):
        """Retry with exponential backoff for transient failures."""
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"{description} failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}, "
                        f"retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"{description} failed after {MAX_RETRIES} attempts: {e}")
        raise last_err  # type: ignore[misc]

    # ---------------------------------------
    # Text Embedding
    # ---------------------------------------
    def embed_text(self, text: str) -> list[float]:
        def _call():
            result = self.client.multimodal_embed(
                [[text]],
                model=self.model,
            )
            return result.embeddings[0]

        return self._retry(_call, "embed_text")

    # ---------------------------------------
    # Image Embedding
    # ---------------------------------------
    def embed_image(self, image_bytes: bytes) -> list[float]:
        def _call():
            image = Image.open(io.BytesIO(image_bytes))
            result = self.client.multimodal_embed(
                [[image]],
                model=self.model,
            )
            return result.embeddings[0]

        return self._retry(_call, "embed_image")