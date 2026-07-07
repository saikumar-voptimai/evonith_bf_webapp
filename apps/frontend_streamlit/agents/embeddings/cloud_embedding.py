"""Voyage multimodal embedding client for the Knowledge Hub.

Produces 1024-dim embeddings from text or images (and future video inputs)
via the Voyage AI multimodal API.
"""

import io
from typing import List

import voyageai
from PIL import Image

from apps.frontend_streamlit.agents.embeddings.config import load_cloud_embedding_config


class CloudEmbeddingClient:
    """Voyage Multimodal Embedding Client.

    Supports text and image embedding via the ``voyage-multimodal`` model
    family.  All settings (model name, API key, dimension) are read from
    :data:`~utils.settings.settings`.

    Attributes:
        model:     Voyage model identifier.
        dimension: Expected embedding dimension (1024 by default).
        client:    Authenticated :class:`voyageai.Client` instance.
    """

    def __init__(self) -> None:
        """Initialise the client using cloud embedding settings from
        :func:`~agents.embeddings.config.load_cloud_embedding_config`.
        """
        cfg = load_cloud_embedding_config()
        self.model = cfg.model_name
        self.dimension = cfg.dimension

        self.client = voyageai.Client(api_key=cfg.api_key)

    # ---------------------------------------
    # 🔹 Text Embedding
    # ---------------------------------------
    def embed_text(self, text: str, *, input_type: str | None = None) -> List[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.
            input_type: Optional Voyage retrieval mode (``"query"`` or
                ``"document"``).

        Returns:
            A float list of length ``self.dimension``.
        """
        inputs = [[text]]

        kwargs = {"model": self.model}
        if input_type is not None:
            kwargs["input_type"] = input_type

        result = self.client.multimodal_embed(inputs, **kwargs)

        return result.embeddings[0]

    # ---------------------------------------
    # 🔹 Image Embedding
    # ---------------------------------------
    def embed_image(
        self, image_bytes: bytes, *, input_type: str | None = None
    ) -> List[float]:
        """Embed a single image from raw bytes.

        The bytes are decoded with ``PIL.Image`` before being sent to the
        Voyage multimodal endpoint.

        Args:
            image_bytes: Raw bytes of any PIL-supported image format.
            input_type: Optional Voyage retrieval mode (``"query"`` or
                ``"document"``).

        Returns:
            A float list of length ``self.dimension``.
        """
        image = Image.open(io.BytesIO(image_bytes))

        inputs = [[image]]

        kwargs = {"model": self.model}
        if input_type is not None:
            kwargs["input_type"] = input_type

        result = self.client.multimodal_embed(inputs, **kwargs)

        return result.embeddings[0]
