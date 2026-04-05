"""Local sentence-transformer embedding client.

Used by :class:`~memory.vector_store.QdrantVectorStore` to produce 384-dim
cosine-normalized embeddings for shift/day/week summaries.
"""

# embeddings/sentence_embedding.py

from typing import List

from sentence_transformers import SentenceTransformer


class SentenceEmbedding:
    """Thin wrapper around ``sentence-transformers`` for batch text embedding.

    Always runs on CPU to avoid CUDA/driver issues on Streamlit Cloud.

    Attributes:
        model: The loaded :class:`~sentence_transformers.SentenceTransformer`
               instance.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load the sentence-transformer model.

        Args:
            model_name: HuggingFace model identifier
                (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``).
            device:     Ignored; always forces CPU to prevent CUDA issues.
        """
        # Always use CPU to avoid CUDA issues on Streamlit Cloud
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into L2-normalised vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float-lists, one per input text.  Each vector has the
            same dimension as the loaded model (typically 384).
        """
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return embeddings.tolist()
