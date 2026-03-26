# embeddings/sentence_embedding.py

from sentence_transformers import SentenceTransformer
from typing import List

class SentenceEmbedding:
    def __init__(self, model_name: str, device: str = "cpu"):
        # Always use CPU to avoid CUDA issues on Streamlit Cloud
        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()