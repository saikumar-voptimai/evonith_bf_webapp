# memory/vector_store.py

from typing import Dict, List, Optional

from utils.payload_helpers import window_id_to_uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
)

from utils.settings import settings
from embeddings.sentence_embedding import SentenceEmbedding


class QdrantVectorStore:
    """
    Vector store for window-based summaries (shift/day/week).
    Uses LOCAL embeddings (sentence_transformer) + SHIFT Qdrant collection (384-dim).
    """

    def __init__(self):
        qcfg = settings.qdrant_shift  # ✅ shift collection config

        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )

        self.collection_name = qcfg.collection_name

        emb_cfg = settings.embedding["local"]  # ✅ local embedding config
        self.embedding_dim = emb_cfg.dimension

        if qcfg.embedding_dim != self.embedding_dim:
            raise RuntimeError(
                f"SHIFT_QDRANT_EMBED_DIM ({qcfg.embedding_dim}) does not match "
                f"LOCAL_EMBEDDING_DIM ({self.embedding_dim}). Fix your .env values."
            )

        self.embedding = SentenceEmbedding(model_name=emb_cfg.model_name)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        existing = {c.name for c in collections}

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            return

        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors

        if not hasattr(vectors, "size"):
            raise RuntimeError(
                f"Invalid vector schema for collection '{self.collection_name}': {vectors}. "
                "Expected unnamed VectorParams."
            )

        if vectors.size != self.embedding_dim:
            raise RuntimeError(
                f"Vector dimension mismatch for collection '{self.collection_name}': "
                f"expected {self.embedding_dim}, got {vectors.size}"
            )

    # Write operations
    def add_window(
        self,
        *,
        window_id: str,
        embedding_text: str,
        payload: Dict,
    ) -> None:
        embedding = self.embedding.embed([embedding_text])[0]

        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} "
                f"does not match expected {self.embedding_dim}"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=window_id_to_uuid(window_id),
                    vector=embedding,
                    payload=payload,
                )
            ],
            wait=True,
        )

    # Read operations
    def search_similar_windows(
        self,
        *,
        query_text: str,
        top_k: int = 3,
        window_type: Optional[str] = None,
        stability_filter: Optional[str] = None,
    ) -> List[Dict]:

        query_embedding = self.embedding.embed([query_text])[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        filtered = []
        for p in results.points:
            payload = p.payload or {}

            if window_type and payload.get("window_type") != window_type:
                continue

            if stability_filter and payload.get("overall_stability") != stability_filter:
                continue

            filtered.append({"score": p.score, "payload": payload})

        return filtered

    def get_window_by_id(self, window_id: str) -> Optional[Dict]:
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[window_id_to_uuid(window_id)],
            with_payload=True,
        )

        if not points:
            return None

        return points[0].payload
