# FurnaceMind/memory/vector_store.py
# Purpose: Vector store for shift/day/week window-based summaries
# Fixed: Singleton embedding model (no duplicate loading),
#        proper error handling, dimension validation

import logging
from typing import Dict, List, Optional

from FurnaceMind.utils.payload_helpers import window_id_to_uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
)

from FurnaceMind.utils.settings import settings
from FurnaceMind.embeddings.sentence_embedding import SentenceEmbedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton embedding model (loaded once, shared across instances)
# ---------------------------------------------------------------------------
_embedding_instance: Optional[SentenceEmbedding] = None


def _get_shared_embedding() -> SentenceEmbedding:
    """Return a singleton SentenceEmbedding to avoid loading the model multiple times."""
    global _embedding_instance
    if _embedding_instance is None:
        emb_cfg = settings.embedding["local"]
        _embedding_instance = SentenceEmbedding(model_name=emb_cfg.model_name)
        logger.info(f"Loaded sentence embedding model: {emb_cfg.model_name}")
    return _embedding_instance


class QdrantVectorStore:
    """
    Vector store for window-based summaries (shift/day/week).
    Uses LOCAL embeddings (sentence_transformer) + SHIFT Qdrant collection.
    """

    def __init__(self):
        qcfg = settings.qdrant_shift

        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )

        self.collection_name = qcfg.collection_name

        emb_cfg = settings.embedding["local"]
        self.embedding_dim = emb_cfg.dimension

        if qcfg.embedding_dim != self.embedding_dim:
            raise RuntimeError(
                f"SHIFT_QDRANT_EMBED_DIM ({qcfg.embedding_dim}) does not match "
                f"LOCAL_EMBEDDING_DIM ({self.embedding_dim}). Fix your .env values."
            )

        # Use shared singleton instead of creating a new model
        self.embedding = _get_shared_embedding()
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
        if not embedding_text or not embedding_text.strip():
            logger.warning(f"Skipping empty embedding_text for window_id={window_id}")
            return

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