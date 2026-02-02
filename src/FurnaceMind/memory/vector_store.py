# memory/vector_store.py

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


class QdrantVectorStore:
    """
    Vector store for window-based summaries (shift/day/week).
    """

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            timeout=settings.qdrant.timeout,
        )

        self.collection_name = settings.qdrant.collection_name
        self.embedding_dim = settings.qdrant.embedding_dim

        self.embedding = SentenceEmbedding(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
        )

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

        # Reject invalid schemas early
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

            filtered.append(
                {
                    "score": p.score,
                    "payload": payload,
                }
            )

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