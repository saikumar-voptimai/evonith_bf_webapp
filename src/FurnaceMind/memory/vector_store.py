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


class QdrantVectorStore:
    """
    Vector store for window-based summaries (shift/day/week).
    Handles fetching and metadata management for the Reports tab.
    """

    def __init__(self):
        qcfg = settings.qdrant_shift  # ✅ shift collection config

        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )

        self.collection_name = qcfg.collection_name
        self.embedding_dim = qcfg.embedding_dim

        from FurnaceMind.embeddings.local_embedding import LocalEmbeddingClient
        self.embedding = LocalEmbeddingClient()

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models
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
            # Create indices for metadata filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="date",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="shift",
                field_schema=models.PayloadSchemaType.KEYWORD,
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
            
        # Ensure indices exist even if collection already existed
        payload_schema = info.payload_schema or {}
        if "date" not in payload_schema:
            self.client.create_payload_index(
                self.collection_name, "date", models.PayloadSchemaType.KEYWORD
            )
        if "shift" not in payload_schema:
            self.client.create_payload_index(
                self.collection_name, "shift", models.PayloadSchemaType.KEYWORD
            )

    # Read operations
    def get_window_by_id(self, window_id: str) -> Optional[Dict]:
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[window_id_to_uuid(window_id)],
            with_payload=True,
        )

        if not points:
            return None

        return points[0].payload

    def get_report_by_metadata(self, date_str: str, shift_label: str) -> Optional[Dict]:
        """
        Fetch a report by date and shift metadata.
        """
        from qdrant_client import models

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="date",
                        match=models.MatchValue(value=date_str),
                    ),
                    models.FieldCondition(
                        key="shift",
                        match=models.MatchValue(value=shift_label),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return None

        return results[0].payload

    def search_similar_shifts(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search for similar historical shifts.
        """
        query_embedding = self.embedding.embed_text(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [{"score": p.score, "payload": p.payload} for p in results.points]
