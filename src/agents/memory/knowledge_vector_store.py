"""Qdrant vector store for Knowledge Hub documents.

Uses Voyage multimodal embeddings (1024-dim, cosine similarity)
against the ``furnacemind_knowledge`` Qdrant collection.
"""

# memory/knowledge_vector_store.py

import uuid
from typing import Dict, Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from utils.settings import settings

_INDEXED_PAYLOAD_FIELDS = ("user_id", "document_id")


def _embed_text(embedding_client, text: str, *, input_type: str) -> list[float]:
    """Embed text with retrieval mode when the client supports it."""
    try:
        return embedding_client.embed_text(text, input_type=input_type)
    except TypeError:
        return embedding_client.embed_text(text)


class KnowledgeVectorStore:
    """
    Vector store for Knowledge Hub documents (PDF/DOCX/PPTX/Excel + images).
    Uses CLOUD embeddings (Voyage/OpenAI) + KNOWLEDGE Qdrant collection (1024-dim).
    """

    def __init__(self, embedding_client) -> None:
        """Connect to Qdrant and configure the cloud embedding client.

        Args:
            embedding_client: Any embedding client that exposes
                ``embed_text(text: str) -> List[float]`` and a ``dimension``
                integer attribute (e.g.
                :class:`~embeddings.cloud_embedding.CloudEmbeddingClient`).

        Raises:
            RuntimeError: If ``KNOWLEDGE_QDRANT_EMBED_DIM`` does not match the
                dimension reported by *embedding_client*, or if the existing
                Qdrant collection has a different vector size.
        """
        qcfg = settings.qdrant_knowledge  # ✅ knowledge collection config

        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )

        self.collection_name = qcfg.collection_name

        self.embedding = embedding_client
        self.embedding_dim = embedding_client.dimension

        # Safety: ensure Qdrant dimension matches embedding client dimension
        if qcfg.embedding_dim != self.embedding_dim:
            raise RuntimeError(
                f"KNOWLEDGE_QDRANT_EMBED_DIM ({qcfg.embedding_dim}) does not match "
                f"CLOUD_EMBEDDING_DIM ({self.embedding_dim}). Fix your .env values."
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create or validate the Qdrant knowledge collection.

        Raises:
            RuntimeError: If the existing collection has an incompatible vector
                schema or mismatched dimension.
        """
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
            self._ensure_payload_indexes()
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

        self._ensure_payload_indexes(getattr(info, "payload_schema", None) or {})

    def _ensure_payload_indexes(self, payload_schema: dict | None = None) -> None:
        """Ensure Qdrant can filter MRAG points by ownership and document id."""
        existing = payload_schema or {}
        for field_name in _INDEXED_PAYLOAD_FIELDS:
            if field_name in existing:
                continue
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def add_document(self, content: str, metadata: Dict) -> None:
        """Embed *content* and upsert the document into Qdrant.

        A random UUID is generated for each document (documents are not
        deduplicated automatically).

        Args:
            content:  Text content to embed and store.
            metadata: Arbitrary metadata dict stored alongside the vector
                      (e.g. ``source``, ``page``, ``file_name``).

        Raises:
            ValueError: If the generated embedding dimension does not match
                the expected dimension for this collection.
        """
        embedding = _embed_text(self.embedding, content, input_type="document")

        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match expected {self.embedding_dim}"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={**metadata, "content": content},
                )
            ],
            wait=True,
        )

    def delete_points(self, point_ids: Iterable[str]) -> int:
        """Delete Qdrant points for a removed knowledge document."""
        ids = [str(point_id) for point_id in point_ids if str(point_id).strip()]
        if not ids:
            return 0
        self._ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
            wait=True,
        )
        return len(ids)

    def delete_document(
        self,
        *,
        document_id: str,
        user_id: str | None = None,
    ) -> None:
        """Delete Qdrant points by MRAG document id when point ids are unavailable."""
        conditions = [
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            )
        ]
        if user_id:
            conditions.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            )
        self._ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=conditions),
            wait=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        active_document_ids: Iterable[str] | None = None,
    ) -> List[Dict]:
        """Semantic search over the knowledge document collection.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.
            user_id: Optional owner filter for user-scoped retrieval.
            active_document_ids: Optional active SQL document ids to post-filter
                Qdrant matches.

        Returns:
            List of ``{"score": float, "payload": dict}`` dicts sorted by
            descending cosine similarity score.
        """
        self._ensure_collection()
        query_embedding = _embed_text(self.embedding, query, input_type="query")
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    ),
                ]
            )

        active_ids = set(active_document_ids or [])
        limit = top_k * 5 if active_ids else top_k

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        matches = []
        for point in results.points:
            payload = point.payload or {}
            if active_ids and payload.get("document_id") not in active_ids:
                continue
            matches.append({"score": point.score, "payload": payload})
            if len(matches) >= top_k:
                break
        return matches
