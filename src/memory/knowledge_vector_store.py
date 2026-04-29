"""Qdrant vector store for Knowledge Hub documents.

Uses cloud (Voyage/OpenAI) multimodal embeddings (1024-dim, cosine similarity)
against the ``knowledge_docs_voyage_1024`` Qdrant collection.
"""

# memory/knowledge_vector_store.py

import uuid
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from utils.settings import settings


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
        embedding = self.embedding.embed_text(content)

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

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search over the knowledge document collection.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of ``{"score": float, "payload": dict}`` dicts sorted by
            descending cosine similarity score.
        """
        query_embedding = self.embedding.embed_text(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [{"score": p.score, "payload": p.payload} for p in results.points]

    def delete_points(self, point_ids: list[str]) -> None:
        """Delete knowledge points by Qdrant point id."""
        if not point_ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )
