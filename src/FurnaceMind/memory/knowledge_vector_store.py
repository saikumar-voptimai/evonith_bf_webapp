# memory/knowledge_vector_store.py

import uuid
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from FurnaceMind.utils.settings import settings


class KnowledgeVectorStore:
    """
    Vector store for Knowledge Hub documents (PDF/DOCX/PPTX/Excel + images).
    Uses CLOUD embeddings (Voyage/OpenAI) + KNOWLEDGE Qdrant collection (1024-dim).
    """

    def __init__(self, embedding_client):
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

    def _ensure_collection(self):
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

    def add_document(self, content: str, metadata: Dict):
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
        query_embedding = self.embedding.embed_text(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [{"score": p.score, "payload": p.payload} for p in results.points]

