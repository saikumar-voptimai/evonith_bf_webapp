# import uuid
# from typing import Dict, List
# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct, VectorParams, Distance
# from FurnaceMind.utils.settings import settings


# class KnowledgeVectorStore:

#     def __init__(self, embedding_client):
#         self.client = QdrantClient(
#             url=settings.qdrant.url,
#             api_key=settings.qdrant.api_key,
#             timeout=settings.qdrant.timeout,
#         )

#         self.collection_name = "knowledge_docs"
#         self.embedding = embedding_client
#         self.embedding_dim = embedding_client.dimension

#         self._ensure_collection()

#     def _ensure_collection(self):
#         collections = self.client.get_collections().collections
#         existing = {c.name for c in collections}

#         if self.collection_name not in existing:
#             self.client.create_collection(
#                 collection_name=self.collection_name,
#                 vectors_config=VectorParams(
#                     size=self.embedding_dim,
#                     distance=Distance.COSINE,
#                 ),
#             )

#     def add_document(self, content: str, metadata: Dict):
#         embedding = self.embedding.embed_text(content)

#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=[
#                 PointStruct(
#                     id=str(uuid.uuid4()),
#                     vector=embedding,
#                     payload={**metadata, "content": content},
#                 )
#             ],
#             wait=True,
#         )

#     def search(self, query: str, top_k: int = 5) -> List[Dict]:
#         query_vector = self.embedding.embed_text(query)

#         results = self.client.query_points(
#             collection_name=self.collection_name,
#             query=query_vector,
#             limit=top_k,
#             with_payload=True,
#         )

#         return [
#             {"score": p.score, "payload": p.payload}
#             for p in results.points
#         ]

import uuid
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from FurnaceMind.utils.settings import settings


class KnowledgeVectorStore:
    """
    Vector store for uploaded knowledge documents (text only).
    Uses same 384-dim SentenceEmbedding as shift summaries.
    """

    def __init__(self, embedding_client):
        self.client = QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            timeout=settings.qdrant.timeout,
        )

        self.collection_name = "knowledge_docs"

        self.embedding = embedding_client
        self.embedding_dim = settings.embedding["local"].dimension  # 384

        self._ensure_collection()

    # ----------------------------------------------------
    # Ensure Qdrant Collection Exists
    # ----------------------------------------------------
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

        # Validate dimension
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors

        if vectors.size != self.embedding_dim:
            raise RuntimeError(
                f"Vector dimension mismatch for collection "
                f"'{self.collection_name}': "
                f"expected {self.embedding_dim}, got {vectors.size}"
            )

    # ----------------------------------------------------
    # Add Document
    # ----------------------------------------------------
    def add_document(self, content: str, metadata: Dict):

        embedding = self.embedding.embed([content])[0]

        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} "
                f"does not match expected {self.embedding_dim}"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        **metadata,
                        "content": content,
                    },
                )
            ],
            wait=True,
        )

    # ----------------------------------------------------
    # Search
    # ----------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:

        query_embedding = self.embedding.embed([query])[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "score": p.score,
                "payload": p.payload,
            }
            for p in results.points
        ]