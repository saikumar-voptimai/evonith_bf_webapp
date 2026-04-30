"""Qdrant vector store for shift/day/week/bi-week operational summaries.

Uses local sentence-transformer embeddings (384-dim, cosine similarity)
against the ``furnace_shift_summaries`` Qdrant collection.
"""

# memory/vector_store.py

from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)
0
from utils.payload_helpers import window_id_to_uuid
from utils.settings import settings


class QdrantVectorStore:
    """
    Vector store for window-based summaries (shift/day/week).
    Handles fetching and metadata management for the Reports tab.
    """

    def __init__(self) -> None:
        """Connect to Qdrant, configure the sentence-transformer embedder, and
        create the collection if it does not already exist.

        Reads all settings from :data:`~utils.settings.settings`:

        * ``settings.qdrant_shift`` – Qdrant Cloud URL, API key, collection name,
          expected embedding dimension, and request timeout.
        * ``settings.embedding["local"]`` – sentence-transformer model name and
          target dimension.

        Raises:
            RuntimeError: If the configured ``SHIFT_QDRANT_EMBED_DIM`` does not
                match ``LOCAL_EMBEDDING_DIM``, or if the existing collection was
                created with a different vector size.
        """
        qcfg = settings.qdrant_shift  # ✅ shift collection config

        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )

        self.collection_name = qcfg.collection_name
        self.embedding_dim = qcfg.embedding_dim

        from agents.embeddings.local_embedding import LocalEmbeddingClient
        self.embedding = LocalEmbeddingClient()

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection with cosine-distance IVFFLAT config if it
        does not exist, or validate the dimension of an existing collection.

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
            # Create indices for metadata filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="date",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="shift",
                field_schema=PayloadSchemaType.KEYWORD,
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
                self.collection_name, "date", PayloadSchemaType.KEYWORD
            )
        if "shift" not in payload_schema:
            self.client.create_payload_index(
                self.collection_name, "shift", PayloadSchemaType.KEYWORD
            )

    # Write operations
    def add_window(
        self,
        *,
        window_id: str,
        embedding_text: str,
        payload: Dict,
    ) -> None:
        """Upsert a summary window into the Qdrant collection.

        The *window_id* is deterministically converted to a UUID so repeated
        writes for the same window are idempotent.

        Args:
            window_id:      Human-readable ID (e.g. ``"2025-04-05_Shift_B"``).
            embedding_text: Text to embed; typically the LLM summary string.
            payload:        Arbitrary JSON-serialisable metadata to store alongside
                            the vector.

        Raises:
            ValueError: If the generated embedding dimension does not match the
                expected dimension configured for this store.
        """
        embedding = self.embedding.embed([embedding_text])[0]

        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} "
                f"does not match expected {self.embedding_dim}"
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
        """Semantic search for similar operational windows.

        Embeds *query_text* and queries Qdrant for the *top_k* nearest
        neighbours.  Optional post-query filters are applied in Python.

        Args:
            query_text:       Natural-language search query.
            top_k:            Maximum number of results to return.
            window_type:      If set, only return payloads whose
                              ``window_type`` matches (e.g. ``"shift"``).
            stability_filter: If set, only return payloads whose
                              ``overall_stability`` matches (e.g.
                              ``"unstable"``).

        Returns:
            List of ``{"score": float, "payload": dict}`` dicts sorted by
            descending similarity score.
        """
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

            if (
                stability_filter
                and payload.get("overall_stability") != stability_filter
            ):
                continue

            filtered.append({"score": p.score, "payload": payload})

        return filtered

    def get_window_by_id(self, window_id: str) -> Optional[Dict]:
        """Retrieve a single window payload by exact *window_id*.

        Args:
            window_id: Human-readable window ID (converted to UUID internally).

        Returns:
            The payload dict if found, or ``None``.
        """
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
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="date",
                        match=MatchValue(value=date_str),
                    ),
                    FieldCondition(
                        key="shift",
                        match=MatchValue(value=shift_label),
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