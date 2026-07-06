from __future__ import annotations

import os
import sys
from io import BytesIO
from types import SimpleNamespace

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

# Other page-level tests install lightweight stubs into sys.modules during
# collection. These MRAG tests need the real vector-store module.
sys.modules.pop("agents.memory.knowledge_vector_store", None)

from agents.memory.knowledge_vector_store import KnowledgeVectorStore  # noqa: E402
from agents.multimodal.ingestion import process_file  # noqa: E402
from furnace_data.relational.models import MemoryDocument  # noqa: E402


class FakeUpload(BytesIO):
    """Small Streamlit-upload stand-in for MRAG ingestion tests."""

    name = "operator_notes.txt"


class NamedUpload(BytesIO):
    """Streamlit-upload stand-in with a configurable filename."""

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name


class FakeEmbedding:
    """Deterministic embedding client that validates retrieval mode."""

    dimension = 3

    def embed_text(self, text: str, *, input_type: str | None = None) -> list[float]:
        assert text.strip()
        assert input_type == "document"
        return [0.1, 0.2, 0.3]


class FakeQdrantClient:
    """Qdrant client stand-in recording upsert/delete payloads."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def upsert(self, **kwargs) -> None:
        self.upsert_calls.append(kwargs)

    def delete(self, **kwargs) -> None:
        self.delete_calls.append(kwargs)


class FakeDocumentRepository:
    """SQL repository stand-in recording document metadata writes."""

    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.store_calls: list[dict] = []

    def create_document(self, **kwargs) -> object:
        self.create_calls.append(kwargs)
        return SimpleNamespace(document_id="doc-row-1")

    def store_document_file(self, **kwargs) -> None:
        self.store_calls.append(kwargs)


class FakeChunkRepository:
    """SQL chunk repository stand-in recording chunk metadata writes."""

    def __init__(self) -> None:
        self.create_calls: list[dict] = []

    def create_chunks(self, **kwargs) -> int:
        self.create_calls.append(kwargs)
        return len(kwargs["parts"])


def test_process_file_persists_qdrant_points_and_sql_metadata() -> None:
    """Verify MRAG ingestion writes vectors and SQL document metadata."""
    client = FakeQdrantClient()
    store = SimpleNamespace(
        client=client,
        collection_name="furnacemind_knowledge",
        embedding_dim=3,
    )
    repository = FakeDocumentRepository()
    chunk_repository = FakeChunkRepository()
    upload = FakeUpload(b"blast furnace procedure\n" * 120)

    parts = process_file(
        upload,
        store,
        FakeEmbedding(),
        user_id="user-1",
        document_repository=repository,
        chunk_repository=chunk_repository,
    )

    assert parts
    assert client.upsert_calls
    assert client.upsert_calls[0]["collection_name"] == "furnacemind_knowledge"
    point_ids = [point.id for point in client.upsert_calls[0]["points"]]
    payload = client.upsert_calls[0]["points"][0].payload
    assert payload["user_id"] == "user-1"

    assert len(repository.create_calls) == 1
    create_call = repository.create_calls[0]
    assert create_call["user_id"] == "user-1"
    assert create_call["filename"] == "operator_notes.txt"
    assert create_call["file_type"] == "txt"
    assert create_call["qdrant_collection"] == "furnacemind_knowledge"
    assert create_call["qdrant_point_ids"] == point_ids
    assert create_call["metadata"]["source"] == "furnacemind_knowledge"
    assert create_call["metadata"]["user_id"] == "user-1"
    assert create_call["metadata"]["chunk_count"] == len(parts)
    assert create_call["metadata"]["modalities"] == ["text"]
    assert create_call["metadata"]["visual_chunk_count"] == 0

    assert len(repository.store_calls) == 1
    store_call = repository.store_calls[0]
    assert store_call["document_id"] == "doc-row-1"
    assert store_call["user_id"] == "user-1"
    assert store_call["filename"] == "operator_notes.txt"
    assert store_call["file_type"] == "txt"
    assert store_call["file_bytes"] == upload.getvalue()

    assert len(chunk_repository.create_calls) == 1
    chunk_call = chunk_repository.create_calls[0]
    assert chunk_call["document"].document_id == "doc-row-1"
    assert chunk_call["parts"] == parts
    assert chunk_call["qdrant_collection"] == "furnacemind_knowledge"


def test_process_file_embeds_uploaded_images_with_multimodal_client() -> None:
    """Verify image uploads create image vectors, not text-only vectors."""
    from PIL import Image

    class RecordingEmbedding:
        dimension = 3

        def __init__(self) -> None:
            self.text_calls: list[str] = []
            self.image_calls: list[bytes] = []

        def embed_text(
            self, text: str, *, input_type: str | None = None
        ) -> list[float]:
            self.text_calls.append(text)
            return [0.1, 0.2, 0.3]

        def embed_image(
            self, image_bytes: bytes, *, input_type: str | None = None
        ) -> list[float]:
            assert input_type == "document"
            assert image_bytes
            self.image_calls.append(image_bytes)
            return [0.7, 0.8, 0.9]

    image_buffer = BytesIO()
    Image.new("RGB", (12, 12), color="white").save(image_buffer, format="PNG")
    client = FakeQdrantClient()
    store = SimpleNamespace(
        client=client,
        collection_name="furnacemind_knowledge",
        embedding_dim=3,
    )
    embedding = RecordingEmbedding()

    repository = FakeDocumentRepository()
    upload = NamedUpload(image_buffer.getvalue(), "stove_diagram.png")

    parts = process_file(
        upload,
        store,
        embedding,
        user_id="user-1",
        document_repository=repository,
    )

    assert len(parts) == 1
    assert parts[0].modality == "image"
    assert embedding.image_calls
    assert embedding.text_calls == []

    point = client.upsert_calls[0]["points"][0]
    payload = point.payload
    assert point.vector == [0.7, 0.8, 0.9]
    assert payload["modality"] == "image"
    assert payload["has_visual"] is True
    assert "image_path" not in payload
    assert parts[0].image_bytes == upload.getvalue()
    assert repository.store_calls[0]["file_bytes"] == upload.getvalue()


def test_memory_document_reads_point_ids_from_metadata() -> None:
    """Verify ORM compatibility with the live metadata-jsonb table shape."""
    document = MemoryDocument(metadata_json={"qdrant_point_ids": ["p1", "p2"]})

    assert document.qdrant_point_ids == ["p1", "p2"]


def test_knowledge_search_filters_by_user_and_active_documents() -> None:
    """Verify Qdrant knowledge search is user-scoped and active-doc filtered."""

    class FakeSearchEmbedding:
        dimension = 3

        def embed_text(
            self, text: str, *, input_type: str | None = None
        ) -> list[float]:
            assert text == "burden SOP"
            assert input_type == "query"
            return [0.1, 0.2, 0.3]

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.query_kwargs = None

        def query_points(self, **kwargs):
            self.query_kwargs = kwargs
            points = [
                SimpleNamespace(
                    score=0.91,
                    payload={"document_id": "doc-active", "content": "active"},
                ),
                SimpleNamespace(
                    score=0.89,
                    payload={"document_id": "doc-inactive", "content": "inactive"},
                ),
            ]
            return SimpleNamespace(points=points)

    client = FakeQdrantClient()
    store = KnowledgeVectorStore.__new__(KnowledgeVectorStore)
    store.embedding = FakeSearchEmbedding()
    store.client = client
    store.collection_name = "furnacemind_knowledge"
    store._ensure_collection = lambda: None

    matches = store.search(
        "burden SOP",
        top_k=5,
        user_id="user-1",
        active_document_ids={"doc-active"},
    )

    assert matches == [
        {
            "score": 0.91,
            "payload": {"document_id": "doc-active", "content": "active"},
        }
    ]
    conditions = {
        condition.key: condition.match.value
        for condition in client.query_kwargs["query_filter"].must
    }
    assert conditions == {"user_id": "user-1"}


def test_knowledge_store_creates_payload_indexes_for_filters() -> None:
    """Verify Qdrant filter fields are indexed for MRAG search/remove."""

    class FakeIndexedQdrantClient:
        def __init__(self) -> None:
            self.index_calls: list[dict] = []

        def get_collections(self):
            return SimpleNamespace(
                collections=[SimpleNamespace(name="furnacemind_knowledge")]
            )

        def get_collection(self, collection_name: str):
            assert collection_name == "furnacemind_knowledge"
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=SimpleNamespace(size=3))
                ),
                payload_schema={},
            )

        def create_payload_index(self, **kwargs) -> None:
            self.index_calls.append(kwargs)

    client = FakeIndexedQdrantClient()
    store = KnowledgeVectorStore.__new__(KnowledgeVectorStore)
    store.client = client
    store.collection_name = "furnacemind_knowledge"
    store.embedding_dim = 3

    store._ensure_collection()

    assert [call["field_name"] for call in client.index_calls] == [
        "user_id",
        "document_id",
    ]
    assert all(
        call["collection_name"] == "furnacemind_knowledge"
        for call in client.index_calls
    )


def test_knowledge_store_deletes_qdrant_points() -> None:
    """Verify selected document removal deletes stored Qdrant point ids."""

    client = FakeQdrantClient()
    store = KnowledgeVectorStore.__new__(KnowledgeVectorStore)
    store.client = client
    store.collection_name = "furnacemind_knowledge"
    store._ensure_collection = lambda: None

    deleted = store.delete_points(["point-1", "point-2"])

    assert deleted == 2
    assert len(client.delete_calls) == 1
    delete_call = client.delete_calls[0]
    assert delete_call["collection_name"] == "furnacemind_knowledge"
    assert delete_call["points_selector"].points == ["point-1", "point-2"]
    assert delete_call["wait"] is True
