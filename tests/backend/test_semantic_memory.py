from __future__ import annotations

from types import SimpleNamespace

from apps.frontend_streamlit.agents.memory.semantic_memory import (
    SemanticMemoryRecord,
    SemanticMemoryService,
    SemanticMemoryVectorStore,
    _parse_facts_response,
)
from apps.frontend_streamlit.utils.settings import SemanticMemoryConfig


def _config(*, enabled: bool = True) -> SemanticMemoryConfig:
    """Build a compact semantic-memory config for unit tests."""
    return SemanticMemoryConfig(
        enabled=enabled,
        collection_name="test_furnacemind_memory",
        llm_model="openai/gpt-4o-mini",
        max_memories=2,
        search_threshold=0.25,
    )


class FakeVectorStore:
    """In-memory Qdrant stand-in used to verify search and upsert calls."""

    def __init__(self, *, events: list[str] | None = None) -> None:
        """Create the fake vector store with optional event recording."""
        self.collection_name = "test_furnacemind_memory"
        self.search_calls = []
        self.upsert_calls = []
        self.events = events if events is not None else []

    def search_memories(self, **kwargs):
        """Record a search call and return one relevant memory."""
        self.search_calls.append(kwargs)
        return [
            SemanticMemoryRecord(
                memory="Operator prefers concise action-first answers.",
                score=0.91,
                metadata={"source": "test"},
                memory_id="fact-1",
            )
        ]

    def upsert_fact(self, **kwargs):
        """Record a Qdrant upsert call and return a deterministic point id."""
        self.upsert_calls.append(kwargs)
        self.events.append(f"qdrant:{kwargs['fact_id']}")
        return f"point-{kwargs['fact_id']}"


class FakeRepository:
    """In-memory SQL repository stand-in for memory fact persistence."""

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        fail_create: bool = False,
        existing_fact_texts: set[str] | None = None,
    ) -> None:
        """Create the fake repository with optional SQL failure behavior."""
        self.create_calls = []
        self.exists_calls = []
        self.mark_calls = []
        self.events = events if events is not None else []
        self.fail_create = fail_create
        self.existing_fact_texts = existing_fact_texts or set()

    def create_fact(self, **kwargs):
        """Record a SQL create call and return a fake persisted fact row."""
        self.create_calls.append(kwargs)
        if self.fail_create:
            raise RuntimeError("postgres unavailable")

        fact_id = f"fact-{len(self.create_calls)}"
        self.events.append(f"sql:{fact_id}")
        return SimpleNamespace(fact_id=fact_id)

    def fact_exists(self, **kwargs):
        """Return whether a fake SQL fact already exists."""
        self.exists_calls.append(kwargs)
        normalized = " ".join(str(kwargs["fact_text"]).split()).lower()
        existing = {" ".join(item.split()).lower() for item in self.existing_fact_texts}
        return normalized in existing

    def mark_fact_indexed(self, **kwargs):
        """Record that a saved SQL fact was indexed into Qdrant."""
        self.mark_calls.append(kwargs)


class FakeLLM:
    """Fake extraction LLM returning JSON facts."""

    model = "test-memory-model"

    def __init__(self, response: str | None = None) -> None:
        """Create the fake LLM with a configurable extraction response."""
        self.response = response or (
            '{"facts": ['
            '"Operating preference | Furnace=BF2 | Guidance=keep O2 conservative.",'
            '"Monitoring rule | Watch=Si slope before increasing O2."'
            "]}"
        )
        self.generate_calls = []

    def generate(self, **kwargs):
        """Record the extraction prompt and return the configured response."""
        self.generate_calls.append(kwargs)
        return self.response


def _service(
    *,
    repository=None,
    vector_store=None,
    llm=None,
    enabled: bool = True,
) -> SemanticMemoryService:
    """Build a semantic-memory service wired to fake dependencies."""
    return SemanticMemoryService(
        config=_config(enabled=enabled),
        repository=repository if repository is not None else FakeRepository(),
        vector_store=vector_store if vector_store is not None else FakeVectorStore(),
        extraction_llm=llm if llm is not None else FakeLLM(),
    )


def test_semantic_memory_context_is_qdrant_scoped_and_prompt_ready() -> None:
    """Verify runtime context is fetched from Qdrant and prompt formatted."""
    vector_store = FakeVectorStore()
    service = _service(vector_store=vector_store)

    context = service.context_for_query(query="How should I answer?", user_id="user-1")

    assert "LONG-TERM SEMANTIC MEMORY" in context
    assert "Operator prefers concise action-first answers." in context
    assert "score=0.910" in context
    assert vector_store.search_calls == [
        {
            "query": "How should I answer?",
            "user_id": "user-1",
            "limit": 2,
        }
    ]


def test_semantic_memory_saves_to_postgres_before_qdrant() -> None:
    """Verify each extracted memory is saved to SQL before Qdrant indexing."""
    events: list[str] = []
    repository = FakeRepository(events=events)
    vector_store = FakeVectorStore(events=events)
    llm = FakeLLM()
    service = _service(
        repository=repository,
        vector_store=vector_store,
        llm=llm,
    )

    saved_ids = service.add_summary(
        user_id="user-1",
        conversation_id="conv-1",
        summary=(
            "The operator prefers conservative BF2 oxygen changes when hot "
            "metal silicon is falling."
        ),
        source_message_id_start="msg-1",
        source_message_id_end="msg-8",
        summarized_message_count=8,
        metadata={"summary_trigger": "window"},
    )

    assert saved_ids == ["fact-1", "fact-2", "fact-3"]
    assert events == [
        "sql:fact-1",
        "qdrant:fact-1",
        "sql:fact-2",
        "qdrant:fact-2",
        "sql:fact-3",
        "qdrant:fact-3",
    ]
    assert len(repository.create_calls) == 3
    assert len(vector_store.upsert_calls) == 3
    assert len(repository.mark_calls) == 3

    first_sql_call = repository.create_calls[0]
    assert first_sql_call["source_conversation_id"] == "conv-1"
    assert first_sql_call["metadata"]["source"] == "furnacemind_summary"
    assert first_sql_call["metadata"]["conversation_id"] == "conv-1"
    assert first_sql_call["metadata"]["source_message_id_start"] == "msg-1"
    assert first_sql_call["metadata"]["source_message_id_end"] == "msg-8"
    assert first_sql_call["metadata"]["summarized_message_count"] == 8
    assert first_sql_call["metadata"]["summary_trigger"] == "window"
    assert first_sql_call["metadata"]["extraction_model"] == "test-memory-model"

    snapshot_sql_call = repository.create_calls[-1]
    assert snapshot_sql_call["metadata"]["source"] == ("furnacemind_summary_snapshot")
    assert "cumulative conversation summary snapshot" in snapshot_sql_call["fact_text"]

    first_qdrant_call = vector_store.upsert_calls[0]
    assert first_qdrant_call["fact_id"] == "fact-1"
    assert first_qdrant_call["user_id"] == "user-1"
    assert first_qdrant_call["metadata"]["source"] == "furnacemind_summary"


def test_semantic_memory_skips_existing_sql_facts() -> None:
    """Verify cumulative summaries do not repeatedly write the same fact."""
    existing_fact = (
        "Operating preference | Furnace=BF2 | Guidance=keep O2 conservative."
    )
    repository = FakeRepository(existing_fact_texts={existing_fact})
    vector_store = FakeVectorStore()
    service = _service(repository=repository, vector_store=vector_store)

    saved_ids = service.add_summary(
        user_id="user-1",
        conversation_id="conv-1",
        summary="BF2 O2 guidance was discussed.",
    )

    assert saved_ids == ["fact-1", "fact-2"]
    assert repository.exists_calls[0]["fact_text"] == existing_fact
    assert [call["fact_text"] for call in repository.create_calls] == [
        "Monitoring rule | Watch=Si slope before increasing O2.",
        (
            "FurnaceMind cumulative conversation summary snapshot:\n\n"
            "BF2 O2 guidance was discussed."
        ),
    ]
    assert len(vector_store.upsert_calls) == 2


def test_semantic_memory_does_not_write_qdrant_when_sql_save_fails() -> None:
    """Verify Qdrant is not mutated when the PostgreSQL save fails."""
    repository = FakeRepository(fail_create=True)
    vector_store = FakeVectorStore()
    service = _service(repository=repository, vector_store=vector_store)

    saved_ids = service.add_summary(
        user_id="user-1",
        conversation_id="conv-1",
        summary="Operator prefers conservative O2 changes.",
    )

    assert saved_ids == []
    assert repository.create_calls
    assert vector_store.upsert_calls == []
    assert service.last_error == "postgres unavailable"


def test_semantic_memory_disabled_is_noop() -> None:
    """Verify the service is a no-op when semantic memory is disabled."""
    repository = FakeRepository()
    vector_store = FakeVectorStore()
    service = _service(
        repository=repository,
        vector_store=vector_store,
        enabled=False,
    )

    assert service.context_for_query(query="anything", user_id="user-1") == ""
    assert (
        service.add_summary(
            user_id="user-1",
            conversation_id="conv-1",
            summary="hello",
        )
        == []
    )
    assert repository.create_calls == []
    assert vector_store.search_calls == []
    assert vector_store.upsert_calls == []


def test_semantic_memory_search_failure_degrades_cleanly() -> None:
    """Verify Qdrant search failures return empty context without raising."""

    class FailingVectorStore:
        """Vector store stand-in that always fails search."""

        def search_memories(self, **kwargs):
            """Raise a deterministic Qdrant-style failure."""
            raise RuntimeError("qdrant unavailable")

    service = _service(vector_store=FailingVectorStore())

    assert service.context_for_query(query="anything", user_id="user-1") == ""
    assert service.last_error == "qdrant unavailable"


def test_vector_store_search_filters_by_user() -> None:
    """Verify Qdrant runtime search is scoped to one user."""

    class FakeEmbedding:
        """Tiny embedding client for direct vector-store search tests."""

        dimension = 3

        def embed_text(self, text: str) -> list[float]:
            """Return a deterministic query vector."""
            return [0.1, 0.2, 0.3]

    class FakeQdrantClient:
        """Qdrant client stand-in that records query filters."""

        def __init__(self) -> None:
            """Create the fake client with no captured filter."""
            self.query_filter = None

        def query_points(self, **kwargs):
            """Record query arguments and return no points."""
            self.query_filter = kwargs["query_filter"]
            return SimpleNamespace(points=[])

    store = SemanticMemoryVectorStore.__new__(SemanticMemoryVectorStore)
    store.embedding = FakeEmbedding()
    store.client = FakeQdrantClient()
    store.collection_name = "test_furnacemind_memory"
    store.search_threshold = None

    records = store.search_memories(query="O2 guidance", user_id="user-1", limit=3)

    assert records == []
    conditions = {
        condition.key: condition.match.value
        for condition in store.client.query_filter.must
    }
    assert conditions == {"user_id": "user-1"}


def test_parse_facts_response_handles_json_and_deduplicates() -> None:
    """Verify fact parsing handles fenced JSON and removes duplicates."""
    facts = _parse_facts_response(
        '```json\n{"facts": ["Keep O2 conservative.", "Keep O2 conservative."]}\n```'
    )

    assert facts == ["Keep O2 conservative."]


def test_semantic_memory_deletes_document_related_vectors_before_sql() -> None:
    """Verify document revocation removes memory vectors before SQL facts."""
    events: list[str] = []
    fact = SimpleNamespace(fact_id="fact-1", qdrant_point_id="point-1")

    class CleanupRepository:
        """Repository fake for document-related memory cleanup."""

        def __init__(self) -> None:
            self.lookup_calls: list[dict] = []
            self.delete_calls: list[list[str]] = []

        def list_document_related_facts(self, **kwargs):
            self.lookup_calls.append(kwargs)
            return [fact]

        def delete_facts(self, fact_ids: list[str]) -> int:
            events.append("sql-delete")
            self.delete_calls.append(fact_ids)
            return len(fact_ids)

    class CleanupVectorStore:
        """Vector-store fake that records deleted point ids."""

        collection_name = "test_furnacemind_memory"

        def __init__(self) -> None:
            self.delete_calls: list[list[str]] = []

        def delete_points(self, point_ids: list[str]) -> int:
            events.append("qdrant-delete")
            self.delete_calls.append(point_ids)
            return len(point_ids)

    repository = CleanupRepository()
    vector_store = CleanupVectorStore()
    service = _service(repository=repository, vector_store=vector_store)

    deleted = service.delete_document_related_memories(
        user_id="user-1",
        sql_document_id="doc-row-1",
        mrag_document_id="doc-bmo",
        filename="BMO_Analysis.pptx",
    )

    assert deleted == 1
    assert events == ["qdrant-delete", "sql-delete"]
    assert vector_store.delete_calls == [["point-1"]]
    assert repository.delete_calls == [["fact-1"]]
    assert repository.lookup_calls[0]["mrag_document_id"] == "doc-bmo"
