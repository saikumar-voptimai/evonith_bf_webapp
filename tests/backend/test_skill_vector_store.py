from __future__ import annotations

from types import SimpleNamespace

from apps.frontend_streamlit.agents.furnacemind.skill_vector_store import SkillVectorStore


class FakeEmbeddingClient:
    """Embedding stand-in that records whether a query vector was requested."""

    dimension = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def embed_text(self, text: str, *, input_type: str | None = None) -> list[float]:
        self.calls.append((text, input_type))
        return [1.0, 0.0]


class FakeQdrantClient:
    """Qdrant stand-in that returns one stored skill point."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query_points(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={"skill_id": "skill_heatload", "slug": "heatload"},
                    score=0.91,
                )
            ]
        )


def _store() -> SkillVectorStore:
    """Build a SkillVectorStore without opening a real Qdrant connection."""
    store = object.__new__(SkillVectorStore)
    store.embedding = FakeEmbeddingClient()
    store.client = FakeQdrantClient()
    store.collection_name = "furnacemind_skills"
    return store


def test_search_returns_no_matches_when_active_filter_is_empty() -> None:
    """An explicit empty active-id list means no SQL skills are injectable."""
    store = _store()

    matches = store.search(query="check heatloads", active_skill_ids=[], top_k=3)

    assert matches == []
    assert store.embedding.calls == []
    assert store.client.calls == []


def test_search_allows_matches_when_no_active_filter_is_provided() -> None:
    """A missing active-id filter keeps the vector store usable for admin probes."""
    store = _store()

    matches = store.search(query="check heatloads", active_skill_ids=None, top_k=3)

    assert [match.skill_id for match in matches] == ["skill_heatload"]
    assert store.embedding.calls == [("check heatloads", "query")]
    assert store.client.calls[0]["collection_name"] == "furnacemind_skills"


def test_search_filters_matches_to_active_sql_skill_ids() -> None:
    """Runtime search keeps only skill ids that SQL currently marks active."""
    store = _store()

    assert (
        store.search(
            query="check heatloads",
            active_skill_ids=["skill_unit_cost"],
            top_k=3,
        )
        == []
    )
