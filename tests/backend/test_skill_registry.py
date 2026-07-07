from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from apps.frontend_streamlit.agents.furnacemind.skill_registry import SkillRegistry


class FakeEngine:
    """SkillEngine stand-in that records which handler was called."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def optimise_prompt(self) -> str:
        self.calls.append(("optimise_prompt",))
        return "optimise prompt"

    def shift_to_best_prompt(self, shift_date: str, label: str) -> str:
        self.calls.append(("shift_to_best_prompt", shift_date, label))
        return f"shift prompt {shift_date} {label}"

    def heatload_prompt(self) -> str:
        self.calls.append(("heatload_prompt",))
        return "heatload prompt"


class FakeSkillRepository:
    """Repository stand-in for database-backed skill rows."""

    def __init__(self, rows: list[SimpleNamespace], *, fail: bool = False) -> None:
        self.rows = rows
        self.fail = fail
        self.active_only_values: list[bool] = []

    def list_skills(self, *, active_only: bool = False) -> list[SimpleNamespace]:
        self.active_only_values.append(active_only)
        if self.fail:
            raise RuntimeError("database unavailable")
        return self.rows


class FakeEmbeddingClient:
    """Small deterministic embedding client for semantic skill-search tests."""

    def embed_text(self, text: str, *, input_type: str | None = None) -> list[float]:
        lower = str(text or "").lower()
        if "heatload" in lower or "skin temperature" in lower:
            return [1.0, 0.0]
        if "cost" in lower or "sinter" in lower or "ore" in lower:
            return [0.0, 1.0]
        return [0.0, 0.0]


class FakeSkillVectorStore:
    """Qdrant skill-index stand-in for registry routing tests."""

    def __init__(self, matches: list[SimpleNamespace], *, fail: bool = False) -> None:
        self.matches = matches
        self.fail = fail
        self.calls: list[dict] = []

    def search(self, **kwargs) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("qdrant unavailable")
        return self.matches


def _row(
    *,
    skill_id: str,
    name: str,
    instruction: str = "",
    symbol: str = "",
    is_active: bool = True,
    metadata: dict | None = None,
) -> SimpleNamespace:
    """Build a minimal ORM-like skill row for registry tests."""
    return SimpleNamespace(
        skill_id=skill_id,
        name=name,
        symbol=symbol,
        description=None,
        instruction=instruction,
        source_type="built_in",
        is_active=is_active,
        metadata_json=metadata or {},
    )


def test_registry_falls_back_to_builtin_skills_when_database_is_empty() -> None:
    """Empty skills table keeps the existing built-in quick skills available."""
    repository = FakeSkillRepository([])
    registry = SkillRegistry(engine=FakeEngine(), repository=repository)

    skills = registry.available_skills()

    assert [skill.slug for skill in skills] == ["optimise", "shift_to_best", "heatload"]
    assert registry.using_fallback is True
    assert repository.active_only_values == [False]


def test_registry_uses_active_database_skills_in_configured_order() -> None:
    """Active DB rows replace built-ins and are ordered by metadata."""
    rows = [
        _row(
            skill_id="skill_disabled",
            name="Disabled",
            is_active=False,
            metadata={"slug": "disabled", "order": 1, "handler": "heatload_prompt"},
        ),
        _row(
            skill_id="skill_heatload",
            name="Heatload Check",
            metadata={"slug": "heatload", "order": 20, "handler": "heatload_prompt"},
        ),
        _row(
            skill_id="skill_custom",
            name="Custom SOP",
            instruction="Follow the custom SOP.",
            symbol="CS",
            metadata={"slug": "custom_sop", "order": 10},
        ),
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    skills = registry.available_skills()

    assert [skill.slug for skill in skills] == ["custom_sop", "heatload"]
    assert skills[0].button_label == "CS Custom SOP"
    assert registry.using_fallback is False


def test_registry_respects_all_skills_disabled_without_fallback() -> None:
    """When DB rows exist but all are inactive, the UI should show no skills."""
    rows = [
        _row(
            skill_id="skill_disabled",
            name="Disabled",
            is_active=False,
            metadata={"slug": "disabled", "handler": "heatload_prompt"},
        )
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    assert registry.available_skills() == []
    assert registry.using_fallback is False


def test_registry_dispatches_whitelisted_engine_handler() -> None:
    """DB metadata can choose a safe Python handler without executing DB code."""
    engine = FakeEngine()
    rows = [
        _row(
            skill_id="skill_shift",
            name="Shift Skill",
            metadata={
                "slug": "shift_to_best",
                "handler": "shift_to_best_prompt",
                "display_template": "Shift to Best: {shift_date}, Shift {shift_label}",
            },
        )
    ]
    registry = SkillRegistry(engine=engine, repository=FakeSkillRepository(rows))

    execution = registry.execute(
        "shift_to_best",
        shift_date=date(2026, 6, 25),
        shift_label="A",
    )

    assert execution.prompt == "shift prompt 2026-06-25 A"
    assert execution.display == "Shift to Best: 2026-06-25, Shift A"
    assert execution.skill_id == "shift_to_best"
    assert engine.calls == [("shift_to_best_prompt", "2026-06-25", "A")]


def test_registry_builds_prompt_only_database_skill_context() -> None:
    """A DB skill without a handler can still queue its instruction safely."""
    rows = [
        _row(
            skill_id="skill_prompt_only",
            name="Prompt Only",
            instruction="Use this instruction from SQL.",
            metadata={
                "slug": "prompt_only",
                "display_template": "Prompt Only Skill",
            },
        )
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    skill = registry.available_skills()[0]
    execution = registry.execute_definition(
        skill,
        shift_date=date(2026, 6, 25),
        shift_label="B",
    )

    assert execution.prompt == "SKILL: Prompt Only\nUse this instruction from SQL."
    assert execution.display == "Prompt Only Skill"
    assert "database skill instruction" in execution.skill_context
    assert "Use this instruction from SQL." in execution.skill_context


def test_registry_falls_back_when_database_is_unavailable() -> None:
    """Database failures do not remove existing built-in quick skills."""
    repository = FakeSkillRepository([], fail=True)
    registry = SkillRegistry(engine=FakeEngine(), repository=repository)

    skills = registry.available_skills()

    assert [skill.slug for skill in skills] == ["optimise", "shift_to_best", "heatload"]
    assert registry.using_fallback is True
    assert registry.last_error == "database unavailable"


def test_turn_skill_context_always_includes_selected_skill_context() -> None:
    """Clicked skill context is injected even when semantic search is unavailable."""
    rows = [
        _row(
            skill_id="skill_prompt_only",
            name="Prompt Only",
            instruction="Use selected SQL instruction.",
            metadata={"slug": "prompt_only"},
        )
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    context = registry.turn_skill_context(
        query="unrelated operator question",
        selected_skill_id="prompt_only",
        selected_skill_context="Selected context from the clicked skill.",
        embedding_client=None,
    )

    assert "EXPLICITLY SELECTED SKILL CONTEXT" in context
    assert "Selected skill: Prompt Only" in context
    assert "Selected context from the clicked skill." in context
    assert "SEMANTICALLY RELEVANT SKILL CONTEXT" not in context


def test_turn_skill_context_adds_semantic_skills_from_recent_messages() -> None:
    """Recent chat context can add relevant active skills without duplicating the selected one."""
    rows = [
        _row(
            skill_id="skill_unit_cost",
            name="Unit Cost",
            instruction="Cost instruction for sinter and ore economics.",
            metadata={"slug": "unit_cost", "order": 10},
        ),
        _row(
            skill_id="skill_heatload",
            name="Heatloads",
            instruction="Heatload instruction for skin temperature checks.",
            metadata={"slug": "heatload", "order": 20},
        ),
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    context = registry.turn_skill_context(
        query="What should I check next?",
        recent_messages=[
            {
                "role": "user",
                "content": "The skin temperature and heatload rows are rising.",
            },
            {
                "role": "assistant",
                "content": "Review thermal behavior before changing burden.",
            },
        ],
        selected_skill_id="unit_cost",
        selected_skill_context="Selected Unit Cost context.",
        embedding_client=FakeEmbeddingClient(),
        max_retrieved=2,
        min_score=0.8,
    )

    assert "EXPLICITLY SELECTED SKILL CONTEXT" in context
    assert "Selected Unit Cost context." in context
    assert "SEMANTICALLY RELEVANT SKILL CONTEXT" in context
    assert "Skill: Heatloads" in context
    assert "Heatload instruction for skin temperature checks." in context
    assert context.count("Skill: Unit Cost") == 0


def test_turn_skill_context_uses_qdrant_skill_vector_matches() -> None:
    """Additional relevant skills should come from the persisted vector index when available."""
    rows = [
        _row(
            skill_id="skill_unit_cost",
            name="Unit Cost",
            instruction="Cost instruction for sinter economics.",
            metadata={"slug": "unit_cost", "order": 10},
        ),
        _row(
            skill_id="skill_heatload",
            name="Heatloads",
            instruction="Heatload instruction from SQL.",
            metadata={"slug": "heatload", "order": 20},
        ),
    ]
    vector_store = FakeSkillVectorStore(
        [SimpleNamespace(skill_id="skill_heatload", slug="heatload", score=0.91)]
    )
    registry = SkillRegistry(
        engine=FakeEngine(),
        repository=FakeSkillRepository(rows),
        vector_store=vector_store,
    )

    context = registry.turn_skill_context(
        query="What should I check next?",
        recent_messages=[
            {"role": "user", "content": "Skin temperature is climbing."},
        ],
        selected_skill_id="unit_cost",
        selected_skill_context="Selected Unit Cost context.",
        embedding_client=None,
        max_retrieved=2,
        min_score=0.8,
    )

    assert "Selected Unit Cost context." in context
    assert "Skill: Heatloads" in context
    assert "Heatload instruction from SQL." in context
    assert vector_store.calls[0]["active_skill_ids"] == [
        "skill_unit_cost",
        "skill_heatload",
    ]
    assert "unit_cost" in vector_store.calls[0]["exclude_skill_ids"]
    assert registry.last_relevance_error is None


def test_turn_skill_context_records_injected_skill_identifiers() -> None:
    """Skill metadata should identify every skill injected into one turn."""
    rows = [
        _row(
            skill_id="skill_unit_cost",
            name="Unit Cost",
            instruction="Cost instruction for sinter economics.",
            metadata={"slug": "unit_cost", "order": 10},
        ),
        _row(
            skill_id="skill_heatload",
            name="Heatloads",
            instruction="Heatload instruction from SQL.",
            metadata={"slug": "heatload", "order": 20},
        ),
    ]
    vector_store = FakeSkillVectorStore(
        [SimpleNamespace(skill_id="skill_heatload", slug="heatload", score=0.91)]
    )
    registry = SkillRegistry(
        engine=FakeEngine(),
        repository=FakeSkillRepository(rows),
        vector_store=vector_store,
    )

    registry.turn_skill_context(
        query="Skin temperature is climbing.",
        selected_skill_id="unit_cost",
        selected_skill_context="Selected Unit Cost context.",
        embedding_client=None,
        max_retrieved=2,
        min_score=0.8,
    )

    assert registry.last_context_skill_ids == ("skill_unit_cost", "skill_heatload")
    assert registry.last_context_skill_slugs == ("unit_cost", "heatload")


def test_turn_skill_context_does_not_inject_inactive_qdrant_skill() -> None:
    """Inactive SQL rows must not be injected even when Qdrant still has points."""
    rows = [
        _row(
            skill_id="skill_cast_house",
            name="Cast House SOP",
            instruction="Cast house instruction should stay disabled.",
            is_active=False,
            metadata={"slug": "cast_house_sop"},
        )
    ]
    vector_store = FakeSkillVectorStore(
        [
            SimpleNamespace(
                skill_id="skill_cast_house",
                slug="cast_house_sop",
                score=0.99,
            )
        ]
    )
    registry = SkillRegistry(
        engine=FakeEngine(),
        repository=FakeSkillRepository(rows),
        vector_store=vector_store,
    )

    context = registry.turn_skill_context(
        query="Before tapping, what should I check if taphole flow is unstable?",
        embedding_client=None,
        max_retrieved=2,
        min_score=0.8,
    )

    assert context == ""
    assert vector_store.calls[0]["active_skill_ids"] == []
    assert registry.last_context_skill_ids == ()
    assert registry.last_context_skill_slugs == ()
