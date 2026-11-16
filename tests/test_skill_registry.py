from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from agents.furnacemind.skill_registry import SkillRegistry

#: The plant skills in their button order (user decision 13 Nov 2026).
PLANT_SKILLS = [
    "shift_pulse",
    "thermal_watch",
    "gas_flow_lens",
    "wall_heat_map",
    "slag_burden_check",
    "fuel_drivers",
    "trial_board",
]


class FakeEngine:
    """SkillEngine stand-in that records which handler was called."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def shift_pulse_prompt(self, shift_date: str, label: str) -> str:
        self.calls.append(("shift_pulse_prompt", shift_date, label))
        return f"shift pulse {shift_date} {label}"

    def __getattr__(self, name: str):
        if name.endswith("_prompt"):

            def handler() -> str:
                self.calls.append((name,))
                return f"{name} result"

            return handler
        raise AttributeError(name)


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
    source_type: str = "uploaded",
) -> SimpleNamespace:
    """Build a minimal ORM-like skill row for registry tests."""
    return SimpleNamespace(
        skill_id=skill_id,
        name=name,
        symbol=symbol,
        description=None,
        instruction=instruction,
        source_type=source_type,
        is_active=is_active,
        metadata_json=metadata or {},
    )


def _review_rows() -> list[SimpleNamespace]:
    """Two uploaded prompt skills for the semantic-context tests."""
    return [
        _row(
            skill_id="skill_cost_review",
            name="Cost Review",
            instruction="Cost instruction for sinter and ore economics.",
            metadata={"slug": "cost_review", "order": 10},
        ),
        _row(
            skill_id="skill_heat_review",
            name="Heat Review",
            instruction="Heatload instruction for skin temperature checks.",
            metadata={"slug": "heat_review", "order": 20},
        ),
    ]


def test_registry_offers_the_plant_skills_when_the_database_is_empty() -> None:
    """An empty skills table leaves the code's seven plant skills, in their order."""
    repository = FakeSkillRepository([])
    registry = SkillRegistry(engine=FakeEngine(), repository=repository)

    skills = registry.available_skills()

    assert [skill.slug for skill in skills] == PLANT_SKILLS
    assert [skill.name for skill in skills] == [
        "Shift Pulse",
        "Thermal Watch",
        "Gas Flow Lens",
        "Wall Heat Map",
        "Slag & Burden Check",
        "Fuel Drivers",
        "Trial Board",
    ]
    assert len({skill.name for skill in skills}) == len(skills)
    assert all(skill.handler == f"{skill.slug}_prompt" for skill in skills)
    assert registry.using_fallback is True
    assert repository.active_only_values == [False]


def test_table_rows_add_skills_rename_code_skills_and_hide_them_when_inactive() -> None:
    """A row with a code skill's slug changes its presentation (never its handler); an inactive one hides it; other
    rows add their own skills."""
    rows = [
        _row(skill_id="skill_thermal", name="Thermal Watch", is_active=False, metadata={"slug": "thermal_watch"}),
        _row(
            skill_id="skill_pulse",
            name="Shift Review",
            symbol="SR",
            metadata={
                "slug": "shift_pulse",
                "order": 5,
                "handler": "trial_board_prompt",
                "display_template": "Review {shift_date} {shift_label}",
            },
        ),
        _row(
            skill_id="skill_custom",
            name="Custom SOP",
            instruction="Follow the custom SOP.",
            symbol="CS",
            metadata={"slug": "custom_sop", "order": 65},
        ),
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    skills = registry.available_skills()

    assert [skill.slug for skill in skills] == [
        "shift_pulse",
        "gas_flow_lens",
        "wall_heat_map",
        "slag_burden_check",
        "fuel_drivers",
        "custom_sop",
        "trial_board",
    ]
    pulse = skills[0]
    assert (pulse.name, pulse.symbol, pulse.handler, pulse.order) == ("Shift Review", "SR", "shift_pulse_prompt", 5)
    assert registry.display_text(pulse, shift_date=date(2026, 11, 13), shift_label="B") == "Review 2026-11-13 B"
    assert skills[5].button_label == "CS Custom SOP"
    assert registry.using_fallback is False


def test_rows_of_the_retired_skills_are_ignored() -> None:
    """Unit Cost, Shift to Best and Heatloads are not offered any more, active or not, by slug or by handler."""
    rows = [
        _row(
            skill_id="skill_optimise_unit_cost",
            name="Unit Cost",
            source_type="built_in",
            metadata={"slug": "optimise", "handler": "optimise_prompt"},
        ),
        _row(
            skill_id="skill_shift_to_best",
            name="Shift to Best",
            source_type="built_in",
            metadata={"slug": "shift_to_best", "handler": "shift_to_best_prompt"},
        ),
        _row(
            skill_id="skill_heatloads",
            name="Heatloads",
            source_type="built_in",
            metadata={"slug": "heatload", "handler": "heatload_prompt"},
        ),
        _row(
            skill_id="skill_renamed", name="Old heat skill", metadata={"slug": "old_heat", "handler": "heatload_prompt"}
        ),
        _row(skill_id="skill_unit", name="Unit cost", instruction="old", metadata={"slug": "unit_cost"}),
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    assert [skill.slug for skill in registry.available_skills()] == PLANT_SKILLS
    with pytest.raises(ValueError):
        registry.execute("heatload", shift_date=date(2026, 11, 13), shift_label="A")


def test_inactive_rows_hide_only_their_own_skills() -> None:
    """Inactive rows no longer empty the list: the code's skills stay unless a row with their slug is inactive."""
    rows = [
        _row(
            skill_id="skill_disabled", name="Disabled", is_active=False, instruction="x", metadata={"slug": "disabled"}
        ),
        _row(skill_id="skill_trial", name="Trial Board", is_active=False, metadata={"slug": "trial_board"}),
    ]
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(rows))

    assert [skill.slug for skill in registry.available_skills()] == PLANT_SKILLS[:-1]
    assert registry.using_fallback is False


@pytest.mark.parametrize("slug", PLANT_SKILLS)
def test_registry_dispatches_every_plant_skill_to_its_engine_handler(slug: str) -> None:
    """Each plant skill runs its own whitelisted engine method; Shift Pulse gets the page's shift."""
    engine = FakeEngine()
    registry = SkillRegistry(engine=engine, repository=FakeSkillRepository([]))

    execution = registry.execute(slug, shift_date=date(2026, 6, 25), shift_label="A")

    expected = ("shift_pulse_prompt", "2026-06-25", "A") if slug == "shift_pulse" else (f"{slug}_prompt",)
    assert engine.calls == [expected]
    assert execution.skill_id == slug
    assert execution.prompt == ("shift pulse 2026-06-25 A" if slug == "shift_pulse" else f"{slug}_prompt result")
    assert execution.display
    assert "plant" in execution.skill_context.lower() or execution.skill_context


def test_shift_pulse_display_names_the_shift() -> None:
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository([]))

    execution = registry.execute("shift_pulse", shift_date=date(2026, 11, 13), shift_label="B")

    assert execution.display == "Shift Pulse: 2026-11-13, Shift B against the week before"


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

    skill = next(s for s in registry.available_skills() if s.slug == "prompt_only")
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
    """Database failures do not remove the plant skills."""
    repository = FakeSkillRepository([], fail=True)
    registry = SkillRegistry(engine=FakeEngine(), repository=repository)

    skills = registry.available_skills()

    assert [skill.slug for skill in skills] == PLANT_SKILLS
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
    """Recent chat context can add relevant active skills without duplicating the selected one; the code's plant
    skills are buttons, never semantic context."""
    registry = SkillRegistry(engine=FakeEngine(), repository=FakeSkillRepository(_review_rows()))

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
        selected_skill_id="cost_review",
        selected_skill_context="Selected Cost Review context.",
        embedding_client=FakeEmbeddingClient(),
        max_retrieved=2,
        min_score=0.8,
    )

    assert "EXPLICITLY SELECTED SKILL CONTEXT" in context
    assert "Selected Cost Review context." in context
    assert "SEMANTICALLY RELEVANT SKILL CONTEXT" in context
    assert "Skill: Heat Review" in context
    assert "Heatload instruction for skin temperature checks." in context
    assert context.count("Skill: Cost Review") == 0
    assert "Wall Heat Map" not in context


def test_turn_skill_context_uses_qdrant_skill_vector_matches() -> None:
    """Additional relevant skills should come from the persisted vector index when available."""
    vector_store = FakeSkillVectorStore([SimpleNamespace(skill_id="skill_heat_review", slug="heat_review", score=0.91)])
    registry = SkillRegistry(
        engine=FakeEngine(),
        repository=FakeSkillRepository(_review_rows()),
        vector_store=vector_store,
    )

    context = registry.turn_skill_context(
        query="What should I check next?",
        recent_messages=[
            {"role": "user", "content": "Skin temperature is climbing."},
        ],
        selected_skill_id="cost_review",
        selected_skill_context="Selected Cost Review context.",
        embedding_client=None,
        max_retrieved=2,
        min_score=0.8,
    )

    assert "Selected Cost Review context." in context
    assert "Skill: Heat Review" in context
    assert "Heatload instruction for skin temperature checks." in context
    assert vector_store.calls[0]["active_skill_ids"] == [
        "skill_cost_review",
        "skill_heat_review",
    ]
    assert "cost_review" in vector_store.calls[0]["exclude_skill_ids"]
    assert registry.last_relevance_error is None


def test_turn_skill_context_records_injected_skill_identifiers() -> None:
    """Skill metadata should identify every skill injected into one turn."""
    vector_store = FakeSkillVectorStore([SimpleNamespace(skill_id="skill_heat_review", slug="heat_review", score=0.91)])
    registry = SkillRegistry(
        engine=FakeEngine(),
        repository=FakeSkillRepository(_review_rows()),
        vector_store=vector_store,
    )

    registry.turn_skill_context(
        query="Skin temperature is climbing.",
        selected_skill_id="cost_review",
        selected_skill_context="Selected Cost Review context.",
        embedding_client=None,
        max_retrieved=2,
        min_score=0.8,
    )

    assert registry.last_context_skill_ids == ("skill_cost_review", "skill_heat_review")
    assert registry.last_context_skill_slugs == ("cost_review", "heat_review")


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
