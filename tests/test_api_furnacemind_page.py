"""What the FurnaceMind SPA page needs from the API (SV-5.2, #313): page 7's quick skills, the full dataset as CSV,
the owner's feedback on an answer (stored with its audit row), and the first token of a streamed answer within 3 s
when the model answers at once.

Parity with page 7: a skill turn stores the prompt page 7's ``SkillRegistry`` builds for the same skill, shift and
data (exact text, so every pre-computed number agrees), and the CSV is page 7's ``to_csv(index=True)`` of the turn's
DataFrame (exact text).
"""

from __future__ import annotations

import time
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from agents.furnacemind import tool_state
from agents.furnacemind.skill_registry import SkillRegistry
from agents.furnacemind.skills import SkillEngine
from services import furnacemind as fm
from tests import furnacemind_fakes as fakes
from tests.test_api_furnacemind import PLOT, api, ask, call, sqlite_store

SHIFT = (date(2026, 10, 15), "B")  # inside the synthetic static dataset (1 Sep - 20 Oct 2026)


class RecordingPrompt:
    """System-prompt dependency that records the skill keywords of each call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kw: Any) -> str:
        self.calls.append(kw)
        return f"test prompt skill={kw.get('skill_id')}\n{kw.get('skill_context') or ''}"


def deps(llm: Any, tools: list[dict[str, Any]] | None = None, *, registry: Any = None, prompt: Any = None):
    return fm.Dependencies(
        llm=lambda level: llm,
        tools=lambda: tools or [],
        system_prompt=prompt or RecordingPrompt(),
        resources=fakes.resources,
        history_window=lambda: 8,
        skills=lambda: registry if registry is not None else SkillRegistry(engine=SkillEngine(), repository=None),
    )


@pytest.fixture()
def store(tmp_path):
    twin = sqlite_store(tmp_path / "twin.db")
    yield twin
    twin.dispose()


@pytest.fixture()
def plant(monkeypatch, tmp_path):
    import agents.furnace_tools as ft

    fakes.install_plant_data(monkeypatch, tmp_path)
    monkeypatch.setattr(fm, "default_shift", lambda: SHIFT)
    return ft.get_openai_tool_schemas()


def ask_skill(client, conversation_id: str, skill: str, user: str = "op") -> list[dict[str, Any]]:
    r = call(client, user, "POST", f"/api/fm/conversations/{conversation_id}/messages", json={"skill": skill})
    assert r.status_code == 200, r.text
    return fm.parse_events(r.text)


# ------------------------------------------------------------------------------------------------------- skills


def test_skills_are_page_7s_built_in_ones_without_a_skill_table(store, plant) -> None:
    with api(store, deps(fakes.ScriptedLLM(), plant)) as client:
        body = call(client, "op", "GET", "/api/fm/skills").json()
        assert body["source"] == "built_in" and (body["shift_date"], body["shift_label"]) == ("2026-10-15", "B")
        assert [(s["slug"], s["name"]) for s in body["skills"]] == [
            ("optimise", "Unit Cost"),
            ("shift_to_best", "Shift to Best"),
            ("heatload", "Heatloads"),
        ]
        assert body["skills"][1]["display"] == "Shift to Best: 2026-10-15, Shift B"
        assert client.get("/api/fm/skills").status_code == 401


def test_skills_come_from_the_skill_table_when_it_has_rows(store, plant) -> None:
    rows = [
        SimpleNamespace(
            skill_id="s1",
            name="Custom SOP",
            symbol="",
            description="Follow the SOP.",
            instruction="Follow the custom SOP.",
            is_active=True,
            metadata_json={"slug": "custom_sop", "order": 5, "display_template": "SOP for {shift_date} {shift_label}"},
        ),
        SimpleNamespace(
            skill_id="s2", name="Off", symbol="", description="", instruction="x", is_active=False, metadata_json={}
        ),
    ]
    repository = SimpleNamespace(list_skills=lambda active_only=False: rows)
    registry = SkillRegistry(engine=SkillEngine(), repository=repository)
    llm = fakes.ScriptedLLM()
    with api(store, deps(llm, plant, registry=registry)) as client:
        body = call(client, "op", "GET", "/api/fm/skills").json()
        assert body["source"] == "database"
        assert [(s["slug"], s["display"]) for s in body["skills"]] == [("custom_sop", "SOP for 2026-10-15 B")]
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        events = ask_skill(client, conversation["id"], "custom_sop")
        question = events[0]["message"]
        assert question["content"] == "SKILL: Custom SOP\nFollow the custom SOP."  # page 7's prompt-only skill
        assert question["meta"]["display"] == "SOP for 2026-10-15 B" and question["meta"]["skill"] == "custom_sop"


def test_a_skill_turn_stores_page_7s_prompt_and_puts_its_context_in_the_system_prompt(store, plant) -> None:
    llm = fakes.ScriptedLLM()
    prompt = RecordingPrompt()
    with api(store, deps(llm, plant, prompt=prompt)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        events = ask_skill(client, conversation["id"], "shift_to_best")
        types = [e["type"] for e in events if e["type"] != "token"]
        assert types == ["message.user", "message.assistant"]  # the skill's own chart and DataFrame are dropped
        question, answer = events[0]["message"], events[-1]["message"]

        # Page 7: the same registry, skill, shift and data give the same prompt, to the character.
        registry = SkillRegistry(engine=SkillEngine(), repository=None)
        definition = next(d for d in registry.available_skills() if d.slug == "shift_to_best")
        with tool_state.bind({}):
            expected = registry.execute_definition(definition, shift_date=SHIFT[0], shift_label=SHIFT[1])
        assert question["content"] == expected.prompt and "TIER 1 (ranked by impact score)" in expected.prompt
        assert question["meta"]["display"] == expected.display == "Shift to Best: 2026-10-15, Shift B"
        assert question["meta"]["skill"] == "shift_to_best" and question["meta"]["exclude_from_memory"] is True
        assert question["meta"]["skill_context_skill_slugs"] == ["shift_to_best"]

        assert prompt.calls[-1]["skill_id"] == "shift_to_best"
        assert "EXPLICITLY SELECTED SKILL CONTEXT" in prompt.calls[-1]["skill_context"]
        assert "SKILLS_BESTSHIFT.md" in prompt.calls[-1]["skill_context"]
        model_input = llm.calls[-1]
        assert model_input[0]["content"].startswith("test prompt skill=shift_to_best")
        assert model_input[-1] == {"role": "user", "content": expected.prompt}
        assert answer["seq"] == 2 and answer["status"] == "complete"

        state = fm._STATES[conversation["id"]]
        assert state.get("fm_fig") is None and state.get("fm_df") is None  # page 7's _queue_skill
        assert any(k.startswith("ml_static_") for k in state["fm_datasets"])  # the fetched data stays for follow-ups

        stored = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
        assert stored["messages"][0] == question and stored["title"].startswith("SKILL: Shift to Best")


def test_a_skill_request_is_validated(store, plant) -> None:
    with api(store, deps(fakes.ScriptedLLM(), plant)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        url = f"/api/fm/conversations/{conversation['id']}/messages"
        both = call(client, "op", "POST", url, json={"content": "hi", "skill": "heatload"})
        assert both.status_code == 422
        unknown = call(client, "op", "POST", url, json={"skill": "nope"})
        assert unknown.status_code == 422 and "nope" in unknown.json()["detail"]
        view = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
        assert view["message_count"] == 0 and not view["running"]  # nothing stored, the conversation is free


class FailingEngine:
    def heatload_prompt(self) -> str:
        raise RuntimeError("skill_params.yml unreadable")


def test_a_failing_skill_stores_nothing_and_ends_the_stream_with_an_error(store, plant) -> None:
    registry = SkillRegistry(engine=FailingEngine(), repository=None)
    with api(store, deps(fakes.ScriptedLLM(), plant, registry=registry)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        events = ask_skill(client, conversation["id"], "heatload")
        assert events == [
            {
                "type": "error",
                "status": 500,
                "title": "The skill could not start",
                "detail": "Nothing was stored; see the API log.",
            }
        ]
        view = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
        assert view["message_count"] == 0 and not view["running"]


# ------------------------------------------------------------------------------------------------------ datasets


def test_the_full_dataset_downloads_as_page_7s_csv_while_the_process_holds_it(store, plant) -> None:
    with api(store, deps(fakes.ScriptedLLM(), plant)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        events = ask(client, conversation["id"], PLOT)
        part = next(e["part"] for e in events if e["type"] == "artifact" and e["part"]["type"] == "dataset")
        assert len(part["artifact_id"]) == 16 and part["rows"] == 145 and part["preview_rows"] == 50
        url = f"/api/fm/conversations/{conversation['id']}/datasets/{part['artifact_id']}/csv"
        r = call(client, "op", "GET", url)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
        assert r.headers["content-disposition"] == f'attachment; filename="{part["dataset_id"]}.csv"'
        frame = fm._STATES[conversation["id"]]["fm_datasets"][part["dataset_id"]]["df"]
        assert r.text == frame.to_csv(index=True)  # page 7's download of the same DataFrame
        assert len(r.text.splitlines()) == part["rows"] + 1

        assert call(client, "sup", "GET", url).status_code == 404  # another user's conversation
        assert call(client, "adm", "GET", url).status_code == 200  # an administrator may read it
        missing = call(client, "op", "GET", url.replace(part["artifact_id"], "0" * 16))
        assert missing.status_code == 404 and "no longer holds" in missing.json()["detail"]
        assert call(client, "op", "GET", url.replace(part["artifact_id"], "not-hex")).status_code == 422

    fm.forget_states()  # a restart: the stored part stays, the DataFrame is gone
    with api(store, deps(fakes.ScriptedLLM(), plant)) as client:
        assert call(client, "op", "GET", url).status_code == 404


def test_held_datasets_are_bounded_per_conversation() -> None:
    state: dict[str, Any] = {}
    ids = [fm._hold_frame(state, i, f"d{i}") for i in range(fm.FRAMES_PER_CONVERSATION + 3)]
    assert list(state[fm.FRAMES_KEY]) == ids[3:]


# ------------------------------------------------------------------------------------------------------ feedback


def test_feedback_is_saved_per_answer_by_the_owner_with_its_audit_row(store, plant) -> None:
    with api(store, deps(fakes.ScriptedLLM(), plant)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        ask(client, conversation["id"], PLOT)
        url = f"/api/fm/conversations/{conversation['id']}/messages/2/feedback"
        saved = call(client, "op", "PUT", url, json={"rating": "down", "comment": "  Wrong week.  "})
        assert saved.status_code == 200, saved.text
        assert saved.json()["rating"] == "down" and saved.json()["comment"] == "Wrong week."
        again = call(client, "op", "PUT", url, json={"rating": "up"}).json()
        assert (again["rating"], again["comment"], again["author"]) == ("up", "", "op")

        view = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
        assert view["messages"][0]["feedback"] is None and view["messages"][1]["feedback"] == again
        audit = store.audit.query(action="fm.feedback")
        assert [(a.actor, a.before, a.after) for a in audit] == [
            ("op", {"rating": "down", "comment": "Wrong week."}, {"rating": "up", "comment": ""}),
            ("op", None, {"rating": "down", "comment": "Wrong week."}),
        ]

        question = url.replace("/messages/2/", "/messages/1/")
        assert call(client, "op", "PUT", question, json={"rating": "up"}).status_code == 404  # not an answer
        assert call(client, "op", "PUT", url.replace("/2/", "/9/"), json={"rating": "up"}).status_code == 404
        assert call(client, "op", "PUT", url.replace("/2/", "/0/"), json={"rating": "up"}).status_code == 422
        assert call(client, "sup", "PUT", url, json={"rating": "up"}).status_code == 404
        assert call(client, "adm", "PUT", url, json={"rating": "up"}).status_code == 404  # reading is not owning
        assert call(client, "op", "PUT", url, json={"rating": "meh"}).status_code == 422
        long = {"rating": "up", "comment": "x" * (fm.FEEDBACK_COMMENT_CHARS + 1)}
        assert call(client, "op", "PUT", url, json=long).status_code == 422
        assert len(store.audit.query(action="fm.feedback")) == 2


# ------------------------------------------------------------------------------------------------- first token


def first_token_seconds(client, conversation_id: str, content: str, sent: list[tuple[str, float]]) -> float:
    """Seconds from sending the question to the API writing the first ``token`` frame of its stream."""
    sent.clear()
    started = time.perf_counter()
    events = ask(client, conversation_id, content)
    assert events[-1]["type"] == "message.assistant"
    return next(t for kind, t in sent if kind == "token") - started


def test_first_token_within_3_s_when_the_model_answers_at_once(store, plant, monkeypatch) -> None:
    sent: list[tuple[str, float]] = []
    frame = fm.stream_frame

    def timed(event: Any) -> str:
        sent.append((event.type, time.perf_counter()))
        return frame(event)

    monkeypatch.setattr(fm, "stream_frame", timed)
    with api(store, deps(fakes.ScriptedLLM(), plant)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        first = first_token_seconds(client, conversation["id"], "Explain RAFT in one paragraph.", sent)
        second = first_token_seconds(client, conversation["id"], "Explain ETA CO in one paragraph.", sent)
        print(f"first token: first turn {first:.3f} s, second turn {second:.3f} s")
        assert first < 3.0 and second < 3.0


def test_the_agent_stack_loads_in_the_background_at_start(monkeypatch) -> None:
    """The first turn after a start does not wait for the agent-stack imports (4.4 s on the laptop)."""
    import sys

    from fastapi.testclient import TestClient

    from api.main import create_app
    from api.settings import ApiSettings

    monkeypatch.setattr(fm, "_WARM_STARTED", False)
    thread = fm.warm_up()
    assert thread is not None and fm.warm_up() is None  # once per process
    thread.join(120)
    assert all(name in sys.modules for name in fm.AGENT_STACK)

    started: list[bool] = []
    monkeypatch.setattr(fm, "warm_up", lambda: started.append(True))
    for flag in (False, True):
        with TestClient(create_app(ApiSettings(static_dir=None, fm_warm_up=flag))):
            pass
    assert started == [True]
