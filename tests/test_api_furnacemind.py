"""FurnaceMind routes (SV-5.1, #312): conversations per user, the answer streamed as server-sent events and stored
with its parts, one turn per conversation, 503 without a store or an agent, and conversations that survive a restart
of the API (SQLite file, and PostgreSQL migrated with ``alembic -n twin upgrade head``)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from alembic import command
from fastapi.testclient import TestClient

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import furnacemind as fm
from services import store as store_service
from tests import furnacemind_fakes as fakes
from tests.pg_server import fresh_database, needs_postgres, postgres_server
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_twin_store import alembic_config

PLOT = "Plot the fuel rate for the first week of October."
FOLLOW_UP = "Plot that again as a bar chart."


def deps(llm: Any, tools: list[dict[str, Any]] | None = None) -> fm.Dependencies:
    return fm.Dependencies(
        llm=lambda level: llm,
        tools=lambda: tools or [],
        system_prompt=lambda: "test prompt",
        resources=fakes.resources,
        history_window=lambda: 8,
    )


@contextmanager
def api(store: TwinStore | None, dependencies: fm.Dependencies | None) -> Iterator[TestClient]:
    """A fresh API process (app, twin store, agent dependencies, no tool state): what a restart gives."""
    store_service.use(store)
    fm.use(dependencies)
    app = build()
    try:
        with TestClient(app, base_url="https://testserver") as client:
            client.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
            yield client
    finally:
        fm.use(None)
        store_service.reset()


def sqlite_store(path) -> TwinStore:
    engine = build_engine(f"sqlite:///{path}")
    TwinBase.metadata.create_all(engine)
    return TwinStore(session_factory(engine), engine)


def call(client: TestClient, user: str, method: str, url: str, **kw: Any):
    return client.request(method, url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


def ask(client: TestClient, conversation_id: str, content: str, user: str = "op") -> list[dict[str, Any]]:
    r = call(client, user, "POST", f"/api/fm/conversations/{conversation_id}/messages", json={"content": content})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream") and r.headers["cache-control"].startswith(
        "no-cache"
    )
    return fm.parse_events(r.text)


@pytest.fixture()
def plant(monkeypatch, tmp_path):
    import agents.furnace_tools as ft

    fakes.install_plant_data(monkeypatch, tmp_path)
    return ft.get_openai_tool_schemas()


@pytest.fixture()
def store(tmp_path):
    twin = sqlite_store(tmp_path / "twin.db")
    yield twin
    twin.dispose()


# ------------------------------------------------------------------------------------------------------- routes


def test_conversations_belong_to_their_owner(store) -> None:
    with api(store, deps(fakes.ScriptedLLM())) as client:
        created = call(client, "op", "POST", "/api/fm/conversations", json={"reasoning_level": "High"})
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["owner"] == "op" and body["reasoning_level"] == "High" and body["messages"] == []
        assert body["message_count"] == 0 and body["title"] == "" and not body["running"]
        default = call(client, "op", "POST", "/api/fm/conversations", json={"title": "  Coke  rate "}).json()
        assert default["reasoning_level"] == "Medium" and default["title"] == "Coke rate"
        call(client, "sup", "POST", "/api/fm/conversations", json={})

        mine = call(client, "op", "GET", "/api/fm/conversations").json()
        assert [c["id"] for c in mine] == [default["id"], body["id"]]  # most recently updated first
        assert len(call(client, "sup", "GET", "/api/fm/conversations").json()) == 1

        url = f"/api/fm/conversations/{body['id']}"
        assert call(client, "sup", "GET", url).status_code == 404  # another user's conversation does not exist
        assert call(client, "adm", "GET", url).status_code == 200  # an administrator may read it
        r = call(client, "adm", "POST", f"{url}/messages", json={"content": "hello"})
        assert r.status_code == 404 and r.headers["content-type"] == "application/problem+json"
        assert call(client, "op", "GET", "/api/fm/conversations/not-a-uuid").status_code == 404
        assert call(client, "op", "POST", f"{url}/messages", json={"content": ""}).status_code == 422
        assert call(client, "op", "POST", "/api/fm/conversations", json={"reasoning_level": "Max"}).status_code == 422
        assert client.get("/api/fm/conversations").status_code == 401  # no session


def test_ask_streams_the_turn_and_stores_the_answer_with_its_parts(store, plant) -> None:
    llm = fakes.ScriptedLLM()
    with api(store, deps(llm, plant)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        events = ask(client, conversation["id"], PLOT)

        types = [e["type"] for e in events]
        assert types[0] == "message.user" and types[-1] == "message.assistant"
        ordered = [t for t in types if t not in {"token"}]
        assert ordered == [
            "message.user",
            "tool.call",
            "tool.result",
            "artifact",
            "tool.call",
            "tool.result",
            "artifact",
            "message.assistant",
        ]
        assert events[0]["message"]["seq"] == 1 and events[0]["message"]["content"] == PLOT
        call_event = events[1]["part"]
        assert call_event["name"] == "fetch_ml_data" and call_event["label"] == "Reading ML dataset..."
        assert call_event["arguments"]["columns"] == ["fuel rate"]
        dataset = next(e["part"] for e in events if e["type"] == "artifact" and e["part"]["type"] == "dataset")
        assert dataset["rows"] == 145 and dataset["columns"] == 1 and dataset["preview_rows"] == 50
        assert dataset["dataset_id"].startswith("ml_static_") and dataset["preview"]["columns"] == [
            "ACT. FUEL RATEKG/THM."
        ]
        figure = next(e["part"] for e in events if e["type"] == "artifact" and e["part"]["type"] == "plotly")
        assert figure["figure"]["layout"]["title"]["text"] == "Fuel rate" and len(figure["figure"]["data"][0]["y"]) > 0

        answer = events[-1]["message"]
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        assert tokens == answer["content"] and "<think>" not in tokens and answer["content"].startswith("Answer to")
        assert answer["seq"] == 2 and answer["status"] == "complete" and answer["model"] == "fake/scripted"
        assert answer["meta"]["tool_calls"] == 2 and answer["meta"]["duration_s"] >= 0
        streamed_parts = [e["part"] for e in events if e["type"] in {"tool.call", "tool.result", "artifact"}]
        assert answer["parts"][:-1] == streamed_parts and answer["parts"][-1] == {"type": "text", "text": tokens}

        stored = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
        assert stored["title"] == PLOT and stored["message_count"] == 2
        assert stored["messages"][1] == answer and stored["messages"][0] == events[0]["message"]

        # The follow-up works on the conversation's DataFrame (tool state of this process).
        again = ask(client, conversation["id"], FOLLOW_UP)
        assert [p["type"] for p in again[-1]["message"]["parts"]] == ["tool_call", "tool_result", "plotly", "text"]
        history = llm.calls[-1]
        assert [m["content"] for m in history if m["role"] in {"user", "assistant"} and not m.get("tool_calls")][
            :3
        ] == [
            PLOT,
            stored["messages"][1]["content"],
            FOLLOW_UP,
        ]


def test_a_model_failure_is_stored_as_a_failed_answer(store) -> None:
    llm = fakes.ScriptedLLM()
    with api(store, deps(llm)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        events = ask(client, conversation["id"], "This question finds the model busy.")
        answer = events[-1]["message"]
        assert answer["status"] == "failed" and answer["content"] == llm.unavailable_message()
        assert answer["meta"]["model_status"] == "failed" and "429" in answer["meta"]["model_error"]


class BlockingLLM(fakes.ScriptedLLM):
    """Holds its first call until released (a turn that is still running)."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def chat_completions_stream(self, **kw: Any) -> Any:
        self.started.set()
        assert self.release.wait(30)
        return super().chat_completions_stream(**kw)


def test_one_turn_per_conversation(store) -> None:
    llm = BlockingLLM()
    with api(store, deps(llm)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        first: dict[str, Any] = {}
        worker = threading.Thread(target=lambda: first.update(events=ask(client, conversation["id"], "Explain RAFT.")))
        worker.start()
        assert llm.started.wait(30)
        url = f"/api/fm/conversations/{conversation['id']}"
        assert call(client, "op", "GET", url).json()["running"] is True
        busy = call(client, "op", "POST", f"{url}/messages", json={"content": "second"})
        assert busy.status_code == 409 and "running" in busy.json()["detail"]
        llm.release.set()
        worker.join(30)
        assert first["events"][-1]["type"] == "message.assistant"
        deadline = time.monotonic() + 5
        while call(client, "op", "GET", url).json()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        view = call(client, "op", "GET", url).json()
        assert not view["running"] and view["message_count"] == 2  # the refused question was not stored


class BrokenUsageLLM(fakes.ScriptedLLM):
    def usage_metadata(self) -> dict[str, Any]:
        raise KeyError("usage")


def test_a_failing_turn_ends_the_stream_with_an_error_and_frees_the_conversation(store) -> None:
    with api(store, deps(BrokenUsageLLM())) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        events = ask(client, conversation["id"], "Explain RAFT in one paragraph.")
        assert [e["type"] for e in events][0] == "message.user"
        assert events[-1] == {
            "type": "error",
            "status": 500,
            "title": "The turn failed",
            "detail": "See the API log for this conversation.",  # no exception text on the stream
        }
        view = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
        assert view["message_count"] == 1 and not view["running"]  # the question stays, the conversation is free


def test_without_an_agent_or_a_store_the_routes_answer_503(store) -> None:
    def broken(level: str) -> Any:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    with api(store, fm.Dependencies(broken, list, str, dict, lambda: 8)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={}).json()
        r = call(client, "op", "POST", f"/api/fm/conversations/{conversation['id']}/messages", json={"content": "hi"})
        assert r.status_code == 503 and "OPENROUTER_API_KEY" in r.json()["detail"]
        view = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
        assert view["message_count"] == 0 and not view["running"]  # nothing stored, the conversation is free
    with api(None, deps(fakes.ScriptedLLM())) as client:
        r = call(client, "op", "GET", "/api/fm/conversations")
        assert r.status_code == 503 and "BF_TWIN_DB_URL" in r.json()["detail"]


# ------------------------------------------------------------------------------------------------------ restart


def survive_restart(open_store, plant) -> None:
    """Ask in one API process, restart (new app, new store connection, no tool state), read and continue."""
    first_llm = fakes.ScriptedLLM()
    store = open_store()
    with api(store, deps(first_llm, plant)) as client:
        conversation = call(client, "op", "POST", "/api/fm/conversations", json={"reasoning_level": "Low"}).json()
        before = ask(client, conversation["id"], PLOT)[-1]["message"]
        snapshot = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
    store.dispose()

    second_llm = fakes.ScriptedLLM()
    store = open_store()
    try:
        with api(store, deps(second_llm, plant)) as client:
            after = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
            assert after == snapshot and after["messages"][1] == before and after["reasoning_level"] == "Low"
            assert [c["id"] for c in call(client, "op", "GET", "/api/fm/conversations").json()] == [conversation["id"]]

            # The next turn replays the stored history; the DataFrame of the first process is gone, so the plot
            # tool reports it and the model can fetch again.
            events = ask(client, conversation["id"], FOLLOW_UP)
            replayed = [m for m in second_llm.calls[0] if m["role"] in {"user", "assistant"}]
            assert [m["content"] for m in replayed] == [PLOT, before["content"], FOLLOW_UP]
            result = next(e["part"] for e in events if e["type"] == "tool.result")
            assert result["name"] == "execute_python_plot" and "Error" in result["content"]
            final = call(client, "op", "GET", f"/api/fm/conversations/{conversation['id']}").json()
            assert final["message_count"] == 4 and [m["seq"] for m in final["messages"]] == [1, 2, 3, 4]
    finally:
        store.dispose()


def test_conversations_survive_an_api_restart_on_sqlite(tmp_path, plant) -> None:
    survive_restart(lambda: sqlite_store(tmp_path / "restart.db"), plant)


@pytest.fixture(scope="module")
def pg_url():
    with postgres_server() as server, fresh_database(server) as url:
        command.upgrade(alembic_config(url), "head")
        yield url


@needs_postgres
def test_conversations_survive_an_api_restart_on_postgresql(pg_url, plant) -> None:
    survive_restart(lambda: TwinStore.from_url(pg_url), plant)
