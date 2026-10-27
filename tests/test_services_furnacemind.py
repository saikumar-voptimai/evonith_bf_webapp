"""FurnaceMind agent service (SV-5.1, #312): streamed model calls, page 7's history window, the tools' state without
Streamlit, message parts, and the reference set of 20 questions through page 7's turn and through the API.

The parity test types the questions into page 7 itself (``render_ai_cooperate`` under Streamlit's AppTest: chat
history in the session, the page's history window and artifact handling, the model call not streamed) and posts them
to the API (``POST /api/fm/conversations/{id}/messages``: history from the twin store, the model call streamed, the
tools in the turn pool), both on the real tools over synthetic plant data and a scripted model
(``tests/furnacemind_fakes.py``). Both must dispatch the same tool calls and show the model the same messages.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents.furnacemind import tool_state
from agents.furnacemind.graph import ThinkFilter
from agents.furnacemind.turn import history_to_messages
from agents.llm.llm_client import assemble_stream
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import furnacemind as fm
from services import store as store_service
from tests import furnacemind_fakes as fakes
from tests.test_api_auth import build
from tests.test_api_live import session_cookie

WINDOW = 8  # page 7: settings.memory_summary_message_window (MEMORY_SUMMARY_MESSAGE_WINDOW, default 8)


# ------------------------------------------------------------------------------------------------ building blocks


def test_think_filter_hides_reasoning_split_across_chunks() -> None:
    text = "<think>plan the fetch</think>Fuel rate rose 4 kg/tHM.<THINK>again</THINK> Done."
    for size in (1, 2, 3, 5, 7, len(text)):
        f = ThinkFilter()
        visible = "".join(f.feed(text[i : i + size]) for i in range(0, len(text), size)) + f.flush()
        assert visible == "Fuel rate rose 4 kg/tHM. Done.", size
    f = ThinkFilter()
    assert f.feed("a < b and <thi") == "a < b and "
    assert f.feed("nk>hidden") == "" and f.flush() == ""  # a stream that ends inside a think block shows nothing


def test_assemble_stream_joins_content_and_tool_call_deltas() -> None:
    tokens: list[str] = []
    chunks = [
        {"model": "m/1", "choices": [{"delta": {"content": "Let me "}}]},
        {"choices": [{"delta": {"content": "check."}}]},
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "two", "arguments": "{"}}]}}
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "one", "arguments": ""}}]}}
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": '"x": 1}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": []},  # usage chunk
    ]
    completion = assemble_stream(chunks, on_token=tokens.append, default_model="fallback")
    message = completion.choices[0].message
    assert completion.model == "m/1" and tokens == ["Let me ", "check."] and message.content == "Let me check."
    assert [c["id"] for c in message.tool_calls] == ["a", "b"]
    assert message.tool_calls[1]["function"] == {"name": "two", "arguments": '{"x": 1}'}
    assert completion.choices[0].finish_reason == "tool_calls"
    empty = assemble_stream([], default_model="fallback")
    assert empty.model == "fallback" and empty.choices[0].message.content is None


def test_history_window_counts_artifact_entries_like_page_7() -> None:
    """Page 7 keeps the last 8 chat entries; an answer's figure and table are entries of their own (not replayed)."""

    def row(seq: int, role: str, parts: list[dict[str, Any]] | None = None) -> Any:
        return type("Row", (), {"seq": seq, "role": role, "content": f"{role} {seq}", "parts": parts or [], "meta": {}})

    rows = [
        row(1, "user"),
        row(2, "assistant", [{"type": "tool_call"}, {"type": "plotly"}, {"type": "dataset"}, {"type": "plotly"}]),
        row(3, "user"),
        row(4, "assistant", [{"type": "dataset"}]),
        row(5, "user"),
        row(6, "assistant"),
        row(7, "user"),
    ]
    entries = fm.chat_entries(rows)
    assert [e["type"] for e in entries] == [
        "text",
        "plotly",
        "dataframe",
        "text",
        "text",
        "dataframe",
        "text",
        "text",
        "text",
        "text",
    ]
    replay = history_to_messages(entries, max_messages=WINDOW)
    assert [m["content"] for m in replay] == ["assistant 2", "user 3", "assistant 4", "user 5", "assistant 6", "user 7"]
    assert history_to_messages(entries, max_messages=3, skip=lambda e: e["content"] == "assistant 6") == [
        {"role": "user", "content": "user 5"},
        {"role": "user", "content": "user 7"},
    ]


def test_tool_state_is_bound_per_context_and_never_loads_streamlit() -> None:
    a: dict[str, Any] = {}
    b: dict[str, Any] = {}
    with tool_state.bind(a):
        tool_state.current()["fm_df"] = 1
        with tool_state.bind(b):
            tool_state.current()["fm_df"] = 2
        assert tool_state.current() is a
    assert a == {"fm_df": 1} and b == {"fm_df": 2}
    assert not tool_state.owned_by_turn({"fm_fig_turn_id": "t1"}, "fm_fig_turn_id", "t2")
    assert tool_state.owned_by_turn({}, "fm_fig_turn_id", None)


def test_figure_and_dataset_parts() -> None:
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=[1, 2], y=[3.5, 4.5], name="fuel"))
    part = fm.figure_part(figure)
    assert part.figure["data"][0]["y"] == [3.5, 4.5] and part.bytes > 0 and not part.omitted
    index = pd.date_range("2026-10-20 06:00", periods=120, freq="1h")
    frame = pd.DataFrame({"fuel_rate": range(120), "si": [None] * 120}, index=index, dtype=float)
    data = fm.dataset_part(frame, {"dataset_id": "ml_static_x_1", "start": pd.Timestamp("2026-10-20")})
    assert (data.rows, data.columns, data.preview_rows, data.dataset_id) == (120, 2, 50, "ml_static_x_1")
    assert data.preview.columns == ["fuel_rate", "si"] and data.preview.data[1] == [1.0, None]
    assert data.preview.index[0].startswith("2026-10-20T06:00") and data.meta["start"].startswith("2026-10-20")
    wide = pd.DataFrame([[1.0] * 1000] * 60)
    assert fm.dataset_part(wide, {}).preview_rows == 20  # 20,000 cells at most


def test_service_prompt_is_page_7s_prompt_without_the_user_layers() -> None:
    from agents.furnacemind import prompts
    from agents.furnacemind.context import SystemPromptContext
    from agents.furnacemind.turn import system_prompt

    context = SystemPromptContext()
    page7 = context.build(extra=prompts.TOOL_POLICY, skill_id=None, skill_context=None)
    assert system_prompt(context, tool_policy=prompts.TOOL_POLICY) == page7
    assert fm._default_system_prompt() == page7
    layered = system_prompt(context, tool_policy="P", feedback_context="F", semantic_context="S")
    assert layered.endswith("P\n\nF\n\nS")


# -------------------------------------------------------------------------------------------- reference questions


def figure_json(figure: Any) -> dict[str, Any]:
    import plotly.io as pio

    return json.loads(pio.to_json(figure, validate=False))


class NoSemanticMemory:
    storage_available = False
    last_error = "not configured in the test"

    def context_for_query(self, **_: Any) -> str:
        return ""


def page_7_app(monkeypatch, llm: fakes.ScriptedLLM):
    """Page 7's AI Co-Operate tab (``render_ai_cooperate``) under Streamlit's AppTest, with the scripted model and
    without PostgreSQL, Qdrant and the embedding provider (the page runs without them, as it does when they are
    down): no signed-in user id, so no SQL history, feedback or semantic memory."""
    from streamlit.testing.v1 import AppTest

    import agents.furnacemind.ai_cooperate_page as page

    monkeypatch.setattr(page, "OpenRouterClient", lambda **_: llm)
    monkeypatch.setattr(page, "current_user_id", lambda: None)
    for name in (
        "_cached_skill_vector_store",
        "_cached_skill_repository",
        "_cached_history_store",
        "_cached_document_repository",
        "_cached_chunk_repository",
        "_cached_retrieval_trace_repository",
        "_cached_feedback_service",
        "_cached_feedback_llm",
    ):
        monkeypatch.setattr(page, name, lambda: None)
    monkeypatch.setattr(page, "_cached_embedding_client", object)
    monkeypatch.setattr(page, "_cached_knowledge_store", fakes.KnowledgeStore)
    monkeypatch.setattr(page, "_cached_shift_store", fakes.ShiftStore)
    monkeypatch.setattr(page, "_cached_semantic_memory_service", NoSemanticMemory)

    def script() -> None:
        from agents.furnacemind.ai_cooperate_page import render_ai_cooperate

        render_ai_cooperate(field_labels={})

    return AppTest.from_function(script, default_timeout=120)


def run_page_7(monkeypatch, llm: fakes.ScriptedLLM, calls: list[tuple[str, dict[str, Any]]]):
    """The 20 questions typed into page 7's chat input, one script run each."""
    app = page_7_app(monkeypatch, llm)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    out = []
    for question, _ in fakes.REFERENCE:
        start = len(calls)
        before = len(app.session_state["chat_history"])
        app.chat_input[0].set_value(question).run()
        assert not app.exception, [e.value for e in app.exception]
        shown = app.session_state["chat_history"][before:]
        store = app.session_state["fm_artifact_store"]
        figures = [figure_json(store[e["artifact_key"]]) for e in shown if e["type"] == "plotly"]
        assert shown[0]["content"] == question and shown[-1]["role"] == "assistant"
        out.append(
            {
                "tools": calls[start:],
                "text": shown[-1]["content"],
                "figure": figures[-1] if figures else None,
                "dataframe": any(e["type"] == "dataframe" for e in shown),
            }
        )
    return out


def run_api(llm: fakes.ScriptedLLM, calls: list[tuple[str, dict[str, Any]]], tmp_path):
    """The 20 questions through ``POST /api/fm/conversations/{id}/messages`` on a SQLite twin store (the service's
    own tool schemas, system prompt and history window; the vector stores are the page's doubles)."""
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    defaults = fm.default_dependencies()
    fm.use(
        fm.Dependencies(
            llm=lambda level: llm,
            tools=defaults.tools,
            system_prompt=defaults.system_prompt,
            resources=fakes.resources,
            history_window=defaults.history_window,
        )
    )
    out = []
    try:
        app = build()
        with TestClient(app, base_url="https://testserver") as client:
            client.headers["Cookie"] = session_cookie(app, "op")
            conversation = client.post("/api/fm/conversations", json={}).json()
            for question, _ in fakes.REFERENCE:
                start = len(calls)
                r = client.post(f"/api/fm/conversations/{conversation['id']}/messages", json={"content": question})
                assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream"), r.text
                events = fm.parse_events(r.text)
                answer = events[-1]
                assert answer["type"] == "message.assistant", events[-1]
                parts = answer["message"]["parts"]
                plotly = [p["figure"] for p in parts if p["type"] == "plotly"]
                out.append(
                    {
                        "tools": calls[start:],
                        "streamed_calls": [
                            (e["part"]["name"], e["part"]["arguments"]) for e in events if e["type"] == "tool.call"
                        ],
                        "stored_calls": [(p["name"], p["arguments"]) for p in parts if p["type"] == "tool_call"],
                        "results_ok": [p["ok"] for p in parts if p["type"] == "tool_result"],
                        "text": answer["message"]["content"],
                        "status": answer["message"]["status"],
                        "tokens": "".join(e["text"] for e in events if e["type"] == "token"),
                        "figure": plotly[-1] if plotly else None,
                        "dataframe": any(p["type"] == "dataset" for p in parts),
                        "types": [e["type"] for e in events],
                    }
                )
            stored = client.get(f"/api/fm/conversations/{conversation['id']}").json()
    finally:
        fm.use(None)
        store_service.reset()
        store.dispose()
    return out, stored


def test_reference_questions_make_the_same_tool_calls_through_page_7_and_the_api(monkeypatch, tmp_path) -> None:
    fakes.install_plant_data(monkeypatch, tmp_path)
    calls = fakes.record_dispatch(monkeypatch)
    page_llm, api_llm = fakes.ScriptedLLM(), fakes.ScriptedLLM()

    page = run_page_7(monkeypatch, page_llm, calls)
    api, stored = run_api(api_llm, calls, tmp_path)

    assert len(page) == len(api) == len(fakes.REFERENCE) == 20
    for (question, _), p, a in zip(fakes.REFERENCE, page, api, strict=True):
        assert fakes.normalise(a["tools"]) == fakes.normalise(p["tools"]), question
        assert fakes.normalise(a["streamed_calls"]) == fakes.normalise(a["stored_calls"]) == fakes.normalise(a["tools"])
        assert fakes.normalise(a["text"]) == fakes.normalise(p["text"]), question
        assert a["figure"] == p["figure"], question  # the sandbox's figure, as page 7 renders it
        assert a["dataframe"] == p["dataframe"], question
        assert a["types"][0] == "message.user" and a["types"][-1] == "message.assistant"
    # The model saw the same conversation at every call: system prompt, page 7's history window over the stored
    # messages, the tool turns.
    assert len(page_llm.calls) == len(api_llm.calls) == 53
    assert fakes.normalise(api_llm.calls) == fakes.normalise(page_llm.calls)
    assert page_llm.streamed == 0 and api_llm.streamed == len(api_llm.calls)

    total = sum(len(p["tools"]) for p in page)
    assert total == 36
    by_name = {name for p in page for name, _ in p["tools"]}
    assert by_name >= {
        "fetch_ml_data",
        "fetch_online_data",
        "fetch_offline_data",
        "merge_furnace_data",
        "concat_datasets",
        "load_static_shift_data",
        "search_shift_history",
        "search_knowledge_docs",
        "render_heatload_plot",
        "execute_python_plot",
    }
    busy = [q for q, rounds in fakes.REFERENCE].index("This question finds the model busy.")
    assert api[busy]["status"] == "failed" and api[busy]["text"] == page[busy]["text"] and not api[busy]["tools"]
    explode = [q for q, _ in fakes.REFERENCE].index("Search the shifts until the store explodes.")
    assert api[explode]["results_ok"] == [False]
    assert len(api[-1]["tools"]) == 8  # the iteration cap of the graph
    first = api[0]
    assert "<think>" not in first["tokens"] and first["tokens"].endswith(first["text"])
    assert stored["message_count"] == 40 and [m["seq"] for m in stored["messages"]] == list(range(1, 41))


def test_a_turn_completes_and_is_stored_when_nobody_reads_its_events(tmp_path) -> None:
    """A client that goes away before the response starts leaves neither a reserved conversation nor a lost answer."""
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    fm.use(fm.Dependencies(lambda level: fakes.ScriptedLLM(), list, lambda: "prompt", dict, lambda: WINDOW))
    try:
        row = store.fm.create(owner="op", reasoning_level="Low")
        request = fm.begin_turn(
            store, row, content="Explain RAFT in one paragraph.", reasoning_level=None, user_id=None
        )
        assert fm.is_running(str(row.id)) and request.reasoning_level == "Low"
        with pytest.raises(fm.Busy):
            fm.begin_turn(store, row, content="again", reasoning_level=None, user_id=None)

        async def abandon() -> None:
            fm.start_turn(store, request)  # the events are never read
            for _ in range(400):
                if not fm.is_running(str(row.id)):
                    return
                await asyncio.sleep(0.025)

        asyncio.run(abandon())
        messages = store.fm.messages(row.id)
        assert not fm.is_running(str(row.id))
        assert [(m.seq, m.role, m.status) for m in messages] == [(1, "user", "complete"), (2, "assistant", "complete")]
        assert (
            messages[1].content.startswith("Answer to 'Explain RAFT") and messages[0].meta["reasoning_level"] == "Low"
        )
    finally:
        fm.use(None)
        store.dispose()
