"""Demo mode (demo branch, 16 Nov 2026): the name rules of ``config/demo.yml``, the redaction of every kind of twin-api
response, the version route's labels and what FurnaceMind's model reads."""

from __future__ import annotations

import json

import pytest
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.testclient import TestClient

from utils import demo

SENSITIVE = (
    "evonith",
    "wardha",
    "lloyds",
    "nmdc",
    "geomin",
    "gomti",
    "donimalai",
    "jayaswal",
    "titani",
    "acore",
    "bacheli",
    "kirandul",
    "bf-02",
    "sp-02",
)


@pytest.fixture()
def demo_on(monkeypatch):
    monkeypatch.setenv("BF_DEMO", "1")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BF-02, Evonith Steel", "BF-1, Demo Steel"),
        ("Evonith Steel Limited, Wardha", "Demo Steel, the plant site"),
        ("LLOYDS CLO", "SOURCE B CLO"),
        ("Lloyds pellet", "Source B pellet"),
        ("NMDC DONIMALAI", "SOURCE C"),
        ("Donimalai oversize return", "Source C oversize return"),
        ("SINTER (SP-02)", "SINTER (SP-1)"),
        ("ACORE IND.PVT LTD", "SOURCE H"),
        ("Geomin CLO 14.1 %", "Source A CLO 14.1 %"),
        ("from slide 46 of the client deck BF02_FORGE_TRACE_STRATA, 27 Oct 2026", "from slide 46 of the demo deck"),
        ("bucket bf2_evonith_raw", "bucket plant_data"),
        ("coke rate 302.2 kg/tHM", "coke rate 302.2 kg/tHM"),
        ("disk E:\\Work\\EvonithSteel\\webapp 16 %", "disk the app disk 16 %"),
        ("disk /home/bfapp/evonith_webapp 16 %", "disk the app disk 16 %"),
        ("EvonithSteel", "DemoSteel"),
    ],
)
def test_rules(text, expected):
    assert demo.redact_text(text) == expected


def test_ids_stay_for_the_app_and_change_for_the_model():
    text = "ore ids lloyds_clo, nmdc_rom, geomin_clo; coke_rate stays"
    assert demo.redact_text(text) == text  # the SPA sends these ids back on a save
    model = demo.redact_for_model(text)
    assert "source_b_clo" in model and "source_i_rom" in model and "source_a_clo" in model
    assert "coke_rate stays" in model
    assert not any(word in model.lower() for word in SENSITIVE)


def test_redact_walks_keys_values_and_lists():
    value = {"Lloyds CLO": [{"name": "Gomti CLO", "pct": 4.2}], "count": 3, "ok": True, "none": None}
    assert demo.redact(value) == {
        "Source B CLO": [{"name": "Source D CLO", "pct": 4.2}],
        "count": 3,
        "ok": True,
        "none": None,
    }


def test_people_and_labels(demo_on):
    assert demo.person("sai.kumar") == "engineer"
    assert demo.person("scheduler") == "scheduler"
    assert demo.person(None) is None
    assert demo.labels() == {"product": "CoOperate", "furnace": "BF-1", "site": "Demo Steel"}


def test_labels_outside_demo(monkeypatch):
    monkeypatch.delenv("BF_DEMO", raising=False)
    assert not demo.enabled()
    assert demo.labels() == demo.PLANT


def test_json_bytes_keep_unicode_and_newlines():
    body = json.dumps({"note": "Lloyds CLO\nσ(coke) 2 kg/tHM", "n": 1.5}).encode()
    out = json.loads(demo.redact_json_bytes(body))
    assert out == {"note": "Source B CLO\nσ(coke) 2 kg/tHM", "n": 1.5}
    assert demo.redact_json_bytes(b"not json Lloyds") == b"not json Source B"


def test_sse_event():
    event = b'event: token\nid: 7\ndata: {"text": "Evonith BF-02"}'
    out = demo.redact_sse_event(event).split(b"\n")
    assert out[:2] == [b"event: token", b"id: 7"]
    assert json.loads(out[2][len(b"data: ") :]) == {"text": "Demo BF-1"}
    assert demo.redact_sse_event(b"data: plain Wardha") == b"data: plain the plant site"


@pytest.fixture()
def client():
    from api.main import create_app
    from api.settings import ApiSettings

    app = create_app(ApiSettings(static_dir=None))

    @app.get("/api/_test/json")
    def as_json() -> dict:
        return {"material": "LLOYDS CLO", "id": "lloyds_clo", "site": "Wardha"}

    @app.get("/api/strata/_test")
    def strata_view() -> dict:
        return {"blend": {"lloyds_clo": 12.5, "nmdc_rom": 2.0, "sinter": 61.6}}

    @app.get("/api/_test/csv")
    def as_csv() -> PlainTextResponse:
        return PlainTextResponse("material,pct\nGeomin CLO,14.1\n", media_type="text/csv")

    @app.get("/api/_test/sse")
    def as_sse() -> StreamingResponse:
        def events():
            yield b'event: token\ndata: {"text": "Lloyds"}\n\n'
            yield b'event: token\ndata: {"text": "Gom'  # an event split over two chunks
            yield b'ti CLO"}\n\nevent: done\ndata: {}\n\n'

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/_test/missing")
    def missing() -> dict:
        from api.errors import ApiError

        raise ApiError(404, "No run for Evonith BF-02.")

    return TestClient(app, raise_server_exceptions=False)


def test_responses_unchanged_outside_demo(client, monkeypatch):
    monkeypatch.delenv("BF_DEMO", raising=False)
    assert client.get("/api/_test/json").json()["material"] == "LLOYDS CLO"
    body = client.get("/api/version").json()
    assert body["demo"] is False and body["demo_notice"] is None
    assert (body["product"], body["furnace"], body["site"]) == (demo.PRODUCT, demo.PLANT["furnace"], demo.PLANT["site"])


def test_responses_redacted_in_demo(client, demo_on):
    r = client.get("/api/_test/json")
    assert r.json() == {"material": "SOURCE B CLO", "id": "lloyds_clo", "site": "the plant site"}
    assert int(r.headers["content-length"]) == len(r.content)

    csv = client.get("/api/_test/csv")
    assert csv.text == "material,pct\nSource A CLO,14.1\n"

    sse = client.get("/api/_test/sse").text
    assert '{"text":"Source B"}' in sse and '{"text":"Source D CLO"}' in sse and "event: done" in sse
    assert not any(word in sse.lower() for word in SENSITIVE)

    strata = client.get("/api/strata/_test").json()  # a view that shows ids as text: ids replaced too
    assert strata == {"blend": {"source_b_clo": 12.5, "source_i_rom": 2.0, "sinter": 61.6}}

    problem = client.get("/api/_test/missing")
    assert problem.status_code == 404
    assert problem.json()["detail"] == "No run for Demo BF-1."

    version = client.get("/api/version").json()
    assert version["demo"] is True and version["demo_notice"].startswith("Demo data.")
    assert (version["product"], version["furnace"], version["site"]) == ("CoOperate", "BF-1", "Demo Steel")


def test_furnacemind_tool_results_for_the_model(demo_on, monkeypatch):
    from agents import furnace_tools

    monkeypatch.setattr(
        furnace_tools, "_execute_tool", lambda **_: '{"blend": {"lloyds_clo": 13.4}, "note": "Lloyds CLO at Wardha"}'
    )
    out = furnace_tools.execute_openai_tool_call(name="anything", arguments={})
    assert not any(word in out.lower() for word in SENSITIVE)
    assert json.loads(out) == {"blend": {"source_b_clo": 13.4}, "note": "Source B CLO at the plant site"}


def _sqlite_store(tmp_path):
    from data.twin.engine import build_engine, session_factory
    from data.twin.models import TwinBase
    from data.twin.repositories import TwinStore

    engine = build_engine(f"sqlite:///{(tmp_path / 'twin.sqlite').as_posix()}")
    TwinBase.metadata.create_all(engine)
    return TwinStore(session_factory(engine), engine)


def test_skill_conversation_titled_by_its_label(tmp_path):
    """The demo store is SQLite; a conversation a quick skill starts takes the skill's label as its title, not the
    instructions of its prompt."""
    store = _sqlite_store(tmp_path)
    skill = store.fm.create(owner="op", reasoning_level="Medium")
    store.fm.append(
        skill.id,
        role="user",
        content="SKILL: Slag & Burden Check. The numbers below are computed ... do not recompute them",
        meta={"skill": "slag_burden_check", "display": "Slag & Burden Check: the last 24 h against the week before"},
    )
    assert store.fm.get(skill.id).title == "Slag & Burden Check: the last 24 h against the week before"
    asked = store.fm.create(owner="op", reasoning_level="Medium")
    store.fm.append(asked.id, role="user", content="  Why did the\tfuel rate rise? ", meta={"display": " "})
    assert store.fm.get(asked.id).title == "Why did the fuel rate rise?"
    store.dispose()


def test_seed_head_revision_is_the_last_migration():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "demo_seed.py"
    spec = importlib.util.spec_from_file_location("demo_seed_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    newest = max((path.parents[1] / "alembic" / "twin" / "versions").glob("*.py")).stem  # 20261029_twin_0009_...
    assert module.head_revision() == "_".join(newest.split("_")[1:3])


def test_sqlite_store_gives_aware_times(tmp_path):
    """Times read from a SQLite twin store carry UTC, as from PostgreSQL (the SPA would read naive ones as local)."""
    from datetime import UTC, datetime

    store = _sqlite_store(tmp_path)
    at = datetime(2026, 11, 16, 8, 45, tzinfo=UTC)
    created = store.fm.create(owner="op", reasoning_level="Medium", at=at)
    row = store.fm.get(created.id)
    assert row.created_at == at and row.created_at.tzinfo is not None
    store.dispose()


def test_llm_messages_for_the_model(demo_on):
    from agents.llm.llm_client import messages_for_model

    messages = [
        {"role": "system", "content": "Advisor for Evonith BF-02."},
        {"role": "user", "content": [{"type": "text", "text": "blend lloyds_clo 13.4 %, Geomin CLO 14.1 %"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
    ]
    out = messages_for_model(messages)
    assert out[0]["content"] == "Advisor for Demo BF-1."
    assert out[1]["content"][0]["text"] == "blend source_b_clo 13.4 %, Source A CLO 14.1 %"
    assert out[2] == messages[2]
    assert messages[0]["content"] == "Advisor for Evonith BF-02."  # the caller's list is not changed


def test_llm_messages_unchanged_outside_demo(monkeypatch):
    from agents.llm.llm_client import messages_for_model

    monkeypatch.delenv("BF_DEMO", raising=False)
    messages = [{"role": "user", "content": "lloyds_clo at Wardha"}]
    assert messages_for_model(messages) is messages
