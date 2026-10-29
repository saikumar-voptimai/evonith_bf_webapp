"""FurnaceMind twin tools (SV-5.3, #314): get_twin_snapshot, get_twin_series, get_transit and get_optimiser_result
return what the twin API serves (the same services, on a SQLite twin store), and the agent answers the current
coke-rate gap from the tool and cites the snapshot time (a scripted model through POST /api/fm/conversations/{id}/
messages with the real tool schemas, system prompt and dispatcher)."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents import furnace_tools as ft
from agents import twin_tools
from agents.furnacemind import prompts, tool_state
from agents.furnacemind.graph import _TOOL_LABELS
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore
from services import furnacemind as fm
from services import store as store_service
from services import twin_views
from tests import furnacemind_fakes as fakes
from tests.test_api_auth import build
from tests.test_api_furnacemind import api, ask
from tests.test_api_live import session_cookie

IST = "Asia/Kolkata"
TWIN_TOOLS = ("get_twin_snapshot", "get_twin_series", "get_transit", "get_optimiser_result")
QUESTION = "What is the current coke-rate gap?"
CASE_CAL = os.path.join(os.path.dirname(__file__), "fixtures", "cal_2z_jul_oct_2026_case_report_15g.json")


def ist(text: str) -> datetime:
    return pd.Timestamp(text, tz=IST).tz_convert("UTC").to_pydatetime()


class Frozen(datetime):
    at = datetime(2026, 10, 27, 6, 0, tzinfo=UTC)  # 11:30 IST: the window ending 11:00 IST is the expected one

    @classmethod
    def now(cls, tz=None):
        return cls.at if tz is None else cls.at.astimezone(tz)


def snapshot(end: datetime, **values: Any) -> dict[str, Any]:
    return {
        "window_start": end - timedelta(hours=8),
        "window_end": end,
        "evaluated_at": end + timedelta(minutes=5),
        "status": "complete",
        "k_trace": 305.0,
        "k_forge": 303.0,
        "k_strata": 304.0,
        "k_strata_sd": 2.0,
        "k_charged": 306.0,
        "omega_imp": 0.80,
        "omega_strata": 0.79,
        "omega_strata_sd": 0.01,
        "omega_cal": 0.80,
        "quantities": {"hmt_c": 1478.0, "tau_raceway_h": 6.1},
        "flags": {"notes": []},
        "versions": {"model": "test"},
        **values,
    }


def fill(twin: TwinStore) -> list[datetime]:
    """Hourly windows of 7 days to 27 Oct 2026 11:00 IST: TRACE with a daily swing, FORGE and STRATA around it; the
    latest window has FORGE - TRACE = +4.2 and STRATA - TRACE = -3.4 kg/tHM dry."""
    ends = pd.date_range("2026-10-20 12:00", "2026-10-27 11:00", freq="h", tz=IST)
    for i, end in enumerate(ends):
        trace = 300.0 + 4.0 * ((i % 24) / 24.0)
        values = dict(
            k_trace=trace, k_forge=trace + (1.5 if i % 2 else -1.0), k_strata=trace + (0.8 if i % 3 else -1.2)
        )
        if i == len(ends) - 1:
            values = dict(k_trace=305.0, k_forge=309.2, k_strata=301.6)
        twin.snapshots.upsert(snapshot(end.tz_convert("UTC").to_pydatetime(), **values))
    return [e.tz_convert("UTC").to_pydatetime() for e in ends]


@pytest.fixture()
def twin(tmp_path, monkeypatch):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    monkeypatch.setattr(twin_views, "datetime", Frozen)
    yield store
    store_service.reset()
    store.dispose()


@pytest.fixture()
def client(twin):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        c.headers["Cookie"] = session_cookie(app, "op")
        yield c


def tool(name: str, state: dict[str, Any] | None = None, **arguments: Any) -> dict[str, Any]:
    """A twin tool through page 7's dispatcher, its JSON result decoded."""
    with tool_state.bind({} if state is None else state):
        return json.loads(ft.execute_openai_tool_call(name=name, arguments=arguments))


def instant(text: str | None) -> pd.Timestamp | None:
    return None if text is None else pd.Timestamp(text).tz_convert("UTC")


# ------------------------------------------------------------------------------------------------------ contract
def test_the_twin_tools_are_offered_to_the_model_with_json_schemas() -> None:
    schemas = {s["function"]["name"]: s["function"] for s in ft.get_openai_tool_schemas()}
    assert set(TWIN_TOOLS) <= set(schemas) and len(schemas) == 14
    for name in TWIN_TOOLS:
        params = schemas[name]["parameters"]
        assert params["type"] == "object" and params["additionalProperties"] is False and "required" not in params
        assert schemas[name]["description"] and _TOOL_LABELS[name] == twin_tools.LABELS[name]
    assert schemas["get_twin_series"]["parameters"]["properties"]["resolution"]["enum"] == ["hour", "shift"]
    assert schemas["get_optimiser_result"]["parameters"]["properties"]["mode"]["enum"] == list(twin_tools.MODES)
    assert "coke-rate gap" in schemas["get_twin_snapshot"]["description"]
    for text in (prompts.TOOL_POLICY, prompts.AI_COOPERATE_SYSTEM):
        assert all(name in text for name in TWIN_TOOLS)
    assert "quote the tool's cite line" in prompts.TOOL_POLICY


def test_without_a_store_the_twin_tools_fail_with_the_reason() -> None:
    store_service.use(None)
    try:
        for name in ("get_twin_snapshot", "get_twin_series", "get_optimiser_result"):
            with pytest.raises(twin_tools.TwinToolError, match="BF_TWIN_DB_URL"):
                tool(name)
    finally:
        store_service.reset()


def test_time_arguments_without_offset_are_ist() -> None:
    assert twin_tools.parse_time("2026-10-27 11:00") == ist("2026-10-27 11:00")
    assert twin_tools.parse_time("2026-10-27T05:30:00Z") == ist("2026-10-27 11:00")
    assert twin_tools.parse_time("") is None and twin_tools.parse_time(None) is None
    with pytest.raises(ValueError, match="ISO 8601"):
        twin_tools.parse_time("yesterday noon")
    assert twin_tools.ist(datetime(2026, 10, 27, 5, 30)) == "2026-10-27T11:00:00+05:30"  # naive is UTC, as stored
    assert twin_tools.label(ist("2026-10-27 11:00")) == "27 Oct 2026 11:00 IST"


# ------------------------------------------------------------------------------------- same services as the API
def test_snapshot_tool_reports_what_get_twin_now_serves(twin, client) -> None:
    fill(twin)
    twin.alarms.raise_alarm("omega_drift", subject="high", severity="warning", message="drift")
    api_now = client.get("/api/twin/now").json()
    got = tool("get_twin_snapshot")

    snap = got["snapshot"]
    for key in twin_views.SNAPSHOT_FIELDS:
        assert snap[key] == api_now["snapshot"][key], key
    assert snap["quantities"] == api_now["snapshot"]["quantities"] == {"hmt_c": 1478.0, "tau_raceway_h": 6.1}
    for key in ("window_start", "window_end", "evaluated_at"):
        assert instant(snap[key]) == instant(api_now["snapshot"][key]), key
    assert snap["window_end"] == "2026-10-27T11:00:00+05:30" and snap["shift_key"] is None
    for key, value in api_now["gaps"].items():
        if key.endswith("_at") or key.endswith("_since"):
            assert instant(got["gaps"][key]) == instant(value), key
        else:
            assert got["gaps"][key] == pytest.approx(value, rel=1e-12), key
    assert got["gaps"]["forge_trace_kg_thm"] == pytest.approx(4.2) and got["gaps"][
        "strata_trace_kg_thm"
    ] == pytest.approx(-3.4)
    assert got["gaps"]["n_strata"] == 7 * 24
    fresh = got["freshness"]
    assert (fresh["lag_h"], fresh["stale"], fresh["age_s"]) == (api_now["lag_h"], False, pytest.approx(25 * 60))
    assert got["alarms_open"] == api_now["alarms_open"] == 1
    assert [s["key"] for s in got["sources"]] == [s["key"] for s in api_now["sources"]]
    assert got["cite"] == "Twin window 27 Oct 2026 03:00 IST to 27 Oct 2026 11:00 IST, evaluated 27 Oct 2026 11:05 IST"
    assert "kg/tHM dry" in got["units"]["strata_trace_kg_thm"]
    assert any(w.startswith("Sources stale or down") for w in got["warnings"])  # no source times recorded


def test_snapshot_tool_reads_a_past_window_like_get_twin_window(twin, client) -> None:
    fill(twin)
    end = ist("2026-10-26 14:00")  # shift A of 26 Oct
    api_window = client.get("/api/twin/window", params={"end": end.isoformat()}).json()
    got = tool("get_twin_snapshot", window_end="2026-10-26 14:40")  # IST, inside the next hour: floored to 14:00
    assert instant(got["snapshot"]["window_end"]) == instant(api_window["window_end"])
    assert {k: got["snapshot"][k] for k in twin_views.SNAPSHOT_FIELDS} == {
        k: api_window[k] for k in twin_views.SNAPSHOT_FIELDS
    }
    assert got["snapshot"]["shift_key"] == "2026-10-26_A" and "freshness" not in got
    assert instant(got["gaps"]["strata_at"]) == pd.Timestamp(end)
    assert got["gaps"]["strata_trace_kg_thm"] == pytest.approx(api_window["k_strata"] - api_window["k_trace"])
    missing = tool("get_twin_snapshot", window_end="2026-10-01T00:00:00+05:30")
    assert missing["snapshot"] is None and "01 Oct 2026 00:00 IST" in missing["note"]


def test_snapshot_tool_warns_when_the_twin_is_stale(twin) -> None:
    fill(twin)
    Frozen.at = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)  # 13:30 IST: the 12:00 and 13:00 windows are missing
    try:
        got = tool("get_twin_snapshot")
    finally:
        Frozen.at = datetime(2026, 10, 27, 6, 0, tzinfo=UTC)
    assert got["freshness"]["stale"] and got["freshness"]["lag_h"] == 2
    assert got["warnings"][0].startswith("The twin is stale: the latest window ends 27 Oct 2026 11:00 IST, 2 h behind")


def test_snapshot_tool_before_the_first_evaluation(twin) -> None:
    got = tool("get_twin_snapshot")
    assert got["snapshot"] is None and got["note"] == "No twin evaluation is stored yet." and got["freshness"]["stale"]


def test_series_tool_matches_get_twin_series_and_keeps_a_dataset(twin, client) -> None:
    ends = fill(twin)
    params = {"start": ends[0].isoformat(), "end": ends[-1].isoformat(), "quantities": "hmt_c"}
    for resolution in ("hour", "shift"):
        body = client.get("/api/twin/series", params={**params, "resolution": resolution}).json()
        state: dict[str, Any] = {}
        got = tool(
            "get_twin_series",
            state,
            start_time="2026-10-20 12:00",
            end_time="2026-10-27 11:00",
            resolution=resolution,
            quantities=["hmt_c"],
        )
        assert got["count"] == body["count"] == (168 if resolution == "hour" else 21)
        shown = body["rows"][-twin_tools.ROWS_INLINE :]
        assert got["rows_shown"] == len(shown) and got["rows_not_shown"] == body["count"] - len(shown)
        for mine, theirs in zip(got["rows"], shown, strict=True):
            assert instant(mine["window_end"]) == instant(theirs["window_end"])
            assert mine.get("shift_key") == theirs["shift_key"] and mine["quantities"] == theirs["quantities"]
            assert all(mine[k] == theirs[k] for k in twin_views.SNAPSHOT_FIELDS)
        frame = state["fm_datasets"][got["dataset_id"]]["df"]
        assert state["fm_df"] is frame and frame.index.name == "time (IST)" and len(frame) == body["count"]
        assert frame["k_trace"].tolist() == [r["k_trace"] for r in body["rows"]]
        assert frame.index[-1] == instant(body["rows"][-1]["window_end"]) and "hmt_c" in frame.columns
        assert got["stats"]["k_trace"]["n"] == body["count"]
        assert got["stats"]["k_trace"]["mean"] == pytest.approx(frame["k_trace"].mean(), rel=1e-12)
    with pytest.raises(ValueError, match="start lies after end"):
        tool("get_twin_series", start_time="2026-10-27", end_time="2026-10-26")
    with pytest.raises(ValueError, match="At most 92 days"):
        tool("get_twin_series", start_time="2026-01-01", end_time="2026-10-26")
    empty = tool("get_twin_series", start_time="2026-09-01", end_time="2026-09-02")
    assert empty["count"] == 0 and "dataset_id" not in empty


def test_transit_tool_matches_get_twin_transit(twin, client, monkeypatch) -> None:
    from services import coke_energy as ce
    from tests.twin_data import synthetic_frames

    fd, *_ = synthetic_frames(start="2026-05-01", days=4)
    monkeypatch.setattr(ce, "data_keys", lambda cfg, nonce: ce.DataKeys("x", 1, "c", 0, "2026-05-01", "{}", "d", 30.0))
    monkeypatch.setattr(ce, "CACHED_LOADERS", ce.Loaders(lambda p, m: fd, None, None, None, None))
    body = client.get("/api/twin/transit", params={"at": "2026-05-03T06:30:00Z"}).json()
    got = tool("get_transit", at="2026-05-03 12:00")  # IST = 06:30 UTC
    assert got["cite"] == "Burden column at 03 May 2026 12:00 IST" and instant(got["at"]) == instant(body["at"])
    for key in ("charges_per_hour", "tau_cz_h", "tau_raceway_h", "coke_in_transit_t", "complete", "missing_hours"):
        assert got[key] == body[key], key
    assert got["bands"] == body["bands"] and got["lag"] is None
    if body["blend_change"] is not None:
        assert instant(got["blend_change"]["raceway_at"]) == instant(body["blend_change"]["raceway_at"])

    monkeypatch.setattr(ce, "CACHED_LOADERS", ce.Loaders(lambda p, m: (_ for _ in ()).throw(OSError()), *[None] * 4))
    with pytest.raises(RuntimeError, match="hourly dataset is not available"):
        tool("get_transit")


def run_row(**values: Any) -> dict[str, Any]:
    started = values.pop("started_at", datetime(2026, 10, 27, 0, 45, tzinfo=UTC))
    return {
        "trigger": "schedule",
        "requested_by": "scheduler",
        "mode": "step2",
        "omega_mode": "fixed",
        "started_at": started,
        "finished_at": started + timedelta(seconds=40),
        "status": "ok",
        "production_t_d": 2270.0,
        "cost_rs_thm": 29852.4,
        "saving_rs_thm": 180.0,
        "coke_kg_thm": 303.0,
        "fuel_rate_kg_thm": 564.7,
        "certainty": "Held",
        "inputs": {"mode": "step2", "production_target_t_d": 2270.0},
        "result": {"mode": "fixed"},
        **values,
    }


def test_optimiser_tool_returns_the_latest_plan_and_its_decision(twin, client) -> None:
    from services import optimiser as svc
    from utils.burden_engine import Engine
    from utils.burden_engine.engine import Limits
    from utils.burden_engine.scenarios import run_fixed

    assert tool("get_optimiser_result") == {"run": None, "note": "No optimiser run has a plan yet."}
    eng = Engine(materials=["sinter_sp_02", "geomin_clo", "lloyds_clo", "nmdc_rom", "lloyds_pellet"], cal=CASE_CAL)
    plan = svc.jsonable(run_fixed(eng, 2270.0, eng.cal_settings(), Limits())["with_fuel"])
    run = twin.optimiser.add(run_row(result={"mode": "max", "result": plan}, mode="max_production"))
    twin.optimiser.decide(run.id, "accepted", Actor(name="sup", role="supervisor"), comment="Within trial limits")
    older = twin.optimiser.add(run_row(started_at=datetime(2026, 10, 26, 22, 45, tzinfo=UTC)))

    latest = client.get("/api/optimiser/latest").json()
    result = client.get(f"/api/optimiser/runs/{run.id}/result").json()
    got = tool("get_optimiser_result")
    assert got["run"]["id"] == latest["id"] == str(run.id) and got["run"]["decision"]["decision"] == "accepted"
    for key in ("mode", "omega_mode", "status", "production_t_d", "cost_rs_thm", "saving_rs_thm", "certainty"):
        assert got["run"][key] == latest[key], key
    assert got["result"]["plan"]["cost"]["total_rs_thm"] == pytest.approx(result["plan"]["cost"]["total_rs_thm"])
    assert got["result"]["plan"]["production_t_d"] == result["plan"]["production_t_d"]
    assert [b["id"] for b in got["result"]["plan"]["blend"]] == [b["id"] for b in result["plan"]["blend"]]
    assert "frontier" not in got["result"] and "calibration" not in got["result"]
    assert got["cite"] == f"Optimiser run {run.id}: max_production, fixed-omega, ok, finished 27 Oct 2026 06:15 IST"
    assert "frontier" in tool("get_optimiser_result", detail=True)["result"]

    by_mode = tool("get_optimiser_result", mode="step2")
    assert by_mode["run"]["id"] == str(older.id) and by_mode["run"]["decision"] is None
    assert tool("get_optimiser_result", run_id=str(older.id))["run"]["id"] == str(older.id)
    assert tool("get_optimiser_result", mode="step1")["note"] == "No optimiser run has a plan yet in mode step1."
    assert tool("get_optimiser_result", run_id=str(uuid.uuid4()))["run"] is None
    with pytest.raises(ValueError, match="UUID"):
        tool("get_optimiser_result", run_id="latest")
    with pytest.raises(ValueError, match="mode must be one of"):
        tool("get_optimiser_result", mode="cheapest")


# --------------------------------------------------------------------------------------------- acceptance (SV-5.3)
class GapReader(fakes.ScriptedLLM):
    """Scripted model: asks get_twin_snapshot, then answers from what the tool returned (and only from that)."""

    def _decide(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        text, calls = super()._decide(messages)
        if calls:
            return text, calls
        result = json.loads(next(m for m in reversed(messages) if m.get("role") == "tool")["content"])
        gaps = result["gaps"]
        return (
            "<think>read the snapshot</think>"
            f"Coke-rate gap STRATA − TRACE {gaps['strata_trace_kg_thm']:+.1f} kg/tHM dry, FORGE − TRACE "
            f"{gaps['forge_trace_kg_thm']:+.1f} kg/tHM dry (7-day RMSE {gaps['strata_trace_rmse_kg_thm']:.1f}). "
            f"Source: {result['cite']}."
        ), []


def test_the_agent_answers_the_current_coke_rate_gap_from_the_tool_and_cites_the_snapshot_time(twin) -> None:
    fill(twin)
    llm = GapReader([(QUESTION, [[("get_twin_snapshot", {})]])])
    defaults = fm.default_dependencies()
    deps = fm.Dependencies(
        llm=lambda level: llm,
        tools=defaults.tools,
        system_prompt=defaults.system_prompt,
        resources=fakes.resources,
        history_window=lambda: 8,
    )
    with api(twin, deps) as client:
        conversation = client.post(
            "/api/fm/conversations", json={}, headers={"Cookie": client.cookies_by_user["op"]}
        ).json()
        events = ask(client, conversation["id"], QUESTION)
        api_gaps = client.get("/api/twin/now", headers={"Cookie": client.cookies_by_user["op"]}).json()["gaps"]

    answer = events[-1]
    assert answer["type"] == "message.assistant" and answer["message"]["status"] == "complete"
    calls = [(e["part"]["name"], e["part"]["arguments"]) for e in events if e["type"] == "tool.call"]
    results = [e["part"] for e in events if e["type"] == "tool.result"]
    assert calls == [("get_twin_snapshot", {})] and [r["ok"] for r in results] == [True]
    assert [e["part"]["label"] for e in events if e["type"] == "tool.call"] == ["Reading the live twin..."]
    expected = (
        f"Coke-rate gap STRATA − TRACE {api_gaps['strata_trace_kg_thm']:+.1f} kg/tHM dry, FORGE − TRACE "
        f"{api_gaps['forge_trace_kg_thm']:+.1f} kg/tHM dry (7-day RMSE {api_gaps['strata_trace_rmse_kg_thm']:.1f}). "
        "Source: Twin window 27 Oct 2026 03:00 IST to 27 Oct 2026 11:00 IST, evaluated 27 Oct 2026 11:05 IST."
    )
    assert answer["message"]["content"] == expected
    assert "STRATA − TRACE -3.4 kg/tHM dry, FORGE − TRACE +4.2 kg/tHM dry" in expected
    # The model was offered the tool and the routing, and saw the stored result of the snapshot.
    first_call = llm.calls[0]
    assert "get_twin_snapshot" in first_call[0]["content"] and first_call[-1]["content"] == QUESTION
    stored = json.loads(results[0]["content"])
    assert stored["snapshot"]["k_strata"] == 301.6 and stored["snapshot"]["evaluated_at"] == "2026-10-27T11:05:00+05:30"
    assert not results[0]["truncated"]
