"""Burden & Fuel Optimiser API (SV-3.2, #299): the 16 Oct 2026 case report through the API, the worker function
called in-process, results validated against the schema, and the commentary job on the optimiser notes.

The case report's inputs: the reference yard at its own (UAT) prices, the Jul-Oct 2026 calibration at 15 g/Nm³
(``tests/fixtures/cal_2z_jul_oct_2026_case_report_15g.json``, as ``tests/test_burden_engine.py``), Fixed-ω, HMT
1,470 °C, Si 0.3836 %, PCI 190 and nut coke 70 kg/tHM, the calibration's controls, the register's limits. The
production cases (Step 1, maximum production) are slow: ``BURDEN_ENGINE_FAST=1`` skips them."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.live import job_store
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import optimiser as svc
from services import optimiser_commentary, optimiser_results
from services import store as store_service
from services.events import MemoryEventBus
from services.jobs import CATALOG, JobRecord, OptimiserCommentaryParams, OptimiserRunParams, input_hash
from services.optimiser_results import OptimiserResult
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from utils.burden_engine.commentary import NOTES_PATH, SYSTEM_PROMPT, load_assumptions
from utils.burden_engine.facts import ENGINE_FACTS, load_register
from utils.slag_bands import PLANT_SLAG_BANDS
from worker.kinds import optimiser as optimiser_kind
from worker.runtime import JobContext

CASE_CAL = os.path.join(os.path.dirname(__file__), "fixtures", "cal_2z_jul_oct_2026_case_report_15g.json")
slow = pytest.mark.skipif(os.environ.get("BURDEN_ENGINE_FAST") == "1", reason="production cases are slow")

#: The case report's controls and thermal targets (Settings at the calibration point, PCI 190, nut coke 70).
CASE_SETTINGS = {
    "hmt_c": 1470.0,
    "hm_si_pct": 0.3836,
    "pci_kg_thm": 190.0,
    "nut_coke_kg_thm": 70.0,
    "blast_moisture_g_nm3": 15.0,
    "hbt_c": 1188.4,
    "o2_per_air": 0.05209,
    "hot_blast_pressure_bar_g": 2.79,
    "top_pressure_bar_g": 1.42,
}
CASE = {"omega_mode": "fixed", "calibration": "reference", "yard": "reference", "settings": CASE_SETTINGS}


class FakeJobs:
    """The parts of ``JobStore`` the route and the handler use."""

    def __init__(self) -> None:
        self.records: dict[str, JobRecord] = {}

    async def submit(self, kind: str, params: dict[str, Any], user: str) -> tuple[JobRecord, bool]:
        params = CATALOG[kind].params.model_validate(params).model_dump(mode="json")
        record = JobRecord(id=uuid.uuid4().hex, kind=kind, status="queued", params=params,
                           input_hash=input_hash(kind, params), submitted_by=user,
                           submitted_at=datetime.now(UTC))  # fmt: skip
        self.records[record.id] = record
        return record, False

    async def get(self, job_id: str) -> JobRecord | None:
        return self.records.get(job_id)

    async def update(self, job_id: str, **changes: Any) -> None:
        return None


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    yield store
    store_service.reset()
    store.dispose()


@pytest.fixture()
def api(twin, monkeypatch):
    """(client, cookies, jobs): the app with a fake job queue; the page config without the configured price basis
    and with the case report's calibration as the shipped one."""
    cfg = dict(svc.read_page_config(0))
    cfg.pop("prices", None)
    cfg["calibration"] = dict(cfg.get("calibration") or {}, fallback_path=CASE_CAL)
    monkeypatch.setattr(svc, "page_config", lambda mtime: cfg)
    jobs = FakeJobs()
    app = build()
    app.dependency_overrides[job_store] = lambda: jobs
    with TestClient(app, base_url="https://testserver") as client:
        cookies = {u: session_cookie(app, u) for u in PASSWORDS}
        yield client, cookies, jobs


def call(api, method: str, user: str, url: str, **kw: Any):
    client, cookies, _ = api
    return client.request(method, url, headers={"Cookie": cookies[user]}, **kw)


def work(api, job_id: str) -> dict[str, Any]:
    """Run the queued job's worker function in this process, as the worker does."""
    _, _, jobs = api
    record = jobs.records[job_id]
    kind = CATALOG[record.kind]
    job = JobContext(job_id=job_id, kind=kind, attempt=1, store=jobs, bus=MemoryEventBus(), pool=None)
    handler = {"optimiser.run": optimiser_kind.run, "optimiser.commentary": optimiser_kind.commentary}[record.kind]
    return asyncio.run(handler(job, kind.params.model_validate(record.params)))


def run_through_api(api, body: dict[str, Any]) -> dict[str, Any]:
    started = call(api, "POST", "sup", "/api/optimiser/runs", json=body)
    assert started.status_code == 202, started.text
    out = work(api, started.json()["job"]["id"])
    response = call(api, "GET", "op", f"/api/optimiser/runs/{out['run_id']}/result")
    assert response.status_code == 200, response.text
    OptimiserResult.model_validate(response.json())
    return response.json()


def shares(plan: dict[str, Any]) -> dict[str, float]:
    return {b["id"]: b["share_wet_pct"] for b in plan["blend"]}


def flux(plan: dict[str, Any]) -> dict[str, float]:
    return {f["id"]: f["wet_kg_thm"] for f in plan["flux"]}


# ----------------------------------------------------------------------------------------------- case report
def test_step2_reproduces_cases_1_and_2_of_the_case_report(api) -> None:
    res = run_through_api(api, {"mode": "step2", "production_t_d": 2270, **CASE})
    assert (res["status"], res["mode"], res["omega_mode"], res["yard_source"]) == ("ok", "step2", "fixed", "reference")
    assert res["calibration"]["basis"] == "reference" and res["production_target_t_d"] == 2270.0
    # On page 9's slag basis (held Si 0.60 %, the plant burden sheet's dust row of 530 kg/charge held constant, minors
    # from the analyses, FeO rule, TiO2 left out of the slag as on the UAT page; 11 Nov 2026) the case report's plan
    # moves: sinter 62.5 %, quartz 2.5 kg/tHM, the Al2O3 cap binding (the engine tests without the basis keep the case
    # report's own figures).
    plan = res["plan"]  # case 2: fuel in the objective
    assert plan["feasible"] and plan["production_t_d"] == pytest.approx(2270.0)
    assert plan["fuel"]["coke_dry_kg_thm"] == pytest.approx(302.2, abs=0.3)
    assert plan["slag"]["rate_kg_thm"] == pytest.approx(340.0, abs=0.1)
    assert plan["fuel"]["fuel_rate_kg_thm"] == pytest.approx(563.5, abs=0.3)
    assert plan["cost"]["total_rs_thm"] == pytest.approx(30092, abs=5)
    assert plan["process"]["raft_c"] == pytest.approx(2212, abs=2)
    assert {b["limit"] for b in plan["binding"]} >= {"slag cap", "basicity min", "Al2O3 max"}
    assert plan["warnings"] == [] and plan["shaft_efficiency"]["certainty"] == "Held"
    cost = plan["cost"]
    assert cost["ore_rs_thm"] + cost["flux_rs_thm"] + cost["fuel_rs_thm"] - cost["slag_credit_rs_thm"] == pytest.approx(
        cost["total_rs_thm"]
    )
    assert cost["total_sigma_rs_thm"] is None and plan["fuel"]["coke_sigma_kg_thm"] is None  # Fixed-ω
    ore = res["ore_flux_plan"]  # case 1: page 9's objective
    assert ore["cost"]["total_rs_thm"] == pytest.approx(30175, abs=5)
    sh = shares(ore)
    assert sh["sinter_sp_02"] == pytest.approx(58.0, abs=0.1) and sh["lloyds_pellet"] == pytest.approx(10.0, abs=0.1)
    assert sh["geomin_clo"] == pytest.approx(16.5, abs=0.2) and flux(ore)["dolomite"] == pytest.approx(20.0, abs=0.3)
    assert sum(b["share_wet_pct"] for b in plan["blend"]) == pytest.approx(100.0, abs=0.1)
    # page 9's blend-table columns and Slag Balance Details on the plan (the breakup of the slag rate)
    sinter = next(b for b in plan["blend"] if b["id"] == "sinter_sp_02")
    assert sinter["fe_kg_thm"] == pytest.approx(sinter["dry_kg_thm"] * 0.5446, rel=1e-3)      # dry kg x total Fe %
    assert sinter["slag_kg_thm"] == pytest.approx(sinter["dry_kg_thm"] * 0.2148, rel=2e-3)    # slag-forming oxides
    assert sinter["kg_per_charge"] == pytest.approx(sinter["wet_kg_thm"] * plan["charging"]["hm_per_charge_t"], rel=1e-9)
    assert sinter["cost_rs_thm"] == pytest.approx(sinter["wet_kg_thm"] * sinter["price_rs_t"] / 1000, rel=1e-9)
    comps = {c["key"]: c for c in plan["slag"]["components"]}
    assert comps["Al2O3"]["pct"] == pytest.approx(plan["slag"]["al2o3_pct"], rel=1e-9)
    assert comps["TiO2"]["kg_thm"] == 0.0 and comps["FeO"]["kg_thm"] == pytest.approx(1.2, abs=0.1)  # TiO2 left out
    assert sum(c["kg_thm"] for k, c in comps.items() if k != "Slag") == pytest.approx(340.0, abs=0.1)
    rollup = {x["source"]: x["kg_thm"] for x in plan["slag"]["rollup"]}
    assert rollup["Top-gas dust"] < 0 < rollup["Ore (slag-forming oxides)"] and rollup["Slag"] == pytest.approx(340.0, abs=0.1)
    assert {x["removal"] for x in plan["slag"]["removals"]} >= {"SiO₂ reduced to hot-metal Si", "Fe to the hot metal",
                                                               "TiO₂ left out of the slag"}
    assert "Si 0.60 %" in plan["slag"]["basis"] and plan["slag"]["dust_kg_thm"] == pytest.approx(36.2, abs=0.2)
    # page 9's fuel-ash inputs and dust rows on the plan: the calibration's analyses, the burden sheet's dust row
    ash = {x["fuel"]: x for x in plan["slag"]["fuel_ash"]}
    assert list(ash) == ["Coke", "Nut coke", "PCI"] and ash["PCI"]["wet_kg_thm"] == pytest.approx(plan["fuel"]["pci_kg_thm"])
    assert ash["Coke"]["wet_kg_thm"] == pytest.approx(plan["fuel"]["coke_wet_kg_thm"], rel=1e-6)
    assert ash["Coke"]["ash_kg_thm"] == pytest.approx(ash["Coke"]["dry_kg_thm"] * ash["Coke"]["ash_pct"] / 100, rel=1e-9)
    assert 50 < ash["Coke"]["sio2_pct"] < 60 and ash["Coke"]["slag_kg_thm"] > ash["Nut coke"]["slag_kg_thm"] > 0
    dust = plan["slag"]["dust"]
    assert [(x["id"], x["kg_per_charge"], x["cao_pct"]) for x in dust] == [("bf_gas_dust", 530.0, 5.63)]
    assert dust[0]["kg_thm"] == pytest.approx(plan["slag"]["dust_kg_thm"]) and "burden sheet" in dust[0]["source"]
    assert [(d["id"], d["enabled"], d["kg_per_charge"]) for d in res["yard"]["dust"]] == [
        ("bf_gas_dust", True, 530.0), ("dust_catcher", False, 229.0), ("gcp_dust", False, 235.0)]
    assert res["ore_flux_plan"]["slag"]["components"]  # every plan of the run carries the breakup
    vol = {v["change"]: v["saving_rs_thm"] for v in res["value_of_limits"]}  # the case report's value of limits
    assert vol["Slag cap +10 kg/tHM"] == pytest.approx(43, abs=3)
    assert vol["CaO/SiO₂ floor −0.01"] == pytest.approx(2.8, abs=1)
    assert res["today_plan"] is None and res["saving_rs_thm"] is None  # the reference yard has no charged blend
    assert res["critical_assumptions"] == [] and res["solves"] is None


def test_a_run_can_deduct_its_own_dust(api) -> None:
    """Page 11's *Top-gas dust and fuel ash* editor applied to the run (not saved): the request's kg per charge and
    analysis replace the operator inputs in force for the named streams; the case records what was set."""
    res = run_through_api(api, {"mode": "step2", "production_t_d": 2270, **CASE,
                                "dust": {"bf_gas_dust": {"kg_per_charge": 400, "cao_pct": 6.0}}})
    assert res["status"] == "ok" and res["case"]["dust_set"] == ["bf_gas_dust"]
    yard = {d["id"]: d for d in res["yard"]["dust"]}
    assert (yard["bf_gas_dust"]["kg_per_charge"], yard["bf_gas_dust"]["cao_pct"], yard["bf_gas_dust"]["mgo_pct"]) == (400.0, 6.0, 3.57)
    assert "this run" in yard["bf_gas_dust"]["source"] and yard["dust_catcher"]["kg_per_charge"] == 229.0
    plan = res["plan"]
    (stream,) = plan["slag"]["dust"]
    assert stream["kg_per_charge"] == 400.0 and stream["cao_pct"] == 6.0
    assert plan["slag"]["dust_kg_thm"] == pytest.approx(36.2 * 400 / 530, abs=0.6)   # the blend's charges per tHM move a little
    # an unknown stream is refused before the engine runs
    started = call(api, "POST", "sup", "/api/optimiser/runs", json={"mode": "step2", **CASE, "dust": {"nope": {"kg_per_charge": 1}}})
    assert started.status_code == 202
    with pytest.raises(Exception, match="unknown dust stream"):
        work(api, started.json()["job"]["id"])


@slow
def test_step1_reproduces_case_4_of_the_case_report(api) -> None:
    res = run_through_api(api, {"mode": "step1", **CASE})
    assert res["plan"]["production_t_d"] == pytest.approx(2283, abs=3)
    assert res["plan"]["cost"]["total_rs_thm"] == pytest.approx(29900, abs=10)
    assert res["plan"]["process"]["hbt_c"] == pytest.approx(1230.0)
    assert res["highest_output_plan"]["production_t_d"] == pytest.approx(2404, abs=3)
    assert min(p["cost_rs_thm"] for p in res["frontier"]) >= res["plan"]["cost"]["total_rs_thm"] - 1


@slow
def test_maximum_production_reproduces_case_3_of_the_case_report(api) -> None:
    res = run_through_api(api, {"mode": "max_production", **CASE})
    plan = res["plan"]
    assert plan["production_t_d"] == pytest.approx(2404, abs=3)
    assert plan["cost"]["total_rs_thm"] == pytest.approx(30758, abs=10)
    assert "charging" in {b["limit"] for b in plan["binding"]}
    assert shares(plan)["lloyds_pellet"] == pytest.approx(19.5, abs=0.3)


# ------------------------------------------------------------------------------------------------- requests
def test_settings_and_limits_of_a_request_replace_the_defaults(api) -> None:
    case = svc.default_case(
        "step2", omega_mode="fixed", calibration_basis="reference", yard_basis="reference",
        settings={"hbt_c": 1201.0, "pci_kg_thm": None}, limits={"slag_max_kg_thm": 350.0, "charges_per_h": 6.35},
    )  # fmt: skip
    settings, limits = case.args[3], case.args[4]
    assert settings["HBT"] == 1201.0 and limits["slag_max"] == 350.0 and limits["charges_per_h"] == 6.35
    assert limits["fuel_min"] == 520.0 and limits["fuel_max"] == 600.0  # BAND-01 from the register
    summary = case.summary()
    assert summary["overrides"] == {"settings": {"hbt_c": 1201.0}, "limits": {"slag_max_kg_thm": 350.0,
                                                                            "charges_per_h": 6.35}, "dust": {}}  # fmt: skip
    assert summary["calibration"]["source"] == "fallback" and summary["yard"]["fuel"]["coke_price"] > 0
    with pytest.raises(ValueError, match="unknown limit"):
        svc.default_case("step2", calibration_basis="reference", yard_basis="reference", limits={"fuel_max": 650})
    bad = call(api, "POST", "sup", "/api/optimiser/runs", json={"mode": "step2", "limits": {"fuel_max_kg_thm": 650}})
    assert bad.status_code == 422  # the fuel-rate band is the register's
    bad = call(api, "POST", "sup", "/api/optimiser/runs", json={"mode": "step2", "settings": {"hmt_c": 1700}})
    assert bad.status_code == 422
    assert call(api, "POST", "op", "/api/optimiser/runs", json={"mode": "step2"}).status_code == 403


def test_result_and_commentary_routes_answer_404_and_422(api) -> None:
    missing = uuid.uuid4()
    assert call(api, "GET", "op", f"/api/optimiser/runs/{missing}/result").status_code == 404
    assert call(api, "GET", "op", "/api/optimiser/runs/not-a-uuid/result").status_code == 422
    assert call(api, "GET", "op", f"/api/optimiser/runs/{missing}/commentary").status_code == 404
    assert call(api, "POST", "op", f"/api/optimiser/runs/{missing}/commentary").status_code == 404


# --------------------------------------------------------------------------------------------------- schema
@pytest.fixture(scope="module")
def case_plan():
    """The case report's Step 2 result as stored (``jsonable``), for the Step 1 and maximum-production views."""
    from utils.burden_engine import Engine
    from utils.burden_engine.engine import Limits
    from utils.burden_engine.scenarios import frontier_table, run_fixed

    eng = Engine(materials=["sinter_sp_02", "geomin_clo", "lloyds_clo", "nmdc_rom", "lloyds_pellet"], cal=CASE_CAL)
    res = run_fixed(eng, 2270.0, eng.cal_settings(), Limits())
    lower = run_fixed(eng, 2200.0, eng.cal_settings(), Limits())["with_fuel"]
    return svc.jsonable(res["with_fuel"]), svc.jsonable(frontier_table([lower, res["with_fuel"]]))


@pytest.mark.parametrize("mode", ["step1", "max_production"])
def test_step1_and_maximum_production_views_validate(case_plan, mode) -> None:
    plan, frontier = case_plan
    result = (
        {"mode": "cheapest", "best": plan, "top": plan, "frontier": frontier}
        if mode == "step1"
        else {"mode": "max", "result": plan}
    )
    row = {"id": uuid.uuid4(), "mode": mode, "omega_mode": "fixed", "status": "ok", "inputs": {"materials": {}},
           "result": result, "started_at": datetime(2026, 10, 29, tzinfo=UTC),
           "finished_at": datetime(2026, 10, 29, 0, 1, tzinfo=UTC)}  # fmt: skip
    view = optimiser_results.result_view(row)
    OptimiserResult.model_validate(view.model_dump(mode="json"))
    assert view.plan.cost.total_rs_thm == pytest.approx(30139, abs=5) and view.run_s == 60.0
    if mode == "step1":
        assert view.highest_output_plan is not None and len(view.frontier) == 2
        assert view.frontier[1].marginal_rs_per_t is not None and view.frontier[0].marginal_rs_per_t is None


def test_a_failed_run_and_the_critical_assumptions_validate() -> None:
    row = {"id": uuid.uuid4(), "mode": "step2", "omega_mode": "strata", "status": "infeasible",
           "inputs": {}, "result": {"error": "BandError: HMT outside 1440-1520 °C"}}  # fmt: skip
    view = optimiser_results.result_view(row)
    assert view.plan is None and view.error.startswith("BandError")
    rows = [{"assumption": "RI of Lloyds pellet", "value": "65 %", "alternative": "60 / 70 %", "d_coke_kg": 1.2,
             "d_coke_lo_kg": -1.1, "d_coke_hi_kg": 1.3, "basis": "literature"},
            {"assumption": "Size exponent n", "value": "1.0", "alternative": "1.5", "d_coke_kg": 0.8,
             "symmetric": True, "basis": "GAS-21"}]  # fmt: skip
    found, error = optimiser_results._critical(rows)
    assert [c.coke_effect_kg_thm for c in found] == [1.2, 0.8] and found[1].symmetric and error is None
    assert optimiser_results._critical({"error": "network down"}) == ([], "network down")


UNITS = ("kg/tHM", "t/d", "t/h", "Rs/tHM", "Rs/t", "lakh Rs/d", "°C", "%", "count", "dimensionless", "charges/h",
         "t", "kg", "s", "bar g", "bar", "Nm³/h", "g/Nm³", "Nm³ O₂ per Nm³ air", "in the limit's unit",
         "in the band's unit", "in the fact's unit")  # fmt: skip


@pytest.mark.parametrize(
    "schema",
    ["OptimiserResult", "PlanView", "CostSplit", "FuelSplit", "SlagView", "ChargingView", "ProcessView",
     "ShaftEfficiency", "NetworkSolves", "FrontierPoint", "ValueOfLimit", "CriticalAssumption", "BlendRow", "FluxRow",
     "BindingLimit", "Violation", "OptimiserSettings", "OptimiserLimits", "CommentaryView", "PlanLimits",
     "RunSettings", "TodayPoint", "RunCase", "YardMaterial", "YardFlux", "YardFuel", "PlanLead", "OptimiserReference",
     "SlagBand", "RegisterBand", "FactRow", "ModelFactRow"],
)  # fmt: skip
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = []
    for name, prop in spec.get("properties", {}).items():
        types = {prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}
        if not types & {"number", "integer"}:
            continue
        description = prop.get("description") or ""
        if not ("," in description and description.rsplit(",", 1)[-1].strip() in UNITS):
            missing.append(f"{name}: {description!r}")
    assert not missing, missing


# ------------------------------------------------------------------------------------------------ commentary
class FakeLLM:
    def __init__(self, failures: int) -> None:
        self.failures, self.calls = failures, []
        self.last_actual_model = "test/model"

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if len(self.calls) <= self.failures:
            raise TimeoutError("LLM timed out")
        return "<think>draft</think>**Furnace and model today** - on track."


def test_the_commentary_job_uses_the_optimiser_notes_and_retries_once(api, monkeypatch) -> None:
    llm = FakeLLM(failures=1)
    monkeypatch.setattr(optimiser_commentary, "commentary_client", lambda: llm)
    monkeypatch.setattr(optimiser_commentary, "LLM_RETRY_DELAY_S", 0.0)
    res = run_through_api(api, {"mode": "step2", "production_t_d": 2270, **CASE})
    started = call(api, "POST", "op", f"/api/optimiser/runs/{res['run_id']}/commentary")
    assert started.status_code == 202, started.text
    out = work(api, started.json()["job"]["id"])
    assert (out["status"], out["tries"]) == ("ok", 2) and "try 1" not in (out["error"] or "")
    system, user = llm.calls[-1]
    notes = NOTES_PATH.read_text(encoding="utf-8").strip()
    assert system == SYSTEM_PROMPT and notes and notes in user
    assert user.startswith("# PART 1 - OPTIMISER NOTES") and "# PART 2 - FACT REGISTER" in user
    assert "OPTIMISED PLAN - fixed production 2,270 t/d" in user and "Cost Rs/tHM" in user
    assert "Coke Energy shift results" in user  # the model check is listed as not available
    got = call(api, "GET", "op", f"/api/optimiser/runs/{res['run_id']}/commentary", params={"context": True}).json()
    assert got["status"] == "ok" and got["text"] == "**Furnace and model today** - on track."
    assert got["notes_sha1"] == hashlib.sha1(notes.encode("utf-8")).hexdigest() and got["model"] == "test/model"
    assert got["context"] == user and got["context_sha1"] == hashlib.sha1(user.encode("utf-8")).hexdigest()
    assert re.fullmatch(r"\d+\.\d+", got["facts_version"])
    assert call(api, "GET", "op", f"/api/optimiser/runs/{res['run_id']}/commentary").json()["context"] is None


def test_a_commentary_that_fails_twice_is_stored_as_failed(api, twin, monkeypatch) -> None:
    llm = FakeLLM(failures=5)
    monkeypatch.setattr(optimiser_commentary, "commentary_client", lambda: llm)
    monkeypatch.setattr(optimiser_commentary, "LLM_RETRY_DELAY_S", 0.0)
    run = twin.optimiser.add({
        "trigger": "user", "requested_by": "sup", "mode": "step2", "omega_mode": "fixed", "status": "infeasible",
        "started_at": datetime(2026, 10, 29, tzinfo=UTC), "finished_at": datetime(2026, 10, 29, tzinfo=UTC),
    })  # fmt: skip
    assert call(api, "POST", "op", f"/api/optimiser/runs/{run.id}/commentary").status_code == 409  # no plan
    res = run_through_api(api, {"mode": "step2", "production_t_d": 2270, **CASE})
    _, _, jobs = api
    record, _ = asyncio.run(jobs.submit("optimiser.commentary", {"run_id": res["run_id"]}, "op"))
    out = work(api, record.id)
    assert out["status"] == "failed" and len(llm.calls) == 2
    assert out["error"] == "try 1: TimeoutError: LLM timed out; try 2: TimeoutError: LLM timed out"
    got = call(api, "GET", "op", f"/api/optimiser/runs/{res['run_id']}/commentary").json()
    assert got["status"] == "failed" and got["text"] is None and got["requested_by"] == "op"
    assert OptimiserCommentaryParams(run_id=res["run_id"]).run_id == res["run_id"]
    with pytest.raises(ValueError):
        OptimiserRunParams(mode="step2", yard="yesterday")


# --------------------------------------------------------------------------- SV-3.3: the SPA's optimiser page (#300)
FIXTURE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib" / "optimiser" / "fixtures" / "result.json"


def _close(actual: float | None, expected: float | None) -> None:
    assert actual is not None and expected is not None
    assert abs(actual - expected) <= 1e-6 * max(1.0, abs(expected)), (actual, expected)


def test_the_recorded_result_equals_page_11() -> None:
    """The recorded run as the API serves it carries page 11's figures of the same engine result (``plan_tiles``,
    ``plan_sentence``): relative difference <= 1e-6."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    res, exp = data["step2"], data["expected"]["step2"]
    OptimiserResult.model_validate(res)
    plan, tiles = res["plan"], exp["tiles"]
    pairs = {
        ("overview", "total"): plan["cost"]["total_rs_thm"],
        ("overview", "P_tday"): plan["production_t_d"],
        ("overview", "P_th"): plan["production_t_h"],
        ("overview", "fe_wet_pct"): plan["burden_fe_wet_pct"],
        ("overview", "ferrous_kg"): plan["ferrous_kg_thm"],
        ("cost", "ore"): plan["cost"]["ore_rs_thm"],
        ("cost", "flux"): plan["cost"]["flux_rs_thm"],
        ("cost", "fuel"): plan["cost"]["fuel_rs_thm"],
        ("cost", "coke"): plan["cost"]["coke_rs_thm"],
        ("cost", "nut"): plan["cost"]["nut_coke_rs_thm"],
        ("cost", "pci"): plan["cost"]["pci_rs_thm"],
        ("cost", "gross"): plan["cost"]["gross_rs_thm"],
        ("fuel", "coke_dry"): plan["fuel"]["coke_dry_kg_thm"],
        ("fuel", "coke_wet"): plan["fuel"]["coke_wet_kg_thm"],
        ("fuel", "total"): plan["fuel"]["fuel_rate_kg_thm"],
        ("slag", "rate"): plan["slag"]["rate_kg_thm"],
        ("slag", "t_day"): plan["slag"]["t_d"],
        ("slag", "b2"): plan["slag"]["cao_sio2"],
        ("slag", "t_basicity"): plan["slag"]["t_basicity"],
        ("slag", "ib4"): plan["slag"]["ib4"],
        ("slag", "mgo_al2o3"): plan["slag"]["mgo_al2o3"],
        ("charging", "coke_t"): plan["charging"]["coke_t_d"],
        ("charging", "ferrous_t"): plan["charging"]["ferrous_t_d"],
        ("charging", "hm_per_charge"): plan["charging"]["hm_per_charge_t"],
        ("charging", "flux_kg"): plan["charging"]["flux_kg_thm"],
    }
    for (group, key), value in pairs.items():
        _close(value, tiles[group][key])
    _close(res["saving_lakh_rs_d"], exp["saving_lakh_rs_d"])
    # page 9's per-material columns and the slag breakup (the Blend Mix Optimiser's basis) travel with the plan
    rows = {m["id"]: m for m in exp["material_rows"]}
    for b in [*plan["blend"], *plan["flux"]]:
        m = rows[b["id"]]
        for key, mine in (("fe_kg", "fe_kg_thm"), ("slag_kg", "slag_kg_thm"), ("kg_per_charge", "kg_per_charge"),
                          ("cost_rs_thm", "cost_rs_thm")):
            if mine in b:
                _close(b[mine], m[key])
    comps = plan["slag"]["components"]
    assert [c["label"] for c in comps] == [c["component"] for c in exp["slag_components"]]
    for c, e in zip(comps, exp["slag_components"]):
        _close(c["kg_thm"], e["kg_per_thm"])
        _close(c["pct"], e["pct"])
    assert comps[-1]["key"] == "Slag" and _close(comps[-1]["kg_thm"], plan["slag"]["rate_kg_thm"]) is None
    assert sum(c["kg_thm"] for c in comps[:-1]) == pytest.approx(plan["slag"]["rate_kg_thm"], rel=1e-9)
    for x, e in zip(plan["slag"]["balance"], exp["slag_balance"]):
        assert x["component"] == e["component"]
        for key in ("ore", "flux", "fuel_ash", "dust", "hot_metal_gas", "slag"):
            _close(x[f"{key}_kg_thm"], e[key])
    for x, e in zip(plan["slag"]["rollup"], exp["slag_rollup"]):
        assert x["source"] == e["source"] and _close(x["kg_thm"], e["kg_per_thm"]) is None
    assert sum(x["kg_thm"] for x in plan["slag"]["rollup"][:-1]) == pytest.approx(plan["slag"]["rate_kg_thm"], rel=1e-9)
    assert [x["removal"] for x in plan["slag"]["removals"]] == [x["removal"] for x in exp["slag_removals"]]
    assert plan["slag"]["basis"].startswith("Slag on the Blend Mix Optimiser's basis") and plan["slag"]["dust_kg_thm"] > 0
    assert [res["lead"]["kind"], res["lead"]["text"]] == exp["lead"]
    bands = {b["key"]: b for b in data["reference"]["slag_bands"]}
    assert {k: (b["lo"], b["hi"]) for k, b in bands.items()} == {k: tuple(v) for k, v in PLANT_SLAG_BANDS.items()}


def test_the_case_and_yard_of_a_run_are_served_in_the_api_names(api) -> None:
    res = run_through_api(api, {"mode": "step2", "production_t_d": 2270, **CASE, "limits": {"slag_max_kg_thm": 345}})
    case = res["case"]
    assert case["settings"] == CASE_SETTINGS  # the request's values, in the request's names
    assert case["settings_set"] == sorted(CASE_SETTINGS) and case["limits_set"] == ["slag_max_kg_thm"]
    assert case["limits"]["slag_max_kg_thm"] == 345.0 and case["limits"]["fuel_max_kg_thm"] == 600.0
    assert case["limits"]["process"] is False and case["certainty_filter"] is None  # Fixed-ω
    assert case["production_source"] == "requested" and case["today"] is None  # the shipped calibration
    plan = res["plan"]
    assert plan["limits"] == case["limits"] and plan["limited_by"]
    prices = {m["id"]: m["price_rs_t"] for m in res["yard"]["materials"]}
    assert all(b["price_rs_t"] == prices[b["id"]] for b in plan["blend"])
    assert res["yard"]["fuel"]["coke_rs_t"] > 0 and {f["id"] for f in res["yard"]["fluxes"]} >= {"limestone"}
    assert res["lead"]["kind"] == "plain" and res["lead"]["text"].startswith("**Charge** ")
    assert all(v["binds_after_label"] for v in res["value_of_limits"])


def test_the_reference_serves_the_plant_bands_facts_and_assumptions(api) -> None:
    got = call(api, "GET", "op", "/api/optimiser/reference")
    assert got.status_code == 200, got.text
    ref = got.json()
    reg = load_register()
    assert {b["key"]: (b["lo"], b["hi"]) for b in ref["slag_bands"]} == {
        k: tuple(v) for k, v in PLANT_SLAG_BANDS.items()
    }
    assert {b["fact_id"]: (b["lo"], b["hi"]) for b in ref["bands"]} == {
        f: reg.band(f) for f in ("BAND-01", "BAND-02", "BAND-03")
    }
    assert [f["id"] for f in ref["facts"]] == list(ENGINE_FACTS)
    assert len(ref["model_facts"]) == len(reg.model_table()) and ref["facts_version"] == str(reg.version)
    assert ref["assumptions"] == load_assumptions() and ref["assumptions"].startswith("- **Shaft efficiency**")
    client, _, _ = api
    assert client.get("/api/optimiser/reference").status_code == 401


def test_a_run_started_on_the_page_appears_on_the_live_twin_plan_card(api) -> None:
    """The page's request (``runParams`` of the form: Step 2, Fixed-ω, a target and a thermal setting) goes through the
    endpoint and the job the Live Twin's "Run now" uses; the worker stores it in the same runs table, so the plan card's
    ``GET /api/optimiser/latest`` and the runs list return it; the job's result names the run the page then shows."""
    body = {
        "mode": "step2",
        "omega_mode": "fixed",
        "certainty": "any",
        "risk_k": 0,
        "auto": False,
        "calibration": "reference",
        "yard": "reference",
        "production_t_d": 2270,
        "settings": CASE_SETTINGS,
    }
    started = call(api, "POST", "sup", "/api/optimiser/runs", json=body)
    assert started.status_code == 202, started.text
    job_id = started.json()["job"]["id"]
    out = work(api, job_id)
    latest = call(api, "GET", "op", "/api/optimiser/latest")
    assert latest.status_code == 200 and latest.json()["id"] == out["run_id"]
    assert latest.json()["trigger"] == "user" and latest.json()["requested_by"] == "sup"
    assert call(api, "GET", "op", "/api/optimiser/runs").json()[0]["id"] == out["run_id"]
    assert out["run_id"] and out["status"] == "ok"  # the job's result names the run (read by job id with Redis)
    result = call(api, "GET", "op", f"/api/optimiser/runs/{out['run_id']}/result").json()
    assert result["case"]["settings_set"] == sorted(CASE_SETTINGS) and result["production_target_t_d"] == 2270.0
