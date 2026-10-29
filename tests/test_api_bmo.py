"""Blend Mix Optimiser API (SV-3.5, #302): LP runs queued as ``bmo.run`` jobs and read back with their plan, the Si
and charged-coke forecasts read from the worker's ledgers, page 9's snapshots, roles, and units in the schemas.

The plan of a run equals page 9's on the five recorded days: ``tests/test_services_bmo_case.py`` (the run result is
``services.bmo_views.run_result`` of ``services.bmo_case.run_lp``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services import bmo_views
from services import forecasts as forecast_service
from services.jobs import CATALOG, BmoRunParams, JobStore
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_api_reports import MemoryRedis, call
from utils.bmo.charged_coke.ledger import ForecastLedger
from utils.bmo.si_forecast.ledger import SiliconForecastLedger
from utils.bmo.si_forecast.service import forecast_origin

ROOT = Path(__file__).resolve().parents[1]
SI_BUNDLE = ROOT / "src/assets/models/bmo_si_forecast/20260924_initial"  # hourly origins (forest.npz)
CHARGED_BUNDLE = ROOT / "src/assets/models/bmo_charged_coke/20261007_0400"
SNAPSHOTS = ROOT / "src/storage/bmo_snapshots"


@pytest.fixture()
def client():
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        app.state.live.store = JobStore(MemoryRedis())
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def _cfg(tmp_path: Path, issuer: str = "auto") -> dict[str, Any]:
    return {
        "forecast_issuer": issuer,
        "si_forecast": {
            "mode": "forecast",
            "default_bundle_dir": str(SI_BUNDLE),
            "storage_dir": str(tmp_path / "si"),
            "static_dataset_path": str(tmp_path / "data" / "furnace_dataset.csv"),
        },
        "data_driven_coke": {
            "dataset_path": str(tmp_path / "data" / "furnace_dataset.csv"),
            "charged_4h": {"storage_dir": str(tmp_path / "cc"), "bundle_dir": str(CHARGED_BUNDLE)},
        },
        "data_sources": {"static_dataset_url": ""},
    }


def _si_issue(storage: Path, origin: pd.Timestamp, si: float) -> None:
    SiliconForecastLedger(storage).record_issue(
        {
            "origin_at": origin.isoformat(),
            "issued_at": (origin + pd.Timedelta(minutes=1)).isoformat(),
            "status": "ok",
            "mode": "forecast",
            "model_version": "20260924_initial",
            "si_pct": si,
            "horizons": [
                {
                    "horizon_minutes": minutes,
                    "label": "Now" if not minutes else f"+{minutes // 60} h",
                    "target_start": (origin + pd.Timedelta(minutes=minutes)).isoformat(),
                    "target_end": (origin + pd.Timedelta(minutes=minutes + 60)).isoformat(),
                    "si_pct": si + minutes / 6000,
                    "lower_pct": si - 0.1,
                    "upper_pct": si + 0.1,
                    "status": "ok",
                }
                for minutes in (0, 60, 120, 180)
            ],
        }
    )


def _charged_issue(storage: Path, row: pd.Timestamp, coke: float) -> None:
    ForecastLedger(storage).record_issue(
        {
            "data_row": row,
            "issued_at": row + pd.Timedelta(hours=1, minutes=7),
            "state": "cruise",
            "shown": True,
            "prediction_kg_thm": coke,
            "lower_kg_thm": coke - 12.0,
            "upper_kg_thm": coke + 12.0,
            "coke_window": [row - pd.Timedelta(hours=3), row + pd.Timedelta(hours=1)],
            "model_version": "20261007_0400",
            "path": [{"horizon": 1, "at": row + pd.Timedelta(hours=1), "prediction_kg_thm": coke + 1.0}],
            "reasons": [],
        }
    )


# ---------------------------------------------------------------------------------------------- runs
def _result(**changes: Any) -> dict[str, Any]:
    plan = bmo_views.BmoPlanView(
        feasible=True,
        total_qty_t_d=3836.2,
        total_cost_rs_thm=30780.2,
        ores=[
            bmo_views.BmoOreRow(ore_id="sinter_sp_02", ore_name="SINTER (SP-02)", share_pct=57.5, quantity_t_d=2205.8)
        ],
    )
    result = bmo_views.BmoRunResult(
        status="ok",
        plan=plan,
        anchor=bmo_views.BmoAnchor(basis="data_driven", rate_kg_thm=392.4),
        targets=bmo_views.BmoRunTargets(
            production_t_d=2250.0,
            fe_t_d=2126.25,
            hm_fe_pct=94.5,
            slag_rate_cap_kg_thm=300.0,
            slag_cap_t_d=675.0,
            model_to_plant_slag_factor=1.0,
            basicity_min=1.05,
            basicity_max=1.2,
            max_charges_per_hour=6.35,
            charge_mass_t=26.4,
            nut_coke_rate_kg_thm=70.0,
            nut_coke_charge_moisture_pct=6.5,
        ),
        fuel_rates=bmo_views.BmoFuelRates(pci_rate_kg_thm=165.0, pci_source="influx_1h_avg.coal_rate_actual_value"),
        coke_model="data_driven",
        chemistry_mode="latest",
        chemistry_window_days=30,
        computed_at=pd.Timestamp("2026-10-29T05:00:00Z").to_pydatetime(),
        seconds=9.4,
    )
    return result.model_copy(update=changes).model_dump(mode="json")


def test_a_run_is_queued_as_a_bmo_job_and_read_back_with_its_plan(client) -> None:
    body = {
        "target_production_mt": 2250,
        "coke_model": "data_driven",
        "ore_overrides": {"lump": {"price_rs_per_mt": 4500}},
    }
    r = call(client, "POST", "op", "/api/bmo/runs", json=body)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert job["kind"] == "bmo.run" and job["status"] == "queued" and job["submitted_by"] == "op"
    assert (
        job["params"]["target_production_mt"] == 2250
        and job["params"]["ore_overrides"]["lump"]["price_rs_per_mt"] == 4500
    )
    assert job["params"] == BmoRunParams.model_validate(body).model_dump(mode="json")
    again = call(client, "POST", "sup", "/api/bmo/runs", json=body).json()
    assert again["duplicate"] and again["job"]["id"] == job["id"]  # an identical run is not queued twice
    defaults = call(client, "POST", "op", "/api/bmo/runs")
    assert defaults.status_code == 202 and defaults.json()["job"]["params"] == BmoRunParams().model_dump()

    pending = call(client, "GET", "op", f"/api/bmo/runs/{job['id']}").json()
    assert pending["job"]["status"] == "queued" and pending["result"] is None
    jobs = client.app.state.live.store  # type: ignore[attr-defined]
    asyncio.run(jobs.finish(job["id"], "done", result=_result()))  # the in-memory queue has no loop
    done = call(client, "GET", "op", f"/api/bmo/runs/{job['id']}").json()
    assert done["result"]["status"] == "ok" and done["result"]["plan"]["ores"][0]["share_pct"] == 57.5
    assert done["result"]["anchor"] == {"basis": "data_driven", "rate_kg_thm": 392.4}

    missing = call(client, "GET", "op", "/api/bmo/runs/no-such-job")
    assert missing.status_code == 404 and missing.headers["content-type"].startswith("application/problem+json")
    other, _ = asyncio.run(jobs.submit("report.live", {"narrative": False}, "op"))
    assert call(client, "GET", "op", f"/api/bmo/runs/{other.id}").status_code == 404  # not a BMO run


@pytest.mark.parametrize(
    "body",
    [
        {"target_production_mt": 500},  # below 1000 t/day
        {"coke_model": "xgboost"},
        {"unknown": 1},
        {"ore_overrides": {"lump": {"price_rs_per_mt": -1}}},
        {"hm_chemistry": {"silicon_pct": 9}},
    ],
)
def test_a_run_request_out_of_range_is_refused(client, body) -> None:
    r = call(client, "POST", "op", "/api/bmo/runs", json=body)
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")


def test_the_run_kind_is_open_to_users_and_unscheduled() -> None:
    kind = CATALOG["bmo.run"]
    assert kind.role == "user" and kind.schedule is None and kind.params is BmoRunParams and kind.max_tries == 1


def test_the_worker_runs_the_case_and_returns_its_result(monkeypatch) -> None:
    from services.events import MemoryEventBus
    from worker.kinds import bmo as bmo_kind
    from worker.runtime import HANDLERS, JobContext

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(bmo_views, "run_case", lambda params: calls.append(params) or _result())
    bus = MemoryEventBus()
    job = JobContext(job_id="j1", kind=CATALOG["bmo.run"], attempt=1, store=JobStore(MemoryRedis()), bus=bus, pool=None)
    out = asyncio.run(HANDLERS["bmo.run"](job, BmoRunParams(target_production_mt=2250)))
    assert HANDLERS["bmo.run"] is bmo_kind.run and calls == [BmoRunParams(target_production_mt=2250).model_dump()]
    assert bmo_views.BmoRunResult.model_validate(out).status == "ok"
    assert [e.type for e in asyncio.run(bus.recent(5))] == ["job.progress"]


def test_the_params_reach_the_case_inputs() -> None:
    params = BmoRunParams(
        target_production_mt=2300,
        target_slag_t_basicity_min=0,
        coke_model="energy_balance",
        pci_override_kg_thm=150,
        force_data_driven=True,
        selected_ore_ids=["sinter_sp_02", "lump"],
        ore_overrides={"lump": {"max_share_pct": 20}},
        hm_chemistry={"silicon_pct": 0.45},
        fuel_prices={"pci": 13500},
    )
    inputs = bmo_views.inputs_from_params(params)
    assert inputs.target_production_mt == 2300 and inputs.target_slag_t_basicity_min == 0  # 0 = not enforced
    assert inputs.coke_model == "energy_balance" and inputs.pci_override_kg_thm == 150
    assert inputs.force_data_driven_prediction and inputs.nut_coke_override_kg_thm is None
    assert inputs.selected_ore_ids == ["sinter_sp_02", "lump"] and inputs.ore_overrides == {
        "lump": {"max_share_pct": 20}
    }
    assert inputs.hm_chemistry == {"silicon_pct": 0.45} and inputs.fuel_prices_rs_per_mt == {"pci": 13500}
    assert inputs.target_slag_basicity_min is None and inputs.ore_table is None  # page 9's defaults


# ---------------------------------------------------------------------------------------------- forecasts
def test_the_si_forecast_is_read_from_the_worker_ledger(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    storage = tmp_path / "si"
    origin = pd.Timestamp("2026-10-29T10:00:00+05:30").tz_convert("UTC")
    now = origin + pd.Timedelta(minutes=10)
    empty = bmo_views.si_forecast(cfg, now=now)
    assert empty.forecast is None and not empty.current and "No HM Si forecast" in empty.note

    _si_issue(storage, origin - pd.Timedelta(hours=1), 0.45)
    previous = bmo_views.si_forecast(cfg, now=now)
    assert previous.current and "being issued" in previous.note and previous.forecast.si_pct == 0.45
    _si_issue(storage, origin, 0.52)
    forecast_service.AttemptLog(storage).record(
        {"forecast": "si", "at": now - pd.Timedelta(minutes=2), "issuer": "worker", "status": "ok"}, None
    )
    view = bmo_views.si_forecast(cfg, now=now)
    assert view.current and view.note is None and view.worker_active and view.issuer == "auto"
    si = view.forecast
    assert si.recorded and si.si_pct == 0.52 and si.origin_at == origin.to_pydatetime()
    assert [h.label for h in si.horizons] == ["Now", "+1 h", "+2 h", "+3 h"] and si.horizons[
        2
    ].lower_pct == pytest.approx(0.42)

    late = bmo_views.si_forecast(cfg, now=origin + pd.Timedelta(hours=3))
    assert not late.current and late.forecast.si_pct is None and "since 29 Oct 10:00 IST" in late.note
    assert len(SiliconForecastLedger(storage).issues()) == 2  # reading issued nothing
    off = bmo_views.si_forecast({**cfg, "si_forecast": {**cfg["si_forecast"], "mode": "legacy"}}, now=now)
    assert off.mode == "legacy" and off.forecast is None and not off.current


def test_the_charged_coke_forecast_is_read_from_the_worker_ledger(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    storage = tmp_path / "cc"
    row = pd.Timestamp("2026-10-29 09:00")  # the row the 10:06 build holds complete
    assert bmo_views.charged_coke_forecast(cfg, now=pd.Timestamp("2026-10-29 10:30")).forecast is None
    _charged_issue(storage, row, 392.0)
    view = bmo_views.charged_coke_forecast(cfg, now=pd.Timestamp("2026-10-29 10:30"))
    assert view.current and view.note is None and not view.worker_active
    cc = view.forecast
    assert cc.prediction_kg_thm == 392.0 and cc.shown and cc.state == "cruise"
    assert cc.data_row == pd.Timestamp("2026-10-29 09:00", tz="Asia/Kolkata").to_pydatetime()
    assert cc.path[0].prediction_kg_thm == 393.0 and cc.coke_end == pd.Timestamp("2026-10-29 10:00", tz="Asia/Kolkata")
    pending = bmo_views.charged_coke_forecast(cfg, now=pd.Timestamp("2026-10-29 11:10"))
    assert pending.current and "being issued" in pending.note and pending.forecast.prediction_kg_thm == 392.0
    old = bmo_views.charged_coke_forecast(cfg, now=pd.Timestamp("2026-10-29 13:10"))
    assert not old.current and old.forecast is None and "since the 29 Oct 09:00 row" in old.reasons[0]
    assert old.last_issued_row == pd.Timestamp("2026-10-29 09:00", tz="Asia/Kolkata")


def test_the_forecast_routes_read_the_ledgers_now(client, tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, issuer="worker")
    monkeypatch.setattr(forecast_service, "bmo_config", lambda: cfg)
    now = pd.Timestamp.now(tz="UTC")
    _si_issue(tmp_path / "si", forecast_origin(now, cadence_minutes=60), 0.61)
    _charged_issue(tmp_path / "cc", forecast_service.expected_row(now), 401.5)
    si = call(client, "GET", "op", "/api/bmo/forecasts/si").json()
    assert si["current"] and si["issuer"] == "worker" and si["forecast"]["si_pct"] == 0.61
    assert si["forecast"]["recorded"] and len(si["forecast"]["horizons"]) == 4
    cc = call(client, "GET", "op", "/api/bmo/forecasts/charged-coke").json()
    assert cc["current"] and cc["forecast"]["prediction_kg_thm"] == 401.5 and cc["forecast"]["shown"]
    assert not list(tmp_path.glob("**/attempts/*.jsonl"))  # reading issues nothing and logs no attempt


# ---------------------------------------------------------------------------------------------- snapshots
def test_the_snapshots_of_page_9_are_listed_and_read(client) -> None:
    listing = call(client, "GET", "op", "/api/bmo/snapshots").json()
    ids = [s["id"] for s in listing["snapshots"]]
    assert ids == sorted((p.stem for p in SNAPSHOTS.glob("*.json")), reverse=True) and listing["unreadable"] == []
    assert (
        call(client, "GET", "op", "/api/bmo/snapshots", params={"limit": 2}).json()["snapshots"]
        == listing["snapshots"][:2]
    )
    first = listing["snapshots"][0]
    assert first["created_at"].endswith("+05:30") and first["summary"]["production_t_d"] == 2250.0

    detail = call(client, "GET", "op", f"/api/bmo/snapshots/{first['id']}").json()
    stored = json.loads((SNAPSHOTS / f"{first['id']}.json").read_text(encoding="utf-8"))
    lp = detail["lp"]
    assert detail["recommended"] == "Balanced Optimizer" and detail["run"]["pinned"] == stored["run"]["pinned"]
    assert lp["total_cost_rs_thm"] == pytest.approx(first["summary"]["total_cost_rs_thm"], rel=1e-9)
    assert lp["fuel_rate_kg_thm"] == pytest.approx(first["summary"]["fuel_rate_kg_thm"], rel=1e-9)
    assert sum(r["share_pct"] for r in lp["ores"]) == pytest.approx(100.0)
    assert {r["ore_name"]: round(r["share_pct"], 2) for r in lp["ores"] if r["share_pct"] > 0.05} == first["summary"][
        "blend_pct"
    ]
    ore_rows = detail["inputs"]["applied_ore_editor_df"]
    assert isinstance(ore_rows, list) and {"ore_id", "price_rs_per_mt"} <= set(ore_rows[0])
    assert detail["inputs"]["target_production_mt"] == 2250.0 and "frozen" not in detail
    assert len(json.dumps(detail)) < len(json.dumps(stored)) / 10  # without the frozen plant data

    for bad in ("no_such_snapshot", "..%2F..%2Fconfig%2Fsetting_bmo"):
        r = call(client, "GET", "op", f"/api/bmo/snapshots/{bad}")
        assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")


# ---------------------------------------------------------------------------------------------- roles, units
def test_every_bmo_route_needs_a_session(client) -> None:
    for method, url in (
        ("POST", "/api/bmo/runs"),
        ("GET", "/api/bmo/runs/x"),
        ("GET", "/api/bmo/forecasts/si"),
        ("GET", "/api/bmo/forecasts/charged-coke"),
        ("GET", "/api/bmo/snapshots"),
        ("GET", "/api/bmo/snapshots/x"),
    ):
        assert call(client, method, None, url).status_code == 401, url


def test_every_numeric_field_of_the_bmo_schemas_states_its_unit(client) -> None:
    schemas = client.get("/api/openapi.json").json()["components"]["schemas"]
    names = [n for n in schemas if n.startswith("Bmo")]
    assert {"BmoRunParams", "BmoRunResult", "BmoPlanView", "BmoSiForecastView", "BmoSnapshotDetail"} <= set(names)
    missing = []
    for name in names:
        for field, spec in schemas[name].get("properties", {}).items():
            kinds = {s.get("type") for s in spec.get("anyOf", [spec])}
            if kinds & {"number", "integer"} and not spec.get("description"):
                missing.append(f"{name}.{field}")
    assert not missing, missing
    paths = client.get("/api/openapi.json").json()["paths"]
    assert "202" in paths["/api/bmo/runs"]["post"]["responses"]
