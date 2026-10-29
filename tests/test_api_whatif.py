"""What-if API of the admin space (SV-4.3, #306): administrators only, cases saved per user with their audit rows,
runs of up to four cases through the worker functions (called in-process), the deck's what-if and the case report's
plan through the API, and the register version stated by every result. SQLite twin store, fake job queue."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from api.live import job_store
from services import day_calibration
from services import optimiser as optimiser_svc
from services import whatif as svc
from services.events import MemoryEventBus
from services.jobs import CATALOG
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_api_optimiser import CASE_CAL, CASE_SETTINGS, FakeJobs
from tests.test_api_optimiser import twin as twin  # noqa: F401 - the SQLite twin store fixture
from tests.test_services_whatif import CASE2_BLEND
from worker.kinds import whatif as whatif_kind
from worker.runtime import JobContext

URL = "/api/admin/whatif"


@pytest.fixture()
def api(twin, monkeypatch, tmp_path):  # noqa: F811
    """(client, cookies, jobs): the shipped calibration is the case report's (15 g/Nm³), the yard's prices the UAT
    ones, and no day calibration is stored."""
    cfg = dict(optimiser_svc.read_page_config(0))
    cfg.pop("prices", None)
    cfg["calibration"] = dict(cfg.get("calibration") or {}, fallback_path=CASE_CAL)
    monkeypatch.setattr(optimiser_svc, "page_config", lambda mtime: cfg)
    monkeypatch.setattr(day_calibration, "STORE_DIR", tmp_path / "calibration")
    jobs = FakeJobs()
    app = build()
    app.dependency_overrides[job_store] = lambda: jobs
    with TestClient(app, base_url="https://testserver") as client:
        cookies = {u: session_cookie(app, u) for u in PASSWORDS}
        yield client, cookies, jobs


def call(api, method: str, user: str | None, url: str, **kw: Any):
    client, cookies, _ = api
    headers = {"Cookie": cookies[user]} if user else {}
    return client.request(method, url, headers=headers, **kw)


def work(api, job_id: str) -> dict[str, Any]:
    """The queued job's worker function, in this process (as the worker runs it)."""
    _, _, jobs = api
    record = jobs.records[job_id]
    kind = CATALOG[record.kind]
    job = JobContext(job_id=job_id, kind=kind, attempt=1, store=jobs, bus=MemoryEventBus(), pool=None)
    handler = {"whatif.forge": whatif_kind.forge, "whatif.strata": whatif_kind.strata}[record.kind]
    return asyncio.run(handler(job, kind.params.model_validate(record.params)))


def levers(case: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in case["forge"].items() if v is not None}


def save(api, user: str, body: dict[str, Any]) -> dict[str, Any]:
    r = call(api, "POST", user, f"{URL}/cases", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_every_what_if_route_requires_the_admin_role() -> None:
    routes = sorted(
        (r.path, *sorted(r.methods)) for r in build().routes if isinstance(r, APIRoute) and r.path.startswith(URL)
    )
    assert routes == [
        (f"{URL}/basis", "GET"),
        (f"{URL}/cases", "GET"),
        (f"{URL}/cases", "POST"),
        (f"{URL}/cases/{{case_id}}", "DELETE"),
        (f"{URL}/cases/{{case_id}}", "PUT"),
        (f"{URL}/runs", "POST"),
    ]


@pytest.mark.parametrize("user,status", [(None, 401), ("op", 403), ("sup", 403)])
def test_non_administrators_are_refused(api, user, status) -> None:
    for method, url, body in [
        ("GET", f"{URL}/basis", None),
        ("GET", f"{URL}/cases", None),
        ("POST", f"{URL}/cases", {"kind": "forge", "name": "x", "forge": {}}),
        ("POST", f"{URL}/runs", {"kind": "forge", "case_ids": ["00000000-0000-0000-0000-000000000000"]}),
    ]:
        r = call(api, method, user, url, json=body)
        assert r.status_code == status, (method, url, r.text)
        assert r.headers["content-type"].startswith("application/problem+json")


def test_the_basis_lists_days_the_calibration_point_and_materials(api) -> None:
    r = call(api, "GET", "adm", f"{URL}/basis")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["days"] == [] and body["basis"] is None and "No day calibration" in body["message"]
    assert body["facts_version"] == "1.8" and body["max_cases"] == 4
    assert {"sinter_sp_02", "lloyds_pellet", "gomti_clo"} <= {m["id"] for m in body["materials"]}
    ref = call(api, "GET", "adm", f"{URL}/basis", params={"calibration": "reference"}).json()
    assert ref["basis"]["calibration"] == "reference" and ref["basis"]["omega"] == pytest.approx(0.80, abs=0.001)
    assert ref["base"]["two_zone"]["coke_kg_thm"] == pytest.approx(305.5, abs=0.05)
    assert ref["base"]["inputs"]["hbt_c"] == pytest.approx(1188.4, abs=0.1)


def test_cases_are_saved_changed_and_deleted_per_user(api) -> None:
    a = save(api, "adm", {"kind": "forge", "name": "PCI +10", "forge": {"pci_kg_thm": 196.8}})
    assert levers(a) == {"pci_kg_thm": 196.8} and a["strata"] is None and a["forge_run"] is None
    r = call(api, "PUT", "adm", f"{URL}/cases/{a['id']}", json={"kind": "forge", "name": "PCI +12",
                                                              "forge": {"pci_kg_thm": 198.8}})  # fmt: skip
    assert r.status_code == 200 and r.json()["name"] == "PCI +12" and levers(r.json()) == {"pci_kg_thm": 198.8}
    assert [c["name"] for c in call(api, "GET", "adm", f"{URL}/cases").json()] == ["PCI +12"]
    assert call(api, "GET", "adm", f"{URL}/cases", params={"kind": "strata"}).json() == []
    audit = call(api, "GET", "adm", "/api/admin/audit", params={"object_type": "whatif_case"}).json()
    assert [e["action"] for e in audit["entries"]] == ["whatif.save", "whatif.save"]
    assert {c["path"] for c in audit["entries"][0]["changes"]} >= {"name", "inputs.levers.pci_kg_thm"}
    assert call(api, "DELETE", "adm", f"{URL}/cases/{a['id']}").status_code == 204
    assert call(api, "DELETE", "adm", f"{URL}/cases/{a['id']}").status_code == 404
    assert call(api, "GET", "adm", f"{URL}/cases").json() == []


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"kind": "forge", "name": "x", "forge": {"hmt_c": 1700}}, "less than or equal to 1600"),
        ({"kind": "forge", "name": "x", "strata": {"blend": None}}, "forge inputs only"),
        ({"kind": "strata", "name": "x", "strata": {"blend": {"sinter_sp_02": 120}}}, "(0, 100]"),
        ({"kind": "strata", "name": "x", "strata": {"blend": {"sinter_sp_02": 50}}}, "not 95-105 %"),
        ({"kind": "forge", "name": "", "forge": {}}, "at least 1 character"),
    ],
)
def test_invalid_cases_are_refused(api, body, fragment) -> None:
    r = call(api, "POST", "adm", f"{URL}/cases", json=body)
    assert r.status_code == 422 and fragment in r.text, r.text


def test_a_forge_run_reproduces_the_deck_and_states_the_register_version(api) -> None:
    pci = save(api, "adm", {"kind": "forge", "name": "PCI +10", "forge": {"pci_kg_thm": 196.8}})
    hbt = save(api, "adm", {"kind": "forge", "name": "HBT +50", "forge": {"hbt_c": 1238.4}})
    r = call(api, "POST", "adm", f"{URL}/runs", json={"kind": "forge", "case_ids": [pci["id"], hbt["id"]],
                                                      "calibration": "reference"})  # fmt: skip
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert job["kind"] == "whatif.forge" and [c["name"] for c in job["params"]["cases"]] == ["PCI +10", "HBT +50"]
    out = work(api, job["id"])
    assert out["stored"] == [pci["id"], hbt["id"]]
    cases = {c["name"]: c for c in call(api, "GET", "adm", f"{URL}/cases").json()}
    for name, two_zone, fixed_eta in (("PCI +10", -8.7, -9.9), ("HBT +50", -6.7, -4.1)):
        run = cases[name]["forge_run"]
        assert run["facts_version"] == "1.8" and run["job_id"] == job["id"]
        assert run["basis"]["calibration"] == "reference" and run["basis"]["blast_moisture_g_nm3"] == 15.0
        assert run["result"]["two_zone"]["d_coke_kg_thm"] == pytest.approx(two_zone, abs=0.06)
        assert run["result"]["fixed_eta"]["d_coke_kg_thm"] == pytest.approx(fixed_eta, abs=0.06)


def test_a_strata_run_reproduces_the_case_report(api) -> None:
    cheapest = save(api, "adm", {"kind": "strata", "name": "Cheapest", "strata": {"settings": CASE_SETTINGS}})
    strata = {"blend": CASE2_BLEND, "settings": CASE_SETTINGS}
    forced = save(api, "adm", {"kind": "strata", "name": "Case 2 blend", "strata": strata})
    body = {"kind": "strata", "case_ids": [cheapest["id"], forced["id"]], "calibration": "reference",
            "omega_mode": "fixed", "yard": "reference"}  # fmt: skip
    r = call(api, "POST", "adm", f"{URL}/runs", json=body)
    assert r.status_code == 202, r.text
    out = work(api, r.json()["job"]["id"])
    assert (
        out["facts_version"] == "1.8" and out["basis"]["omega_mode"] == "fixed" and out["basis"]["yard"] == "reference"
    )
    runs = [c["strata_run"] for c in call(api, "GET", "adm", f"{URL}/cases").json()]
    by_name = {r["result"]["name"]: r["result"] for r in runs}
    for name in ("Cheapest", "Case 2 blend"):
        res = by_name[name]
        assert res["status"] == "ok" and res["facts_version"] == "1.8"
        assert res["production_t_d"] == pytest.approx(2270.0)
        assert res["coke_kg_thm"] == pytest.approx(305.7, abs=0.3) and res["cost_rs_thm"] == pytest.approx(30139, abs=5)
    assert by_name["Case 2 blend"]["forced"] and not by_name["Cheapest"]["forced"]


def test_a_run_names_only_the_users_cases(api) -> None:
    mine = save(api, "adm", {"kind": "forge", "name": "mine", "forge": {}})
    r = call(api, "POST", "adm", f"{URL}/runs", json={"kind": "strata", "case_ids": [mine["id"]]})
    assert r.status_code == 422 and "not strata cases" in r.json()["detail"]
    r = call(api, "POST", "adm", f"{URL}/runs", json={"kind": "forge", "case_ids": [mine["id"]] * 2})
    assert r.status_code == 422 and "named twice" in r.json()["detail"]
    r = call(api, "POST", "adm", f"{URL}/runs", json={"kind": "forge", "case_ids": ["0" * 32]})
    assert r.status_code == 404
    r = call(api, "POST", "adm", f"{URL}/runs", json={"kind": "forge", "case_ids": [mine["id"]] * 5})
    assert r.status_code == 422


def test_a_day_run_without_the_days_calibration_fails_with_the_reason(api) -> None:
    case = save(api, "adm", {"kind": "forge", "name": "x", "forge": {}})
    r = call(api, "POST", "adm", f"{URL}/runs", json={"kind": "forge", "case_ids": [case["id"]], "day": "2026-10-20"})
    assert r.status_code == 202
    with pytest.raises(svc.WhatIfInputError, match="No calibration is stored for 2026-10-20"):
        work(api, r.json()["job"]["id"])
