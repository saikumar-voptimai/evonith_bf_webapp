"""Burnt against charged coke, admin route (SV-4.4, #307): administrators only, 503 before the first build, the
values of the stored analysis equal ``transit`` on the same shift table for both bases, units in every numeric
field."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services import burnt_vs_charged as bvc
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_api_twin import UNITS, _numeric
from tests.test_services_burnt_vs_charged import BUILT, CFG, close, result_of, shift_table
from utils.coke_energy import transit as tr

URL = "/api/admin/burnt-vs-charged"
UNITS_HERE = (*UNITS, "d", "shifts", "kg/h", "g/Nm³", "percentage")


@pytest.fixture(scope="module")
def table_and_analysis():
    T, S = shift_table(days=90)
    result = result_of(T, S)
    return bvc.shift_table(result), bvc.build(result, CFG, now=BUILT)


@pytest.fixture()
def stored(table_and_analysis, tmp_path, monkeypatch):
    path = tmp_path / bvc.STORE_NAME
    monkeypatch.setattr(bvc, "STORE_PATH", path)
    bvc.save(table_and_analysis[1], path)
    return table_and_analysis


@pytest.fixture()
def client():
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def get(client: TestClient, user: str | None, url: str = URL, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.get(url, headers=headers, **kw)


@pytest.mark.parametrize("actor,status", [(None, 401), ("op", 403), ("sup", 403), ("adm", 200)])
def test_administrators_only(client, stored, actor, status) -> None:
    assert get(client, actor).status_code == status


def test_503_before_the_first_build(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bvc, "STORE_PATH", tmp_path / "missing.json")
    r = get(client, "adm")
    assert r.status_code == 503 and r.headers["content-type"].startswith("application/problem+json")
    assert "twin.trace" in r.json()["detail"]


def test_a_bad_basis_is_refused(client, stored) -> None:
    assert get(client, "adm", params={"basis": "unstable"}).status_code == 422


@pytest.mark.parametrize("basis", ["all", "stable"])
def test_values_equal_transit_on_the_same_data(client, stored, basis) -> None:
    full, _ = stored
    T = full if basis == "all" else full[full["stable"]]
    body = get(client, "adm", params={"basis": basis}).json()
    assert body["basis"] == basis and body["shifts"] == len(T) and body["stale"] is True  # built in October 2026
    table = tr.window_agreement(T, bvc.WINDOWS).to_dict("records")
    assert [w["window"] for w in body["windows"]] == list(bvc.WINDOWS)
    for g, e in zip(body["windows"], table, strict=True):
        assert g["n"] == e["n"]
        for key in ("ccc", "r", "cb", "bias", "sd", "loa_lo", "loa_hi", "slope"):
            assert close(g[key], e[key]), (g["window"], key)
    d = tr.decompose_window_sd(tr.window_agreement(T, bvc.WINDOWS))
    assert all(close(body["split"][k], d[k]) for k in ("transit_sd", "random_sd", "floor_sd"))
    c = tr.closure(T)
    assert body["closure"]["n"] == c["n"] and close(body["closure"]["ratio"], c["ratio"])
    assert close(body["closure"]["burnt_t"], c["burnt_t"]) and close(body["closure"]["charged_t"], c["charged_t"])
    scan = tr.lag_scan(T, bvc.LAGS, detrend_shifts=bvc.DETREND_SHIFTS)
    assert all(close(g["r"], e) for g, e in zip(body["lag"]["rows"], scan["r"], strict=True))
    assert [p["window"] for p in body["points"]] == list(bvc.WINDOWS)
    assert body["points"][1]["shifts"] == 3 and len(body["points"][1]["points"]) == len(tr.window_pairs(T, 3))
    leak = body["leak"]
    assert leak["state"] in {"none", "gap_only", "h2_only"} and leak["settings"]["baseline_days"] == 14
    assert [s["key"] for s in leak["steps"]] == ["gap", "h2", "raft", "moisture"]
    assert all(not math.isnan(s["n_recent"]) for s in leak["steps"])
    assert pd.Timestamp(body["built_at"]) == BUILT


@pytest.mark.parametrize(
    "schema",
    [
        "BurntVsCharged",
        "BvcSplitWindow",
        "BvcSplit",
        "BvcLagRow",
        "BvcLag",
        "BvcClosureMonth",
        "BvcClosurePoint",
        "BvcClosure",
        "LeakModel",
        "LeakShift",
        "LeakSettings",
        "BvcWindowPoints",
    ],
)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = []
    for name, prop in spec.get("properties", {}).items():
        if not _numeric(prop):
            continue
        unit = (prop.get("description") or "").rsplit(",", 1)[-1].strip()
        if not (prop.get("description") and "," in prop["description"] and unit.split(" ")[0] in UNITS_HERE):
            missing.append(f"{name}: {prop.get('description')!r}")
    assert not missing, missing


def test_the_step_values_name_their_unit_field() -> None:
    props = build().openapi()["components"]["schemas"]["LeakStep"]["properties"]
    for name in ("recent", "baseline", "step", "se"):
        assert props[name]["description"].endswith("in the unit field's unit"), name
    assert props["z"]["description"].endswith(", dimensionless") and props["n_recent"]["description"].endswith("count")
