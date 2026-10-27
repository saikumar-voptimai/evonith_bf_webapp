"""STRATA page API (SV-1.12, #292): ``/api/strata/omega``, ``/summary`` and ``/live-point`` on a SQLite twin store,
the units of every numeric field, and the live point written by the worker as the route serves it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import ApiSettings
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import store as store_service
from services import strata
from tests.test_api_auth import PASSWORDS, SECRET, build
from tests.test_api_live import session_cookie


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
def client(twin, tmp_path, monkeypatch):
    monkeypatch.setattr(strata, "STORE_DIR", tmp_path / "strata")
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def get(client: TestClient, url: str, user: str = "op", **kw):
    return client.get(url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


def fill(store: TwinStore, hours: int, *, flags_at: int | None = None) -> datetime:
    """Hourly windows up to the last full hour: omega_imp = omega_STRATA + an AR(1) residual."""
    rng = np.random.default_rng(5)
    last = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    r = np.zeros(hours)
    for i in range(hours):
        r[i] = (0.6 * r[i - 8] if i >= 8 else 0.0) + rng.normal(0, 0.01)
    for i in range(hours):
        end = last - timedelta(hours=hours - 1 - i)
        flags = (
            {"strata": {"certainty": "Low", "support": "outside", "outside": ["clo 41.0 %"]}} if i == flags_at else {}
        )
        store.snapshots.upsert(
            {
                "window_start": end - timedelta(hours=8),
                "window_end": end,
                "evaluated_at": end + timedelta(minutes=12),
                "status": "complete",
                "omega_imp": 0.80 + r[i],
                "omega_strata": 0.80,
                "omega_strata_sd": 0.01,
                "omega_cal": 0.80,
                "k_strata": 301.0,
                "k_strata_sd": 3.4,
                "flags": flags,
            }
        )
    return last


def test_omega_returns_the_windows_the_band_and_the_cusum(twin, client) -> None:
    last = fill(twin, 24 * 10)
    body = get(client, "/api/strata/omega?days=2").json()
    assert body["count"] == len(body["rows"]) == 2 * 24
    rows = body["rows"]
    assert rows[-1]["window_end"].startswith(last.strftime("%Y-%m-%dT%H:00"))
    c = body["cusum"]
    assert c["state"] in ("in_control", "high", "low", "both") and c["sigma_e"] > 0 and c["n_pairs"] >= 72
    assert c["h"] == 5.0 and c["k"] == 0.5 and c["history_days"] == 56 and c["band_sigmas"] == 2.0
    for row in rows:
        assert row["band_high"] - row["omega_strata"] == pytest.approx(c["mu"] + 2 * c["sigma_e"], rel=1e-9)
        assert row["r"] == pytest.approx(row["omega_imp"] - row["omega_strata"], abs=1e-12)
    assert get(client, "/api/strata/omega?days=60").status_code == 422


def test_summary_states_the_validation_tiers_and_the_current_blend(twin, client) -> None:
    fill(twin, 12, flags_at=9)
    body = get(client, "/api/strata/summary").json()
    assert body["version"] == "STRATA 1.5"
    v = body["validation"]
    assert (v["expanding"]["rmse_strata"], v["expanding"]["rmse_fixed"]) == (5.99, 6.39)
    assert (v["strict"]["rmse_strata"], v["strict"]["rmse_fixed"]) == (8.0, 8.23)
    assert len(body["families"]) == 12 and body["reference"][0]["id"] == "MOD-01"
    assert body["tiers"]["levels"][3]["tier"] == "Investigate"
    cur = body["current"]
    assert cur["certainty"] == "Low" and cur["outside"] == ["clo 41.0 %"] and body["current_note"] is None

    store_service.use(None)  # without a twin store: the model part, and why there is no current tier
    alone = get(client, "/api/strata/summary").json()
    assert alone["current"] is None and "BF_TWIN_DB_URL" in alone["current_note"]
    assert get(client, "/api/strata/omega").status_code == 503


def test_live_point_is_empty_before_the_job_and_then_the_saved_document(client) -> None:
    empty = get(client, "/api/strata/live-point").json()
    assert empty["made_at"] is None and empty["responses"] == [] and empty["kind"] == "strata.live_point"
    ring = dict(ring=1, name="centre", gas_share=0.24, fe_share=0.08, residence_h=0.9, gas_capacity=0.17,
                ore_oxygen=0.08, ntu=2.1, efficiency=0.7, omega_share=0.08, limited_by="ore")  # fmt: skip
    doc = strata.live_point_document(
        dict(
            version="STRATA 1.5",
            responses=[
                dict(
                    id="MOD-01",
                    statement="Pellet for sinter",
                    unit="kg/tHM dry coke",
                    fixed_omega=-1.8,
                    strata_omega=-6.1,
                    difference=-4.3,
                    sigma_strata=1.2,
                    d_omega=0.016,
                )
            ],
            reference_point=dict(production_t_d=2270.0, blend_wet_pct={"sinter_sp_02": 64.0}, yard="reference"),
            rings=dict(
                state="calibration",
                distance=0.0,
                omega=0.81,
                capacity_ratio=0.7,
                k_blend=0.87,
                kappa=3.9,
                gas_in_excess=0.25,
                inputs=dict(blend_wet_pct={"sinter": 64.0}, classes_wet_pct={"sinter": 64.0}),
                rings=[ring],
            ),
            network=dict(solves_run=31),
            seconds=40.0,
        ),  # fmt: skip
        dict(
            source="live", window=["2026-10-13", "2026-10-26"], n_shifts=40, omega=0.81, basis="day", day="2026-10-28"
        ),
        made_at=datetime(2026, 10, 27, 21, 47, tzinfo=UTC),
        job_id="j1",
    )
    strata.save_live_point(doc)
    body = get(client, "/api/strata/live-point").json()
    assert body["made_at"].startswith("2026-10-27T21:47") and body["calibration"]["basis"] == "day"
    r = body["responses"][0]
    assert r["strata_omega"] == -6.1 and r["reference_strata_omega"] == -5.96 and r["reference_sigma_strata"] == 1.19
    assert body["rings"]["rings"][0]["limited_by"] == "ore"


def test_every_numeric_field_of_the_strata_schemas_states_its_unit() -> None:
    schema = create_app(ApiSettings(static_dir=None, jwt_secret=SECRET)).openapi()
    used = {"StrataOmega", "StrataSummary", "LivePoint"}
    names: set[str] = set()

    def walk(name: str) -> None:
        if name in names:
            return
        names.add(name)
        for prop in schema["components"]["schemas"][name].get("properties", {}).values():
            for ref in [prop.get("$ref")] + [x.get("$ref") for x in prop.get("anyOf", []) + [prop.get("items") or {}]]:
                if ref:
                    walk(ref.rsplit("/", 1)[1])

    for name in used:
        walk(name)
    missing = []
    for name in names:
        for field, prop in schema["components"]["schemas"][name].get("properties", {}).items():
            types = {prop.get("type")} | {x.get("type") for x in prop.get("anyOf", [])}
            if types & {"number", "integer"} and not prop.get("description"):
                missing.append(f"{name}.{field}")
    assert not missing, missing
