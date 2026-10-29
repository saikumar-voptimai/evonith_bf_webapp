"""STRATA page service (SV-1.12, #292): the omega panel against the drift alarm, the model summary, the tier of the
current blend, and the live point (MOD facts at the current calibration, ring view of the thermal reserve zone).

The live-point tests replace the STRATA-Flow network by a smooth deterministic function of the network state (the
solves of ``model_facts`` would take minutes) and keep its Jacobian cache in a temporary folder; the slow test at the
end runs the network itself (skipped with ``BURDEN_ENGINE_FAST=1``)."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.burden_engine.calibration import CalibrationInfo, fallback_calibration
from services import drift, optimiser, strata
from services.events import MemoryEventBus
from services.jobs import CATALOG, JobRecord, StrataLivePointParams, input_hash
from utils.burden_engine import model_facts
from utils.burden_engine import strata_network as sn
from utils.burden_engine.facts import load_register
from utils.burden_engine.omega_model import OmegaModel
from worker.kinds import strata as strata_kind
from worker.runtime import JobContext

BASE_F = np.array([0.24, 0.17, 0.15, 0.14, 0.14, 0.16])
BASE_O = np.array([0.08, 0.15, 0.18, 0.19, 0.19, 0.21])
BASE_T = np.array([0.9, 1.1, 1.25, 1.35, 1.45, 1.6])
RING = np.linspace(-1.0, 1.0, 6)


def fake_solve(state):
    """A smooth stand-in for the network: ring quantities that move with coke, production, blend and moisture."""
    blend = state["blend_wet"]
    lump = sum(v for m, v in blend.items() if m not in ("sinter", "pellet"))
    pellet = blend.get("pellet", 0.0)
    dc = (state["coke_kg_thm"] - 330.0) / 100.0
    dp = (state["production_t_h"] - 95.0) / 100.0
    dn = (state["nut_coke_kg_thm"] - 60.0) / 100.0
    dh = (state["blast_moisture"] - 15.0) / 100.0
    f = BASE_F * (1 + 0.3 * dc * RING + 0.2 * dp * RING + 0.1 * dh * RING)
    o = BASE_O * (1 + 0.2 * lump * RING - 0.1 * pellet * RING + 0.1 * dn * RING)
    t = BASE_T * (1 - 0.5 * dp + 0.2 * dc + 0.1 * pellet)
    return [float(x) for x in (*f, *o, *t)], 0.001


@pytest.fixture()
def fake_network(monkeypatch, tmp_path):
    """The fake network, its Jacobians cached in ``tmp_path``, the process's network pool emptied before and after."""
    monkeypatch.setattr(sn, "solve_state", fake_solve)
    monkeypatch.setattr(sn.NetworkParts.__init__, "__defaults__", (tmp_path / "strata_flow", True))
    sn.clear_pool()
    yield
    sn.clear_pool()


def calibration() -> dict:
    """The shipped calibration moved off the reference: a 'current' calibration with a higher omega."""
    cal = copy.deepcopy(fallback_calibration())
    cal["omega"] = float(cal["omega"]) + 0.004
    return cal


# ---------------------------------------------------------------------------------------------------- live point
@pytest.fixture()
def live(fake_network):
    cal = calibration()
    out = strata.compute_live_point(json.dumps(cal))
    return cal, out


def test_mod_responses_equal_model_facts_at_the_same_calibration(live) -> None:
    cal, out = live
    sn.clear_pool()  # model_facts as a fresh process runs it, on the same calibration
    rows = {r["id"]: r for r in model_facts.model_facts(model_facts.reference_engine(cal))}
    got = {r["id"]: r for r in out["responses"]}
    assert set(got) == set(rows) == {f"MOD-{i:02d}" for i in range(1, 12)}
    for fid, row in rows.items():
        p = row["params"]
        r = got[fid]
        assert (r["statement"], r["unit"]) == (row["statement"], row["unit"])
        for key in ("fixed_omega", "strata_omega", "difference", "sigma_strata", "d_omega"):
            assert r[key] == pytest.approx(p[key], rel=1e-6, abs=1e-12), (fid, key)
    assert out["version"] == "STRATA 1.5" and out["reference_point"]["production_t_d"] == pytest.approx(2270.0)
    assert out["network"]["solves_run"] > 0 and out["network"]["network"] == sn.NET_VERSION


def test_the_ring_view_equals_the_network_solve_at_the_calibration_state(live) -> None:
    cal, out = live
    view = out["rings"]
    eng = model_facts.reference_engine(cal)
    m = eng.omega_model()  # the same calibration: the pool's parts, anchored at the calibration state
    net = sn.NetworkParts(m.net_ctx, cache_dir=None, jacobian=False)
    u = net._fill(net.vec(m.ref))
    f, o, t = sn.NetworkParts._split(np.asarray(sn.solve_state(net.network_state(u))[0], float))
    assert [r["ring"] for r in view["rings"]] == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose([r["gas_share"] for r in view["rings"]], f, rtol=1e-9)
    np.testing.assert_allclose([r["fe_share"] for r in view["rings"]], o, rtol=1e-9)
    np.testing.assert_allclose([r["residence_h"] for r in view["rings"]], t, rtol=1e-9)
    assert view["distance"] == 0.0 and view["state"] == "calibration"
    # kappa is refitted on the calibration: w of the ring model at its state is the calibrated omega
    assert view["omega"] == pytest.approx(cal["omega"], abs=1e-9)
    assert sum(r["omega_share"] for r in view["rings"]) == pytest.approx(view["omega"], rel=1e-12)
    assert view["capacity_ratio"] == pytest.approx(m.R_ref) and view["kappa"] == pytest.approx(m.kappa)
    for r in view["rings"]:
        assert r["gas_capacity"] == pytest.approx(m.R_ref * r["gas_share"], rel=1e-12)
        assert r["limited_by"] == ("gas" if r["gas_capacity"] < r["ore_oxygen"] else "ore")
        assert 0 < r["efficiency"] < 1
    assert view["inputs"]["production_t_h"] == pytest.approx(m.ref.prod)


def test_rings_and_w_of_the_omega_model_agree() -> None:
    """``OmegaModel.w`` is the sum of the ring parts of ``OmegaModel.rings`` (the refactor changed no number)."""
    m = OmegaModel.load(backend="surrogate")
    st = type("S", (), {})()
    st.fe_frac = {"sinter": 0.7, "pellet": 0.3}
    parts = (BASE_F / BASE_F.sum(), BASE_O / BASE_O.sum(), BASE_T)
    rg = m.rings(st, 0.7, parts=parts)
    assert m.w(st, 0.7, parts=parts) == rg["w"] == pytest.approx(float(np.sum(rg["share"])), rel=1e-12)
    k = m.k_blend(st.fe_frac)
    cg = 0.7 * parts[0]
    cmin = np.minimum(cg, parts[1])
    cr = cmin / np.maximum(np.maximum(cg, parts[1]), 1e-12)
    from utils.burden_engine.omega_model import counter_current_eff

    eff = counter_current_eff(m.kappa * k * parts[2] * parts[1] / cmin, cr)
    assert rg["w"] == float(np.sum(eff * cmin) / np.sum(cg))


class FakeJobs:
    """The parts of ``JobStore`` the handler uses."""

    def __init__(self, rec: JobRecord) -> None:
        self.rec = rec

    async def get(self, job_id: str) -> JobRecord:
        return self.rec

    async def update(self, job_id: str, **changes) -> None:
        return None


def job_context(kind: str) -> JobContext:
    now = datetime.now(UTC)
    rec = JobRecord(id="j-strata", kind=kind, status="running", input_hash=input_hash(kind, {}),
                    submitted_by="scheduler", submitted_at=now, scheduled_for=now)  # fmt: skip
    return JobContext(job_id=rec.id, kind=CATALOG[kind], attempt=1, store=FakeJobs(rec), bus=MemoryEventBus(),
                      pool=None)  # fmt: skip


def shipped_info() -> CalibrationInfo:
    return CalibrationInfo(fallback_calibration(), None, "live", ("2026-10-13", "2026-10-26"), 40, "")


def test_the_worker_job_saves_the_live_point_that_the_api_reads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(strata, "STORE_DIR", tmp_path)
    seen = {}

    def case_calibration(cfg, nonce, basis):
        seen["basis"] = basis
        return shipped_info(), {"basis": "day", "day": "2026-10-28", "made_at": "2026-10-27T21:01:00+00:00"}

    def compute(cal_json):
        seen["cal"] = json.loads(cal_json)
        return dict(
            version="STRATA 1.5",
            responses=[dict(id="MOD-01", statement="Pellet for sinter", unit="kg/tHM dry coke", fixed_omega=-1.8,
                            strata_omega=-6.1, difference=-4.3, sigma_strata=1.2, d_omega=0.016)],
            reference_point=dict(production_t_d=2270.0, blend_wet_pct={"sinter_sp_02": 64.0}, yard="ref"),
            rings=dict(omega=0.81, rings=[]),
            network=dict(solves_run=31, solve_seconds=34.0),
            seconds=40.0,
        )  # fmt: skip

    monkeypatch.setattr(optimiser, "case_calibration", case_calibration)
    monkeypatch.setattr(strata, "compute_live_point", compute)
    job = job_context("strata.live_point")
    out = asyncio.run(strata_kind.live_point(job, StrataLivePointParams()))
    assert seen["basis"] == "day" and seen["cal"]["omega"] == pytest.approx(fallback_calibration()["omega"])
    assert out["responses_strata"] == {"MOD-01": -6.1} and out["network"] == {"solves": 31, "seconds": 34.0}
    doc = strata.read_live_point()
    assert doc["job_id"] == "j-strata" and doc["calibration"]["basis"] == "day" and doc["calibration"]["n_shifts"] == 40
    view = strata.live_point_view(doc)
    mod01 = load_register()["MOD-01"]
    assert view["responses"][0]["reference_strata_omega"] == mod01.params["strata_omega"]
    assert view["responses"][0]["reference_sigma_strata"] == mod01.params["sigma_strata"]


def test_the_live_point_kind_is_scheduled_after_the_nightly_calibration() -> None:
    kind = CATALOG["strata.live_point"]
    assert kind.schedule == ({3}, {15}) and kind.role == "supervisor"
    nightly = CATALOG["calibration.nightly"]
    last_nightly_try_min = 2 * 60 + 30 + (nightly.max_tries - 1) * nightly.retry_delay_s / 60 + nightly.timeout_s / 60
    assert last_nightly_try_min <= 3 * 60 + 15  # the day's calibration and its Jacobian are made by then
    assert strata.live_point_view(None) == {
        "kind": "strata.live_point",
        "made_at": None,
        "responses": [],
        "rings": None,
    }


# --------------------------------------------------------------------------------------------------- omega panel
def snapshots(n_hours: int = 24 * 12, shift_after: int | None = None, seed: int = 3) -> list[dict]:
    """Hourly windows: omega_imp = omega_STRATA + an AR(1) residual (8-h lag structure), shifted up near the end."""
    rng = np.random.default_rng(seed)
    end0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    r = np.zeros(n_hours)
    for i in range(n_hours):
        r[i] = (0.6 * r[i - 8] if i >= 8 else 0.0) + rng.normal(0, 0.01)
    if shift_after is not None:
        r[shift_after:] += 0.03
    rows = []
    for i in range(n_hours):
        strata_omega = 0.80 + 0.005 * np.sin(i / 20)
        rows.append(
            dict(
                window_start=end0 + timedelta(hours=i - 8),
                window_end=end0 + timedelta(hours=i),
                evaluated_at=end0 + timedelta(hours=i, minutes=12),
                status="complete",
                omega_imp=strata_omega + r[i],
                omega_strata=strata_omega,
                omega_strata_sd=0.01,
                omega_cal=0.80,
                k_strata=300.0,
                k_strata_sd=3.0,
                flags={},
            )
        )
    return rows


def test_the_omega_panel_is_the_drift_alarms_evaluation() -> None:
    rows = snapshots(shift_after=24 * 10)
    cfg = drift.DriftConfig.from_config({})
    now = rows[-1]["window_end"] + timedelta(minutes=20)
    start = rows[-1]["window_end"] - timedelta(days=3)
    panel = strata.omega_panel(rows, cfg, start=start, now=now)
    report = drift.evaluate(rows, cfg, now)  # what the alarm acts on
    assert report.ar1 is not None
    assert panel["count"] == len(panel["rows"]) == 3 * 24 + 1
    assert all(r["window_end"] >= start for r in panel["rows"])
    sig, mu = report.ar1.sigma_e, report.ar1.mu
    for row in panel["rows"]:
        p = report.points.loc[pd.Timestamp(row["window_end"])]
        assert row["r"] == pytest.approx(row["omega_imp"] - row["omega_strata"], abs=1e-12)
        assert (row["r"], row["s_pos"], row["s_neg"]) == pytest.approx((p.r, p.s_pos, p.s_neg), rel=1e-12, abs=1e-15)
        assert row["band_high"] - row["omega_strata"] == pytest.approx(mu + 2 * sig, rel=1e-12)
        assert row["omega_strata"] - row["band_low"] == pytest.approx(2 * sig - mu, rel=1e-12)
    c = panel["cusum"]
    assert (c["sigma_e"], c["rho"], c["k"], c["h"]) == (sig, report.ar1.rho, 0.5, 5.0)
    assert c["state"] == "high" and c["episodes"][0]["direction"] == "high"  # the +3 sigma shift of the last days
    assert any(r["signal"] == "high" for r in panel["rows"])
    assert c["latest"]["s_pos"] == pytest.approx(report.latest()["s_pos"])


def test_windows_older_than_the_alarm_history_show_omega_without_the_cusum() -> None:
    rows = snapshots(shift_after=24 * 10)
    cfg = drift.DriftConfig.from_config({"twin": {"drift": {"history_days": 5}}})
    now = rows[-1]["window_end"]
    panel = strata.omega_panel(rows, cfg, start=rows[0]["window_end"], now=now)
    since = now - timedelta(days=5)
    report = drift.evaluate([r for r in rows if r["window_end"] >= since], cfg, now)  # the alarm's own rows
    assert panel["count"] == len(rows) and panel["cusum"]["sigma_e"] == report.ar1.sigma_e
    old = [r for r in panel["rows"] if r["window_end"] < since]
    assert old and all(r["s_pos"] is None and r["omega_strata"] is not None for r in old)
    assert panel["cusum"]["latest"]["s_pos"] == pytest.approx(report.latest()["s_pos"])


def test_without_enough_history_the_cusum_is_collecting() -> None:
    rows = snapshots(n_hours=48)
    panel = strata.omega_panel(rows, drift.DriftConfig.from_config({}), start=rows[0]["window_end"],
                               now=rows[-1]["window_end"])  # fmt: skip
    c = panel["cusum"]
    assert c["state"] == "collecting" and c["sigma_e"] is None and "need 72" in c["note"]
    assert all(r["band_low"] is None and r["s_pos"] is None for r in panel["rows"])
    assert panel["rows"][-1]["omega_strata"] is not None


def test_the_state_follows_the_open_episodes_of_the_alarm() -> None:
    rows = snapshots(seed=11)
    cfg = drift.DriftConfig.from_config({})
    now = rows[-1]["window_end"]
    panel = strata.omega_panel(rows, cfg, start=rows[-24]["window_end"], now=now)
    open_eps = drift.evaluate(rows, cfg, now).open_episodes()
    expected = "in_control" if not open_eps else ("both" if len(open_eps) == 2 else next(iter(open_eps)))
    assert panel["cusum"]["state"] == expected and len(panel["rows"]) == 24


# ------------------------------------------------------------------------------------------------- current blend
def test_the_current_blend_is_the_latest_window_with_strata() -> None:
    rows = snapshots(n_hours=5)
    rows[2]["flags"] = {"strata": {"certainty": "Investigate", "support": "outside", "investigate": ["pellet 27.0 %"]}}
    rows[3]["flags"] = {"strata": {"certainty": "Medium", "support": "margin", "outside": ["pellet 16.0 %"]}}
    cur = strata.current_blend(rows)
    assert cur["certainty"] == "Medium" and cur["window_end"] == rows[3]["window_end"]
    assert cur["outside"] == ["pellet 16.0 %"] and cur["k_strata_sd"] == 3.0
    assert strata.current_blend(snapshots(n_hours=3)) is None


# ------------------------------------------------------------------------------------------------- model summary
def test_the_model_summary_holds_the_validation_tiers_and_reference_responses() -> None:
    s = strata.model_summary()
    reg = load_register()
    assert s["version"] == "STRATA 1.5" and s["register_version"] == reg.version
    v = s["validation"]
    assert (v["expanding"]["rmse_strata"], v["expanding"]["rmse_fixed"]) == (5.99, 6.39)
    assert (v["strict"]["rmse_strata"], v["strict"]["rmse_fixed"]) == (8.0, 8.23)
    assert {p["period"] for p in v["strict"]["periods"]} >= {"All days"} and len(v["time_splits"]) == 2
    fams = s["families"]
    assert [f["id"] for f in fams] == [f.id for f in reg.facts(group="VAL", include_retired=False)]
    assert fams[0]["family"] == "N1 No pellet, sinter + CLO" and fams[0]["verdict"] == reg["VAL-01"].status
    ref = {r["id"]: r for r in s["reference"]}
    assert (ref["MOD-01"]["strata_omega"], ref["MOD-01"]["sigma_strata"]) == (-5.96, 1.19)
    t = s["tiers"]
    assert [lv["tier"] for lv in t["levels"]] == ["High", "Medium", "Low", "Investigate"]
    assert [lv["sigma_max_kg_thm"] for lv in t["levels"][:2]] == [2.0, 5.0]
    assert t["investigate_sigma_kg_thm"] == 8.0 and t["investigate_classes_max_pct"] == {"pellet": 25.0}
    assert "pellet > 25 %" in t["levels"][3]["rule"] and "8 kg/tHM" in t["levels"][3]["rule"]
    m = OmegaModel.load()
    for tier, sigma, support, inv in (("High", 2.0, "inside", []), ("Medium", 4.9, "margin", []),
                                      ("Low", 7.0, "outside", []), ("Investigate", 8.1, "inside", []),
                                      ("Investigate", 1.0, "inside", ["pellet 26 %"])):  # fmt: skip
        assert m.certainty(sigma, support, inv) == tier  # the rules the page states are the model's


# ------------------------------------------------------------------------------------------- slow: real network
@pytest.mark.skipif(os.environ.get("BURDEN_ENGINE_FAST") == "1", reason="network solves (BURDEN_ENGINE_FAST)")
def test_the_live_point_at_the_shipped_calibration_reproduces_the_register() -> None:
    """With the network itself at the shipped calibration: the register's MOD facts (generated by model_facts)."""
    sn.clear_pool()
    out = strata.compute_live_point(json.dumps(fallback_calibration()))
    reg = load_register()
    for r in out["responses"]:
        f = reg[r["id"]]
        assert r["fixed_omega"] == pytest.approx(f.params["fixed_omega"], abs=0.05)
        assert r["strata_omega"] == pytest.approx(f.params["strata_omega"], abs=0.05)
    assert out["rings"]["distance"] == 0.0
    assert out["rings"]["omega"] == pytest.approx(fallback_calibration()["omega"], abs=1e-9)
