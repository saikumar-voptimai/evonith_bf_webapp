"""STRATA-Learn (shadow mode): evidence store, random-walk Kalman filter, gate, drift alarm, register proposals,
constrained exploration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from utils.burden_engine.facts import REGISTER_PATH
from utils.burden_engine.learn.autofill import apply, propose
from utils.burden_engine.learn.evidence import REGRESSORS, EvidenceStore, blend_family
from utils.burden_engine.learn.explore import suggest, thompson
from utils.burden_engine.learn.gate import cusum, dm_test, gate
from utils.burden_engine.learn.kalman import RandomWalkKalman
from utils.burden_engine.learn.run import shadow_run
from utils.burden_engine.omega_model import MODEL_PATH


@pytest.fixture(scope="module")
def spec():
    with open(MODEL_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def reg():
    with open(REGISTER_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_evidence_store():
    st = EvidenceStore()
    assert len(st.df) == 191 and st.df["date"].is_monotonic_increasing
    assert set(st.df["family"]) >= {"P1 Pellet, sinter + CLO", "N1 No pellet, sinter + CLO"}
    row = st.df.iloc[-1].to_dict()
    row["dobs"] = 0.5
    assert st.append([row]) == 0 and st.df["dobs"].iloc[-1] == 0.5           # same date replaced, not added
    new = dict(row, date=pd.Timestamp("2026-10-08"))
    assert st.append([new]) == 1 and len(st.df) == 192
    with pytest.raises(ValueError):
        st.append([{"date": "2026-10-09", "q_pel": 0.0}])
    assert blend_family({"sh_pellet": 30.0, "sh_rom": 0.0}) == "P1 Pellet, sinter + CLO"


def test_kalman_replay_matches_the_out_of_sample_test(spec, reg):
    """From the priors over the record, the one-step predictions reach the skill of the expanding-window test and
    the final state agrees with the deployed posterior."""
    res = shadow_run(EvidenceStore(), spec, reg)
    sk = res["skill"]["challenger (one-step, out of sample)"]
    assert sk["coke_day"] > 0.05 and sk["omega_day"] > 0.05
    m, cs = res["state"]["mean"], spec["channel_scales"]
    for i, p in enumerate(("lam_pellet", "lam_lump", "lam_production", "lam_other")):
        assert m[p] == pytest.approx(cs["mean"][i], abs=0.15)
    assert res["gate"]["status"] == "collecting"
    fams = {r["family"]: r for r in res["families"]}
    assert fams["All days"]["n"] == 175 and fams["All days"]["verdict"] == "Supported"


def _synthetic(theta, n, rho, sig, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, [0.004, 0.005, 0.004, 0.005, 3.5, 5.0], (n, 6))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, sig)
    dates = np.datetime64("2026-11-01") + np.arange(n).astype("timedelta64[D]")
    return dates, X, X @ theta + e


def test_kalman_tracks_a_changed_scale(spec):
    cs = spec["channel_scales"]
    true = np.array(cs["mean"], float)
    true[0] = 0.6                                                            # pellet scale lower than deployed
    d, X, y = _synthetic(true, 400, cs["rho"], cs["sigma_e"])
    kf = RandomWalkKalman.from_asset(spec)
    kf.run(d, X, y)
    s = kf.summary()
    assert abs(s["mean"]["lam_pellet"] - 0.6) < 2.5 * s["sd"]["lam_pellet"]
    assert s["sd"]["lam_pellet"] < cs["cov"][0][0] ** 0.5                     # uncertainty shrinks with evidence


def test_gate_and_drift(spec):
    rng = np.random.default_rng(3)
    y = rng.normal(0, 0.02, 80)
    good, bad = y + rng.normal(0, 0.005, 80), y + rng.normal(0, 0.02, 80)
    assert gate(y[:30], bad[:30], good[:30])["status"] == "collecting"
    g = gate(y, bad, good)
    assert g["promote"] and g["gain"] > 0.5 and g["dm_t"] > 2
    assert not gate(y, bad, good, drift_open=True)["promote"]
    assert not gate(y, good, bad)["promote"]
    assert dm_test(bad - y, good - y)[1] > 2
    sig = spec["channel_scales"]["sigma_e"]
    e = np.zeros(300)
    for t in range(1, 300):
        e[t] = 0.76 * e[t - 1] + rng.normal(0, sig)
    assert not cusum(e, 0.76, sig)["ever_alarm"]
    shifted = e.copy(); shifted[150:] += 0.03                                # a 0.03 step in omega (8 kg/tHM)
    c = cusum(shifted, 0.76, sig)
    assert c["ever_alarm"] and c["first_alarm"] >= 149


def test_register_proposal_rules(spec, reg):
    res = shadow_run(EvidenceStore(), spec, reg)
    prop = res["proposal"]
    assert all(it["automatic"] for it in prop["items"] if it["id"] in ("GAS-14", "GAS-19", "GAS-23"))
    fam = [dict(r) for r in res["families"]]
    for r in fam:
        if r["family"].startswith("N1"):
            r["verdict"] = "Disputed"                                       # a status change
    p2 = propose(reg, res["state"], fam)
    it = next(i for i in p2["items"] if i["id"] == "VAL-01")
    assert it["kind"] == "status" and not it["automatic"]
    r_auto = apply(reg, p2, version="1.7x", today="2026-12-01")
    old_status = next(f for f in reg["facts"] if f["id"] == "VAL-01")["status"]
    assert old_status != "Disputed"
    assert next(f for f in r_auto["facts"] if f["id"] == "VAL-01")["status"] == old_status   # status change needs approval
    r_ok = apply(reg, p2, approved={"VAL-01"}, version="1.8", today="2026-12-01")
    assert next(f for f in r_ok["facts"] if f["id"] == "VAL-01")["status"] == "Disputed"
    assert r_ok["version"] == "1.8" and r_ok["revision_log"][0]["change"].startswith("STRATA-Learn")
    assert reg["version"] == "1.7"                                           # input untouched


def test_exploration():
    C0 = np.array([30163.0, 30170.0, 30217.0, 30298.0])
    H = np.array([[0.0, 0.0], [-20.0, 5.0], [-60.0, 10.0], [-90.0, 12.0]])
    t = thompson(C0, H, [1.27, 0.47], [[0.19, -0.02], [-0.02, 0.12]], n=5000)
    assert t["p_opt"].sum() == pytest.approx(1.0) and (t["regret"] >= 0).all()
    s = suggest(["7.5 %", "10 %", "20 %", "30 %"], C0, H, [1.27, 0.47], [[0.19, -0.02], [-0.02, 0.12]],
                ["High", "High", "Medium", "Investigate"], regret_budget=15.0, n=5000)
    assert s["choice"] == "7.5 %" and s["explore"] in ("7.5 %", "10 %")
    assert not next(r for r in s["rows"] if r["plan"] == "30 %")["eligible"]


def test_regressors_order():
    assert REGRESSORS == ("q_pel", "q_lump", "q_prod", "q_oper_rest", "d_humidity_g", "d_nut")
