"""Blend-dependent shaft efficiency (#262): omega model, engine blend mode, certainty filter, risk aversion, auto mode.

Reference: the 16 Oct 2026 UAT yard and the shipped Jul-Oct 2026 calibration. Fixed mode must keep reproducing the
case report (305.7 kg/tHM, Rs 30,139/tHM at 2,270 t/d); blend mode is checked for consistency and direction.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from utils.burden_engine.auto_review import (CANDIDATES, parse_review, pick, review_context, ungrounded_numbers)
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH, Engine, Limits, Settings
from utils.burden_engine.omega_model import assumption_sensitivity, difference_sigma
from utils.coke_energy.rist import solve_2z

P_REF = 2270 / 24


@pytest.fixture(scope="module")
def eng():
    return Engine()


@pytest.fixture(scope="module")
def inputs():
    with open(REFERENCE_INPUTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------- model
def test_model_anchors_on_the_calibration(eng):
    m = eng.omega_model()
    assert m.kappa > 0 and 0.6 < m.R_ref < 0.8
    assert m.w_ref == pytest.approx(eng.cal["omega"], abs=1e-6)
    po = m.plan(m.ref)
    assert po.omega(m.ref.coke, m.R_ref) == pytest.approx(eng.cal["omega"], abs=1e-9)
    assert po.sigma(m.ref.coke, m.R_ref) == pytest.approx(0.0, abs=1e-9)
    assert eng._dKdw < -200        # loop-inclusive dK/domega, kg/tHM per unit omega


def test_physics_directions(eng):
    """Pellet and sinter reduce faster than lump ore; more production and more gas lower omega."""
    m = eng.omega_model()
    r = m.ref
    def state(**wet):
        sh, fe, cls = m.blend_parts(wet)
        return replace(r, shares=sh, fe_frac=fe, cls=cls)
    base = state(sinter=63.6, pellet=10.6, lloyds_clo=13.6, geomin=8.8, nmdc_rom=2.2, donimalai=1.2)
    more_pel = state(sinter=63.6, pellet=20.6, lloyds_clo=3.6, geomin=8.8, nmdc_rom=2.2, donimalai=1.2)
    w0, w1 = m.w(base, m.R_ref), m.w(more_pel, m.R_ref)
    assert w1 > w0 + 0.01                                       # pellet for CLO: about +0.02 per 10 pt
    assert m.w(replace(base, prod=base.prod * 1.05), m.R_ref) < w0
    assert m.w(base, m.R_ref * 1.03) < w0


def test_solve_2z_omega_fn_constant_equals_fixed(eng):
    c = eng._cal_case(); cal = eng.cal; P = eng.case0["prod_th"]
    args = (c, eng.p, cal["omega"], cal["Q_LZ_GJh"] * 1000 / P, cal["Q_UZ_GJh"] * 1000 / P, cal["r_H"])
    a = solve_2z(*args, T_R=cal["T_R"], x0=eng._x0)
    b = solve_2z(*args, T_R=cal["T_R"], x0=eng._x0, omega_fn=lambda K, z: cal["omega"])
    assert b["K"] == pytest.approx(a["K"], abs=1e-6)
    assert b["omega"] == pytest.approx(cal["omega"]) and 0.6 < b["R_cap"] < 0.8 and b["R_cap_C"] < b["R_cap"]


def test_gas_capacity_channel_retired(eng):
    """GAS-17 retired (STRATA 1.1+): the capacity ratio the omega model sees is the calibration's for every plan, while
    the balance's own R still moves with the gas (here: blast moisture)."""
    m = eng.omega_model()
    assert m.gas_capacity is False
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    dry = eng.evaluate(z, replace(s, humidity_g=10.0), P_REF, full=True)
    wet = eng.evaluate(z, replace(s, humidity_g=20.0), P_REF, full=True)
    assert wet["omega_terms"]["R"] == pytest.approx(m.R_ref) == pytest.approx(dry["omega_terms"]["R"])
    assert wet["omega_terms"]["d_R"] == 0.0 and dry["omega_terms"]["d_R"] == 0.0
    assert wet["R_cap"] > dry["R_cap"]


def test_moisture_term(eng):
    """GAS-19: omega rises with the blast moisture of the weather record by a_h per g/Nm3, so the coke rate responds
    to moisture much less than with omega held (the balance alone: about +0.8 kg/tHM per g/Nm3)."""
    m = eng.omega_model()
    assert 0.002 < m.a_moist < 0.007 and m.a_moist_se > 0
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    dry = eng.evaluate(z, replace(s, humidity_g=10.0), P_REF, full=True)
    wet = eng.evaluate(z, replace(s, humidity_g=20.0), P_REF, full=True)
    assert wet["omega_terms"]["d_moist"] - dry["omega_terms"]["d_moist"] == pytest.approx(10 * m.a_moist, rel=1e-9)
    assert wet["omega_terms"]["sigma_parts"]["moisture"] > 0
    held_dry = eng.evaluate(z, replace(Settings(), humidity_g=10.0), P_REF)
    held_wet = eng.evaluate(z, replace(Settings(), humidity_g=20.0), P_REF)
    d_held, d_blend = held_wet["K"] - held_dry["K"], wet["K"] - dry["K"]
    assert d_held > 5.0                                   # balance: moisture is endothermic
    assert abs(d_blend) < 0.6 * d_held


def test_asset_strata12():
    from utils.burden_engine.omega_model import MODEL_PATH
    with open(MODEL_PATH, encoding="utf-8") as fh:
        spec = json.load(fh)
    assert spec["version"].startswith("STRATA 1.2") and spec["gas_capacity_channel"] is False
    assert spec["brand"]["name"] == "STRATA"
    ri = {m: v["ri_pct"] for m, v in spec["materials"].items()}
    assert ri == {"sinter": 68.7, "pellet": 65.0, "lloyds_clo": 65.0, "geomin": 75.0, "nmdc_rom": 65.0,
                  "donimalai": 60.0, "lloyds_oversize": 65.0, "gomti": 75.0}                      # GAS-21, set S8
    assert all(v.get("ri_basis") for v in spec["materials"].values())
    ops = spec["operating_scales"]
    assert 0.3 < ops["production"] < 1.0 < ops["other"] < 2.0 and ops["production_sd"] > 0
    inv = spec["certainty"]["investigate"]
    assert inv["classes_max"] == {"pellet": 25.0} and inv["sigma_kg"] == 8.0
    po, cs = spec["posterior"], spec["channel_scales"]
    assert po["names"][0] == "x_blend" and po["mean"][0] == 1.0
    g = cs["g_per_pt"]
    i = po["names"].index("sh_pellet")
    assert po["mean"][i] == pytest.approx((cs["lam_pellet"] - 1) * g["sh_pellet"], rel=1e-9)
    for c in ("sh_clo", "sh_rom", "sh_gomti", "sh_oversize"):
        assert po["mean"][po["names"].index(c)] == pytest.approx((cs["lam_lump"] - 1) * g[c], rel=1e-9)
    v = spec["validation"]
    assert v["expanding"]["coke_day"] > 0.15 and v["expanding"]["omega_day"] > 0.2
    assert all(x["coke_day"] > 0.1 for x in v["time_split"].values())
    fams = {f["family"]: f for f in v["by_blend"]}
    assert fams["All days"]["verdict"] == "Supported" and fams["All days"]["n"] == 175
    assert all(f["verdict"] == "Modelled only" for f in v["by_blend"] if f["n"] < 10)


def test_production_split(eng):
    """STRATA 1.2: the operating channel is a production part (TRZ residence) and the rest, scaled apart; the parts
    add up to the operating channel; production alone moves only the production part."""
    m = eng.omega_model()
    assert m.split_oper and m.lam_prod < 1.0 < m.lam_rest
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    a = eng.evaluate(z, s, 90.0, full=True)["omega_terms"]
    b = eng.evaluate(z, s, 95.0, full=True)["omega_terms"]
    for t in (a, b):
        assert t["d_prod"] + t["d_rest"] == pytest.approx(t["d_oper"], abs=1e-12)
        assert t["d_prod"] == pytest.approx(m.lam_prod * t["d_prod_phys"], rel=1e-9)
    assert b["d_prod_phys"] < a["d_prod_phys"]                               # faster: shorter TRZ residence
    slope = (b["d_prod_phys"] - a["d_prod_phys"]) / 5.0
    assert -0.004 < slope < -0.001                                           # GAS-16 physics about -0.0023 per t/h


def test_investigate_tier(eng, inputs):
    """GAS-18: pellet above 25 % of ferrous is Investigate whatever sigma; "low" removes Investigate plans."""
    m = eng.omega_model()
    assert m.certainty(0.5, "inside", ["pellet 30 % > 25 %"]) == "Investigate"
    assert m.certainty(9.0, "inside") == "Investigate"
    assert m.certainty(1.0, "inside") == "High" and m.certainty(6.0, "outside") == "Low"
    assert m.investigate_reasons({"pellet": 26.0}) and not m.investigate_reasons({"pellet": 24.0})
    lim = m.limits_for("low")
    assert lim["classes"]["pellet"] == (0.0, 25.0) and lim["sigma_max"] == 8.0
    r30 = _pinned(inputs, "blend", 30)
    assert r30["omega"]["certainty"] == "Investigate" and r30["omega"]["investigate"]
    assert any(w.startswith("Investigate tier") for w in r30["warnings"])
    mats = [x["id"] for x in inputs["materials"] if x.get("selected_in_uat", True)]
    bounds = {x["id"]: (x["lo"], x["hi"]) for x in inputs["materials"]}
    bounds["lloyds_pellet"] = (10.0, 40.0)
    e = Engine(inputs, materials=mats, bounds=bounds)
    r = e._try(Settings(omega_mode="blend", certainty="low"), P_REF, Limits(), True)
    assert r["share_wet"]["lloyds_pellet"] <= 25.0 + 0.1 and r["omega"]["certainty"] != "Investigate"


def test_critical_assumptions(eng):
    """Each critical assumption re-anchors on the calibration: zero effect at the calibration state, non-zero for a
    plan away from it; RI rows for every material in the plan; scale rows symmetric."""
    m = eng.omega_model()
    rows0 = assumption_sensitivity(m, m.ref, m.ref.coke, m.R_ref)
    assert all(abs(r["d_omega"]) < 1e-9 for r in rows0 if r["kind"] in ("RI", "size"))
    s = Settings(omega_mode="blend")
    r = eng.min_cost(prod_t_day=2270, s=s, L=Limits())
    rows = eng.critical_assumptions(r, s)
    names = {x["assumption"] for x in rows}
    assert {"RI pellet", "RI sinter", "size exponent of the reduction rate", "plant-data scale, pellet channel"} <= names
    assert all(x["d_coke_kg"] >= 0 for x in rows if x.get("symmetric"))
    assert abs(rows[0]["d_coke_kg"]) >= abs(rows[-1]["d_coke_kg"])
    assert eng.critical_assumptions(eng.min_cost(prod_t_day=2270, s=Settings(), L=Limits()), Settings()) == []


# ---------------------------------------------------------------------------------------------- engine
def test_fixed_mode_unchanged(eng):
    r = eng.min_cost(prod_t_day=2270, s=Settings(), L=Limits())
    assert r["out"]["K"] == pytest.approx(305.7, abs=0.15)
    assert r["cost"]["total"] == pytest.approx(30139, abs=5)
    assert r["out"]["sig_K"] == 0.0 and r["omega"]["mode"] == "fixed" and r["omega"]["certainty"] == "Held"
    assert r["cost"]["sigma"] == 0.0 and r["cost"]["risk_adjusted"] == pytest.approx(r["cost"]["total"])


def test_blend_mode_result(eng):
    r = eng.min_cost(prod_t_day=2270, s=Settings(omega_mode="blend"), L=Limits())
    om = r["omega"]
    assert r["feasible"] and om["mode"] == "blend"
    assert 0.70 < om["value"] < 0.90 and om["sigma_K"] > 0
    assert om["certainty"] in ("High", "Medium", "Low", "Investigate")
    parts = om["d_blend"] + om["d_disc"] + om["d_oper"] + om["d_R"] + om["d_moist"]
    assert om["value"] == pytest.approx(om["omega_cal"] + parts, abs=1e-9)
    assert r["cost"]["sigma"] == pytest.approx(om["sigma_K"] * eng.prices_fuel["coke"] / 1000)


def _pinned(inputs, mode, pel):
    mats = [m["id"] for m in inputs["materials"] if m.get("selected_in_uat", True)]
    bounds = {m["id"]: (m["lo"], m["hi"]) for m in inputs["materials"]}
    bounds["lloyds_pellet"] = (pel - 0.02, pel + 0.02)
    e = Engine(inputs, materials=mats, bounds=bounds)
    return e._try(Settings(omega_mode=mode), P_REF, Limits(), True)


def test_pellet_lowers_coke_in_blend_mode_only(inputs):
    """Pellet 10 -> 20 % with the blend re-optimised (pellet mostly displaces Lloyds CLO): about flat coke with omega
    held; several kg/tHM less with STRATA-omega (about -5 kg/tHM, GAS-15, MOD-08)."""
    f10, f20 = _pinned(inputs, "fixed", 10), _pinned(inputs, "fixed", 20)
    b10, b20 = _pinned(inputs, "blend", 10), _pinned(inputs, "blend", 20)
    d_fixed = f20["out"]["K"] - f10["out"]["K"]
    d_blend = b20["out"]["K"] - b10["out"]["K"]
    assert abs(d_fixed) < 1.0
    assert d_blend < d_fixed - 3.0
    assert b20["omega"]["sigma_K"] > b10["omega"]["sigma_K"]            # further from the calibration blend


def test_production_lowers_omega(eng):
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    lo = eng.evaluate(z, s, 88.0, full=True)
    hi = eng.evaluate(z, s, 96.0, full=True)
    assert hi["omega"] < lo["omega"] - 0.005


def test_certainty_filter_high(inputs):
    mats = [m["id"] for m in inputs["materials"] if m.get("selected_in_uat", True)]
    bounds = {m["id"]: (m["lo"], m["hi"]) for m in inputs["materials"]}
    bounds["lloyds_pellet"] = (10.0, 30.0)
    e = Engine(inputs, materials=mats, bounds=bounds)
    s = Settings(omega_mode="blend", certainty="high")
    r = e._try(s, P_REF, Limits(), True)
    assert r is not None
    if r["feasible"]:
        assert r["omega"]["certainty"] == "High"
        assert r["share_wet"]["lloyds_pellet"] <= 15.4 + 0.1
        assert r["omega"]["sigma_K"] <= 2.0 + 0.05
    any_ = e._try(replace(s, certainty="any"), P_REF, Limits(), True)
    assert any_["cost"]["total"] <= r["cost"]["total"] + 1.0              # the filter can only cost money


def test_risk_aversion_reduces_sigma(inputs):
    mats = [m["id"] for m in inputs["materials"] if m.get("selected_in_uat", True)]
    bounds = {m["id"]: (m["lo"], m["hi"]) for m in inputs["materials"]}
    bounds["lloyds_pellet"] = (10.0, 30.0)
    e = Engine(inputs, materials=mats, bounds=bounds)
    a = e._try(Settings(omega_mode="blend"), P_REF, Limits(), True)
    b = e._try(Settings(omega_mode="blend", risk_k=1.64), P_REF, Limits(), True)
    assert b["cost"]["sigma"] <= a["cost"]["sigma"] + 1.0
    assert b["cost"]["total"] >= a["cost"]["total"] - 1.0
    assert b["cost"]["risk_adjusted"] <= a["cost"]["total"] + 1.64 * a["cost"]["sigma"] + 1.0


def test_saving_sigma_and_difference(eng):
    s = Settings(omega_mode="blend")
    r = eng.min_cost(prod_t_day=2270, s=s, L=Limits())
    same = eng.saving_sigma(r, r)
    assert same["sigma_coke"] == pytest.approx(r["omega"]["terms"]["sigma_parts"]["surrogate"] * math.sqrt(2)
                                               * abs(eng._dKdw) / (1 - eng.case0["coke"]["moist_pct"] / 100), rel=1e-6)
    m = eng.omega_model()
    t = r["omega"]["terms"]
    assert difference_sigma(m, t, t) == pytest.approx(math.sqrt(2) * t["sigma_parts"]["surrogate"])
    assert eng.saving_sigma(r, eng.min_cost(prod_t_day=2270, s=Settings(), L=Limits())) is None


def test_settings_validation(eng):
    with pytest.raises(ValueError):
        eng.min_cost(prod_t_day=2270, s=Settings(omega_mode="magic"), L=Limits())
    with pytest.raises(ValueError):
        eng.min_cost(prod_t_day=2270, s=Settings(omega_mode="blend", certainty="certain"), L=Limits())


# ---------------------------------------------------------------------------------------------- auto mode
def _row(cid, cost, sig, cert="High", feasible=True, saving=None, sig_s=None):
    r = dict(id=cid, label=cid, available=True, feasible=feasible, cost=cost, sigma_cost=sig, cost_plus_sigma=cost + sig,
             coke_dry=300.0, sigma_coke=sig / 32.0, omega=0.8, certainty=cert, support="inside", outside=[],
             shares={"sinter": 60.0}, P_tday=2270.0)
    if saving is not None:
        r.update(saving_vs_today=saving, sigma_saving=sig_s)
    return r


def test_rule_pick():
    rows = [_row("A", 30000, 150, "Low"), _row("B", 30050, 40, "Medium"), _row("C", 30080, 20, "High")]
    assert pick(rows)[0] == "B"                            # cost + 1 sigma: A 30,150, B 30,090, C 30,100
    rows = [_row("A", 30000, 150, "Low"), _row("B", 30050, 60, "Medium"), _row("C", 30080, 20, "High")]
    assert pick(rows)[0] == "C"                            # A 30,150, B 30,110, C 30,100
    rows = [_row("A", 30000, 10, saving=40, sig_s=60), _row("T", 30040, 0, "High")]
    assert pick(rows)[0] == "T"                            # saving inside one sigma: hold today's blend
    rows = [_row("A", 30000, 10, feasible=False), _row("T", 30040, 0)]
    assert pick(rows)[0] == "T"
    assert [c[0] for c in CANDIDATES] == ["A", "B", "C"]
    assert all(c[2]["certainty"] in ("low", "high") for c in CANDIDATES)          # never "any": no Investigate plan


def test_rule_never_picks_investigate():
    rows = [_row("A", 29000, 50, "Investigate"), _row("B", 30050, 40, "Medium")]
    assert pick(rows)[0] == "B"


def test_rule_pick_ties_to_certainty():
    rows = [_row("A", 30000, 50, "Low"), _row("C", 30000, 50, "High")]
    assert pick(rows)[0] == "C"


def test_parse_review_and_grounding():
    ctx = review_context([_row("A", 30000, 150), _row("B", 30050, 40)], "B", "rule", names={}, mode_label="test")
    text = '```json\n{"choice": "B", "agrees_with_rule": true, "reasons": ["B costs 30,050 +/- 40, 50 more than A"], ' \
           '"risks": ["coke 300.0 could be 1.25 higher"], "monitoring": ["check eta CO after 2 shifts"], ' \
           '"summary": "Take B; it saves 777 Rs/tHM."}\n```'
    rev, problems = parse_review(text, ["A", "B"])
    assert rev["choice"] == "B" and not problems
    bad = ungrounded_numbers(rev, ctx)
    assert "777" in bad and "30050" not in bad and "50" not in bad
    rev2, problems2 = parse_review('{"choice": "Z", "reasons": [], "risks": [], "monitoring": [], "summary": ""}', ["A"])
    assert problems2 and "not one of the candidates" in problems2[0]
    assert parse_review("no json here", ["A"])[0] is None


def test_run_candidates(eng):
    from utils.burden_engine.scenarios import run_candidates
    out = run_candidates(eng, 2270.0, Settings(), Limits())
    ids = [r["id"] for r in out["rows"]]
    assert ids == ["A", "B", "C"] and out["pick"] in ids
    a = next(r for r in out["rows"] if r["id"] == "A")
    assert all(r["cost"] >= a["cost"] - 1.0 for r in out["rows"] if r.get("available") and r.get("feasible"))
