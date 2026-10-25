"""Blend-dependent shaft efficiency (#262): omega model, engine blend mode, certainty filter, risk aversion, auto mode.

Reference: the 16 Oct 2026 UAT yard and the shipped Jul-Oct 2026 calibration. Fixed mode must keep reproducing the
case report (305.7 kg/tHM, Rs 30,139/tHM at 2,270 t/d); blend mode is checked for consistency and direction.
STRATA 1.3 runs the STRATA-Flow network (about 1.1 s per solve; the Jacobian at the calibration is cached in
src/storage/strata_flow after the first run, about 20 s).
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


def test_moisture_acts_through_the_balance(eng):
    """GAS-19 retired (STRATA 1.5, Open-Meteo daily moisture in the balance): no moisture data term on omega; the coke
    rate responds to moisture through the balance (about +0.8 kg/tHM per g/Nm3) and the network's small response in
    the other-operation channel only."""
    m = eng.omega_model()
    assert m.a_moist == 0.0 and m.a_moist_se == 0.0
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    dry = eng.evaluate(z, replace(s, humidity_g=10.0), P_REF, full=True)
    wet = eng.evaluate(z, replace(s, humidity_g=20.0), P_REF, full=True)
    assert wet["omega_terms"]["d_moist"] == 0.0 and dry["omega_terms"]["d_moist"] == 0.0
    assert wet["omega_terms"]["sigma_parts"].get("moisture", 0.0) == 0.0
    held_dry = eng.evaluate(z, replace(Settings(), humidity_g=10.0), P_REF)
    held_wet = eng.evaluate(z, replace(Settings(), humidity_g=20.0), P_REF)
    d_held, d_blend = held_wet["K"] - held_dry["K"], wet["K"] - dry["K"]
    assert d_held > 5.0                                   # balance: moisture is endothermic
    assert abs(d_blend - d_held) < 0.3 * d_held           # network response to moisture is small (MOD-10)


def test_asset_strata15():
    from utils.burden_engine.omega_model import MODEL_PATH
    with open(MODEL_PATH, encoding="utf-8") as fh:
        spec = json.load(fh)
    assert spec["version"].startswith("STRATA 1.5") and spec["gas_capacity_channel"] is False
    assert spec["brand"]["name"] == "STRATA" and spec["parts_backend"] == "network"
    mats = spec["materials"]
    ri = {m: v["ri_pct"] for m, v in mats.items()}
    assert ri["sinter"] == 66.0 and ri["gomti"] == 65.0              # sinter: latest strength test; Gomti: no test
    assert 80 < ri["pellet"] < 85 and 55 < ri["nmdc_rom"] < 65 and 75 < ri["geomin"] < 82   # laboratory tests (T26)
    kin = spec["kinetics"]
    assert kin["ri_exponent"] == 0.0 and kin["n_size"] == 1.0                               # link calibrated (GAS-21)
    assert all(v.get("ri_basis") and v.get("ri_sd", 0) > 0 for v in mats.values())
    assert all(v.get("ri_tests") for m, v in mats.items() if m not in ("sinter", "gomti"))
    assert mats["lloyds_oversize"]["sauter_mm"] == mats["lloyds_clo"]["sauter_mm"]          # no size credit
    assert mats["lloyds_oversize"]["sauter_sieve_mm"] < 13.0
    assert "donimalai" in spec["plant_map"]["ore_11"]                                       # two oversize types
    ctx = spec["network"]["context"]
    assert {"matrix", "top_eta_co", "top_h2", "top_gas_T_c", "hbt_c"} <= set(ctx)
    assert spec["nut_coke"]["coef"] > 0 and spec["nut_coke"]["se"] > 0
    ops = spec["operating_scales"]
    assert ops["production_channel"] is False and ops["production"] == 0.0 and ops["production_sd"] == 0.0  # STRATA 1.4
    assert 0.3 < ops["other"] < 2.0 and ops["other_sd"] > 0
    cs0 = spec["channel_scales"]
    for name in ("q_prod", "d_humidity_g"):                     # STRATA 1.4: no production channel; 1.5: no moisture term
        i = cs0["names"].index(name)
        assert cs0["mean"][i] == 0.0 and all(abs(x) == 0.0 for x in cs0["cov"][i])
    assert cs0["fixed"] == {"q_prod": 0.0, "d_humidity_g": 0.0}
    mo = spec["moisture"]
    assert mo["coef"] == 0.0 and mo["se"] == 0.0 and mo["source"] == "open_meteo"   # moisture acts through the balance
    from utils.coke_energy.weather import _settings
    assert _settings()["source"] == mo["source"]                 # pipeline moisture = the basis of the fit
    val = spec["validation"]
    assert val["expanding"]["rmse_coke"] < val["expanding"]["rmse_fixed"] and val["expanding"]["coke_day"] > 0.05
    strict = val["strict_2023_24"]
    assert strict["rmse"]["All days"]["STRATA 1.5"] < strict["rmse"]["All days"]["Fixed-omega"]
    assert strict["skill_day"]["STRATA 1.5"] > 0.0
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
    assert v["expanding"]["coke_day"] > 0.05 and v["expanding"]["omega_day"] > 0.05
    assert all(x["coke_day"] > 0.0 for x in v["time_split"].values())
    fams = {f["family"]: f for f in v["by_blend"]}
    assert fams["All days"]["verdict"] == "Supported" and fams["All days"]["n"] == 175
    assert all(f["verdict"] == "Modelled only" for f in v["by_blend"] if f["n"] < 10)


def test_production_split(eng):
    """STRATA 1.2+: the operating channel is a production part (TRZ residence) and the rest, scaled apart; the parts
    add up to the operating channel. STRATA 1.4: the production part has scale 0 (no production channel); its physics
    is still solved and reported, and production alone leaves the scaled operating channel unchanged."""
    m = eng.omega_model()
    assert m.split_oper and m.lam_prod == 0.0 and m.sd_prod == 0.0 and not m.prod_channel
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    a = eng.evaluate(z, s, 90.0, full=True)["omega_terms"]
    b = eng.evaluate(z, s, 95.0, full=True)["omega_terms"]
    for t in (a, b):
        assert t["d_prod"] + t["d_rest"] == pytest.approx(t["d_oper"], abs=1e-12)
        assert t["d_prod"] == pytest.approx(m.lam_prod * t["d_prod_phys"], rel=1e-9)
        assert t["d_prod"] == 0.0
    assert b["d_prod_phys"] < a["d_prod_phys"]                               # faster: shorter TRZ residence (physics)
    slope = (b["d_prod_phys"] - a["d_prod_phys"]) / 5.0
    assert -0.004 < slope < -0.001                                           # network about -0.0023 per t/h (GAS-16)


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
    plan away from it; with the calibrated link (beta = 0) the RI enters through the beta row, not per material;
    scale rows symmetric."""
    m = eng.omega_model()
    rows0 = assumption_sensitivity(m, m.ref, m.ref.coke, m.R_ref)
    assert all(abs(r["d_omega"]) < 1e-9 for r in rows0 if r["kind"] in ("RI", "size"))
    s = Settings(omega_mode="blend")
    r = eng.min_cost(prod_t_day=2270, s=s, L=Limits())
    rows = eng.critical_assumptions(r, s)
    names = {x["assumption"] for x in rows}
    assert {"RI exponent of the reduction rate (laboratory RI weight)", "size exponent of the reduction rate",
            "plant-data scale, pellet channel"} <= names
    assert "plant-data scale, production channel" not in names                    # STRATA 1.4: no production channel
    if m.spec["kinetics"].get("ri_exponent", 1.0) == 0.0:
        assert not any(n.startswith("RI ") and "exponent" not in n for n in names)
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
    parts = om["d_blend"] + om["d_disc"] + om["d_oper"] + om["d_R"] + om["d_moist"] + om["d_nut"]
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


def test_production_has_no_residence_penalty(eng):
    """STRATA 1.4 (GAS-16 retired): production alone does not lower omega through the TRZ residence; the other-
    operation channel (blast, coke layers per tonne) moves it by less than 0.002 over 88 -> 96 t/h."""
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    lo = eng.evaluate(z, s, 88.0, full=True)
    hi = eng.evaluate(z, s, 96.0, full=True)
    assert abs(hi["omega"] - lo["omega"]) < 0.002
    assert hi["omega_terms"]["d_prod_phys"] < lo["omega_terms"]["d_prod_phys"] - 0.01      # physics still solved


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
    assert same["sigma_coke"] == pytest.approx(0.0, abs=1e-9)          # network: no independent surrogate error
    m = eng.omega_model()
    t = r["omega"]["terms"]
    assert t["sigma_parts"]["surrogate"] == 0.0
    assert difference_sigma(m, t, t) == pytest.approx(0.0, abs=1e-12)
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


# ---------------------------------------------------------------------------------------------- STRATA 1.3: network
def test_network_backend_at_calibration(eng):
    """The ring quantities at the calibration are a network solve (not a surrogate); the Jacobian is the network's."""
    m = eng.omega_model()
    assert m.backend == "network" and m.net is not None and m.net.J.shape == (18, 16)
    from utils.strata_flow.omega_eval import evaluate
    f, o, t = m.parts(m.ref)
    r = evaluate(m.net.network_state(m.net._fill(m.net.vec(m.ref))))
    assert f == pytest.approx(r["f"], abs=1e-9) and t == pytest.approx(r["t"], abs=1e-9)
    assert m.net.distance(m.ref) == 0.0


def test_reported_plan_is_solved_by_the_network(eng):
    """Every SLP iterate is solved at its blend (one solve per iterate); a reported plan at its blend, production and
    full-operation states, so its omega comes from network solves at the plan itself."""
    m = eng.omega_model()
    n0 = m.net.n_solves
    r = eng.min_cost(prod_t_day=2200, s=Settings(omega_mode="blend"), L=Limits())
    net = r["omega"]["terms"]["network"]
    assert net["backend"] == "network" and m.net.n_solves - n0 >= 2
    assert net["distance"]["blend"] == 0.0 and net["distance"]["production"] == 0.0
    assert net["distance"]["operation"] < 0.01             # the balance's coke moved < 0.05 kg/tHM after the last solve
    assert r["feasible"]


def test_surrogate_selectable_but_not_default():
    from utils.burden_engine.omega_model import OmegaModel
    assert OmegaModel.load().backend == "network"
    assert OmegaModel.load(backend="surrogate").backend == "surrogate"


def test_nut_coke_term(eng):
    """GAS-23: omega moves by a_n per kg/tHM of nut coke against the calibration; sigma carries its se."""
    m = eng.omega_model()
    s = Settings(omega_mode="blend")
    z = eng.min_cost(prod_t_day=2270, s=s, L=Limits())["z"]
    lo = eng.evaluate(z, replace(s, nut_plant=60.0), P_REF, full=True)["omega_terms"]
    hi = eng.evaluate(z, replace(s, nut_plant=80.0), P_REF, full=True)["omega_terms"]
    assert hi["d_nut"] - lo["d_nut"] == pytest.approx(20 * m.a_nut, rel=1e-9) and m.a_nut > 0
    assert hi["sigma_parts"]["nut_coke"] > 0


def test_fuel_defaults_from_regime(eng):
    """PCI and nut coke default to the regime window's medians (as injected / charged), else the calibration."""
    from utils.burden_engine.scenarios import fuel_defaults, settings_from_regime
    cal = eng.cal
    d = fuel_defaults({"pci_kgthm_dry": 170.0, "nut_kgthm_dry": 60.0}, {}, cal)
    assert d["pci_kg_thm"] == pytest.approx(170.0 / (1 - cal["case"]["pci"]["moist_pct"] / 100))
    assert d["nut_kg_thm_source"] == "regime median"
    d0 = fuel_defaults(None, {}, cal)
    assert d0["pci_kg_thm_source"] == "calibration"
    assert d0["pci_kg_thm"] == pytest.approx(cal["case"]["pci_kg"] / (1 - cal["case"]["pci"]["moist_pct"] / 100))
    s = settings_from_regime({"pci_kgthm_dry": 170.0, "nut_kgthm_dry": 60.0}, {}, cal)
    assert s.pci_plant == pytest.approx(d["pci_kg_thm"]) and s.nut_plant == pytest.approx(d["nut_kg_thm"])


def test_search_tolerance_and_refine(eng):
    """Production searches solve the network coarsely (reuse within SEARCH_TOL scaled units) and re-solve the plan they
    return exactly; the tolerance is restored after the search."""
    assert eng.net_tol_iter == 0.0 and eng.net_tol_final == 0.0
    with eng.network_tolerance(Engine.SEARCH_TOL):
        assert eng.net_tol_iter == eng.net_tol_final == Engine.SEARCH_TOL
    assert eng.net_tol_iter == 0.0 and eng.net_tol_final == 0.0
    s = Settings(omega_mode="blend")
    with eng.network_tolerance(Engine.SEARCH_TOL):
        coarse = eng.min_cost(prod_t_day=2250, s=s, L=Limits())
    exact = eng.refine(coarse, Limits())
    assert exact["omega"]["terms"]["network"]["distance"]["blend"] == 0.0
    assert abs(exact["out"]["K"] - coarse["out"]["K"]) < 1.0


def test_slag_credit(inputs):
    """ASM-20: the slag sold is credited in the objective and the total; with a slag price the optimiser does not take
    less slag than without it (more saleable slag is preferred at nearly the same cost)."""
    base = Engine(inputs)
    r0 = base.min_cost(prod_t_day=2270, s=Settings(), L=Limits())
    assert r0["cost"]["slag_credit"] == 0.0 and r0["cost"]["total"] == pytest.approx(r0["cost"]["gross"])
    inp = dict(inputs, fuel=dict(inputs["fuel"], slag_price=1500.0))
    e = Engine(inp)
    r1 = e.min_cost(prod_t_day=2270, s=Settings(), L=Limits())
    c = r1["cost"]
    assert c["slag_credit"] == pytest.approx(r1["out"]["slag"] * 1.5, rel=1e-9)
    assert c["total"] == pytest.approx(c["gross"] - c["slag_credit"], abs=1e-9)
    assert r1["out"]["slag"] >= r0["out"]["slag"] - 0.5


def test_sinter_strength_from_calibration():
    """Sinter RI / RDI come from the calibration's latest raw-material strength data, not from the laboratory report."""
    e = Engine()
    assert e.cal["sinter_strength"]["rdi_pct"] == pytest.approx(30.06)
    m = e.omega_model()
    assert m.rdi == pytest.approx(30.06) and m.ref.rdi == pytest.approx(30.06)
    assert m.spec["kinetics"]["ri_ref_pct"] == pytest.approx(e.cal["sinter_strength"]["ri_pct"])
    assert m.rate["sinter"] == pytest.approx((m.spec["kinetics"]["d_test_mm"] / m.spec["materials"]["sinter"]["sauter_mm"])
                                             ** m.spec["kinetics"]["n_size"])


def test_blend_mode_restarts_from_fixed_optimum(inputs):
    """Sinter >= 65 % with the pellet floor: the first STRATA-omega linearisation from the default start has no feasible
    LP; the engine re-starts from the Fixed-omega optimum and returns a feasible STRATA plan."""
    mats = [m["id"] for m in inputs["materials"] if m.get("selected_in_uat", True)]
    bounds = {m["id"]: (m["lo"], m["hi"]) for m in inputs["materials"]}
    bounds["sinter_sp_02"] = (65.0, 70.0)
    e = Engine(inputs, materials=mats, bounds=bounds)
    r = e._try(Settings(omega_mode="blend"), P_REF, Limits(), True)
    assert r is not None and r["feasible"] and r["share_wet"]["sinter_sp_02"] >= 65.0 - 1e-6
    assert r["omega"]["mode"] == "blend" and r["omega"]["certainty"] in ("High", "Medium")


def test_search_re_solves_near_feasible_points_exactly(inputs, monkeypatch):
    """Inside a coarse production search a STRATA-omega point that stops with a small limit breach is re-solved once
    with exact network solves; a large breach (above the ceiling) is not."""
    e = Engine(inputs)
    calls = []

    def fake_try(s, P, L, fuel=True, z0=None, ore6_on=False):
        calls.append(e.net_tol_iter)
        if e.net_tol_iter > 0:
            return dict(feasible=False, viol={"basicity min": 0.009 if P < 100 else 5.0}, z=[0.0])
        return dict(feasible=True, viol={}, z=[1.0])

    monkeypatch.setattr(e, "_try", fake_try)
    monkeypatch.setattr(e, "_network_on", lambda s: True)
    s = Settings(omega_mode="blend")
    with e.network_tolerance(e.SEARCH_TOL):
        near = e._try_search(s, 90.0, Limits(), True)
        far = e._try_search(s, 105.0, Limits(), True)
    assert near["feasible"] and calls[:2] == [e.SEARCH_TOL, 0.0]
    assert not far["feasible"] and calls[2:] == [e.SEARCH_TOL]
    with e.network_tolerance(e.SEARCH_TOL):
        low_end = e._try_search(s, 105.0, Limits(), True, force=True)
    assert low_end["feasible"] and calls[3:] == [e.SEARCH_TOL, 0.0]
    assert e.net_tol_iter == 0.0
