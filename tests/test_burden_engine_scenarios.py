"""Scenario layer of the Burden & Fuel Optimiser page (#255-#257): tables and value of limits against the case report."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from utils.burden_engine import Engine, Limits, Settings, load_register
from utils.burden_engine.scenarios import (binding_table, blend_table, cost_table, frontier_table, fuel_defaults,
                                           fuel_split, limits_from, nut_as_charged, nut_moisture_pct, pci_as_injected,
                                           pci_moisture_pct, plan_sentence, plan_tiles, process_table,
                                           run_cheapest_level, run_fixed,
                                           settings_from_regime, slag_table, value_of_limits, viu_table)

THERMAL = dict(HMT=1470.0, hm_Si=0.3836, pci_kg_thm=190.0, nut_kg_thm=70.0)


# the case report's calibration and blast moisture (15 g/Nm3): these tests check the scenario layer against it
CASE_CAL = os.path.join(os.path.dirname(__file__), "fixtures", "cal_2z_jul_oct_2026_case_report_15g.json")


@pytest.fixture(scope="module")
def eng():
    return Engine(cal=CASE_CAL)


@pytest.fixture(scope="module")
def fixed(eng):
    s = settings_from_regime(None, THERMAL, eng.cal, pci_plant=190.0, nut_plant=70.0,
                             humidity_g=eng.cal["case"]["humidity_g"])  # case report
    return s, run_fixed(eng, 2270.0, s, limits_from(load_register()))


def test_settings_from_regime_prefers_regime(eng):
    s = settings_from_regime(dict(HBT=1201.0, O2_per_air=None, Pb=2.8), THERMAL, eng.cal)
    assert s.HBT == 1201.0 and s.Pb == 2.8
    assert s.O2_per_air == pytest.approx(eng.cal["case"]["O2_per_air"])
    assert s.HMT == 1470.0
    # PCI: the regime's median when given, else the calibration's (dry, held as injected), not 190; nut coke: the
    # config's rate, which is dry, held as charged at the calibration's nut-coke moisture
    case = eng.cal["case"]
    assert s.pci_plant == pytest.approx(case["pci_kg"] / (1 - case["pci"]["moist_pct"] / 100))
    s2 = settings_from_regime(dict(pci_kgthm_dry=175.0, nut_kgthm_dry=65.0), THERMAL, eng.cal)
    assert s2.pci_plant == pytest.approx(175.0 / (1 - case["pci"]["moist_pct"] / 100))
    assert s2.nut_plant == pytest.approx(70.0 / (1 - case["nut"]["moist_pct"] / 100)) and s2.nut_plant > 70.0


def test_nut_coke_input_is_the_plants_dry_rate(eng):
    """The nut-coke input (config, page, API) is the plant's nut-coke rate, dry; ``nut_as_charged`` gives the engine's
    as-charged ``nut_plant`` at the calibration's nut-coke moisture, and an explicit ``nut_plant`` stays as charged."""
    shipped = Engine().cal                                           # total moisture from the laboratory
    assert nut_moisture_pct(shipped) == pytest.approx(shipped["case"]["nut"]["moist_pct"]) and nut_moisture_pct({}) == 0.0
    assert 3.0 < nut_moisture_pct(shipped) < 9.0
    wet = nut_as_charged(70.0, shipped)
    assert wet == pytest.approx(70.0 / (1 - nut_moisture_pct(shipped) / 100)) and 72.0 < wet < 77.0
    assert nut_as_charged(70.0, {}) == 70.0
    assert fuel_defaults(None, THERMAL, shipped)["nut_kg_thm"] == 70.0   # the default stays the dry rate
    assert settings_from_regime(None, THERMAL, shipped).nut_plant == pytest.approx(wet)
    assert settings_from_regime(None, THERMAL, shipped, nut_plant=70.0).nut_plant == 70.0
    # the engine takes the as-charged value back to the dry rate: the plan's nut coke is the input
    s = settings_from_regime(None, THERMAL, eng.cal, humidity_g=eng.cal["case"]["humidity_g"])
    out = eng.evaluate(eng.start_point(s, 2270.0 / 24), s, 2270.0 / 24, full=True)
    assert out["nut_dry"] == pytest.approx(70.0) and out["nut_wet"] == pytest.approx(s.nut_plant)


def test_limits_from_register_and_saved():
    L = limits_from(load_register(), dict(slag_max=350.0), dict(charges_per_h=6.35, charge_t=26.0))
    assert (L.slag_max, L.charges_per_h, L.charge_t) == (350.0, 6.35, 26.0)
    assert (L.fuel_min, L.fuel_max) == (520.0, 600.0)


def test_fixed_tables(eng, fixed):
    s, res = fixed
    r, r0 = res["with_fuel"], res["ore_only"]
    assert r["cost"]["total"] == pytest.approx(30139, abs=5) and r0["cost"]["total"] == pytest.approx(30139, abs=5)
    B = blend_table(eng, r, r0)
    assert B.share_wet_pct.dropna().sum() == pytest.approx(100.0, abs=0.01)
    assert cost_table(r).set_index("item").loc["Total", "rs_per_thm"] == pytest.approx(r["cost"]["total"])
    basis = cost_table(r).set_index("item").basis
    assert basis["Coke"] == f"{r['out']['K']:.1f} kg/tHM dry ({r['out']['K_wet']:.1f} as charged)"
    assert basis["Nut coke"] == f"{r['out']['nut_dry']:.1f} kg/tHM dry (70.0 as charged)"
    assert basis["PCI"] == f"{r['out']['pci_dry']:.1f} kg/tHM dry (190.0 as injected)"
    fuel_row = process_table(r).set_index("quantity").loc["Fuel rate (coke dry + nut coke dry + PCI)"]
    assert fuel_row["value"] == f"{r['out']['fuel_rate']:.1f} kg/tHM" and "BAND-01" in fuel_row["limit"]
    assert len(slag_table(r)) == 5 and len(process_table(r)) >= 10
    bt = binding_table(r)
    assert set(bt.limit) >= {"slag cap", "basicity min"}
    assert (bt.set_index("limit").loc["slag cap", "rs_thm_per_unit"]) > 0
    v = viu_table(eng, r)
    assert v.iloc[0]["name"] == "GEOMIN CLO"      # cheapest all-in Fe unit on the UAT yard (case report)


def test_pci_input_is_the_plants_dry_coal_rate(eng):
    """The PCI input (config, page, API, live tag, regime median, calibration) is the plant's coal rate, dry;
    ``pci_as_injected`` gives the engine's as-injected ``pci_plant`` at the coal moisture of the calibration, and an
    explicit ``pci_plant`` stays as injected."""
    shipped = Engine().cal
    moist = shipped["case"]["pci"]["moist_pct"]
    assert pci_moisture_pct(shipped) == pytest.approx(moist) and 0.5 < moist < 4.0 and pci_moisture_pct({}) == 0.0
    assert pci_as_injected(185.0, shipped) == pytest.approx(185.0 / (1 - moist / 100)) and pci_as_injected(185.0, {}) == 185.0
    # every default is dry: the live tag, the regime median and the calibration's coal rate, then the config
    assert fuel_defaults({"pci_kgthm_dry": 175.0}, THERMAL, shipped, live_pci=182.0)["pci_kg_thm"] == 182.0
    assert fuel_defaults({"pci_kgthm_dry": 175.0}, THERMAL, shipped)["pci_kg_thm"] == 175.0
    assert fuel_defaults(None, THERMAL, shipped)["pci_kg_thm"] == shipped["case"]["pci_kg"]
    assert fuel_defaults(None, THERMAL, {}) == dict(pci_kg_thm=190.0, pci_kg_thm_source="config", nut_kg_thm=70.0,
                                                    nut_kg_thm_source="config")
    assert settings_from_regime({"pci_kgthm_dry": 175.0}, THERMAL, shipped).pci_plant == pytest.approx(
        pci_as_injected(175.0, shipped))
    assert settings_from_regime(None, THERMAL, shipped, pci_plant=190.0).pci_plant == 190.0
    # the engine takes the as-injected value back to the dry rate: the plan's PCI is the input
    s = settings_from_regime({"pci_kgthm_dry": 175.0}, THERMAL, eng.cal, humidity_g=eng.cal["case"]["humidity_g"])
    out = eng.evaluate(eng.start_point(s, 2270.0 / 24), s, 2270.0 / 24, full=True)
    assert out["pci_dry"] == pytest.approx(175.0) and out["pci_wet"] == pytest.approx(s.pci_plant) and s.pci_plant > 177.0
    assert out["fuel_rate"] == pytest.approx(out["K"] + out["nut_dry"] + 175.0)


def test_fuel_figures_lead_with_the_plant_basis(eng, fixed):
    """Dry coke + dry nut coke + PCI add up to the fuel rate as shown; as charged beside; charging stays as charged."""
    _, res = fixed
    r = res["with_fuel"]
    o, f = r["out"], fuel_split(r)
    assert plan_tiles(r)["fuel"] == f
    assert (f["coke_dry"], f["coke_wet"], f["nut"], f["nut_wet"]) == (o["K"], o["K_wet"], o["nut_dry"], 70.0)
    assert (f["pci"], f["pci_wet"]) == (o["pci_dry"], 190.0) and f["pci"] == pytest.approx(187.16, abs=0.01)
    assert f["coke_dry"] + f["nut"] + f["pci"] == f["total"] == pytest.approx(o["fuel_rate"], abs=1e-9)
    assert f["coke_wet"] + f["nut_wet"] + f["pci_wet"] == f["total_charged"] == pytest.approx(o["fuel_charged"], abs=1e-9)
    assert f["total"] == pytest.approx(562.5, abs=0.3) and f["total_charged"] == pytest.approx(567.0, abs=0.3)
    g = plan_tiles(r)["charging"]
    assert g["coke_t"] == pytest.approx(o["K_wet"] * 2.27) and g["nut_t"] == pytest.approx(70.0 * 2.27)
    assert g["pci_t"] == pytest.approx(190.0 * 2.27)                 # tonnes as injected
    # a result stored before the engine reported the plant basis (fuel_rate = the as-charged sum) reads the same
    old = dict(r, out={k: v for k, v in o.items() if k not in ("fuel_charged", "nut_wet", "pci_wet")}
               | {"fuel_rate": o["fuel_charged"]})
    assert fuel_split(old) == f
    kind, text = plan_sentence(r, {}, [])
    assert kind == "plain" and text.endswith(f"coke {o['K']:.2f} kg/tHM dry ({o['K_wet']:.2f} as charged).")
    T = frontier_table([r])
    assert T.fuel_rate.iloc[0] == o["fuel_rate"] and T.fuel_charged.iloc[0] == o["fuel_charged"]


def test_value_of_limits_matches_case_report(eng, fixed):
    s, res = fixed
    V = value_of_limits(eng, 2270.0, s, limits_from(load_register()), res["with_fuel"]).set_index("change")
    assert V.loc["Slag cap +10 kg/tHM", "saving_rs_thm"] == pytest.approx(53, abs=3)
    assert V.loc["LLOYDS PELLET floor −2 pt", "saving_rs_thm"] == pytest.approx(44, abs=3)
    assert V.loc["CaO/SiO₂ floor −0.01", "saving_rs_thm"] == pytest.approx(27, abs=3)
    assert V.loc["Charges per hour +0.1", "saving_rs_thm"] == pytest.approx(0, abs=0.5)


def test_frontier_table_marginal_cost():
    def r(P, c):
        return dict(P_tday=P, feasible=True, cost=dict(total=c, ore=c / 2, flux=0.0, fuel=c / 2), out=dict(K=300.0, fuel_rate=560.0, slag=330.0),
                    settings=dict(HBT=1230.0), enrich_tag=4.2, blast_nm3h=1e5, O2_nm3h=5e3, Pb_needed=2.8, charges_per_h=6.4,
                    binding=["slag cap", "min share sinter_sp_02"])
    T = frontier_table([r(2000.0, 30000.0), r(2100.0, 29900.0), r(2100.0, 29950.0)])
    assert list(T.prod_t_day) == [2000.0, 2100.0]
    assert T.cost_rs_thm.iloc[1] == 29900.0
    assert T.marginal_rs_per_t.iloc[1] == pytest.approx((2100 * 29900 - 2000 * 30000) / 100)
    assert T.binding.iloc[0] == "slag cap"


@pytest.mark.skipif(os.environ.get("BURDEN_ENGINE_FAST") == "1", reason="slow")
def test_cheapest_level_matches_case4(eng):
    s = settings_from_regime(None, THERMAL, eng.cal, pci_plant=190.0, nut_plant=70.0,
                             humidity_g=eng.cal["case"]["humidity_g"])  # case report
    res = run_cheapest_level(eng, s, limits_from(load_register()))
    assert res["best"]["P_tday"] == pytest.approx(2290, abs=3)
    assert res["best"]["cost"]["total"] == pytest.approx(29935, abs=10)
    assert res["top"]["P_tday"] == pytest.approx(2412, abs=3)
    F = res["frontier"]
    assert F.cost_rs_thm.idxmin() == F.index[(F.prod_t_day - res["best"]["P_tday"]).abs().argmin()] or \
        F.cost_rs_thm.min() >= res["best"]["cost"]["total"] - 1
