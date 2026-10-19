"""Scenario layer of the Burden & Fuel Optimiser page (#255-#257): tables and value of limits against the case report."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from utils.burden_engine import Engine, Limits, Settings, load_register
from utils.burden_engine.scenarios import (binding_table, blend_table, cost_table, frontier_table, limits_from,
                                           process_table, run_cheapest_level, run_fixed, settings_from_regime,
                                           slag_table, value_of_limits, viu_table)

THERMAL = dict(HMT=1470.0, hm_Si=0.3836, pci_kg_thm=190.0, nut_kg_thm=70.0)


@pytest.fixture(scope="module")
def eng():
    return Engine()


@pytest.fixture(scope="module")
def fixed(eng):
    s = settings_from_regime(None, THERMAL, eng.cal)
    return s, run_fixed(eng, 2270.0, s, limits_from(load_register()))


def test_settings_from_regime_prefers_regime(eng):
    s = settings_from_regime(dict(HBT=1201.0, O2_per_air=None, Pb=2.8), THERMAL, eng.cal)
    assert s.HBT == 1201.0 and s.Pb == 2.8
    assert s.O2_per_air == pytest.approx(eng.cal["case"]["O2_per_air"])
    assert s.pci_plant == 190.0 and s.HMT == 1470.0


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
    assert len(slag_table(r)) == 5 and len(process_table(r)) >= 10
    bt = binding_table(r)
    assert set(bt.limit) >= {"slag cap", "basicity min"}
    assert (bt.set_index("limit").loc["slag cap", "rs_thm_per_unit"]) > 0
    v = viu_table(eng, r)
    assert v.iloc[0]["name"] == "GEOMIN CLO"      # cheapest all-in Fe unit on the UAT yard (case report)


def test_value_of_limits_matches_case_report(eng, fixed):
    s, res = fixed
    V = value_of_limits(eng, 2270.0, s, limits_from(load_register()), res["with_fuel"]).set_index("change")
    assert V.loc["Slag cap +10 kg/tHM", "saving_rs_thm"] == pytest.approx(53, abs=3)
    assert V.loc["LLOYDS PELLET floor −2 pt", "saving_rs_thm"] == pytest.approx(44, abs=3)
    assert V.loc["CaO/SiO2 floor −0.01", "saving_rs_thm"] == pytest.approx(27, abs=3)
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
    s = settings_from_regime(None, THERMAL, eng.cal)
    res = run_cheapest_level(eng, s, limits_from(load_register()))
    assert res["best"]["P_tday"] == pytest.approx(2290, abs=3)
    assert res["best"]["cost"]["total"] == pytest.approx(29935, abs=10)
    assert res["top"]["P_tday"] == pytest.approx(2412, abs=3)
    F = res["frontier"]
    assert F.cost_rs_thm.idxmin() == F.index[(F.prod_t_day - res["best"]["P_tday"]).abs().argmin()] or \
        F.cost_rs_thm.min() >= res["best"]["cost"]["total"] - 1
