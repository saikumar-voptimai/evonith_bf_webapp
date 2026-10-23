"""Inputs adapter (#253): BMO typed records + operator preferences -> engine inputs reproduce the reference yard."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from data.burden_engine.inputs import (MN_IN_MNO, build_engine_inputs, catalogue_fluxes, flux_rows, fuel_prices,
                                       kind_for, load_yard, repo_path, saved_charging, slag_limits, yard_rows)
from utils.bmo.types import FluxInput, OreChemistry, OreInput
from utils.burden_engine import Engine
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH

REF = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))
PREFS = yaml.safe_load(open(repo_path("src/config/bmo_operator_inputs.yml"), encoding="utf-8"))


def _uat_prefs():
    """The operator inputs at the 16 Oct 2026 UAT prices (the case report's basis); the file itself now carries the
    22 Oct 2026 cost-sheet prices (DATA-14)."""
    p = copy.deepcopy(PREFS)
    for m in REF["materials"]:
        p["ore_editor"]["rows"].setdefault(m["id"], {})["price_rs_per_mt"] = float(m["price"])
    for f in REF["fluxes"]:
        p["flux_editor"]["rows"].setdefault(f["id"], {})["price_rs_per_mt"] = float(f["price"])
    for k, name in (("coke_price", "coke"), ("nut_price", "nut_coke"), ("pci_price", "pci")):
        p["fuel_ash_editor"]["rows"][name]["price_rs_per_mt"] = float(REF["fuel"][k])
    return p


UAT_PREFS = _uat_prefs()


def _ore(m, source="lab"):
    c = m["chem"]
    chem = OreChemistry(fe_t_pct=c["Fe"], moisture_pct=m["moist"], feo_pct=c.get("FeO", 0.0), sio2_pct=c["SiO2"],
                        al2o3_pct=c["Al2O3"], cao_pct=c["CaO"], mgo_pct=c["MgO"], mno_pct=c["MnO"], tio2_pct=c["TiO2"],
                        p_pct=c["P"], loi_pct=c.get("LOI", 0.0))
    return OreInput(m["id"], m["name"], 1e6, 0.0, 0.0, 100.0, chem,
                    metadata=dict(material_key=m["code"], chemistry_source=source))


def _flux(f):
    c = f["chem"]
    return FluxInput(f["id"], f["name"], moisture_pct=f["moist"], sio2_pct=c["SiO2"], al2o3_pct=c["Al2O3"], cao_pct=c["CaO"],
                     mgo_pct=c["MgO"], fe2o3_pct=c["Fe"] / (2 * 55.845 / 159.69), loi_pct=c["LOI"], stock_mt=1e6,
                     price_rs_per_mt=0.0)


@pytest.fixture(scope="module")
def inputs():
    mats = yard_rows([_ore(m) for m in REF["materials"]], UAT_PREFS,
                     k_rules=dict(gomti_clo="gomti", lloyds_oversize_clo="oversize"))
    fl = flux_rows([_flux(f) for f in REF["fluxes"]], UAT_PREFS)
    return build_engine_inputs(mats, fl, fuel_prices(UAT_PREFS))


def test_kind_for():
    assert [kind_for(k) for k in ("sinter_3", "pellet_1", "ore_6", "")] == ["sinter", "pellet", "ore", "ore"]


def test_operator_prices_and_shares_applied(inputs):
    m = {x["id"]: x for x in inputs["materials"]}
    assert m["sinter_sp_02"]["price"] == 8724.0 and m["sinter_sp_02"]["lo"] == 58.0
    assert m["lloyds_pellet"]["lo"] == 10.0
    assert m["geomin_clo"]["chem"]["LOI"] == pytest.approx(7.37, abs=0.01)
    assert m["gomti_clo"]["k_rule"] == "gomti" and not m["gomti_clo"]["selected"]
    assert {f["id"]: f["loi_type"] for f in inputs["fluxes"]} == dict(limestone="CO2", dolomite="CO2", quartz="H2O")
    assert inputs["fuel"] == dict(coke_price=31992.0, nut_price=26074.0, pci_price=22120.0)
    assert slag_limits(PREFS)["slag_max"] == 340.0
    assert saved_charging(PREFS)["charges_per_h"] == pytest.approx(6.35)


def test_operator_file_carries_plant_prices():
    """DATA-14: plant prices of 23 Oct 2026; fuels from the BF-2 cost sheet (FTD Costing 22 Oct 2026)."""
    rows = PREFS["ore_editor"]["rows"]
    given = dict(nmdc_rom=10323.0, lloyds_clo=8877.0, nmdc_donimalai=6141.0, geomin_clo=4204.0, sinter_sp_02=8784.0,
                 lloyds_pellet=10515.0)
    assert {k: rows[k]["price_rs_per_mt"] for k in given} == given
    fl = PREFS["flux_editor"]["rows"]
    assert (fl["limestone"]["price_rs_per_mt"], fl["dolomite"]["price_rs_per_mt"], fl["quartz"]["price_rs_per_mt"]) == \
        (2369.0, 1196.0, 1792.0)
    assert fuel_prices(PREFS) == dict(coke_price=31960.0, nut_price=26292.0, pci_price=21168.0)
    cfg = yaml.safe_load(open(repo_path("src/config/burden_optimiser.yml"), encoding="utf-8"))["burden_optimiser"]
    assert cfg["prices"]["slag_price_rs_per_t"] == pytest.approx(457.85)
    assert cfg["prices"]["materials"]["lloyds_clo"] == 8877


def test_adapter_reproduces_case2(inputs):
    r = Engine(inputs).min_cost(prod_t_day=2270, fuel=True)
    assert r["out"]["K"] == pytest.approx(305.7, abs=0.3)
    assert r["cost"]["total"] == pytest.approx(30139, abs=5)


def test_fallback_chemistry_left_out():
    rows = yard_rows([_ore(REF["materials"][1], source="fallback")], PREFS)
    assert rows[0]["carried"] is False and "fallback" in rows[0]["reason"]


def test_stock_caps_and_zero_stock(inputs):
    mats = [dict(m) for m in inputs["materials"]]
    fl = [dict(f, stock_mt=1e6, max_qty_mt=100.0, carried=True) for f in inputs["fluxes"]]
    for m in mats:
        m["stock_mt"] = 0.0 if m["id"] == "nmdc_rom" else 1e6
        m["carried"] = True
    inp = build_engine_inputs(mats, fl, inputs["fuel"])
    assert inp["not_carried"]["nmdc_rom"] == "zero stock"
    geomin_cap = 50.0
    for m in mats:
        if m["id"] == "geomin_clo":
            m["stock_mt"] = geomin_cap
    inp = build_engine_inputs(mats, fl, inputs["fuel"])
    r = Engine(inp).min_cost(prod_t_day=2270, fuel=True)
    assert r["t_day"]["geomin_clo"] <= geomin_cap + 1e-6


def test_flux_catalogue_is_page_9s():
    """Fluxes are page 9's catalogue, not the last charge's flux (codes flux_1.., no price): limestone is offered
    even on a day it was not charged, with the operator's price."""
    cfg = yaml.safe_load(open(repo_path("src/config/setting_bmo.yml"), encoding="utf-8"))["bmo"]["flux_inputs"]
    rows = {r["id"]: r for r in flux_rows(catalogue_fluxes(cfg), PREFS)}
    assert {"limestone", "dolomite", "quartz"} <= set(rows)
    assert rows["limestone"]["price"] == PREFS["flux_editor"]["rows"]["limestone"]["price_rs_per_mt"]
    assert rows["dolomite"]["loi_type"] == "CO2" and rows["quartz"]["loi_type"] == "H2O"
    extra = [dict(flux_id="hid", hidden_by_default=True, optimizable=True),
             dict(flux_id="fixed", optimizable=False, cao_pct=50.0),
             dict(flux_id="mn", optimizable=True, mn_basis="mn", mn_pct=7.745)]
    got = {f.flux_id: f for f in catalogue_fluxes(extra)}
    assert "hid" not in got and not got["fixed"].enabled
    assert got["mn"].mno_pct == pytest.approx(7.745 / MN_IN_MNO)


def test_load_yard_takes_fluxes_from_the_catalogue():
    class Provider:  # ores only: load_yard must not ask the provider for fluxes
        def build_ore_inputs(self, mode, window_days):
            return [_ore(m) for m in REF["materials"]], {"warnings": []}

    y = load_yard({"yard": {}}, provider=Provider())
    assert {f["id"] for f in y["fluxes"]} == {"limestone", "dolomite", "quartz"}
    assert all(f["price"] > 0 for f in y["fluxes"])
