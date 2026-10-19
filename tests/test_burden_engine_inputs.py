"""Inputs adapter (#253): BMO typed records + operator preferences -> engine inputs reproduce the reference yard."""

from __future__ import annotations

import json

import pytest
import yaml

from data.burden_engine.inputs import (build_engine_inputs, flux_rows, fuel_prices, kind_for, repo_path, saved_charging,
                                       slag_limits, yard_rows)
from utils.bmo.types import FluxInput, OreChemistry, OreInput
from utils.burden_engine import Engine
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH

REF = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))
PREFS = yaml.safe_load(open(repo_path("src/config/bmo_operator_inputs.yml"), encoding="utf-8"))


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
    mats = yard_rows([_ore(m) for m in REF["materials"]], PREFS,
                     k_rules=dict(gomti_clo="gomti", lloyds_oversize_clo="oversize"))
    fl = flux_rows([_flux(f) for f in REF["fluxes"]], PREFS)
    return build_engine_inputs(mats, fl, fuel_prices(PREFS))


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
