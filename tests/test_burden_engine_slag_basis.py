"""The engine's slag on the Blend Mix Optimiser's basis (user decision 11 Nov 2026): one blend, one slag.

The same wet tonnes of ore, flux, fuel and dust go through page 9's ``calculate_full_slag_balance`` and the engine's
``slag_and_flux`` with ``slag_basis``; every slag component must agree to 1e-9 relative. Without the basis the engine
keeps the calibration's rule (the 16 Oct 2026 case report, ``tests/test_burden_engine.py``).
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest
import yaml

from data.burden_engine.inputs import (build_engine_inputs, describe_slag_basis, reference_slag_basis, repo_path,
                                       slag_basis)
from utils.bmo.calculations import compute_charge_count
from utils.bmo.slag_balance import calculate_full_slag_balance
from utils.bmo.types import DustInput, FluxInput, FuelAshInput, OreChemistry, OreInput, SlagBalanceSettings
from utils.burden_engine import Engine, Limits
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH
from utils.coke_energy.bf_balance import slag_and_flux
from utils.coke_energy.slag_basis import FE_IN_FE2O3, MN_IN_MNO, TI_IN_TIO2

UAT = ["sinter_sp_02", "geomin_clo", "lloyds_clo", "nmdc_rom", "lloyds_pellet"]
CASE_CAL = os.path.join(os.path.dirname(__file__), "fixtures", "cal_2z_jul_oct_2026_case_report_15g.json")
BMO = yaml.safe_load(open(repo_path("src/config/setting_bmo.yml"), encoding="utf-8"))["bmo"]
REF = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))
CHARGE_T = float(REF["charging"]["uat_charge_t"])


@pytest.fixture(scope="module")
def basis():
    return reference_slag_basis(charge_t=CHARGE_T)


@pytest.fixture(scope="module")
def eng(basis):
    inputs = dict(REF, slag_basis=basis)
    return Engine(inputs, cal=CASE_CAL, materials=UAT)


def _page9_inputs(eng: Engine, case: dict, K: float):
    """Page 9's typed records from the engine's case: the same wet tonnes per tHM (hot_metal_mt = 1 t)."""
    ores, qty, fluxes = [], {}, []
    for v, m in zip(eng.vars, case["burden"]):
        c = m["comp"]
        wet_t = m["dry_kg"] / (1 - m["moist_pct"] / 100) / 1000
        if v.kind == "flux":
            fluxes.append(FluxInput(v.id, v.name, enabled=True, wet_qty_mt=wet_t, moisture_pct=m["moist_pct"],
                                    sio2_pct=c["SiO2"], al2o3_pct=c["Al2O3"], cao_pct=c["CaO"], mgo_pct=c["MgO"],
                                    fe2o3_pct=c["Fe"] / FE_IN_FE2O3, mno_pct=c["MnO"], tio2_pct=c["TiO2"],
                                    p_pct=c["P"], loi_pct=c["LOI"], s_pct=c.get("S", 0.0), zn_pct=c.get("Zn", 0.0),
                                    na2o_pct=c.get("Na2O", 0.0), k2o_pct=c.get("K2O", 0.0), caf2_pct=c.get("CaF2", 0.0),
                                    price_rs_per_mt=v.price, stock_mt=1e6, optimizable=True))
            continue
        chem = OreChemistry(fe_t_pct=c["Fe"], moisture_pct=m["moist_pct"], feo_pct=c.get("FeO", 0.0), sio2_pct=c["SiO2"],
                            al2o3_pct=c["Al2O3"], cao_pct=c["CaO"], mgo_pct=c["MgO"], mno_pct=c["MnO"],
                            tio2_pct=c["TiO2"], p_pct=c["P"], loi_pct=c.get("LOI", 0.0), s_pct=c.get("S", 0.0),
                            zn_pct=c.get("Zn", 0.0), na2o_pct=c.get("Na2O", 0.0), k2o_pct=c.get("K2O", 0.0))
        ores.append(OreInput(v.id, v.name, 1e6, v.price, 0.0, 100.0, chem))
        qty[v.id] = wet_t
    fuels = []
    for key, fid, name, dry in (("coke", "coke", "Coke", K), ("nut", "nut_coke", "Nut coke", case["nut_kg"]),
                                ("pci", "pci", "PCI", case["pci_kg"])):
        f = case[key]
        a = f["ash_comp"]
        fuels.append(FuelAshInput(fid, name, rate_kg_per_thm=dry / (1 - f["moist_pct"] / 100), rate_basis="wet",
                                  moisture_pct=f["moist_pct"], vm_pct=f.get("VM", 0.0), ash_pct=f["ash"],
                                  sio2_pct=a["SiO2"], al2o3_pct=a["Al2O3"], cao_pct=a["CaO"], mgo_pct=a["MgO"],
                                  fe2o3_pct=a["Fe2O3"], tio2_pct=a["TiO2"], mno_pct=a.get("MnO", 0.0),
                                  alkali_pct=a.get("alk", 0.0), s_pct=f.get("S_pct", 0.0), p_pct=f.get("P_pct", 0.0)))
    dust = [DustInput(d["id"], d["name"], enabled=d["enabled"], wet_qty_mt=d["kg"] / 1000, rate_basis="mt_per_day",
                      moisture_pct=d["moist_pct"], sio2_pct=d["SiO2_pct"], al2o3_pct=d["Al2O3_pct"], cao_pct=d["CaO_pct"],
                      mgo_pct=d["MgO_pct"], fe_pct=d["Fe_pct"], mn_pct=d["Mn_pct"], p_pct=d["P_pct"], s_pct=d["S_pct"],
                      ti_pct=d["Ti_pct"], zn_pct=d["Zn_pct"], na2o_pct=d["Na2O_pct"], k2o_pct=d["K2O_pct"],
                      caf2_pct=d["CaF2_pct"]) for d in case["dust"]["streams"]]
    r = eng.p["slag_basis"]
    settings = SlagBalanceSettings(enabled=True, carbon_pct=r["C_pct"], silicon_pct=r["Si_pct"], sulphur_pct=r["S_pct"],
                                   other_pct=r["other_pct"], mn_pct=case["hm"]["Mn"], ti_pct=case["hm"]["Ti"],
                                   pi_loss_pct=r["pi_loss_pct"], fe_to_pig_iron_fraction=r["fe_to_pig_iron_fraction"],
                                   mn_recovery_pct=r["mn_recovery_pct"], sulphur_gas_loss_pct=r["sulphur_gas_loss_pct"],
                                   alkali_to_slag_fraction=r["alkali_to_slag_fraction"],
                                   si_to_sio2_factor=r["si_to_sio2_factor"], fe_to_feo_factor=r["fe_to_feo_factor"],
                                   mn_to_mno_factor=r["mn_to_mno_factor"],
                                   slag_correction_factor=r["slag_correction_factor"], tio2_to_slag=r["tio2_to_slag"])
    return dict(ores=ores, quantities_mt=qty, hot_metal_mt=1.0, settings=settings, fuel_ash_inputs=fuels,
                flux_inputs=fluxes, dust_inputs=dust)


def _same_blend(eng: Engine, z, s, P):
    o = eng.evaluate(z, s, P, full=True)
    case = eng.make_case(z, s, P)
    sl = slag_and_flux(o["K"], case, eng.p)
    p9 = calculate_full_slag_balance(**_page9_inputs(eng, case, o["K"]))
    return o, sl, p9


PAIRS = (("alk", "alkali"), ("SiO2", "sio2"), ("Al2O3", "al2o3"), ("CaO", "cao"), ("MgO", "mgo"), ("FeO", "feo"),
         ("MnO", "mno"), ("TiO2", "tio2"), ("S", "s"), ("CaF2", "caf2"))


def test_constants_are_page_9s():
    from utils.bmo.slag_balance import FE_FROM_FE2O3_FACTOR
    from utils.bmo.types import MN_IN_MNO_FRACTION, TI_IN_TIO2_FRACTION

    assert (MN_IN_MNO, TI_IN_TIO2, FE_IN_FE2O3) == (MN_IN_MNO_FRACTION, TI_IN_TIO2_FRACTION, FE_FROM_FE2O3_FACTOR)


def test_same_blend_one_slag_at_the_start_point(eng):
    s, P = eng.cal_settings(), 2270.0 / 24
    z = eng.start_point(s, P)
    o, sl, p9 = _same_blend(eng, z, s, P)
    assert o["slag"] == pytest.approx(p9.total_slag_mt * 1000, rel=1e-9)
    for ek, pk in PAIRS:
        assert sl[ek] == pytest.approx(p9.slag_components_mt[pk] * 1000, rel=1e-9, abs=1e-9), ek
    d = p9.diagnostics
    assert sl["pig_iron"]["sio2_used_kg"] == pytest.approx(d["sio2_consumed_by_si_mt"] * 1000, rel=1e-9)
    assert sl["Fe_slag"] == pytest.approx(d["fe_remaining_mt"] * 1000, rel=1e-9)
    # held Si for the slag side, the thermal Si for the coke: page 9's split
    assert eng.p["slag_basis"]["Si_pct"] == pytest.approx(0.6) and s.hm_Si == pytest.approx(0.3836)
    # the Fe closes on the pig iron: Fe charged - dust - FeO's Fe = the hot metal's
    assert abs(o["Fe_res"]) < 1e-6


def test_same_blend_one_slag_at_the_plan(eng):
    s, P = eng.cal_settings(), 2270.0 / 24
    r = eng.min_cost(prod_t_day=2270.0, s=s, L=Limits(), fuel=True)
    assert r["feasible"]
    o, sl, p9 = _same_blend(eng, r["z"], s, P)
    assert o["slag"] == pytest.approx(p9.total_slag_mt * 1000, rel=1e-9)
    assert o["B2"] == pytest.approx(p9.slag_components_mt["cao"] / p9.slag_components_mt["sio2"], rel=1e-9)
    assert o["Al2O3_pct"] == pytest.approx(100 * p9.slag_components_mt["al2o3"] / p9.total_slag_mt, rel=1e-9)
    assert o["MgO_pct"] == pytest.approx(100 * p9.slag_components_mt["mgo"] / p9.total_slag_mt, rel=1e-9)
    assert o["slag"] <= Limits().slag_max + 0.05


def test_dust_follows_the_charges_of_the_blend(eng):
    """Each stream's kg/tHM = the DPR's (here the configured) kg per charge x page 9's charge count per tHM."""
    s, P = eng.cal_settings(), 2270.0 / 24
    z = eng.start_point(s, P)
    case = eng.make_case(z, s, P)
    inputs = _page9_inputs(eng, case, 300.0)
    charges = compute_charge_count(inputs["quantities_mt"], flux_inputs=inputs["flux_inputs"],
                                   fuel_ash_inputs=inputs["fuel_ash_inputs"], hot_metal_mt=1.0, charge_mass_mt=CHARGE_T)
    for d in case["dust"]["streams"]:
        assert d["kg"] == pytest.approx(d["kg_per_charge"] * charges if d["enabled"] else 0.0, rel=1e-9)
    assert case["dust"]["dc_kg"] + case["dust"]["gcp_kg"] == pytest.approx(sum(d["kg"] for d in case["dust"]["streams"]))
    # the plant's burden-sheet row, constant (user decision 11 Nov 2026); the measured streams stay disabled
    assert [(d["id"], d["kg"] > 0) for d in case["dust"]["streams"]] == [("bf_gas_dust", True), ("dust_catcher", False),
                                                                        ("gcp_dust", False)]
    assert case["dust"]["streams"][0]["kg_per_charge"] == 530.0 and case["dust"]["streams"][0]["MgO_pct"] == 3.57
    assert case["dust"]["streams"][0]["kg"] == pytest.approx(530.0 * charges)   # about 36 kg/tHM
    assert case["dust"]["gcp_kg"] == 0.0


def test_without_the_basis_the_calibrations_rule_stands():
    e = Engine(materials=UAT, cal=CASE_CAL)
    assert e.slag_basis is None and "slag_basis" not in e.p
    s, P = e.cal_settings(), 2270.0 / 24
    case = e.make_case(e.start_point(s, P), s, P)
    assert "streams" not in case["dust"]
    sl = slag_and_flux(300.0, case, e.p)
    assert sl["FeO"] == pytest.approx(sl["slag_kg"] * case["slag_FeO_pct"] / 100)
    assert sl["minor"] == pytest.approx(sl["slag_kg"] * e.p["slag_minor_pct"] / 100)


def test_the_breakup_is_page_9s_slag_balance_details(eng):
    """The plan's slag breakup (``slag_detail``) equals page 9's Slag Balance Details on the same blend: the
    components charged by the ores, the fluxes and the fuel ash, carried away by the dust, after the dust, each
    material's Fe (MT) and Slag (MT), the fuel ash's slag, and the worked hot-metal removals; the per-source rollup
    and the components add up to the slag rate."""
    s, P = eng.cal_settings(), 2270.0 / 24
    r = eng.min_cost(prod_t_day=2270.0, s=s, L=Limits(), fuel=True)
    o, sl, p9 = _same_blend(eng, r["z"], s, P)
    d = o["slag_detail"]
    assert d is sl["detail"] or d == sl["detail"]
    keys = ("sio2", "al2o3", "cao", "mgo", "mn", "p", "s", "ti", "zn", "alkali", "fe", "caf2")
    for src, p9_key in (("ore", "ore_components_mt"), ("flux", "flux_components_mt"), ("fuel_ash", "fuel_ash_components_mt"),
                        ("dust", "dust_components_mt")):
        for k in keys:
            assert d["sources"][src][k] == pytest.approx(p9.__dict__[p9_key][k] * 1000, rel=1e-9, abs=1e-9), (src, k)
    for k in keys:
        assert d["net"][k] == pytest.approx(p9.net_into_bf_mt[k] * 1000, rel=1e-9, abs=1e-9), k
    # page 9's blend-table columns: Fe (MT) and Slag (MT) per ore, the fuel ash's and the flux's slag
    from utils.bmo.calculations import (compute_fe_contribution_mt, compute_flux_slag_contributions_mt,
                                        compute_fuel_ash_slag_contributions_mt, compute_slag_contribution_mt)

    case = eng.make_case(r["z"], s, P)
    inputs = _page9_inputs(eng, case, o["K"])
    ores = {x.ore_id: x for x in inputs["ores"]}
    flux_slag = compute_flux_slag_contributions_mt(inputs["flux_inputs"])
    for v, m0 in zip(eng.vars, case["burden"]):
        m = d["by_material"][v.id]
        if v.kind == "flux":
            assert m["slag_kg"] == pytest.approx(flux_slag[v.id] * 1000, rel=1e-9, abs=1e-9)
            continue
        c = ores[v.id].chemistry
        dry_t = m0["dry_kg"] / 1000
        assert m["fe_kg"] == pytest.approx(compute_fe_contribution_mt(dry_t, c.fe_t_pct) * 1000, rel=1e-9, abs=1e-9)
        assert m["slag_kg"] == pytest.approx(1000 * compute_slag_contribution_mt(
            dry_t, sio2_pct=c.sio2_pct, al2o3_pct=c.al2o3_pct, cao_pct=c.cao_pct, mgo_pct=c.mgo_pct, tio2_pct=c.tio2_pct,
            mno_pct=c.mno_pct, na2o_pct=c.na2o_pct, k2o_pct=c.k2o_pct), rel=1e-9, abs=1e-9)
    # page 9's per-fuel column takes Na2O + K2O of the ash, not its alkali_pct (which its balance does count); the
    # engine's column counts the ash alkalis as it does the ores' Na2O + K2O
    fuel = compute_fuel_ash_slag_contributions_mt(inputs["fuel_ash_inputs"], hot_metal_mt=1.0)
    for k, fid in (("coke", "coke"), ("nut", "nut_coke"), ("pci", "pci")):
        f = d["by_fuel"][k]
        assert f["slag_kg"] == pytest.approx(fuel[fid] * 1000 + f["comp"]["alkali"], rel=1e-9, abs=1e-9)
    fb = p9.diagnostics
    rem = d["removals"]
    assert rem["SiO2_to_Si"] == pytest.approx(fb["sio2_consumed_by_si_mt"] * 1000, rel=1e-9)
    assert rem["Fe_to_hot_metal"] == pytest.approx(fb["fe_to_pig_iron_mt"] * 1000, rel=1e-9)
    assert rem["TiO2_to_hot_metal"] == pytest.approx(fb["hm_reduction_tio2_mt"] * 1000, rel=1e-9)
    assert rem["MnO_to_hot_metal"] == pytest.approx(fb["hm_reduction_mno_mt"] * 1000, rel=1e-9)
    assert rem["S_to_hot_metal"] + rem["S_to_gas"] == pytest.approx(fb["hm_reduction_s_mt"] * 1000, rel=1e-9)
    assert rem["alkali_to_gas"] == pytest.approx(fb["hm_reduction_alkali_mt"] * 1000, rel=1e-9)
    assert d["dust_kg"] == pytest.approx(sum(x["kg"] for x in d["dust"].values()))
    # the components and the per-source rollup add up to the slag
    assert sum(d["slag"].values()) == pytest.approx(o["slag"], rel=1e-9)
    ru = d["rollup"]
    assert sum(v for k, v in ru.items() if k != "slag") == pytest.approx(ru["slag"], rel=1e-9) and ru["slag"] == pytest.approx(o["slag"])
    assert ru["dust"] < 0 and ru["hot_metal_gas"] < 0 and ru["ore"] > ru["fuel_ash"] > 0
    # each burden row carries its kind
    assert {m["kind"] for m in d["by_material"].values()} >= {"sinter", "ore", "pellet", "flux"}


def test_the_result_tables_follow_the_breakup(eng):
    from utils.burden_engine.scenarios import (dust_table, fuel_ash_table, material_rows, slag_balance_table,
                                               slag_basis_note, slag_components_table, slag_removals_table,
                                               slag_rollup_table)

    s = eng.cal_settings()
    r = eng.min_cost(prod_t_day=2270.0, s=s, L=Limits(), fuel=True)
    names, prices = {v.id: v.name for v in eng.vars}, {v.id: v.price for v in eng.vars}
    rows = {m["id"]: m for m in material_rows(r, names, prices)}
    d = r["out"]["slag_detail"]
    hm_per_charge = r["P_th"] / r["charges_per_h"]
    for i, m in rows.items():
        assert m["fe_kg"] == d["by_material"][i]["fe_kg"] and m["slag_kg"] == d["by_material"][i]["slag_kg"]
        assert m["kg_per_charge"] == pytest.approx(r["wet_kg"][i] * hm_per_charge)
        assert m["cost_rs_thm"] == pytest.approx(r["wet_kg"][i] * prices[i] / 1000)
        assert (m["share_wet_pct"] is None) == (m["kind"] == "flux")
    assert set(rows) == {i for i, kg in r["wet_kg"].items() if kg >= 0.05}
    C = slag_components_table(r)
    assert list(C["component"])[-1] == "Slag" and C["kg_per_thm"].iloc[:-1].sum() == pytest.approx(r["out"]["slag"], rel=1e-9)
    assert C["pct"].iloc[:-1].sum() == pytest.approx(100.0, rel=1e-9)
    B = slag_balance_table(r)
    assert list(B["component"])[:4] == ["SiO₂", "Al₂O₃", "CaO", "MgO"]
    sio2 = B.iloc[0]
    assert sio2["ore"] + sio2["flux"] + sio2["fuel_ash"] + sio2["dust"] + sio2["hot_metal_gas"] == pytest.approx(sio2["slag"], rel=1e-9)
    al = B.iloc[1]
    assert al["ore"] + al["flux"] + al["fuel_ash"] + al["dust"] == pytest.approx(al["slag"], rel=1e-9) and al["hot_metal_gas"] == 0.0
    R = slag_rollup_table(r)
    assert R["kg_per_thm"].iloc[:-1].sum() == pytest.approx(R["kg_per_thm"].iloc[-1], rel=1e-9)
    M = slag_removals_table(r)
    assert "2.14" in M["rule"].iloc[0] and M["kg_per_thm"].iloc[0] == pytest.approx(d["removals"]["SiO2_to_Si"])
    note = slag_basis_note(r)
    tio2 = d["rules"]["tio2_to_slag"]           # setting_bmo.yml -> slag_balance.tio2_to_slag (False since 11 Nov 2026)
    assert "Si 0.60 %" in note and "keeps Si 0.38 %" in note
    assert ("TiO₂ in the slag" in note) is tio2 and ("TiO₂ left out of the slag" in note) is not tio2
    # the fuel-ash and dust rows page 9's editors show, on the plan
    F = fuel_ash_table(r)
    assert list(F["fuel"]) == ["Coke", "Nut coke", "PCI"]
    coke = F.iloc[0]
    assert coke["dry_kg"] == pytest.approx(r["out"]["K"]) and coke["wet_kg"] == pytest.approx(r["out"]["K"] / (1 - coke["moist_pct"] / 100))
    assert coke["ash_kg"] == pytest.approx(coke["dry_kg"] * coke["ash_pct"] / 100) and coke["slag_kg"] == pytest.approx(d["by_fuel"]["coke"]["slag_kg"])
    assert coke["sio2_pct"] == pytest.approx(eng.case0["coke"]["ash_comp"]["SiO2"])
    assert F["slag_kg"].sum() == pytest.approx(d["rollup"]["fuel_ash"], rel=1e-9)
    D = dust_table(r)
    assert list(D["id"]) == [x["id"] for x in eng.slag_basis["dust"] if x.get("enabled", True)]
    assert D["kg_per_thm"].sum() == pytest.approx(d["dust_kg"]) and D.iloc[0]["kg_per_charge"] == eng.slag_basis["dust"][0]["kg_per_charge"]
    assert D.iloc[0]["cao_pct"] == eng.slag_basis["dust"][0]["CaO_pct"] and D.iloc[0]["fe_kg"] == pytest.approx(d["dust"][D.iloc[0]["id"]]["fe_kg"])
    # without the basis: no breakup, empty tables, the material rows without Fe and slag
    e = Engine(materials=UAT, cal=CASE_CAL)
    r0 = e.min_cost(prod_t_day=2270.0, s=e.cal_settings(), L=Limits(), fuel=True)
    assert r0["out"]["slag_detail"] is None and slag_components_table(r0).empty and slag_basis_note(r0) is None
    assert fuel_ash_table(r0).empty and dust_table(r0).empty
    m0 = material_rows(r0, names, prices)
    assert m0 and all(m["fe_kg"] is None and m["slag_kg"] is None and m["kg_per_charge"] > 0 for m in m0)


def test_basis_from_page_9s_configuration():
    b = slag_basis(BMO, None, charge_t=26.46)
    r = b["rules"]
    assert (r["Si_pct"], r["C_pct"], r["S_pct"]) == (0.6, 4.2, 0.03)
    assert r["fe_to_pig_iron_fraction"] == 0.999 and r["alkali_to_slag_fraction"] == 0.8
    assert r["fe_to_feo_factor"] == pytest.approx(72 / 56, abs=1e-6)
    assert r["tio2_to_slag"] is bool(BMO["slag_balance"].get("tio2_to_slag", True))   # False since 11 Nov 2026
    # the plant's burden-sheet row, constant (user decision 11 Nov 2026); the two measured streams disabled
    assert [(d["id"], d["enabled"], d["kg_per_charge"]) for d in b["dust"]] == [
        ("bf_gas_dust", True, 530.0), ("dust_catcher", False, 229.0), ("gcp_dust", False, 235.0)]
    assert (b["dust"][0]["CaO_pct"], b["dust"][0]["MgO_pct"], b["dust"][0]["Fe_pct"]) == (5.63, 3.57, 33.5)
    assert BMO["data_sources"]["dust_quantity_source"] == "configured"
    assert b["fuel"]["coke"]["S_pct"] == 0.72 and b["fuel"]["pci"]["alk_pct"] == pytest.approx(2.19, abs=0.01)
    assert b["fuel"]["pci"]["MnO_pct"] == pytest.approx(0.05 / MN_IN_MNO)
    # operator inputs and (when read) the DPR lay over the configuration as on page 9
    prefs = dict(hot_metal_chemistry=dict(silicon_pct=0.45), dust_editor=dict(rows=dict(bf_gas_dust=dict(mgo_pct=0.9))))
    rates = dict(dust_catcher=dict(kg_per_charge=201.26, first_day="2026-10-12", last_day="2026-11-10", days=28))
    b2 = slag_basis(BMO, prefs, dust_rates=rates, charge_t=26.46)
    assert b2["rules"]["Si_pct"] == 0.45
    assert b2["dust"][0]["kg_per_charge"] == 530.0 and b2["dust"][0]["MgO_pct"] == 0.9
    assert b2["dust"][1]["kg_per_charge"] == 201.3 and "DPR" in b2["dust"][1]["source"]   # a DPR rate only lands on its stream
    assert "Si 0.45 %" in describe_slag_basis(b2) and "530 kg/charge" in describe_slag_basis(b2)


def test_basis_travels_with_the_inputs(basis):
    mats = [dict(m, carried=True, selected=True) for m in copy.deepcopy(REF["materials"])]
    fl = [dict(f, carried=True) for f in copy.deepcopy(REF["fluxes"])]
    inp = build_engine_inputs(mats, fl, REF["fuel"], slag_basis=basis)
    assert inp["slag_basis"]["rules"]["Si_pct"] == 0.6
    assert "slag_basis" not in build_engine_inputs(mats, fl, REF["fuel"])
    e = Engine(inp, cal=CASE_CAL, materials=UAT)
    assert e._spec["inputs"]["slag_basis"] == basis       # rebuilt the same in a solver process
    assert np.isfinite(e.evaluate(e.start_point(e.cal_settings(), 94.0), e.cal_settings(), 94.0)["slag"])
