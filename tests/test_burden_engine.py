"""Burden engine 2.0 (#250) and fact register (#251).

Reference numbers are the engine runs behind *BF-02 blend optimiser with fuel cost and production: case report*
(16 Oct 2026): UAT yard and prices, PCI 190, nut coke 70, HMT 1,470 degC, Jul-Oct 2026 two-zone calibration.
The production cases take 10-50 s each; set BURDEN_ENGINE_FAST=1 to skip them.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace

import numpy as np
import pytest

from utils.burden_engine import BandError, Engine, Limits, RetiredFactError, Settings, load_register, summary
from utils.burden_engine.facts import REGISTER_PATH, SCHEMA_PATH, load_register_data
from utils.burden_engine.hydraulics import raft_c

UAT = ["sinter_sp_02", "geomin_clo", "lloyds_clo", "nmdc_rom", "lloyds_pellet"]
EXT = UAT + ["nmdc_donimalai", "gomti_clo", "lloyds_oversize_clo"]
P_PAGE = 2270.0 / 24
slow = pytest.mark.skipif(os.environ.get("BURDEN_ENGINE_FAST") == "1", reason="production cases are slow")


@pytest.fixture(scope="module")
def eng():
    return Engine(materials=UAT)


@pytest.fixture(scope="module")
def ext():
    return Engine(materials=EXT)


# ---------------------------------------------------------------------------------------------- fact register
def test_register_matches_schema():
    jsonschema = pytest.importorskip("jsonschema")
    data = load_register_data()
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        jsonschema.validate(data, json.load(fh))
    assert data["version"] == "1.2"
    groups = {}
    for f in data["facts"]:
        groups[f["group"]] = groups.get(f["group"], 0) + 1
    assert groups == {"COKE": 17, "GAS": 11, "FLOW": 14, "DATA": 10, "BAND": 3, "ASM": 13}


def test_retired_facts_are_refused():
    reg = load_register()
    for fid in ("COKE-17", "DATA-10"):
        assert reg[fid].retired
        with pytest.raises(RetiredFactError):
            reg.active(fid)
    assert all(not reg[f].retired for f in ("FLOW-02", "BAND-01", "ASM-11"))


def test_engine_rules_come_from_register(eng):
    reg = load_register()
    th = reg.thumb_rules()
    assert th["beta_pellet"] == pytest.approx(-0.0037)
    assert th["beta_gomti"] == pytest.approx(0.0038)
    assert th["beta_ore6_on"] == pytest.approx(-0.028)
    assert th["beta_o2"] == pytest.approx(0.0214)
    assert th["beta_coke"] == pytest.approx(-0.0005)
    assert th["n_flow"] == pytest.approx(1.8)
    assert eng.th == th
    # the dataclass defaults are the register's values
    assert Limits.from_register(reg) == Limits()


def test_no_hard_coded_rule_duplicates():
    src = open(os.path.join(os.path.dirname(REGISTER_PATH), "..", "utils", "burden_engine", "engine.py"), encoding="utf-8").read()
    for literal in ("-0.0037", "0.0038", "-0.028", "0.0214", "0.30 kg", "k_slag_extra", "2021.0"):
        assert literal not in src, literal


# ---------------------------------------------------------------------------------------------- Step 2 (fixed production)
def test_case1_page_objective(eng):
    r = eng.min_cost(P=P_PAGE, fuel=False)
    assert r["feasible"]
    assert r["cost"]["total"] == pytest.approx(30139, abs=5)
    sh = r["share_wet"]
    assert sh["sinter_sp_02"] == pytest.approx(58.0, abs=0.1)
    assert sh["lloyds_pellet"] == pytest.approx(10.0, abs=0.1)
    assert sh["geomin_clo"] == pytest.approx(13.4, abs=0.2)
    assert r["wet_kg"]["limestone"] == pytest.approx(13.7, abs=0.3)


def test_case2_fuel_added(eng):
    r = eng.min_cost(P=P_PAGE, fuel=True)
    o = r["out"]
    assert r["feasible"] and r["lp_iterations"] <= 5
    assert o["K"] == pytest.approx(305.7, abs=0.3)
    assert o["slag"] == pytest.approx(340.0, abs=0.1)
    assert o["fuel_rate"] == pytest.approx(567.0, abs=0.3)
    assert r["cost"]["total"] == pytest.approx(30139, abs=5)
    assert r["raft"] == pytest.approx(2214, abs=2)
    assert set(r["binding"]) >= {"slag cap", "basicity min"}
    assert r["warnings"] == []


def test_case2b_extended_yard_flags_extrapolation(ext):
    r = ext.min_cost(P=P_PAGE, fuel=True)
    assert r["cost"]["total"] == pytest.approx(29306, abs=5)
    assert r["share_wet"]["gomti_clo"] == pytest.approx(22.0, abs=0.2)
    assert any("Gomti" in w for w in r["warnings"])


def test_calibration_reproduced():
    """At the calibration burden the engine returns the calibration coke and blast (305.5 kg/tHM, 1,191 Nm3/tHM)."""
    e = Engine(materials=UAT)
    c = e.cal
    assert c["coke_model_cal"] == pytest.approx(305.5, abs=0.1)
    assert c["hb_per_t_model_cal"] == pytest.approx(1191.4, abs=0.5)


# ---------------------------------------------------------------------------------------------- bands
def test_band_02_03_refuse_inputs(eng):
    with pytest.raises(BandError):
        eng.min_cost(P=P_PAGE, s=Settings(HMT=1430.0))
    with pytest.raises(BandError):
        eng.min_cost(P=P_PAGE, s=Settings(hm_Si=0.3836 + 0.5))


def test_band_01_ideal_shaft_not_reported(eng):
    """The Rist ideal (omega 1.0) needs fuel below 520 kg/tHM: the fuel-band row makes it infeasible (COKE-14)."""
    r = eng.min_cost(P=P_PAGE, s=Settings(omega=1.0), fuel=True)
    assert not r["feasible"]
    assert r["viol"]["fuel rate min"] > 0


def test_band_01_large_lower_zone_loss_not_reported():
    e = Engine(materials=UAT)
    e.cal = copy.deepcopy(e.cal)
    e.cal["Q_LZ_GJh"] += 60.0
    r = e.min_cost(P=P_PAGE, fuel=True)
    assert not r["feasible"]
    assert r["viol"]["fuel rate max"] > 0


# ---------------------------------------------------------------------------------------------- RAFT (DATA-03/04)
def test_raft_plant_workbook_per_tonne(eng):
    s = Settings()
    base = eng.raft(s)
    assert eng.raft(replace(s, humidity_g=16.0)) - base == pytest.approx(-6.6, abs=0.3)   # DATA-03 at fixed blast per tonne
    assert eng.raft(replace(s, HBT=s.HBT + 100)) - base == pytest.approx(83.9, abs=0.5)
    assert raft_c(1200, 1190, 58, 200, 15) - raft_c(1200, 1190, 58, 190, 15) == pytest.approx(-25.0, abs=3)


def test_process_settings_respect_raft_max(eng):
    L = Limits(process=True, HBT_max=1300.0, RAFT_max=2250.0)
    s = eng.process_settings(L)
    assert eng.raft(s) <= 2250.0 + 1e-6
    assert s.HBT < 1300.0


# ---------------------------------------------------------------------------------------------- production modes
@slow
def test_case3_max_production(eng):
    r = eng.max_production(Limits(process=True))
    assert r["P_tday"] == pytest.approx(2412, abs=3)
    assert r["cost"]["total"] == pytest.approx(30789, abs=10)
    assert "charging" in r["binding"]
    assert r["share_wet"]["lloyds_pellet"] == pytest.approx(19.8, abs=0.3)
    assert any("Pellet" in w for w in r["warnings"])


@slow
def test_case5b_blast_limited_max_production(eng):
    r = eng.max_production(Limits(process=True, V_max=105000.0))
    assert r["P_tday"] == pytest.approx(2257, abs=3)
    assert r["O2_nm3h"] == pytest.approx(6490, abs=15)
    assert "blast volume" in r["binding"]


@slow
def test_case4_min_total_cost(eng):
    L = Limits(process=True)
    LV = list(np.linspace(Settings().O2_per_air, eng.hy.o2_per_air_from_tag(L.enrich_max), 9))
    r = eng.min_total_cost(L, P_lo=84.0, step=1.0, o2_levels=LV, curves=False)
    assert r["P_tday"] == pytest.approx(2290, abs=3)
    assert r["cost"]["total"] == pytest.approx(29935, abs=10)
    assert r["settings"]["HBT"] == pytest.approx(1230.0)


def test_summary_is_flat(eng):
    r = eng.min_cost(P=P_PAGE, fuel=True)
    s = summary(r, eng)
    assert all(not isinstance(v, (dict, list)) for v in s.values())
    assert s["fuel_rate"] == pytest.approx(567.0, abs=0.3)
