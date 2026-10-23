"""Page 11: burnt against charged coke (utils.coke_energy.transit), the page-9-style plan tiles, the shared slag
bands and the furnace commentary context (utils.burden_engine.commentary)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from utils.burden_engine import Engine, load_register
from utils.burden_engine import model_check as mc
from utils.burden_engine.commentary import (NOTES_PATH, SYSTEM_PROMPT, build_context, describe_transit, facts_table,
                                            load_notes)
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH
from utils.burden_engine.scenarios import (limits_from, plan_tiles, run_fixed, run_today, settings_from_regime,
                                           value_of_limits)
from utils.coke_energy import transit as tr
from utils.slag_bands import PLANT_SLAG_BANDS, band_note

THERMAL = dict(HMT=1470.0, hm_Si=0.3836, pci_kg_thm=190.0, nut_kg_thm=70.0)
TODAY = dict(sinter_sp_02=60.8, lloyds_clo=14.1, geomin_clo=9.7, nmdc_rom=5.3, lloyds_pellet=10.1)


# ---------------------------------------------------------------------------------------------- transit
def _shift_table(n_days: int = 60, lag_shifts: int = 0, noise: float = 4.0, offset: float = 0.0, seed: int = 3):
    """Charged coke as a random walk around 305; burnt = charged ``lag_shifts`` earlier + noise + offset."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-08-01", periods=n_days, freq="D")
    keys = [(d, s) for d in days for s in "ABC"]
    k = 305 + np.cumsum(rng.normal(0, 3, len(keys)))
    k = k - (k.mean() - 305)
    b = np.r_[np.full(lag_shifts, np.nan), k[:len(k) - lag_shifts]] + rng.normal(0, noise, len(keys)) + offset
    idx = [f"{d:%Y-%m-%d}_{s}" for d, s in keys]
    return pd.DataFrame(dict(sdate=[d for d, _ in keys], t_mid=[d + pd.Timedelta(hours={"A": 10, "B": 18, "C": 26}[s])
                                                                 for d, s in keys],
                             K_charged=k, K_burnt=b, hm_t=730.0, stable=True), index=idx)


def test_concordance_is_one_only_for_identical_series():
    x = np.linspace(280, 330, 50)
    assert tr.concordance(x, x)["ccc"] == pytest.approx(1.0)
    shifted = tr.concordance(x, x + 10)            # perfect correlation, wrong level
    assert shifted["r"] == pytest.approx(1.0) and shifted["ccc"] < 0.85 and shifted["bias"] == pytest.approx(10)
    scaled = tr.concordance(x, 305 + 1.5 * (x - 305))
    assert scaled["slope"] == pytest.approx(1.5) and scaled["cb"] < 1.0
    assert np.isnan(tr.concordance([1.0], [1.0])["ccc"])


def test_window_agreement_improves_with_the_window():
    T = _shift_table(noise=8.0)
    W = tr.window_agreement(T).set_index("window")
    assert W.loc["Shift (8 h)", "n"] == len(T)
    assert W.loc["1 week", "sd"] < W.loc["1 day", "sd"] < W.loc["Shift (8 h)", "sd"]
    assert W.loc["1 week", "ccc"] > W.loc["Shift (8 h)", "ccc"]
    split = tr.decompose_window_sd(W.reset_index())
    assert split["random_sd"] == pytest.approx(8.0, rel=0.25) and split["floor_sd"] < 3.0


def test_lag_scan_finds_the_shift_the_burnt_coke_follows():
    T = _shift_table(lag_shifts=1, noise=2.0)
    scan = tr.lag_scan(T).set_index("lag_shifts")
    assert scan["r"].idxmax() == 1
    assert tr.peak_lag_hours(scan.reset_index()) == pytest.approx(8.0, abs=4.0)
    assert tr.lag_scan(_shift_table(noise=2.0)).set_index("lag_shifts")["r"].idxmax() == 0


def test_closure_uses_raw_burnt_and_tonnes():
    T = _shift_table(noise=0.0, offset=-9.0).rename(columns={"K_burnt": "K_burnt_raw"})
    c = tr.closure(T)
    assert c["ratio"] == pytest.approx(1 - 9 / 305, abs=0.002)
    assert c["cumulative"].iloc[-1] == pytest.approx(9 * 730 / 1000 * len(T), rel=1e-6)
    assert tr.summary(T.rename(columns={"K_burnt_raw": "K_burnt"}))["closure"] is None


def test_summary_and_its_commentary_lines():
    S = tr.summary(_shift_table(noise=6.0))
    text = describe_transit(S)
    assert "Shift (8 h)" in text and "CCC" in text and "6-7 h" in text


# ---------------------------------------------------------------------------------------------- tiles and bands
@pytest.fixture(scope="module")
def plans():
    eng = Engine()
    s = settings_from_regime(None, THERMAL, eng.cal, pci_plant=190.0, nut_plant=70.0)  # case report
    L = limits_from(load_register())
    inputs = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))
    res = run_fixed(eng, 2270.0, s, L)
    today = run_today(inputs, eng.cal, TODAY, 2270.0, s, L)
    return eng, s, L, inputs, res, today


def test_plan_tiles_match_the_engine_result(plans):
    eng, s, L, inputs, res, today = plans
    r = res["with_fuel"]
    P = plan_tiles(r, [v.id for v in eng.flux])
    assert P["cost"]["total"] == pytest.approx(P["cost"]["ore"] + P["cost"]["flux"] + P["cost"]["fuel"])
    assert P["fuel"]["coke_dry"] == pytest.approx(r["out"]["K"]) and P["fuel"]["total"] == pytest.approx(r["out"]["fuel_rate"])
    o = r["out"]
    assert P["slag"]["b2"] == pytest.approx(o["B2"], abs=1e-6)
    assert P["slag"]["t_basicity"] == pytest.approx((o["CaO"] + o["MgO"]) / o["SiO2"])
    assert P["slag"]["ib4"] == pytest.approx((o["CaO"] + o["MgO"]) / (o["SiO2"] + o["Al2O3"]))
    assert P["slag"]["t_day"] == pytest.approx(o["slag"] * 2270 / 1000)
    assert P["charging"]["hm_per_charge"] == pytest.approx(r["P_th"] / r["charges_per_h"])
    assert P["charging"]["flux_kg"] == pytest.approx(r["wet_kg"]["limestone"])
    assert P["charging"]["coke_t"] == pytest.approx(o["K_wet"] * 2.27)


def test_slag_bands_are_shared_with_page_9():
    assert PLANT_SLAG_BANDS["b2"] == (1.005, 1.155)
    assert band_note(1.05, 1.005, 1.155) == ("✓ plant 1.005–1.155", True)
    assert band_note(1.20, 1.005, 1.155)[1] is False
    import importlib.util
    spec = importlib.util.find_spec("ui.bmo.components")
    src = open(spec.origin, encoding="utf-8").read()
    assert "from utils.slag_bands import PLANT_SLAG_BANDS, band_note" in src


# ---------------------------------------------------------------------------------------------- commentary
def test_notes_document_ships_with_the_app():
    notes = load_notes()
    assert NOTES_PATH.exists() and notes.startswith("# Burden & Fuel Optimiser")
    for heading in ("How coke is computed", "Model check", "Burnt coke against charged coke", "What the engine does not see"):
        assert heading in notes
    assert load_notes("/nonexistent/notes.md") == ""


def test_facts_table_lists_every_fact():
    reg = load_register()
    table = facts_table(reg)
    rows = [ln for ln in table.splitlines() if ln.startswith("| ") and not ln.startswith("| ID")]
    assert len(rows) == len(reg.facts())
    assert any(ln.startswith("| COKE-17 |") and "Retired" in ln for ln in rows)
    assert all(ln.count("|") == 11 for ln in rows)          # no stray pipes inside cells


def _shown():
    T = _shift_table(noise=5.0)
    T["K_energy"] = T["K_burnt"] - 1.0
    T["K_obal"] = T["K_burnt"] + 2.0
    T["HMT"], T["hm_si"], T["prod_th"] = 1490.0, 0.5, 91.0
    sc = mc.scored_shifts(T)
    e, o = mc.summary(mc.window(sc, 14), "K_energy"), mc.summary(mc.window(sc, 14), "K_obal")
    state, reason = mc.verdict(e, mc.Thresholds())
    return dict(latest=mc.latest(sc), latest_name="29 Sep · shift C", period="14 days", energy=e, oxygen=o,
                state=state, reason=reason, age_days=1, transit=tr.summary(T))


def test_commentary_context_carries_notes_facts_and_the_run(plans):
    eng, s, L, inputs, res, today = plans
    r, r0 = res["with_fuel"], res["ore_only"]
    names = {v.id: v.name for v in eng.vars}
    VL = value_of_limits(eng, 2270.0, s, L, r)
    ctx = build_context(
        notes=load_notes(), reg=load_register(), mode="Step 2 · Fixed production", plan=r,
        plan_label="fixed production 2,270 t/d", names=names, flux_ids=[v.id for v in eng.flux],
        settings=r["settings"], limits=r["limits"], yard_source="live",
        calibration=dict(source="live", window="2026-10-05 to 2026-10-18", omega=0.8, Q_LZ_GJh=120.0, Q_UZ_GJh=30.0),
        target_t_day=2270.0, model_check=_shown(), today=today, today_window="from 18 Oct 22:04 to 19 Oct 21:57 IST",
        ore_only=r0, value_of_limits=VL, materials=inputs["materials"], fluxes=inputs["fluxes"], fuel=inputs["fuel"])
    t = ctx.text
    assert ctx.is_usable and ctx.missing == []
    order = [t.index(x) for x in ("# PART 1 - OPTIMISER NOTES", "# PART 2 - FACT REGISTER", "# PART 3 - LIVE RUN",
                                   "## MODEL CHECK", "## BURNT COKE AGAINST CHARGED COKE", "## OPTIMISED PLAN",
                                   "## PLAN AGAINST TODAY'S CHARGED BLEND", "## VALUE OF LIMITS", "## YARD")]
    assert order == sorted(order)
    d = today["cost"]["total"] - r["cost"]["total"]
    assert f"Plan costs Rs {abs(d):,.0f}/tHM less than today's charged blend" in t
    assert f"TOTAL {r['cost']['total']:,.0f}" in t and "CaO/SiO₂ floor" in t and "| BAND-01 |" in t
    assert "No saving (< Rs 0.5/tHM) from:" in t


def test_commentary_context_names_what_is_missing(plans):
    eng, s, L, inputs, res, today = plans
    r = res["with_fuel"]
    ctx = build_context(notes="", reg=load_register(), mode="Step 2 · Fixed production", plan=r, plan_label="x",
                        names={}, flux_ids=[], settings=r["settings"], limits=r["limits"], yard_source="reference",
                        calibration={}, model_check=None, today=None)
    assert set(ctx.missing) >= {"the optimiser notes", "the model check (Coke Energy shift results)",
                                "today's charged blend", "the live yard (reference yard of 16 Oct 2026 used)"}
    assert "## NOT AVAILABLE FOR THIS RUN" in ctx.text
    none = build_context(notes="n", reg=load_register(), mode="m", plan=None, plan_label="", names={}, flux_ids=[],
                         settings=r["settings"], limits=r["limits"], yard_source="live", calibration={})
    assert not none.is_usable


def test_system_prompt_keeps_the_page_9_discipline():
    for rule in ("Use ONLY the figures supplied", "ACTION, REASON and MAGNITUDE", "Start from the model check",
                 "**What to watch**"):
        assert rule in SYSTEM_PROMPT
