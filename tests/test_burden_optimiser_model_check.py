"""Page 11 additions: the model check (burnt coke against the balances), today's charged blend and its comparison
with the plan, and the operator labels of the LP's limit rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.burden_engine.inputs import charged_blend
from utils.burden_engine import Engine, load_register
from utils.burden_engine import model_check as mc
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH
from utils.burden_engine.scenarios import (blend_compare_table, flux_compare_table, limit_label, limit_labels,
                                           limits_from, run_fixed, run_today, settings_from_regime)

THERMAL = dict(HMT=1470.0, hm_Si=0.3836, pci_kg_thm=190.0, nut_kg_thm=70.0)
TODAY = dict(sinter_sp_02=60.8, lloyds_clo=14.1, geomin_clo=9.7, nmdc_rom=5.3, lloyds_pellet=10.1)


# ---------------------------------------------------------------------------------------------- model check
def _shifts(n_days: int = 20, bias: float = 0.0, noise: float = 3.0, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-09-20", periods=n_days, freq="D")
    rows = []
    for d in days:
        for k, (sh, h) in enumerate((("A", 10), ("B", 18), ("C", 26))):
            burnt = 310 + rng.normal(0, 6)
            rows.append(dict(skey=f"{d:%Y-%m-%d}_{sh}", sdate=d, t_mid=d + pd.Timedelta(hours=h), stable=k != 2,
                             K_burnt=burnt, K_energy=burnt + bias + rng.normal(0, noise),
                             K_obal=burnt + 2 + rng.normal(0, noise), K_charged=burnt - 8, HMT=1490.0, hm_si=0.6,
                             prod_th=92.0))
    return pd.DataFrame(rows).set_index("skey")


def test_scored_shifts_need_burnt_and_a_prediction():
    T = _shifts(3)
    T.iloc[0, T.columns.get_loc("K_burnt")] = np.nan
    T.iloc[1, T.columns.get_loc("K_energy")] = np.nan
    T.iloc[1, T.columns.get_loc("K_obal")] = np.nan
    T.iloc[2, T.columns.get_loc("K_energy")] = np.nan          # oxygen balance alone still counts
    S = mc.scored_shifts(T)
    assert len(S) == len(T) - 2 and T.index[2] in S.index
    assert mc.scored_shifts(None).empty and mc.scored_shifts(pd.DataFrame()).empty


def test_window_latest_and_age():
    S = mc.scored_shifts(_shifts(20))
    W = mc.window(S, 7)
    assert pd.to_datetime(W["sdate"]).nunique() == 7 and W["sdate"].max() == S["sdate"].max()
    row = mc.latest(S)
    assert row.name.endswith("_C") and row["sdate"] == S["sdate"].max()
    assert mc.age_days(row, today=row["sdate"] + pd.Timedelta(days=3)) == 3
    assert mc.age_days(None, today=pd.Timestamp("2026-10-20")) is None


def test_summary_scores_stable_shifts_only():
    T = _shifts(14, bias=0.0)
    T.loc[~T["stable"], "K_energy"] += 50.0                     # a bad non-stable shift must not count
    e = mc.summary(mc.scored_shifts(T), "K_energy")
    assert e["n"] == 28 and abs(e["bias"]) < 2.0 and e["sd"] < 5.0
    assert mc.summary(mc.scored_shifts(T), "K_energy", stable_only=False)["bias"] > 10.0


def test_verdict_states():
    th = mc.Thresholds.from_config(dict(ok_bias_kgthm=8, ok_sd_kgthm=12, min_shifts=6))
    ok = mc.verdict(mc.summary(mc.scored_shifts(_shifts(14)), "K_energy"), th)
    assert ok[0] == "ok" and "stable shifts" in ok[1]
    biased = mc.verdict(mc.summary(mc.scored_shifts(_shifts(14, bias=15.0)), "K_energy"), th)
    assert biased[0] == "watch" and "bias +1" in biased[1] and "beyond ±8" in biased[1]
    noisy = mc.verdict(mc.summary(mc.scored_shifts(_shifts(14, noise=20.0)), "K_energy"), th)
    assert noisy[0] == "watch" and "shift-to-shift" in noisy[1]
    few = mc.verdict(mc.summary(mc.scored_shifts(_shifts(2)), "K_energy"), th)
    assert few[0] == "insufficient"
    assert mc.Thresholds.from_config(None) == mc.Thresholds()


def test_page_config_carries_the_model_check():
    import yaml

    from data.burden_engine.inputs import repo_path
    cfg = yaml.safe_load(open(repo_path("src/config/burden_optimiser.yml"), encoding="utf-8"))
    cfg = cfg.get("burden_optimiser", cfg)
    th = mc.Thresholds.from_config(cfg["model_check"])
    assert th.bias_kgthm == 8.0 and th.sd_kgthm == 12.0 and cfg["model_check"]["default_days"] in (7, 14, 30)


# ---------------------------------------------------------------------------------------------- today's blend
def _charges() -> pd.DataFrame:
    idx = pd.date_range("2026-10-18T16:00:00Z", periods=40, freq="h")
    df = pd.DataFrame(index=idx)
    df["sinter_3_mt"] = 16.0
    df["ore_2_mt"] = 3.0
    df["ore_3_mt"] = 2.0
    df["pellet_1_mt"] = 2.5
    df["ore_13_mt"] = 0.0
    df["ore_99_mt"] = 1.5                       # charged, but not in the yard
    df["coke_1_mt"] = 6.0                       # not ferrous
    df.iloc[:16, df.columns.get_loc("ore_2_mt")] = 9.0       # older than 24 h: left out
    return df


def test_charged_blend_shares_of_the_last_24_hours():
    mats = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))["materials"]
    b = charged_blend(_charges(), mats, hours=24)
    total = 16 + 3 + 2 + 2.5 + 1.5
    assert b["charges"] == 24 and b["ferrous_t"] == pytest.approx(24 * total)
    assert b["shares"]["sinter_sp_02"] == pytest.approx(100 * 16 / total)
    assert b["shares"]["lloyds_clo"] == pytest.approx(100 * 3 / total)
    assert b["unmapped"] == {"ore_99": pytest.approx(100 * 1.5 / total)}
    assert sum(b["shares"].values()) + sum(b["unmapped"].values()) == pytest.approx(100.0)
    assert b["window"] == ("19 Oct 13:30", "20 Oct 12:30")      # IST


def test_charged_blend_accepts_naive_utc_and_empty():
    mats = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))["materials"]
    df = _charges()
    naive = charged_blend(df.tz_localize(None), mats)
    assert naive["shares"] == pytest.approx(charged_blend(df, mats)["shares"])
    assert charged_blend(pd.DataFrame(), mats)["shares"] == {}


@pytest.fixture(scope="module")
def plans():
    eng = Engine()
    s = settings_from_regime(None, THERMAL, eng.cal)
    L = limits_from(load_register())
    inputs = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))
    r = run_fixed(eng, 2270.0, s, L)["with_fuel"]
    return eng, s, L, inputs, r


def test_today_at_the_plans_shares_costs_the_plan(plans):
    eng, s, L, inputs, r = plans
    t = run_today(inputs, eng.cal, r["share_wet"], 2270.0, s, L)
    assert t["feasible"]
    for i, sh in r["share_wet"].items():
        assert t["share_wet"].get(i, 0.0) == pytest.approx(sh, abs=0.11)
    assert t["cost"]["total"] == pytest.approx(r["cost"]["total"], abs=10)
    assert t["out"]["K"] == pytest.approx(r["out"]["K"], abs=0.5)


def test_todays_blend_is_dearer_than_the_plan(plans):
    eng, s, L, inputs, r = plans
    t = run_today(inputs, eng.cal, TODAY, 2270.0, s, L)
    total = sum(TODAY.values())
    for i, sh in TODAY.items():
        assert t["share_wet"][i] == pytest.approx(100 * sh / total, abs=0.06)
    assert t["P_tday"] == pytest.approx(2270.0, abs=0.5)
    assert t["cost"]["total"] > r["cost"]["total"]          # the plan is the cheapest blend at this production
    assert run_today(inputs, eng.cal, {}, 2270.0, s, L) is None


def test_compare_tables(plans):
    eng, s, L, inputs, r = plans
    t = run_today(inputs, eng.cal, TODAY, 2270.0, s, L)
    names = {v.id: v.name for v in eng.vars}
    B = blend_compare_table(r, [("Today", t)], names, {v.id: v.price for v in eng.vars})
    assert list(B.columns) == ["Material", "Plan, wet %", "Today, wet %", "Plan, kg/tHM", "Today, kg/tHM", "Plan, t/d",
                               "Price, Rs/t"]
    assert B["Plan, wet %"].sum() == pytest.approx(100.0, abs=0.01)
    assert B["Today, wet %"].sum() == pytest.approx(100.0, abs=0.01)
    assert "NMDC ROM" in set(B["Material"])                   # used only today: still a row
    F = flux_compare_table(r, [("Today", t)], names, [v.id for v in eng.flux])
    assert list(F.columns)[:1] == ["Flux"] and (F.drop(columns="Flux").max(axis=1) >= 0.5).all()


# ---------------------------------------------------------------------------------------------- labels
def test_limit_labels_for_operators():
    names = {"lloyds_pellet": "LLOYDS PELLET"}
    assert limit_label("min share lloyds_pellet", names) == "LLOYDS PELLET share floor"
    assert limit_label("max share nmdc_rom") == "nmdc_rom share cap"
    assert limit_label("basicity min") == "CaO/SiO₂ floor" and limit_label("slag cap") == "slag rate cap"
    assert limit_label("unknown row") == "unknown row"
    rows = ["basicity min", "min share lloyds_pellet", "charging"]
    assert limit_labels(rows, names) == "CaO/SiO₂ floor, charging rate"
    assert limit_labels(rows, names, shares=True) == "CaO/SiO₂ floor, LLOYDS PELLET share floor, charging rate"
