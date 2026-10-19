"""Coke Energy Model page parts: freshness notices, the slag-temperature rule, display smoothing and the figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ui.coke_energy.figures import parity_figure, prediction_figure, residual_figure, rist_animation_figure
from utils.coke_energy.accuracy import rolling_mean, shift_timeline
from utils.coke_energy.bf_balance import DEFAULTS, calibrate_loss
from utils.coke_energy.freshness import assess, format_age, in_use_between, materials_for, parse_limits
from utils.coke_energy.pipeline import PipelineSettings, shift_params, slag_superheat_c
from utils.coke_energy.rist_diagram import shift_rist
from utils.coke_energy.run_shifts import case_from_shift, shift_balance

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "coke_energy"
RULE = ((16.0, 30.0), (21.0, 60.0))


# --------------------------------------------------------------------------------------------- freshness
def _samples() -> pd.DataFrame:
    rows = [
        ("rm_chemistry", "ore_2", "2026-10-14 07:00"),
        ("rm_chemistry", "ore_2", "2026-10-13 07:00"),
        ("rm_chemistry", "coke_2", "2026-09-25 23:00"),
        ("rm_chemistry", "pci_1", "2026-10-15 07:00"),
        ("ash_chemistry", "coke", "2026-09-30 00:00"),   # date 29 Sep, stamped at its end
        ("hm_slag", "", "2026-10-15 06:09"),
        ("hm_slag", "", "2026-10-15 13:30"),            # after the reference: must not count
        ("fines", "pellet_1", "2026-10-01 00:00"),
        ("fines", "ore_2", "2026-10-15 01:00"),
    ]
    return pd.DataFrame(rows, columns=["source", "material", "t"]).assign(t=lambda d: pd.to_datetime(d.t))


def test_freshness_names_the_last_valid_point_and_flags_ages_past_the_limit() -> None:
    ref = pd.Timestamp("2026-10-15 10:00")
    a = assess(_samples(), ref, parse_limits({}), {"ore_2", "coke_2", "pellet_1"})
    by = a.set_index(["source", "material"])
    assert by.loc[("rm_chemistry", "ore_2"), "last"] == pd.Timestamp("2026-10-14 07:00")
    assert bool(by.loc[("rm_chemistry", "ore_2"), "stale"])                          # 27 h > 1 day
    assert not bool(by.loc[("rm_chemistry", "pci_1"), "stale"])                      # PCI is checked whenever injected
    assert bool(by.loc[("rm_chemistry", "coke_2"), "stale"])
    assert by.loc[("hm_slag", ""), "last"] == pd.Timestamp("2026-10-15 06:09")       # the 13:30 sample is later than ref
    assert bool(by.loc[("hm_slag", ""), "stale"])                                    # 3.85 h > 3 h
    assert bool(by.loc[("ash_chemistry", "coke"), "stale"])                          # 15.4 days > 15 days
    assert ("fines", "pellet_1") not in by.index                                     # pellet size is fixed
    assert not bool(by.loc[("fines", "ore_2"), "stale"])
    assert bool(by.loc[("dpr", ""), "stale"]) and pd.isna(by.loc[("dpr", ""), "last"])  # no data at all
    assert format_age(pd.Timedelta(hours=27)) == "27.0 h" and format_age(pd.Timedelta(days=3)) == "3.0 days"


def test_limits_and_materials_in_use() -> None:
    limits = parse_limits({"hm_slag": "2h", "ash_chemistry": "15d"})
    assert limits["hm_slag"] == pd.Timedelta(hours=2) and limits["ash_chemistry"] == pd.Timedelta(days=15)
    assert limits["burden_distribution"] == pd.Timedelta(days=30)
    hours = pd.date_range("2026-10-15 00:00", periods=4, freq="h")
    charged = pd.DataFrame({"charges": [6, 6, 6, 6], "ore_2": [3.0, 0, 0, 0], "coke_2": [0, 0, 1.4, 0], "ore_6": 0.0},
                           index=hours)
    assert in_use_between(charged, hours[1], hours[3]) == {"coke_2"}
    assert in_use_between(charged, hours[0], hours[3] + pd.Timedelta(hours=1)) == {"ore_2", "coke_2"}
    assert materials_for("ash_chemistry", {"coke_1", "nut_coke_1"}, set()) == {"coke", "nutcoke", "pci"}
    assert materials_for("rm_strength", {"coke_1", "ore_2"}, {"coke_1", "sinter_3"}) == {"coke_1"}


# --------------------------------------------------------------------------------------------- slag temperature
def test_slag_superheat_rule() -> None:
    assert slag_superheat_c(16.0, RULE) == pytest.approx(30.0)
    assert slag_superheat_c(18.5, RULE) == pytest.approx(45.0)
    assert slag_superheat_c(14.0, RULE) == pytest.approx(30.0) and slag_superheat_c(23.0, RULE) == pytest.approx(60.0)
    assert slag_superheat_c(18.5, None) == DEFAULTS["slag_T_above_HM"]
    assert slag_superheat_c(float("nan"), RULE) == DEFAULTS["slag_T_above_HM"]


def test_slag_rule_reaches_the_balance() -> None:
    rows = pd.read_csv(FIXTURES / "shift_inputs.csv", index_col=0)
    k = rows.index[-1]
    params = shift_params(rows, rows.index, PipelineSettings(slag_superheat=RULE))
    expected = slag_superheat_c(rows.loc[k, "slag_pct_al2o3"], RULE)
    assert params[k]["slag_T_above_HM"] == pytest.approx(expected)
    hot = shift_balance(rows.loc[k], {**DEFAULTS, "slag_T_above_HM": 60.0})
    cold = shift_balance(rows.loc[k], {**DEFAULTS, "slag_T_above_HM": 30.0})
    slag = hot["slag_kgthm_Al2O3tracer"]
    # 30 degC more slag superheat = slag mass x 1.2 MJ/t.K x 30 K more heat leaving with the slag
    assert hot["E_out_slag"] - cold["E_out_slag"] == pytest.approx(slag * DEFAULTS["slag_cp"] * 30 / 1000, rel=1e-9)
    assert PipelineSettings.from_config({"slag_superheat": {"low": {"al2o3_pct": 16, "superheat_c": 30},
                                                            "high": {"al2o3_pct": 21, "superheat_c": 60}}}).slag_superheat == RULE
    assert PipelineSettings.from_config({"slag_superheat": None}).slag_superheat is None


# --------------------------------------------------------------------------------------------- smoothing
def test_rolling_mean_is_centred_weighted_and_keeps_gaps() -> None:
    grid = shift_timeline(pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-03"))
    T = pd.DataFrame(index=grid.index).assign(sdate=lambda d: pd.to_datetime([k[:10] for k in d.index]), t_mid=grid.t_mid,
                                              hm_t=700.0, K=np.arange(9, dtype=float) * 10 + 300)
    T.loc["2026-10-02_A", "hm_t"] = 1400.0
    T = T.drop(index=["2026-10-03_A", "2026-10-03_B"])
    raw = rolling_mean(T, 1, ["K"])
    assert raw["K"].isna().sum() == 2 and len(raw) == 9
    day = rolling_mean(T, 3, ["K"])
    # centred on 2026-10-01_C: shifts B, C, next A (A has twice the hot metal)
    assert day["K"].iloc[2] == pytest.approx((310 * 700 + 320 * 700 + 330 * 1400) / (700 + 700 + 1400))
    assert np.isnan(day["K"].iloc[7])                         # only 1 of 3 shifts present around 03_B
    assert day.index[0] == grid.t_mid.iloc[0]


# --------------------------------------------------------------------------------------------- figures
@pytest.fixture(scope="module")
def table_and_frames():
    rows = pd.read_csv(FIXTURES / "shift_inputs.csv", index_col=0)
    keys = list(rows.index)
    p = dict(DEFAULTS)
    T = pd.DataFrame(index=keys)
    T["sdate"] = pd.to_datetime([k[:10] for k in keys])
    T["shift"] = [k[-1] for k in keys]
    T["t_mid"] = T.sdate + pd.to_timedelta(T["shift"].map({"A": 10, "B": 18, "C": 26}), unit="h")
    T["hm_t"] = 700.0
    frames = []
    for k in keys:
        R = shift_balance(rows.loc[k])
        case = case_from_shift(rows.loc[k]); case["eta_H2"] = R["eta_H2"] / 100
        K = R["coke_kgthm_dry"] - 5
        _, V, _ = calibrate_loss(case, K, p, eta_CO=R["eta_CO"] / 100)
        frames.append(shift_rist(case, K, V, R["eta_CO"] / 100, prod_th=R["prod_th"], omega_fixed=0.79, Q_LZ_GJh_fixed=40.0,
                                 Q_UZ_GJh_fixed=20.0, r_H_fixed=1.0, skey=k))
        T.loc[k, ["K_charged", "K_burnt", "K_obal", "K_energy"]] = [R["coke_kgthm_dry"], K, K + 2, K - 3]
        T.loc[k, "omega_fixed"] = 0.79
    return T, frames


def test_figures_build(table_and_frames) -> None:
    T, frames = table_and_frames
    cols = ["K_charged", "K_burnt", "K_obal", "K_energy"]
    raw = rolling_mean(T, 1, cols)
    for fig in (prediction_figure(raw, None), prediction_figure(raw, rolling_mean(T, 3, cols), show_charged=True,
                                                                smoothing_label="24 h mean"),
                parity_figure(T), residual_figure(T)):
        assert isinstance(fig, go.Figure) and len(fig.data) > 0
    fig = rist_animation_figure(frames, T, raw)
    assert len(fig.frames) == len(frames)
    assert fig.layout.sliders[0].active == len(frames) - 1
    animated = fig.frames[0].traces
    assert len(fig.frames[0].data) == len(animated)
    # the shift line of the last frame is the drawn one, and it ends at the top-gas point
    a = frames[-1].actual
    line = fig.data[2]
    assert line.x[-1] == pytest.approx(a.x_top) and line.y[-1] == pytest.approx(a.y_top_c)
