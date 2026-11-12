"""HMT tap filter (``utils.coke_energy.hmt_filter``): the recursion against its model, the band, the newest-tap check
and the back-test with publication times."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from utils.coke_energy import hmt_filter as hf

P = hf.Params()


def _simulate(n: int = 4000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Taps of the filter's own model at irregular intervals (1.2-3 h)."""
    rng = np.random.default_rng(seed)
    dt = rng.uniform(1.2, 3.0, n)
    t = np.cumsum(dt)
    level, dev = 1480.0, 0.0
    y = np.empty(n)
    for i in range(n):
        a = math.exp(-dt[i] / P.tau_h)
        level += rng.normal(0.0, math.sqrt(P.q_l_c2_per_h * dt[i]))
        dev = a * dev + rng.normal(0.0, P.s_d_c * math.sqrt(1 - a * a))
        y[i] = level + dev + rng.normal(0.0, P.r_c)
    return t, y


def test_the_innovations_are_standard_on_data_of_the_model() -> None:
    t, y = _simulate()
    steps = hf.run(t, y, P)
    z = np.array([s.z for s in steps[50:]])
    assert abs(z.mean()) < 0.05 and 0.95 < z.std() < 1.05
    # +-1 sigma holds about 68 % of the taps, +-2 sigma about 95 %
    assert 0.65 < np.mean(np.abs(z) <= 1) < 0.71 and 0.94 < np.mean(np.abs(z) <= 2) < 0.965


def test_the_band_widens_with_the_lead_to_the_long_run_spread() -> None:
    t, y = _simulate(400)
    state = hf.run(t, y, P)[-1].state
    now = hf.predict(state, state.t_h, P)
    assert now.mean == pytest.approx(state.level + state.dev)
    assert now.sd_tap**2 - now.sd_state**2 == pytest.approx(P.r_c**2)
    sds = [hf.predict(state, state.t_h + h, P).sd_tap for h in (0, 1, 2, 4, 6, 12, 48)]
    assert all(b > a for a, b in zip(sds, sds[1:], strict=False))
    far = hf.predict(state, state.t_h + 48, P)
    assert far.mean == pytest.approx(state.level, abs=0.01)  # the deviation has died out
    assert far.sd_tap == pytest.approx(math.sqrt(state.p_ll + P.q_l_c2_per_h * 48 + P.s_d_c**2 + P.r_c**2), rel=1e-3)
    assert hf.predict(state, state.t_h - 5, P) == now  # never before the state


def test_the_chance_of_a_cold_tap_is_the_normal_tail() -> None:
    assert hf.p_below(1480.0, 10.0, 1480.0) == pytest.approx(0.5)
    assert hf.p_below(1480.0, 10.0, 1460.0) == pytest.approx(0.02275, abs=1e-4)
    assert hf.p_below(1480.0, 0.0, 1460.0) == 0.0 and hf.p_below(1450.0, 0.0, 1460.0) == 1.0


def test_the_newest_tap_is_checked_against_two_and_one_sigma() -> None:
    state = hf.State(0.0, 1480.0, 0.0, 1.0, 0.0, 1.0)

    def step(value: float) -> hf.Step:
        return hf.Step(1480.0, 10.0, value, state)

    assert hf.tap_check([])["state"] == "unknown"
    assert hf.tap_check([step(1485.0), step(1495.0)])["state"] == "ok"
    single = hf.tap_check([step(1480.0), step(1455.0)])
    assert single["state"] == "watch" and single["rule"] == "single" and single["z"] == pytest.approx(-2.5)
    pair = hf.tap_check([step(1492.0), step(1493.0)])
    assert pair["state"] == "watch" and pair["rule"] == "pair"
    assert hf.tap_check([step(1468.0), step(1493.0)])["state"] == "ok"  # opposite sides


def test_the_tap_series_wrapper_gives_estimates_only_after_min_taps() -> None:
    idx = pd.date_range("2026-11-01 06:00", periods=12, freq="2h")
    temps = pd.Series(1480.0 + np.arange(12) % 3, index=idx)
    f = hf.filter_taps(temps.iloc[::-1], P)  # any order
    assert f.ready and len(f.steps) == 12 and f.tap_interval_h() == pytest.approx(2.0)
    assert f.at(idx[3]) is None  # 4 taps: fewer than min_taps
    est = f.at(idx[-1] + pd.Timedelta(hours=1))
    assert est is not None and 1475 < est.mean < 1485 and est.sd_tap > est.sd_state
    assert not hf.filter_taps(temps.iloc[:5], P).ready and not hf.filter_taps(temps.iloc[:0], P).ready


def test_the_backtest_uses_only_the_taps_published_by_the_origin() -> None:
    t, y = _simulate(300, seed=3)
    idx = pd.Timestamp("2026-10-01") + pd.to_timedelta(t, unit="h")
    temps = pd.Series(y, index=idx)
    received = pd.Series(idx + pd.Timedelta(minutes=40), index=idx)
    late = 250  # this tap is published two days after it was taken
    received.iloc[late] = idx[late] + pd.Timedelta(days=2)
    rows = hf.backtest(temps, received, P, [2.0], since=idx[100], memory_h=1e9)
    assert set(rows.columns) == {"at", "horizon_h", "forecast", "sd", "measured", "persistence"}
    assert len(rows) == 200 and (rows["measured"].to_numpy() == y[100:]).all()
    # the forecast for tap k: the filter over the taps taken and published before k's time less 2 h
    k = late + 1
    origin = idx[k] - pd.Timedelta(hours=2)
    known = temps[(temps.index <= origin) & (received <= origin)]
    assert idx[late] not in known.index and idx[late] <= origin
    state = hf.filter_taps(known, P).steps[-1].state
    lead = (idx[k] - known.index[-1]) / pd.Timedelta(hours=1)
    expected = hf.predict(state._replace(t_h=0.0), lead, P)
    row = rows[rows["at"] == idx[k]].iloc[0]
    assert row["forecast"] == pytest.approx(expected.mean, abs=1e-6)
    assert row["sd"] == pytest.approx(expected.sd_tap, abs=1e-6)
    assert row["persistence"] == known.iloc[-1]
    # on data of the model the forecast beats the last tap held, and its band is calibrated
    stats = hf.score(rows)
    assert stats["n"] == 200 and stats["ratio"] < 0.9 and 0.6 < stats["within_1sd"] < 0.8
    assert hf.score(rows.iloc[:0])["n"] == 0 and hf.score(rows.iloc[:0])["ratio"] is None
    # without publication times a tap counts as published default_delay_h after its time
    late_all = hf.backtest(temps, None, P, [2.0], since=idx[100], default_delay_h=30.0)
    assert hf.score(late_all)["rmse_persistence_c"] > stats["rmse_persistence_c"]


def test_the_parameters_come_from_the_config() -> None:
    p = hf.Params.from_config({"tau_h": 5, "r_c": 7.5, "min_taps": 4})
    assert (p.tau_h, p.r_c, p.min_taps, p.s_d_c) == (5.0, 7.5, 4, hf.Params().s_d_c)
    assert hf.Params.from_config(None) == hf.Params()
