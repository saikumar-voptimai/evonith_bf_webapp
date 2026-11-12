"""Calibration tracker of the Coke Energy pipeline (user decision 12 Nov 2026).

FORGE's omega, Q_LZ, Q_UZ, r_H and TRACE's charged - burnt offset are calibrated over the stable shifts that ended
before the shift began, at most 14 days old, each weighted 0.5 ** (age / 3 d) (``PipelineSettings.calibration``
"exponential"; "median" keeps the web model's 14-day median, which the other twin tests run on). Checked here:

* the weights and the weighted means (``tracker_weights``, ``rolling_pool_tracker``) against direct arithmetic;
* the pipeline's ``omega_fixed`` and ``burnt_offset`` against the weighted means of the shift table itself;
* a window that coincides with a shift returns the shift's values; a window between shifts uses completed shifts only;
* the FORGE calibration view and the TRACE correction view list the shifts and shares that give the pipeline's values.

Sources: the synthetic frames of ``tests/twin_data.py`` (18 days of shifts made from three real shift rows).
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from data.coke_energy import sources
from services import coke_energy as ce
from services import forge, trace
from tests.twin_data import synthetic_frames
from utils.coke_energy.pipeline import (
    TRACKER_MIN_AGE_H,
    PipelineSettings,
    calibration_values,
    rolling_pool_median,
    rolling_pool_tracker,
    run_pipeline,
    tracker_weights,
)
from utils.coke_energy.twin import Window, evaluate_window

TRACKED = PipelineSettings(history_start="2026-04-01", predict_from="2026-04-01", calibration="exponential")
CAL = ["omega_shift", "Q_LZ_GJh_shift", "Q_UZ_GJh_shift", "r_H_fixed"]
TOL_K = 0.1  # kg/tHM, as tests/test_twin_window.py
TOL_OMEGA = 0.0005


@pytest.fixture(scope="module")
def world():
    frames = synthetic_frames(start="2026-05-01", days=18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(*frames, TRACKED)
    assert not result.errors
    return frames, result


def direct(T: pd.DataFrame, t_mid: pd.Timestamp, col: str, half_life_d: float = 3.0, min_n: int = 4) -> float:
    """Weighted mean of ``col`` over the stable shifts that ended before ``t_mid``'s shift began."""
    pool = T[T.stable & T[col].notna()]
    age = (t_mid - pool.t_mid) / pd.Timedelta(hours=1)
    use = (age >= 7.9) & (age <= 14 * 24)
    if use.sum() < min_n:
        return float("nan")
    w = 0.5 ** (age[use] / (24 * half_life_d))
    return float((w * pool.loc[use, col]).sum() / w.sum())


# ---------------------------------------------------------------------------------------------- weights and means
def test_weights_halve_every_half_life_and_exclude_overlapping_and_old_shifts() -> None:
    w = tracker_weights(np.array([0.0, 4.0, TRACKER_MIN_AGE_H, 8.0, 72.0, 144.0, 336.0, 336.5]), 14, 3.0)
    assert w[0] == w[1] == 0.0  # the shift itself and an overlapping window
    assert w[2] > 0 and w[3] == pytest.approx(0.5 ** (8 / 72))  # the shift that ended as the target began
    assert w[4] == pytest.approx(0.5) and w[5] == pytest.approx(0.25)
    assert w[6] == pytest.approx(0.5 ** (336 / 72)) and w[7] == 0.0  # at most 14 days old


def test_the_tracker_is_a_weighted_mean_per_column_with_its_own_count() -> None:
    t = pd.date_range("2026-05-01 10:00", periods=12, freq="8h")
    pool = pd.DataFrame({"t_mid": t, "date": t.normalize(), "a": np.arange(12.0), "b": [np.nan] * 9 + [1.0, 2.0, 3.0]})
    target = pd.DataFrame({"date": [t[-1].normalize()], "t_mid": [t[-1] + pd.Timedelta(hours=8)]}, index=["x"])
    got = rolling_pool_tracker(pool, target["t_mid"], ["a", "b"], 14, 3.0, 4)
    age = np.asarray((target.t_mid.iloc[0] - t) / pd.Timedelta(hours=1), dtype=float)
    w = 0.5 ** (age / 72)
    assert got.loc["x", "a_roll"] == pytest.approx(float((w * pool.a.to_numpy()).sum() / w.sum()))
    assert np.isnan(got.loc["x", "b_roll"])  # three values: fewer than four
    got3 = rolling_pool_tracker(pool, target["t_mid"], ["b"], 14, 3.0, 3)
    wb = w[9:]
    assert got3.loc["x", "b_roll"] == pytest.approx(float((wb * pool.b.to_numpy()[9:]).sum() / wb.sum()))


def test_the_median_setting_is_the_web_rule() -> None:
    t = pd.date_range("2026-05-01 10:00", periods=60, freq="8h")
    pool = pd.DataFrame({"t_mid": t, "date": t.normalize(), "a": np.sin(np.arange(60.0))})
    target = pd.DataFrame({"date": [pd.Timestamp("2026-05-19")], "t_mid": [pd.Timestamp("2026-05-19 10:00")]})
    median = PipelineSettings(calibration="median")
    a = calibration_values(pool, target, ["a"], median)
    b = rolling_pool_median(pool, target["date"], ["a"], 14, 10)
    pd.testing.assert_frame_equal(a, b)


def test_settings_come_from_the_config() -> None:
    shipped = PipelineSettings.from_config(ce.config())
    assert (shipped.calibration, shipped.half_life_days, shipped.tracker_min_shifts) == ("exponential", 3.0, 4)
    assert shipped.calibration_min_shifts == 4 and shipped.window_days == 14
    web = PipelineSettings.from_config({})
    assert web.calibration == "median" and web.calibration_min_shifts == 10
    own = PipelineSettings.from_config({"calibration_tracker": {"method": "exponential", "half_life_days": 2.5,
                                                               "min_shifts": 6}})  # fmt: skip
    assert (own.half_life_days, own.tracker_min_shifts) == (2.5, 6)
    with pytest.raises(ValueError):
        PipelineSettings(calibration="ewma")


# ---------------------------------------------------------------------------------------------- the pipeline
def test_the_pipeline_calibration_is_the_weighted_mean_of_the_shift_table(world) -> None:
    _, result = world
    T = result.shifts
    checked = 0
    for skey, row in T.iterrows():
        omega = direct(T, row.t_mid, "omega_shift")
        offset = direct(T.assign(gap=T.K_charged - T.K_burnt_raw), row.t_mid, "gap")
        assert np.isnan(row.burnt_offset) == np.isnan(offset), skey
        if np.isfinite(offset):
            assert row.burnt_offset == pytest.approx(offset, rel=1e-9)
            assert row.K_burnt == pytest.approx(row.K_burnt_raw + offset, rel=1e-9)
        if np.isfinite(row.omega_fixed):
            assert row.omega_fixed == pytest.approx(omega, rel=1e-9)
            assert row.Q_LZ_GJh_fixed == pytest.approx(direct(T, row.t_mid, "Q_LZ_GJh_shift"), rel=1e-9)
            checked += 1
    assert checked >= 40
    # four stable shifts suffice: the tracker predicts earlier than the 14-day median (10 shifts)
    assert T.K_burnt.notna().sum() > 42


def test_windows_on_completed_shifts_equal_the_pipeline(world) -> None:
    frames, result = world
    T = result.shifts
    for skey in T.index[::3]:
        row = T.loc[skey]
        ev = evaluate_window(*frames, Window.of_shift(skey), result, TRACKED, strata=False)
        for got, want, tol in ((ev.k_trace, row.K_burnt, TOL_K), (ev.k_forge, row.K_energy, TOL_K)):
            assert (got is None and not np.isfinite(want)) or abs(got - want) <= tol, (skey, got, want)
        if ev.k_forge is not None:
            assert ev.omega_cal == pytest.approx(row.omega_fixed, abs=TOL_OMEGA)


def test_a_window_between_shifts_uses_the_shifts_that_ended_before_it(world) -> None:
    frames, result = world
    T = result.shifts
    w = Window.ending("2026-05-15 10:00")  # 02:00-10:00: overlaps C of 14 May (mid 02:00) and A of 15 May
    ev = evaluate_window(*frames, w, result, TRACKED, strata=False)
    assert w.mid == pd.Timestamp("2026-05-15 06:00")
    assert ev.omega_cal == pytest.approx(direct(T, w.mid, "omega_shift"), rel=1e-9)
    pool_last = T[T.stable & (T.t_mid <= w.mid - pd.Timedelta(hours=7.9))].index[-1]
    assert pool_last == "2026-05-14_B"  # ended 22:00, before the window began; C of 14 May overlaps it


# ---------------------------------------------------------------------------------------------- the views
def test_the_forge_calibration_view_lists_the_shifts_and_shares_of_omega_fixed(world, tmp_path) -> None:
    frames, result = world
    data = ce.PageData(result=result, fresh=sources.freshness(*frames), provisional=[],
                       samples=pd.DataFrame(columns=["source", "material", "t"]), charged=pd.DataFrame())  # fmt: skip
    view = forge.build_view(data, TRACKED, forge.ForgeSettings(stats_days=30, rist_days=30),
                            built_at=datetime(2026, 5, 19, 2, 50, tzinfo=UTC), frames_for=lambda ks: [])  # fmt: skip
    forge.save(view, tmp_path)
    stored = forge.load(tmp_path)
    T = result.shifts
    for day in ("2026-05-08", "2026-05-12", "2026-05-17"):
        cal = forge.calibration(stored, pd.Timestamp(day).date())
        row = T.loc[f"{day}_A"]
        assert cal["method"] == "exponential" and cal["half_life_days"] == 3.0 and cal["target_shift"] == f"{day}_A"
        assert cal["omega_cal"] == pytest.approx(row.omega_fixed, rel=1e-9)
        assert cal["q_uz_gj_h"] == pytest.approx(row.Q_UZ_GJh_fixed, rel=1e-9)
        shares = [s["weight"] for s in cal["shifts"] if s["omega"] is not None]
        assert sum(shares) == pytest.approx(1.0)
        assert sum(s["weight"] * s["omega"] for s in cal["shifts"] if s["omega"] is not None) == pytest.approx(
            row.omega_fixed, rel=1e-9
        )
        ages = [
            (T.loc[f"{day}_A", "t_mid"] - T.loc[s["shift_key"], "t_mid"]) / pd.Timedelta(hours=1) for s in cal["shifts"]
        ]
        assert min(ages) >= TRACKER_MIN_AGE_H and max(ages) <= 14 * 24 and cal["complete"]
    latest = forge.calibration(stored)
    assert latest["target_shift"] == "2026-05-19_A"  # the shift after the latest balanced one (18 May C)


def test_the_trace_correction_is_the_weighted_mean_of_its_pool(world) -> None:
    _, result = world
    T = result.shifts
    c = trace.correction(T, TRACKED)
    assert c["method"] == "exponential" and c["half_life_days"] == 3.0 and c["min_shifts"] == 4
    assert c["skey"] == T.sort_values("t_mid").index[-1]
    assert sum(p["weight"] for p in c["pool"]) == pytest.approx(1.0)
    assert sum(p["weight"] * p["gap"] for p in c["pool"]) == pytest.approx(c["offset"], rel=1e-9)
    assert all(bool(T.loc[p["skey"], "stable"]) for p in c["pool"])
