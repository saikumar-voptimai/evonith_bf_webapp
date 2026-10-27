"""Rolling-window twin evaluation (SV-1.1, #281) against the Coke Energy pipeline on synthetic sources.

The synthetic frames (``tests/twin_data.py``) are 18 days of shifts made from three real shift rows. The pipeline is
run on them once; every window that coincides with a shift must return that shift's pipeline values (acceptance:
within 0.1 kg/tHM and 0.0005 in omega), a window between shifts aggregates its own hours, and the data checks and
flags of the window are reported.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from tests.twin_data import synthetic_frames
from utils.coke_energy.build_shifts import build_shift_inputs, window_key
from utils.coke_energy.pipeline import PipelineSettings, run_pipeline
from utils.coke_energy.twin import Window, evaluate_window, window_inputs

SETTINGS = PipelineSettings(history_start="2026-04-01", predict_from="2026-04-01")
TOL_K = 0.1  # kg/tHM
TOL_OMEGA = 0.0005


@pytest.fixture(scope="module")
def world():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(fd, m10, hm, dpr, SETTINGS)
    assert not result.errors
    return (fd, m10, hm, dpr), result


def _close(a, b, tol):
    if a is None:
        return not np.isfinite(b)
    return np.isfinite(b) and abs(a - b) <= tol


# ---------------------------------------------------------------------------------------------- window geometry
def test_window_ends_on_the_full_hour_and_covers_eight_hours() -> None:
    w = Window.ending("2026-05-10 14:05")
    assert (w.start, w.end) == (pd.Timestamp("2026-05-10 06:00"), pd.Timestamp("2026-05-10 14:00"))
    assert w.hours == 8 and w.key == "W2026-05-10T14:00"
    c = Window.of_shift("2026-05-10_C")
    assert (c.start, c.end) == (pd.Timestamp("2026-05-10 22:00"), pd.Timestamp("2026-05-11 06:00"))
    assert c.date == pd.Timestamp("2026-05-10")  # a C shift belongs to the date it starts
    with pytest.raises(ValueError):
        Window(pd.Timestamp("2026-05-10 14:00"), pd.Timestamp("2026-05-10 06:00"))


def test_a_window_on_a_shift_aggregates_like_the_shift(world) -> None:
    (fd, m10, hm, dpr), _ = world
    shifts = build_shift_inputs(fd, m10, hm, dpr, start=SETTINGS.history_start)
    for skey in ("2026-05-08_A", "2026-05-08_C"):
        w = Window.of_shift(skey)
        row = window_inputs(fd, m10, hm, dpr, w, SETTINGS.history_start).loc[w.key]
        expected = shifts.loc[skey]
        assert row.drop(["shift"]).equals(expected.drop(["shift"])), skey
        assert row["shift"] == "W"


def test_a_window_between_shifts_aggregates_its_own_hours(world) -> None:
    (fd, m10, hm, dpr), _ = world
    w = Window.ending("2026-05-09 10:00")  # 02:00-10:00: the last four hours of C and the first four of A
    S = build_shift_inputs(fd, m10, hm, dpr, start=SETTINGS.history_start, key_fn=window_key(w.start, w.end, w.key))
    hours = fd[(fd.time >= w.start) & (fd.time < w.end)]
    assert S.loc[w.key, "n_hours"] == 8
    assert S.loc[w.key, "COKE_CALC_MT"] == pytest.approx(hours.COKE_CALC_MT.sum())
    assert S.loc[w.key, "PRODUCTIONTONNESPERHR"] == pytest.approx(hours.PRODUCTIONTONNESPERHR.sum())
    ten = m10[(m10.ts >= w.start) & (m10.ts < w.end)]
    assert S.loc[w.key, "co_pct"] == pytest.approx(ten.co_pct.mean())
    assert S.loc[w.key, "sdate"] == pd.Timestamp("2026-05-09")


# ------------------------------------------------------------------------------------------ parity with the pipeline
def test_windows_on_completed_shifts_equal_the_pipeline(world) -> None:
    frames, result = world
    T = result.shifts
    checked = 0
    for skey, row in T.iterrows():
        ev = evaluate_window(*frames, Window.of_shift(skey), result, SETTINGS)
        assert _close(ev.k_trace, row.K_burnt, TOL_K), (skey, ev.k_trace, row.K_burnt)
        assert _close(ev.k_forge, row.K_energy, TOL_K), (skey, ev.k_forge, row.K_energy)
        assert _close(ev.omega_imp, row.omega_shift, TOL_OMEGA), (skey, ev.omega_imp, row.omega_shift)
        assert ev.k_charged == pytest.approx(row.K_charged, abs=TOL_K)
        if ev.k_forge is not None:
            assert ev.omega_cal == pytest.approx(row.omega_fixed, abs=TOL_OMEGA)
        assert ev.quantities["hmt_c"] == pytest.approx(row.HMT)
        assert ev.quantities["hm_si_pct"] == pytest.approx(row.hm_si)
        assert ev.quantities["production_t_h"] == pytest.approx(row.prod_th)
        assert ev.quantities["blast_moisture_g_nm3"] == pytest.approx(row.humidity_g)
        assert ev.status == ("complete" if np.isfinite(row.K_burnt) and np.isfinite(row.K_energy) else "partial")
        checked += 1
    assert checked == len(T) == 54
    assert T.K_burnt.notna().sum() == 42  # the first four days have fewer than 10 stable shifts in their pool


def test_early_windows_are_partial_with_the_reason(world) -> None:
    frames, result = world
    ev = evaluate_window(*frames, Window.of_shift("2026-05-02_A"), result, SETTINGS)
    assert ev.status == "partial"
    assert ev.k_trace is None and ev.k_forge is None and ev.omega_imp is not None
    assert any("TRACE: fewer than 10 stable shifts" in n for n in ev.notes)


def test_a_window_between_shifts_is_evaluated(world) -> None:
    frames, result = world
    ev = evaluate_window(*frames, Window.ending("2026-05-15 10:00"), result, SETTINGS)
    assert ev.status == "complete"
    assert 250 < ev.k_trace < 400 and 250 < ev.k_forge < 400 and 0.6 < ev.omega_imp < 0.95
    assert ev.flags["n_hours"] == 8 and ev.flags["n_ok10"] == 48


# ---------------------------------------------------------------------------------------------- flags and data checks
def test_provisional_dpr_dust_is_flagged(world) -> None:
    frames, result = world
    w = Window.of_shift("2026-05-16_B")
    assert evaluate_window(*frames, w, result, SETTINGS).flags["provisional_dpr"] is False
    flagged = evaluate_window(*frames, w, result, SETTINGS, provisional_dates={w.date.date()})
    assert flagged.flags["provisional_dpr"] is True and flagged.status == "complete"


def test_a_window_without_data_fails_with_the_reasons(world) -> None:
    (fd, m10, hm, dpr), result = world
    empty = evaluate_window(fd, m10, hm, dpr, Window.ending("2026-07-01 10:00"), result, SETTINGS)
    assert empty.status == "failed" and empty.notes == ["no hourly data in the window"]
    w = Window.of_shift("2026-05-12_A")
    no_taps = hm[(hm.t < w.start) | (hm.t >= w.end)]
    ev = evaluate_window(fd, m10, no_taps, dpr, w, result, SETTINGS)
    assert ev.status == "failed"
    assert "0 hot-metal taps (need 2)" in ev.notes and "no hot-metal analysis" in ev.notes


def test_a_known_data_error_fails_the_window(world) -> None:
    (fd, m10, hm, dpr), result = world
    w = Window.of_shift("2026-05-12_B")
    wet = fd.copy()
    wet.loc[(wet.time >= w.start) & (wet.time < w.end), "ORE_TM%"] = 129.7
    ev = evaluate_window(wet, m10, hm, dpr, w, result, SETTINGS)
    assert ev.status == "failed" and ev.notes[0].startswith("data error: ore moisture")


def test_one_evaluation_takes_well_under_30_s(world) -> None:
    frames, result = world
    t0 = time.perf_counter()
    evaluate_window(*frames, Window.ending("2026-05-17 13:00"), result, SETTINGS)
    assert time.perf_counter() - t0 < 30
