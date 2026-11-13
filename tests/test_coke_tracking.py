"""Coke tracked to the cohesive zone (user decision 13 Nov 2026; ``utils/coke_energy/coke_tracking.py``).

The tracked coke of a window is the coke of the charges whose ore melts in it, on the window's hot metal:
K_tracked = K_charged x r_melt / r_charged (r = dry coke / ferrous of a set of charges). Checked here:

* the ages at the cohesive-zone top and bottom are the zone depths over the charged volume per hour;
* the blast clock stops with the blast: a stop delays the melting of the charges in the column by its length;
* a step in coke:ferrous reaches the melting window between the two ages, and K_tracked follows it;
* steady composition gives K_tracked = K_charged; a record gap leaves the windows that depend on it without a value;
* the pipeline on the tracked basis: the per-shift calibration rests on the tracked coke with the blast air closed
  again at it, TRACE's offset is the tracker's mean of tracked - raw burnt, a twin window on a completed shift
  reproduces the pipeline, and the TRACE and FORGE views carry the tracked coke and the basis;
* on the charged basis (the web model) the calibration inputs are the charged coke and its reconciled air.

Sources: constructed hourly records (constant charging and blast) and the synthetic frames of ``tests/twin_data.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from data.coke_energy import sources
from services import coke_energy as ce
from services import forge, trace
from tests.twin_data import synthetic_frames
from utils.coke_energy.bf_balance import calibrate_loss
from utils.coke_energy.coke_tracking import (
    BASES,
    Tracker,
    TrackingSettings,
    _fill_gaps,
    tracked_for_windows,
)
from utils.coke_energy.pipeline import (
    PipelineSettings,
    _case,
    basis_columns,
    build_cases,
    run_pipeline,
    shift_params,
)
from utils.coke_energy.rist import calibrate_2z
from utils.coke_energy.twin import Window, evaluate_window

H = pd.Timedelta(hours=1)
# Charges per hour of the constructed record, t/h (about the plant's: 152 m3/h at the bulk densities)
CHARGE = {"COKE_CALC_MT": 32.0, "NUTCOKE_CALC_MT": 4.0, "SINTER_CALC_MT": 110.0, "TOTAL_PELLET_CALC_MT": 30.0,
          "ORE_CALC_MT": 20.0, "FLUX_CALC_MT": 3.0}  # fmt: skip
BLAST = "HOT BLAST VOLUMENM3/HR."
O2_FLOW = "OXYGENFLOWNM3/HR."
START = pd.Timestamp("2026-05-01 00:00")
TRACKED = PipelineSettings(history_start="2026-04-01", predict_from="2026-04-01", calibration="exponential",
                           tracking=TrackingSettings(basis="tracked"))  # fmt: skip
CHARGED = replace(TRACKED, tracking=TrackingSettings(basis="charged"))
TOL_K = 0.1  # kg/tHM, as tests/test_twin_window.py
TOL_OMEGA = 0.0005


def record(days: int = 20, **columns: float) -> pd.DataFrame:
    """Hourly record (row label = end of the charging hour) at constant charging, blast 110,000 Nm3/h with
    10,000 Nm3/h oxygen, dry coke."""
    t = pd.date_range(START, periods=24 * days, freq="h")
    base = {**CHARGE, BLAST: 110_000.0, O2_FLOW: 10_000.0, "COKE_MOIST%": 0.0, **columns}
    return pd.DataFrame({"time": t, **{k: np.full(len(t), v, dtype=float) for k, v in base.items()}})


def volume_per_hour(s: TrackingSettings) -> float:
    d = s.densities
    names = {"COKE_CALC_MT": "coke", "NUTCOKE_CALC_MT": "nut", "SINTER_CALC_MT": "sinter",
             "TOTAL_PELLET_CALC_MT": "pellet", "ORE_CALC_MT": "ore", "FLUX_CALC_MT": "flux"}  # fmt: skip
    return sum(CHARGE[c] / d[m] for c, m in names.items())


def ratio(coke: float) -> float:
    """Coke : ferrous of the constructed charges, kg/t."""
    return 1000.0 * coke / (CHARGE["SINTER_CALC_MT"] + CHARGE["TOTAL_PELLET_CALC_MT"] + CHARGE["ORE_CALC_MT"])


def hours(ns: np.ndarray | float) -> np.ndarray:
    return np.asarray(ns, dtype=float) / 3.6e12


# ---------------------------------------------------------------------------------------------- settings
def test_settings_come_from_the_config() -> None:
    web = TrackingSettings.from_config(None)
    assert web.basis == "charged" and PipelineSettings().coke_basis == "charged"  # the web model by default
    block = ce.config().get("coke_tracking") or {}
    shipped = TrackingSettings.from_config(block)
    assert shipped.basis in BASES and shipped.basis == block.get("basis", "charged")
    dens = block.get("bulk_density_t_m3") or {}
    assert shipped.density_coke == dens.get("coke", web.density_coke)
    assert shipped.density_ore == dens.get("lump_ore", web.density_ore)
    assert PipelineSettings.from_config(ce.config()).coke_basis == shipped.basis
    with pytest.raises(ValueError):
        TrackingSettings(basis="burnt")
    with pytest.raises(ValueError):
        TrackingSettings(cz_top_m3=430.0, cz_bottom_m3=305.0)


def test_only_short_gaps_are_interpolated() -> None:
    x = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0, np.nan])
    got = _fill_gaps(x, 3)
    assert got.iloc[:4].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert got.iloc[4:8].isna().all() and got.iloc[8] == 9.0 and np.isnan(got.iloc[9])  # long and trailing gaps stay


# ---------------------------------------------------------------------------------------------- the path of a charge
def test_ages_are_the_zone_depths_over_the_charged_volume_per_hour() -> None:
    s = TrackingSettings()
    tr = Tracker(record(), s)
    v = volume_per_hour(s)
    ages = tr.ages_h()
    assert ages["cz_top"] == pytest.approx(s.cz_top_m3 / v, rel=1e-9)
    assert ages["cz_bottom"] == pytest.approx(s.cz_bottom_m3 / v, rel=1e-9)
    assert 1.5 < ages["cz_top"] < ages["cz_bottom"] < 3.5  # plant order of magnitude: 2.2 and 3.1 h
    # constant blast: one hour of age is one hour
    ok = tr.ok
    assert hours(tr.t_top[ok] - tr.t[ok]) == pytest.approx(np.full(ok.sum(), ages["cz_top"]), abs=1e-6)


def test_the_blast_clock_stops_with_the_blast() -> None:
    stop = START + pd.Timedelta(days=12)  # rows stop .. stop + 3 h: blast off (tags of [stop - 0.5 h, stop + 3.5 h))
    fd = record()
    off = fd.time.between(stop, stop + 3 * H)
    fd.loc[off, BLAST] = 20_000.0
    base, tr = Tracker(record()), Tracker(fd)
    age = base.ages_h()["cz_top"]
    t = pd.to_datetime(tr.t)
    stop_tag, restart = stop - 0.5 * H, stop + 3.5 * H
    # charges in the column when the blast stopped: their melting moves by the length of the stop
    before = (t > stop_tag - age * H + 0.1 * H) & (t < stop_tag - 0.1 * H)
    assert before.sum() >= 6
    assert hours(tr.t_top[before] - base.t_top[before]) == pytest.approx(np.full(before.sum(), 4.0), abs=1e-6)
    # charges dumped during the stop start their path when the blast returns
    during = (t > stop_tag + 0.1 * H) & (t < restart - 0.1 * H)
    assert hours(tr.t_top[during] - restart.value) == pytest.approx(np.full(during.sum(), age), abs=1e-6)
    # charges dumped after it are not affected
    after = (t > restart + 0.1 * H) & np.isfinite(base.t_top)
    assert np.isfinite(tr.t_top[after]).all()
    assert np.abs(hours(tr.t_top[after] - base.t_top[after])).max() < 1e-6


def test_a_coke_step_reaches_the_melting_between_the_two_ages() -> None:
    row = START + pd.Timedelta(days=12)  # first row with 10 % more coke: charges dumped from row - 1 h
    fd = record()
    fd.loc[fd.time >= row, "COKE_CALC_MT"] *= 1.1
    tr = Tracker(fd)
    ages = tr.ages_h()
    dump = row - H
    old, new = ratio(CHARGE["COKE_CALC_MT"]), ratio(1.1 * CHARGE["COKE_CALC_MT"])
    # melting windows of 15 min on the tag clock, from 1 h before the step's arrival to 1 h after its passage
    edges = pd.date_range(dump + (ages["cz_top"] - 1) * H, dump + (ages["cz_bottom"] + 1) * H, freq="15min")
    r = np.array([tr.melting(a, b)[0] for a, b in zip(edges[:-1], edges[1:], strict=True)])
    mid = edges[:-1] + pd.Timedelta(minutes=7.5)
    first_new = dump + pd.Timedelta(minutes=5) + ages["cz_top"] * H  # the first new parcel reaches the zone top
    last_old = dump - pd.Timedelta(minutes=5) + ages["cz_bottom"] * H  # the last old parcel leaves the zone bottom
    assert r[mid < first_new - 15 * pd.Timedelta(minutes=1)] == pytest.approx(old, rel=1e-9)
    assert r[mid > last_old + 15 * pd.Timedelta(minutes=1)] == pytest.approx(new, rel=1e-9)
    assert np.all(np.diff(r) >= -1e-9)  # the ratio rises monotonically through the zone


def test_tracked_coke_follows_the_melting_and_equals_charged_coke_at_steady_composition() -> None:
    row = START + pd.Timedelta(days=12)
    fd = record()
    fd.loc[fd.time >= row, "COKE_CALC_MT"] *= 1.1
    tr = Tracker(fd)
    k = 400.0
    # steady composition, before the step and well after it
    assert tr.k_tracked(row - 24 * H, row - 16 * H, k) == pytest.approx(k, rel=1e-9)
    assert tr.k_tracked(row + 12 * H, row + 20 * H, k) == pytest.approx(k, rel=1e-9)
    # the first window of new charges (dumped from row - 1 h) melts old ones until the mean age after the step: less
    # coke than charged. The window's tags start 0.5 h before its first row (tag_offset_h), half an hour after the step.
    first = tr.k_tracked(row, row + 8 * H, k)
    ages = tr.ages_h()
    mean_age = 0.5 * (ages["cz_top"] + ages["cz_bottom"])
    old_share = (mean_age - 0.5) / 8  # share of the window's melting from the old charges (linear melting)
    assert first == pytest.approx(k * (old_share / 1.1 + (1 - old_share)), rel=5e-4)
    assert k - first > 5.0
    # windows: the same through the series helper
    got = tracked_for_windows(fd, pd.Series({"w": row}), pd.Series({"w": k}))
    assert got["w"] == pytest.approx(first, rel=1e-12)


def test_a_record_gap_leaves_the_windows_that_melt_it_without_a_value() -> None:
    gap = START + pd.Timedelta(days=12)
    fd = record()
    fd.loc[fd.time.between(gap, gap + 7 * H), ["COKE_CALC_MT", "SINTER_CALC_MT"]] = np.nan  # 8 h of charges missing
    tr = Tracker(fd)
    assert np.isnan(tr.k_tracked(gap + 3 * H, gap + 11 * H, 400.0))  # melts the missing charges: cover < 0.8
    assert tr.k_tracked(gap + 24 * H, gap + 32 * H, 400.0) == pytest.approx(400.0)
    short = record()
    short.loc[short.time.between(gap, gap + 2 * H), ["COKE_CALC_MT", "SINTER_CALC_MT"]] = np.nan  # 3 h: interpolated
    assert Tracker(short).k_tracked(gap + 3 * H, gap + 11 * H, 400.0) == pytest.approx(400.0, rel=1e-9)
    assert np.isnan(Tracker(record().iloc[:0]).k_tracked(gap, gap + 8 * H, 400.0))  # no record


def test_the_latest_window_is_tracked_to_the_end_of_the_record() -> None:
    fd = record()
    tr = Tracker(fd)
    end = fd.time.iloc[-1] + H  # end of the last row's hour
    _, cover = tr.melting(end - 8.5 * H, end - 0.5 * H)
    assert cover == pytest.approx(1.0, abs=0.01)  # the charges that melt across the end of the record count
    assert tr.k_tracked(end - 8 * H, end, 400.0) == pytest.approx(400.0, rel=1e-9)


# ---------------------------------------------------------------------------------------------- the pipeline
@pytest.fixture(scope="module")
def world():
    frames = synthetic_frames(start="2026-05-01", days=18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracked = run_pipeline(*frames, TRACKED)
        charged = run_pipeline(*frames, CHARGED)
    assert not tracked.errors and not charged.errors
    return frames, tracked, charged


def direct(T: pd.DataFrame, t_mid: pd.Timestamp, col: str, half_life_d: float = 3.0, min_n: int = 4) -> float:
    """Weighted mean of ``col`` over the stable shifts that ended before ``t_mid``'s shift began."""
    pool = T[T.stable & T[col].notna()]
    age = (t_mid - pool.t_mid) / pd.Timedelta(hours=1)
    use = (age >= 7.9) & (age <= 14 * 24)
    if use.sum() < min_n:
        return float("nan")
    w = 0.5 ** (age[use] / (24 * half_life_d))
    return float((w * pool.loc[use, col]).sum() / w.sum())


def test_the_charged_basis_is_the_web_model(world) -> None:
    _, tracked, charged = world
    R = charged.R
    assert basis_columns(R) == ("coke_kgthm_basis", "V_air_basis")
    pd.testing.assert_series_equal(R.coke_kgthm_basis, R.coke_kgthm_dry, check_names=False)
    pd.testing.assert_series_equal(R.V_air_basis, R.V_air_reconciled, check_names=False)
    T = charged.shifts
    pd.testing.assert_series_equal(T.K_basis, T.K_charged, check_names=False)
    # the tracked coke is computed on either basis (shown on the pages); it is the same series
    pd.testing.assert_series_equal(T.K_tracked, tracked.shifts.K_tracked)


def test_the_tracked_basis_calibrates_on_the_tracked_coke_with_the_air_closed_again(world) -> None:
    _, result, charged = world
    S, R, T = result.S, result.R, result.shifts
    have = T.K_tracked.notna()
    assert have.sum() >= 0.9 * len(T)  # all but the first shifts, whose melting precedes the record
    assert (T.K_tracked - T.K_charged).abs().max() > 1.0  # the templates' coke:ferrous differ shift to shift
    cases = build_cases(S, R.index)
    params = shift_params(S, R.index, TRACKED)
    checked = 0
    for key in T.index[have & T.omega_shift.notna()][::4]:
        rb = R.loc[key]
        k = T.loc[key, "K_tracked"]
        assert T.loc[key, "K_basis"] == pytest.approx(k, rel=1e-12)
        v = calibrate_loss(_case(cases, key, rb.eta_H2), k, params[key], eta_CO=rb.eta_CO / 100)[1]
        assert rb.V_air_basis == pytest.approx(v, rel=1e-9)
        cal = calibrate_2z(_case(cases, key, rb.eta_H2), k, v, rb.eta_CO / 100, params[key])
        assert T.loc[key, "omega_shift"] == pytest.approx(cal["omega"], rel=1e-9)
        checked += 1
    assert checked >= 10
    # the shifts without a tracked coke keep the charged basis
    for key in T.index[~have]:
        assert T.loc[key, "K_basis"] == pytest.approx(T.loc[key, "K_charged"], rel=1e-12)
    # the raw burnt coke does not depend on the basis; the calibrated omega does
    pd.testing.assert_series_equal(T.K_burnt_raw, charged.shifts.K_burnt_raw)
    assert (T.omega_shift - charged.shifts.omega_shift).abs().max() > 1e-4


def test_trace_offset_is_the_tracker_mean_of_tracked_minus_raw_burnt(world) -> None:
    _, result, _ = world
    T = result.shifts
    gaps = T.assign(gap=T.K_basis - T.K_burnt_raw)
    checked = 0
    for skey, row in T.iterrows():
        offset = direct(gaps, row.t_mid, "gap")
        assert np.isnan(row.burnt_offset) == np.isnan(offset), skey
        if np.isfinite(offset):
            assert row.burnt_offset == pytest.approx(offset, rel=1e-9)
            assert row.K_burnt == pytest.approx(row.K_burnt_raw + offset, rel=1e-9)
            checked += 1
    assert checked >= 40
    c = trace.correction(T, TRACKED)
    assert c["coke_basis"] == "tracked"
    assert all(p["gap"] == pytest.approx(T.loc[p["skey"], "K_basis"] - p["k_burnt_raw"], rel=1e-9) for p in c["pool"])
    assert sum(p["weight"] * p["gap"] for p in c["pool"]) == pytest.approx(c["offset"], rel=1e-9)
    assert any(p["k_tracked"] is not None and p["k_tracked"] != p["k_charged"] for p in c["pool"])


def test_windows_on_completed_shifts_equal_the_pipeline_on_the_tracked_basis(world) -> None:
    frames, result, _ = world
    T = result.shifts
    checked = 0
    for skey in T.index[2::4]:
        row = T.loc[skey]
        ev = evaluate_window(*frames, Window.of_shift(skey), result, TRACKED, strata=False)
        k_trk = ev.quantities.get("k_tracked_kg_thm")
        assert (k_trk is None and not np.isfinite(row.K_tracked)) or k_trk == pytest.approx(row.K_tracked, abs=1e-6)
        for got, want, tol in ((ev.k_trace, row.K_burnt, TOL_K), (ev.k_forge, row.K_energy, TOL_K),
                               (ev.omega_imp, row.omega_shift, TOL_OMEGA)):  # fmt: skip
            assert (got is None and not np.isfinite(want)) or abs(got - want) <= tol, (skey, got, want)
        checked += k_trk is not None
    assert checked >= 8


def test_the_views_carry_the_tracked_coke_and_the_basis(world, tmp_path) -> None:
    frames, result, _ = world
    T = result.shifts
    rows = trace.shift_rows(T)
    assert all(
        r["k_tracked"] == pytest.approx(T.loc[r["skey"], "K_tracked"]) for r in rows if r["k_tracked"] is not None
    )
    assert sum(r["k_tracked"] is not None for r in rows) >= 0.9 * len(rows)
    assert all("k_tracked" in r for r in trace.day_rows(T))
    data = ce.PageData(result=result, fresh=sources.freshness(*frames), provisional=[],
                       samples=pd.DataFrame(columns=["source", "material", "t"]), charged=pd.DataFrame())  # fmt: skip
    view = forge.build_view(data, TRACKED, forge.ForgeSettings(stats_days=30, rist_days=30),
                            built_at=datetime(2026, 5, 19, 2, 50, tzinfo=UTC), frames_for=lambda ks: [])  # fmt: skip
    forge.save(view, tmp_path)
    stored = forge.load(tmp_path)
    assert stored.info()["coke_basis"] == "tracked"
    cal = forge.calibration(stored, pd.Timestamp("2026-05-12").date())
    assert cal["coke_basis"] == "tracked"
    assert all(s["k_tracked"] == pytest.approx(T.loc[s["shift_key"], "K_tracked"]) for s in cal["shifts"]
               if s["k_tracked"] is not None)  # fmt: skip
    shifts = forge.shift_views(stored.shifts)
    assert sum(s["k_tracked"] is not None for s in shifts) >= 0.9 * len(shifts)
