"""The rolling bias correction on the energy balance's coke rate.

The balance predicts coke with the right shape and the wrong level - +19.7
kg/tHM over 239 days, MAPE 7.24%, R2 0.07. One rolling offset takes that to
MAPE 3.37%, R2 0.74 forward. These tests pin the behaviour that makes the
correction safe to put in front of an operator: it must be inspectable, it must
degrade to a no-op rather than to a wrong number, and it must never quietly
hide a growing problem.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from utils.bmo.coke_calibration import (
    NO_CALIBRATION,
    CokeCalibration,
    fit_offset,
    load_calibration,
    save_calibration,
)


def _pairs(offset: float, n: int = 60, noise: float = 0.0):
    """n days where the balance over-predicts by `offset`.

    ``noise`` perturbs the RESIDUAL, not both series - adding it to actual and
    predicted alike would leave the residual constant and quietly make the
    spread assertions vacuous.
    """

    actual = [330.0 + (i % 5) * 3.0 for i in range(n)]
    predicted = [a + offset + ((i % 7) - 3) * noise for i, a in enumerate(actual)]
    return predicted, actual


# --- fitting ---------------------------------------------------------------------


def test_it_recovers_a_known_offset():
    predicted, actual = _pairs(24.5)

    c = fit_offset(predicted, actual)

    assert c.offset_kg_per_thm == pytest.approx(24.5)
    assert c.sample_days == 60
    assert c.is_usable


def test_applying_it_removes_the_bias():
    predicted, actual = _pairs(19.7)
    c = fit_offset(predicted, actual)

    corrected = [c.apply(p) for p in predicted]

    assert corrected == pytest.approx(actual)


def test_a_bad_day_does_not_drag_the_offset():
    """One mis-keyed charge report must not reshape the correction.

    A blowdown or a data-entry slip produces a residual far outside the normal
    spread. Averaging it in would move the offset for every subsequent day.
    """

    predicted, actual = _pairs(20.0, n=60, noise=1.0)
    predicted[17] += 400.0  # a wild day

    c = fit_offset(predicted, actual)

    assert c.outliers_dropped >= 1
    assert c.offset_kg_per_thm == pytest.approx(20.0, abs=1.0)


def test_the_residual_spread_is_reported_not_just_the_offset():
    """The offset fixes the average day. Operators need to know the scatter."""

    predicted, actual = _pairs(20.0, n=60, noise=6.0)

    c = fit_offset(predicted, actual)

    assert c.residual_sd_kg_per_thm > 0.0
    assert c.first_day == "" or isinstance(c.first_day, str)


def test_a_wide_spread_earns_a_warning():
    actual = [330.0] * 40
    predicted = [a + 20.0 + (60.0 if i % 2 else -60.0) for i, a in enumerate(actual)]

    c = fit_offset(predicted, actual)

    assert any("scatter" in n or "spread" in n for n in c.notes)


# --- degrading safely --------------------------------------------------------------


def test_too_few_days_is_flagged_and_not_applied():
    """A handful of days cannot establish a 20 kg/tHM correction."""

    predicted, actual = _pairs(24.0, n=5)

    c = fit_offset(predicted, actual)

    assert not c.is_usable
    assert c.apply(350.0) == 350.0, "an unusable offset must be a no-op"
    assert any("provisional" in n for n in c.notes)


def test_no_data_returns_a_no_op_that_says_so():
    c = fit_offset([], [])

    assert c is NO_CALIBRATION
    assert c.apply(350.0) == 350.0
    assert c.notes


def test_missing_file_is_not_an_error():
    """Nobody having run the refresh yet must not break a recommendation."""

    c = load_calibration("nonexistent-calibration-file.json")

    assert c.apply(350.0) == 350.0


def test_a_corrupt_file_falls_back_rather_than_crashing(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{ truncated")

    assert load_calibration(path).apply(350.0) == 350.0


def test_a_partial_file_falls_back(tmp_path):
    """A file missing the offset itself must not be read as zero."""

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"sample_days": 90}))

    assert load_calibration(path) is NO_CALIBRATION


def test_it_never_returns_a_negative_coke_rate():
    c = CokeCalibration(offset_kg_per_thm=500.0, sample_days=90,
                        residual_sd_kg_per_thm=5.0, window_days=90)

    assert c.apply(300.0) == 0.0


# --- staleness, because the bias drifts --------------------------------------------


def test_a_recent_calibration_is_not_stale():
    c = CokeCalibration(offset_kg_per_thm=20.0, sample_days=90,
                        residual_sd_kg_per_thm=10.0, window_days=90,
                        fitted_on="2026-08-01")

    assert c.age_days(date(2026, 8, 20)) == 19
    assert not c.is_stale(date(2026, 8, 20))


def test_an_old_calibration_is_stale():
    """The bias moves ~2 kg/tHM a quarter, so a stale offset silently decays."""

    c = CokeCalibration(offset_kg_per_thm=20.0, sample_days=90,
                        residual_sd_kg_per_thm=10.0, window_days=90,
                        fitted_on="2026-01-01")

    assert c.is_stale(date(2026, 8, 20))


def test_never_fitted_is_not_reported_as_stale():
    """Absent and expired are different problems and read differently."""

    assert NO_CALIBRATION.age_days() is None
    assert not NO_CALIBRATION.is_stale()


# --- round trip ---------------------------------------------------------------------


def test_it_survives_a_save_and_load(tmp_path):
    predicted, actual = _pairs(24.5, noise=2.0)
    original = fit_offset(predicted, actual, days=["2026-05-05", "2026-08-21"])
    path = tmp_path / "calibration.json"

    save_calibration(original, path)
    loaded = load_calibration(path)

    assert loaded.offset_kg_per_thm == pytest.approx(original.offset_kg_per_thm)
    assert loaded.sample_days == original.sample_days
    assert loaded.first_day == "2026-05-05"


def test_the_shipped_calibration_is_sane_if_present():
    """Guards the checked-in file against a bad refresh being committed."""

    c = load_calibration()
    if c is NO_CALIBRATION:
        pytest.skip("no calibration committed")

    # The measured bias has run +12 to +25 kg/tHM. Far outside that band means
    # the refresh picked up bad data.
    assert 0.0 < c.offset_kg_per_thm < 60.0
    assert c.sample_days >= 20
