"""Measured coke-rate aggregation used by BMO calibration and accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.bmo.coke_history import _daily_measured_coke_rate


def test_measured_coke_rate_is_mass_weighted_across_paired_hours():
    index = pd.date_range("2026-09-01", periods=24, freq="h")
    frame = pd.DataFrame({
        "COKE_CALC_MT": [10.0] * 12 + [40.0] * 12,
        "PRODUCTIONTONNESPERHR": [100.0] * 12 + [200.0] * 12,
    }, index=index)

    result = _daily_measured_coke_rate(frame)

    expected = 1000.0 * frame["COKE_CALC_MT"].sum() / frame[
        "PRODUCTIONTONNESPERHR"
    ].sum()
    assert result.iloc[0]["actual_coke"] == pytest.approx(expected)
    assert result.iloc[0]["actual_coke_hours"] == 24
    assert result.iloc[0]["actual_coke"] != pytest.approx(
        (100.0 + 200.0) / 2.0
    )


def test_partial_day_is_excluded_from_calibration_target():
    index = pd.date_range("2026-09-01", periods=30, freq="h")
    frame = pd.DataFrame({
        "COKE_CALC_MT": 30.0,
        "PRODUCTIONTONNESPERHR": 100.0,
    }, index=index)

    result = _daily_measured_coke_rate(frame)

    assert result.iloc[0]["actual_coke"] == pytest.approx(300.0)
    assert np.isnan(result.iloc[1]["actual_coke"])
    assert result.iloc[1]["actual_coke_hours"] == 6


def test_day_with_one_missing_hour_is_not_called_complete():
    index = pd.date_range("2026-09-01", periods=23, freq="h")
    frame = pd.DataFrame({
        "COKE_CALC_MT": 30.0,
        "PRODUCTIONTONNESPERHR": 100.0,
    }, index=index)

    result = _daily_measured_coke_rate(frame)

    assert np.isnan(result.iloc[0]["actual_coke"])
    assert result.iloc[0]["actual_coke_hours"] == 23


def test_missing_plant_measurement_columns_return_no_target():
    frame = pd.DataFrame(
        {"PRODUCTIONTONNESPERHR": [100.0]},
        index=pd.date_range("2026-09-01", periods=1, freq="h"),
    )

    result = _daily_measured_coke_rate(frame)

    assert result.empty
    assert list(result.columns) == ["actual_coke", "actual_coke_hours"]
