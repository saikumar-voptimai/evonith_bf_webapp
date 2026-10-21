"""Calibration & regime adapter (#252) on the Coke Energy fixture shifts."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from data.burden_engine.calibration import live_calibration
from utils.burden_engine import Engine
from utils.burden_engine.calibration import CalibrationError, calibrate_from_shifts, operating_regime, select_shifts
from utils.coke_energy.pipeline import PipelineSettings, run_shift_balances

FIX = "tests/fixtures/coke_energy/shift_inputs.csv"


@pytest.fixture(scope="module")
def SR():
    S = pd.read_csv(FIX, index_col=0)
    R, errors = run_shift_balances(S, S.index)
    assert not errors
    return S, R


def test_calibration_reproduces_actual_coke(SR):
    S, R = SR
    keys = select_shifts(S, R, start="2026-09-01")
    assert list(keys) == ["2026-09-18_C", "2026-10-14_A"]
    cal = calibrate_from_shifts(S, R, keys)
    assert cal["coke_model_cal"] == pytest.approx(cal["K"], abs=0.5)
    assert 0.6 < cal["omega"] < 1.0
    assert cal["Q_LZ_GJh"] > 0 and cal["Q_UZ_GJh"] > 0
    assert cal["anchor"]["V"] == pytest.approx(S.loc[keys, "hot_blast_vol_nm3h"].mean())
    # the engine runs on it and returns the calibration coke at the calibration controls
    eng = Engine(cal=cal)
    assert eng.hy.base["V"] == pytest.approx(cal["anchor"]["V"])
    r = eng.min_cost(prod_t_day=2270, fuel=True)
    assert r["feasible"]


def test_too_few_shifts(SR):
    S, R = SR
    with pytest.raises(CalibrationError):
        calibrate_from_shifts(S, R, [], min_shifts=10)


def test_regime_is_medians(SR):
    S, R = SR
    keys = S.index
    reg = operating_regime(S, R, keys)
    assert reg["HBT"] == pytest.approx(S["hot_blast_temp"].median())
    assert reg["Pb"] == pytest.approx(S["hot_blast_press"].median())
    assert reg["prod_th"] == pytest.approx(R["prod_th"].median())
    o2 = S["oxygen_flow"] / (S["hot_blast_vol_nm3h"] - S["oxygen_flow"])
    assert reg["O2_per_air"] == pytest.approx(o2.median())
    # blast moisture: the weather record's value for the last shift's date (DATA-11), table outside the record
    from utils.coke_energy.weather import blast_moisture
    assert (reg["humidity_g"], reg["humidity_source"]) == blast_moisture(pd.to_datetime(S["sdate"]).max())


def test_live_calibration_falls_back_with_few_shifts(SR):
    S, R = SR
    res = SimpleNamespace(S=S, R=R, settings=PipelineSettings())
    info = live_calibration(res, {"calibration": {"window_days": 60, "min_shifts": 10}})
    assert info.source == "fallback" and "shipped" in info.message
    assert info.regime is not None and info.regime["n_shifts"] == 2
    info = live_calibration(res, {"calibration": {"window_days": 60, "min_shifts": 2}})
    assert info.source == "live" and info.n_shifts == 2
