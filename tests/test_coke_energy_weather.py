"""Blast moisture from the ambient RH at Wardha (DATA-11, utils/coke_energy/weather.py).

Default source open_meteo (STRATA 1.5): Open-Meteo daily mean RH mapped linearly to 7-21 g/Nm3, no smoothing; days
after the shipped file from the forecast endpoint (no network in the tests), else the Open-Meteo day-of-year mean.
rh_scaled (31-day centred mean of the 2025-26 weather record's RH, the STRATA 1.4 basis) and the psychrometric
estimate (Magnus, keeps the record's seasonal steps) remain selectable."""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from utils.coke_energy import weather as W
from utils.coke_energy.bf_thermo import HUMIDITY_G_NM3


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("COKE_ENERGY_WEATHER_LIVE", "0")
    W.clear_cache()
    yield
    W.clear_cache()


def test_moisture_formula():
    # 30 / 20 degC -> 25 degC; e_s(25) = 31.60 hPa (Magnus, WMO coefficients); RH 80 % -> e 25.3 hPa -> 20.6 g/Nm3
    v = float(W.moisture_g_per_nm3(30.0, 20.0, 80.0))
    es = 6.112 * math.exp(17.62 * 25 / (243.12 + 25))
    assert es == pytest.approx(31.60, abs=0.02)
    assert v == pytest.approx(0.8 * es / 985 * 18.015 / 22.414 * 1000, rel=1e-12)
    assert v == pytest.approx(20.6, abs=0.1)


def test_rh_scaled_map():
    assert W.rh_scaled(26.7) == pytest.approx(7.0) and W.rh_scaled(79.7) == pytest.approx(21.0)
    assert W.rh_scaled(53.2) == pytest.approx(14.0, abs=1e-9)
    assert W.rh_scaled(10.0) == 7.0 and W.rh_scaled(95.0) == 21.0                  # clipped to the plant range
    s = pd.Series([40.0] * 10 + [80.0] * 10, index=pd.date_range("2026-01-01", periods=20))
    sm = W.smoothed_rh(s, 7)
    assert sm.iloc[0] == pytest.approx(40.0) and sm.iloc[-1] == pytest.approx(80.0)
    assert 40.0 < sm.loc["2026-01-10"] < 80.0                                       # the step is spread over the window


def test_open_meteo_default():
    rec = W.daily_record()
    assert W._settings()["source"] == "open_meteo"
    assert rec.index.min() == pd.Timestamp("2023-01-01") and rec.index.max() >= pd.Timestamp("2026-10-15")
    assert rec.min() >= 7.0 - 1e-9 and rec.max() <= 21.0 + 1e-9
    rh = W.open_meteo_rh(refresh=False)
    d = pd.Timestamp("2026-06-11")
    assert rec.loc[d] == pytest.approx(7.0 + 14.0 * (rh.loc[d] - 26.06) / (86.19 - 26.06))      # no smoothing
    assert 0.8 < float(rec.diff().std()) < 2.5                                     # day-to-day variation is kept
    jun = rec.loc["2026-06-01":"2026-06-20"].mean()
    assert 9.0 < jun < 14.0                                   # the 2026 monsoon reached Wardha late in June
    v, src = W.blast_moisture("2026-06-11")
    assert src == "Open-Meteo daily RH" and v == pytest.approx(rec.loc[d])


def test_open_meteo_gap_uses_day_of_year_mean():
    v, src = W.blast_moisture("2030-07-15")
    assert src == "Open-Meteo day-of-year mean" and 14.0 < v < 21.0
    v1, _ = W.blast_moisture("2031-07-15")
    assert v1 == pytest.approx(v)
    v, _ = W.blast_moisture("2030-04-15")
    assert 7.0 <= v < 11.0                                                            # dry season


def test_open_meteo_live_fetch_is_cached(monkeypatch, tmp_path):
    calls = []
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    days = pd.date_range(today - pd.Timedelta(days=3), today)

    def fake(*a, **k):
        calls.append(1)
        return pd.Series([50.0, 60.0, 70.0, 80.0], index=days)

    W.clear_cache()
    om = W._settings()["open_meteo"]
    monkeypatch.setitem(om, "cache_path", str(tmp_path / "rh.csv"))
    monkeypatch.setenv("COKE_ENERGY_WEATHER_LIVE", "1")
    monkeypatch.setattr(W, "fetch_open_meteo_rh", fake)
    v, src = W.blast_moisture(today)
    assert src == "Open-Meteo daily RH" and v == pytest.approx(7.0 + 14.0 * (80.0 - 26.06) / (86.19 - 26.06))
    assert (tmp_path / "rh.csv").exists() and len(calls) == 1
    W.blast_moisture(today - pd.Timedelta(days=1))
    assert len(calls) == 1                                                           # within refresh_hours


def test_open_meteo_fetch_failure_falls_back(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("offline")

    W.clear_cache()
    om = W._settings()["open_meteo"]
    monkeypatch.setitem(om, "cache_path", str(tmp_path / "rh.csv"))
    monkeypatch.setenv("COKE_ENERGY_WEATHER_LIVE", "1")
    monkeypatch.setattr(W, "fetch_open_meteo_rh", boom)
    v, src = W.blast_moisture(pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize())
    assert src == "Open-Meteo day-of-year mean" and 7.0 <= v <= 21.0
    assert W._LIVE["t_fail"] > 0


def test_rh_scaled_record_is_step_free():
    rec = W.daily_record(method="rh_scaled")
    assert len(rec) > 600 and rec.index.min() <= pd.Timestamp("2025-12-01") and rec.index.max() >= pd.Timestamp("2026-10-07")
    assert rec.min() >= 7.0 - 1e-9 and rec.max() <= 21.0 + 1e-9
    assert float(np.abs(np.diff(rec.to_numpy())).max()) < 1.0                       # no overnight jump (largest 0.6)
    june = rec.loc["2026-05-25":"2026-06-07"]
    assert float(np.abs(np.diff(june.to_numpy())).max()) < 1.0                       # no step on 1 Jun
    v, src = W.blast_moisture("2026-06-15", source="rh_scaled")
    assert src == "weather record (RH, smoothed)" and v == pytest.approx(rec.loc[pd.Timestamp("2026-06-15")])


def test_psychrometric_keeps_steps():
    psy = W.daily_record(method="psychrometric")
    assert 3.0 < psy.min() and psy.max() < 35.0
    m = psy.groupby(psy.index.to_period("M")).mean()
    for mon in ("2026-06", "2026-07", "2026-08", "2026-09"):
        assert 22.0 < m.loc[mon] < 26.5
    assert psy.loc["2026-06-01"] - psy.loc["2026-05-31"] > 5.0                    # the record's seasonal step
    v, src = W.blast_moisture("2026-06-15", source="psychrometric")
    assert src == "weather record" and v == pytest.approx(psy.loc[pd.Timestamp("2026-06-15")])


def test_fallback():
    v, src = W.blast_moisture("2030-01-15", source="rh_scaled")  # outside the record: monthly table
    assert src == "monthly table" and v == HUMIDITY_G_NM3[1]
    v, src = W.blast_moisture("2026-06-15", source="table")
    assert src == "monthly table" and v == HUMIDITY_G_NM3[6]


def test_window_mean():
    days = pd.date_range("2026-07-01", "2026-07-14")
    v, src = W.mean_blast_moisture(days)
    assert src == "Open-Meteo daily RH"
    assert v == pytest.approx(float(W.daily_record().loc[days].mean()))
    v, src = W.mean_blast_moisture(days, source="rh_scaled")
    assert src == "weather record (RH, smoothed)"
    assert v == pytest.approx(float(W.daily_record(method="rh_scaled").loc[days].mean()))


# ---------------------------------------------------------------------------------------------- plan moisture (live)
def test_today_value_is_refreshed_after_refresh_hours(monkeypatch, tmp_path):
    """Today's value fills in during the day: it is fetched again once the refresh window has passed, even though the
    day is already stored."""
    calls = []
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()

    def fake(*a, **k):
        calls.append(1)
        return pd.Series([60.0 + len(calls)], index=[today])

    W.clear_cache()
    om = W._settings()["open_meteo"]
    monkeypatch.setitem(om, "cache_path", str(tmp_path / "rh.csv"))
    monkeypatch.setenv("COKE_ENERGY_WEATHER_LIVE", "1")
    monkeypatch.setattr(W, "fetch_open_meteo_rh", fake)
    v1 = W.blast_moisture(today)[0]
    W._LIVE["t_ok"] -= 3600.0 * float(om["refresh_hours"]) + 1.0          # the refresh window has passed
    v2 = W.blast_moisture(today)[0]
    assert len(calls) == 2 and v2 > v1


def test_plan_moisture_is_open_meteo_for_the_plan_day(monkeypatch):
    monkeypatch.setenv("COKE_ENERGY_TODAY", "2026-10-15")
    g, src, day = W.plan_blast_moisture()
    assert day == pd.Timestamp("2026-10-15") and src == "Open-Meteo daily RH"
    assert g == pytest.approx(W.blast_moisture("2026-10-15")[0]) and 7.0 <= g <= 21.0
    assert W.resolve_blast_moisture(None) == pytest.approx(g)
    assert W.resolve_blast_moisture(float("nan")) == pytest.approx(g)
    assert W.resolve_blast_moisture(12.5) == 12.5                                   # an operator entry wins


def test_engine_settings_default_to_live_open_meteo(monkeypatch):
    from utils.burden_engine import Engine, Settings
    monkeypatch.setenv("COKE_ENERGY_TODAY", "2026-10-15")
    assert Settings().humidity_g == pytest.approx(W.plan_blast_moisture()[0])
    assert Settings(humidity_g=9.0).humidity_g == 9.0
    e = Engine()
    assert e.cal_settings().humidity_g == pytest.approx(e.cal["case"]["humidity_g"])   # window's Open-Meteo mean
    assert e.cal["case"]["humidity_g"] == pytest.approx(W.mean_blast_moisture(
        pd.date_range("2026-07-01", "2026-10-07"))[0], abs=0.5)   # stable shifts only: 18.45 against 18.83 all days


def test_energy_balance_moisture_follows_open_meteo(monkeypatch):
    from utils.energy_balance.assumptions import current_values
    from utils.energy_balance.constants import load_config
    monkeypatch.setenv("COKE_ENERGY_TODAY", "2026-10-15")
    assert load_config()["blast_moisture_g_per_nm3"] is None                       # resolved at run time
    row = next(r for r in current_values({}) if r["key"] == "blast_moisture_g_per_nm3")
    assert row["Value"] == pytest.approx(W.plan_blast_moisture()[0]) and row["Source"] == "Open-Meteo"
