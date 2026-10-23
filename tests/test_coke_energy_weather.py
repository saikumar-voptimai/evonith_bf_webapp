"""Blast moisture from the Wardha daily weather record (DATA-11, utils/coke_energy/weather.py).

Default source rh_scaled: 31-day centred mean of RH mapped linearly to 7-21 g/Nm3, no step changes. The psychrometric
estimate (Magnus) remains selectable and keeps the record's seasonal steps."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from utils.coke_energy import weather as W
from utils.coke_energy.bf_thermo import HUMIDITY_G_NM3


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


def test_record_default_is_step_free():
    rec = W.daily_record()
    assert len(rec) > 600 and rec.index.min() <= pd.Timestamp("2025-12-01") and rec.index.max() >= pd.Timestamp("2026-10-07")
    assert rec.min() >= 7.0 - 1e-9 and rec.max() <= 21.0 + 1e-9
    assert float(np.abs(np.diff(rec.to_numpy())).max()) < 1.0                       # no overnight jump (largest 0.6)
    june = rec.loc["2026-05-25":"2026-06-07"]
    assert float(np.abs(np.diff(june.to_numpy())).max()) < 1.0                       # no step on 1 Jun
    v, src = W.blast_moisture("2026-06-15")
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
    v, src = W.blast_moisture("2030-01-15")                      # outside the record: monthly table
    assert src == "monthly table" and v == HUMIDITY_G_NM3[1]
    v, src = W.blast_moisture("2026-06-15", source="table")
    assert src == "monthly table" and v == HUMIDITY_G_NM3[6]


def test_window_mean():
    days = pd.date_range("2026-07-01", "2026-07-14")
    v, src = W.mean_blast_moisture(days)
    assert src == "weather record (RH, smoothed)"
    assert v == pytest.approx(float(W.daily_record().loc[days].mean()))
