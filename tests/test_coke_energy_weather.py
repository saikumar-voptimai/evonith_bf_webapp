"""Blast moisture from the Wardha daily weather record (DATA-11, utils/coke_energy/weather.py)."""

from __future__ import annotations

import math

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


def test_record_and_fallback():
    rec = W.daily_record()
    assert len(rec) > 600 and rec.index.min() <= pd.Timestamp("2025-12-01") and rec.index.max() >= pd.Timestamp("2026-10-07")
    assert 3.0 < rec.min() and rec.max() < 35.0
    v, src = W.blast_moisture("2026-06-15")
    assert src == "weather record" and v == pytest.approx(rec.loc[pd.Timestamp("2026-06-15")])
    v, src = W.blast_moisture("2030-01-15")                      # outside the record: monthly table
    assert src == "monthly table" and v == HUMIDITY_G_NM3[1]
    v, src = W.blast_moisture("2026-06-15", source="table")
    assert src == "monthly table" and v == HUMIDITY_G_NM3[6]


def test_monsoon_level():
    """Jun-Sep 2026 monthly means of the record: 23-25 g/Nm3 (the table held 17-21)."""
    rec = W.daily_record()
    m = rec.groupby(rec.index.to_period("M")).mean()
    for mon in ("2026-06", "2026-07", "2026-08", "2026-09"):
        assert 22.0 < m.loc[mon] < 26.5


def test_window_mean():
    days = pd.date_range("2026-07-01", "2026-07-14")
    v, src = W.mean_blast_moisture(days)
    assert src == "weather record"
    assert v == pytest.approx(float(W.daily_record().loc[days].mean()))
