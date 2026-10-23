"""Blast moisture from the Wardha daily weather record (DATA-11).

The cold blast is ambient air (no steam injection on BF-02), so its moisture follows the ambient humidity. Two
estimates are available from the record (daily maximum and minimum temperature, mean relative humidity):

  rh_scaled (default)    plant estimate from the relative humidity, without step changes:
                           RH_s(d) = centred mean of RH over ``smoothing_days`` (31) days
                           humidity_g = g_lo + (g_hi - g_lo) (RH_s - RH_lo) / (RH_hi - RH_lo), clipped to [g_lo, g_hi]
                         with g_lo = 7, g_hi = 21 g/Nm3 (the plant's moisture range, dry season to monsoon) and
                         RH_lo / RH_hi the 1st / 99th percentile of RH_s over the record (26.7 / 79.7 %). The record's
                         seasonal blocks change RH by 40-60 points overnight (1 Mar, 1 Jun, 1 Oct, 1 Dec); a furnace
                         does not see such steps, so the centred mean spreads each seasonal change over a month
                         (largest day-to-day change 0.6 g/Nm3).
  psychrometric          e = RH/100 x 6.112 exp(17.62 T / (243.12 + T)) hPa, T = (T_max + T_min)/2 (Magnus);
                         humidity_g = e / P x 18.015 / 22.414 x 1000 g H2O per Nm3 humid air, P = 985 hPa. Keeps
                         the record's steps; monsoon 23-25 g/Nm3.

Source selection (``config/coke_energy.yml -> coke_energy.blast_moisture.source``): ``rh_scaled`` | ``psychrometric``
(alias ``weather_file``) | ``table`` (the monthly table HUMIDITY_G_NM3 for every date). Dates the record does not cover
fall back to the monthly table. Pure functions plus one cached file read; no Streamlit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.coke_energy.bf_thermo import HUMIDITY_G_NM3

SRC = Path(__file__).resolve().parents[2]                 # .../src
DEFAULT_PATH = SRC / "assets" / "coke_energy" / "wardha_daily_weather_2025_2026.csv"
CONFIG_PATH = SRC / "config" / "coke_energy.yml"
P_STATION_HPA = 985.0
SOURCES = ("rh_scaled", "psychrometric", "weather_file", "table")
RH_DEFAULTS = {"smoothing_days": 31, "g_range": [7.0, 21.0], "rh_range": [26.7, 79.7]}


def moisture_g_per_nm3(t_max_c, t_min_c, rh_pct, p_hpa: float = P_STATION_HPA):
    """Psychrometric absolute humidity, g H2O per Nm3 of humid air, from daily max / min temperature and mean RH."""
    t = (np.asarray(t_max_c, float) + np.asarray(t_min_c, float)) / 2.0
    e = np.asarray(rh_pct, float) / 100.0 * 6.112 * np.exp(17.62 * t / (243.12 + t))
    return e / p_hpa * 18.015 / 22.414 * 1000.0


def smoothed_rh(rh: pd.Series, smoothing_days: int = 31) -> pd.Series:
    """Centred mean of the daily RH over ``smoothing_days`` calendar days (gaps filled by the mean of the days present)."""
    s = rh.astype(float).copy()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max()))
    return s.rolling(int(smoothing_days), center=True, min_periods=max(3, int(smoothing_days) // 3)).mean()


def rh_scaled(rh_s, g_range=(7.0, 21.0), rh_range=(26.7, 79.7)):
    """Map smoothed RH linearly onto the plant moisture range (array-safe)."""
    g_lo, g_hi = float(g_range[0]), float(g_range[1])
    r_lo, r_hi = float(rh_range[0]), float(rh_range[1])
    v = g_lo + (g_hi - g_lo) * (np.asarray(rh_s, float) - r_lo) / (r_hi - r_lo)
    return np.clip(v, g_lo, g_hi)


@lru_cache(maxsize=4)
def _settings(config_path: str = str(CONFIG_PATH)) -> dict[str, Any]:
    out = {"source": "rh_scaled", "path": str(DEFAULT_PATH), "station_pressure_hpa": P_STATION_HPA, **RH_DEFAULTS}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        bm = (cfg.get("coke_energy", cfg) or {}).get("blast_moisture") or {}
        out.update({k: v for k, v in bm.items() if v is not None})
    except FileNotFoundError:
        pass
    p = Path(out["path"])
    if not p.is_absolute():
        p = SRC.parent / p                                  # repo-relative, like the other data_sources paths
    out["path"] = str(p)
    if out["source"] not in SOURCES:
        raise ValueError(f"blast_moisture.source must be one of {SOURCES}, not {out['source']!r}")
    return out


@lru_cache(maxsize=8)
def daily_record(path: str | None = None, p_hpa: float | None = None, method: str | None = None) -> pd.Series:
    """Daily blast moisture (g/Nm3) indexed by date for the configured method; empty when the file is missing."""
    s = _settings()
    path = path or s["path"]
    method = method or s["source"]
    if method == "weather_file":
        method = "psychrometric"
    if not Path(path).exists() or method == "table":
        return pd.Series(dtype=float, name="humidity_g")
    w = pd.read_csv(path, parse_dates=["Date"])
    w = w.dropna(subset=["Date", "Max_Temperature_C", "Min_Temperature_C", "Average_Humidity_Percent"])
    idx = pd.DatetimeIndex(w.Date).normalize()
    if method == "psychrometric":
        v = moisture_g_per_nm3(w.Max_Temperature_C, w.Min_Temperature_C, w.Average_Humidity_Percent,
                               float(p_hpa or s["station_pressure_hpa"]))
        out = pd.Series(v, index=idx, name="humidity_g")
        return out[~out.index.duplicated(keep="last")].sort_index()
    rh = pd.Series(w.Average_Humidity_Percent.to_numpy(float), index=idx)
    rs = smoothed_rh(rh, int(s["smoothing_days"]))
    out = pd.Series(rh_scaled(rs.to_numpy(), s["g_range"], s["rh_range"]), index=rs.index, name="humidity_g")
    return out.dropna()


def blast_moisture(day: Any, source: str | None = None) -> tuple[float, str]:
    """(g/Nm3, source label) for one date: the weather record's estimate, else the month's table value."""
    d = pd.Timestamp(day).normalize()
    src = source or _settings()["source"]
    if src != "table":
        rec = daily_record(method=src)
        if d in rec.index and np.isfinite(rec.loc[d]):
            label = "weather record (RH, smoothed)" if src == "rh_scaled" else "weather record"
            return float(rec.loc[d]), label
    return float(HUMIDITY_G_NM3[d.month]), "monthly table"


def blast_moisture_g(day: Any, source: str | None = None) -> float:
    return blast_moisture(day, source)[0]


def mean_blast_moisture(days, source: str | None = None) -> tuple[float, str]:
    """Mean over several dates (e.g. a calibration window) and the source(s) used."""
    vals = [blast_moisture(d, source) for d in pd.to_datetime(list(days))]
    if not vals:
        return blast_moisture(pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None), source)
    labels = sorted({lab for _, lab in vals})
    return float(np.mean([v for v, _ in vals])), " + ".join(labels)


def clear_cache() -> None:
    _settings.cache_clear()
    daily_record.cache_clear()
