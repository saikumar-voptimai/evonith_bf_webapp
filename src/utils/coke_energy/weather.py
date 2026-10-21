"""Blast moisture from the Wardha daily weather record (DATA-11).

The cold blast is ambient air, so its moisture is the ambient absolute humidity. The record holds daily maximum and
minimum temperature and mean relative humidity; the moisture of the day is

    T = (T_max + T_min) / 2,   e = RH/100 x 6.112 exp(17.62 T / (243.12 + T))   hPa (Magnus, over water)
    humidity_g = e / P x 18.015 / 22.414 x 1000    g H2O per Nm3 of humid air (the basis of bf_balance)

with the station pressure P = 985 hPa (Wardha, about 250 m above sea level).

Source selection (``config/coke_energy.yml -> coke_energy.blast_moisture``):
  source: weather_file   the day's value from the record; dates the record does not cover fall back to the monthly
                         table HUMIDITY_G_NM3 (bf_thermo)
  source: table          the monthly table for every date (the behaviour before Oct 2026)

Pure functions plus one cached file read; no Streamlit.
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
SOURCES = ("weather_file", "table")


def moisture_g_per_nm3(t_max_c, t_min_c, rh_pct, p_hpa: float = P_STATION_HPA):
    """Absolute humidity, g H2O per Nm3 of humid air, from daily max / min temperature and mean RH (array-safe)."""
    t = (np.asarray(t_max_c, float) + np.asarray(t_min_c, float)) / 2.0
    e = np.asarray(rh_pct, float) / 100.0 * 6.112 * np.exp(17.62 * t / (243.12 + t))
    return e / p_hpa * 18.015 / 22.414 * 1000.0


@lru_cache(maxsize=4)
def _settings(config_path: str = str(CONFIG_PATH)) -> dict[str, Any]:
    out = {"source": "weather_file", "path": str(DEFAULT_PATH), "station_pressure_hpa": P_STATION_HPA}
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


@lru_cache(maxsize=4)
def daily_record(path: str | None = None, p_hpa: float | None = None) -> pd.Series:
    """Daily blast moisture (g/Nm3) indexed by date; empty when the file is missing."""
    s = _settings()
    path = path or s["path"]
    p_hpa = float(p_hpa or s["station_pressure_hpa"])
    if not Path(path).exists():
        return pd.Series(dtype=float, name="humidity_g")
    w = pd.read_csv(path, parse_dates=["Date"])
    w = w.dropna(subset=["Date", "Max_Temperature_C", "Min_Temperature_C", "Average_Humidity_Percent"])
    v = moisture_g_per_nm3(w.Max_Temperature_C, w.Min_Temperature_C, w.Average_Humidity_Percent, p_hpa)
    out = pd.Series(v, index=pd.DatetimeIndex(w.Date).normalize(), name="humidity_g")
    return out[~out.index.duplicated(keep="last")].sort_index()


def blast_moisture(day: Any, source: str | None = None) -> tuple[float, str]:
    """(g/Nm3, source label) for one date: the weather record's day, else the month's table value."""
    d = pd.Timestamp(day).normalize()
    src = source or _settings()["source"]
    if src == "weather_file":
        rec = daily_record()
        if d in rec.index and np.isfinite(rec.loc[d]):
            return float(rec.loc[d]), "weather record"
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
