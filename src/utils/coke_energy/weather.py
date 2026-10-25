"""Blast moisture from the ambient relative humidity at Wardha (DATA-11).

The cold blast is ambient air (no steam injection on BF-02), so its moisture follows the ambient humidity.

  open_meteo (default)   daily mean relative humidity at Wardha from Open-Meteo (mean of the 24 hourly values of the
                         ERA5-based archive and the forecast model), mapped linearly onto the plant moisture range
                           humidity_g = g_lo + (g_hi - g_lo) (RH - RH_lo) / (RH_hi - RH_lo), clipped to [g_lo, g_hi]
                         with g_lo = 7, g_hi = 21 g/Nm3 and RH_lo / RH_hi = 26.1 / 86.2 % (1st / 99th percentile of the
                         31-day centred Open-Meteo series, 2023 - Oct 2026); no smoothing (``smoothing_days`` 1).
                         Validated on the top-gas H2: +0.041 +/- 0.003 pt H2 per g/Nm3 within months in 2026 and
                         +0.038 +/- 0.003 in 2023-24, the physically expected slope (the RH-scaled record: +0.020 +/-
                         0.011). Days come from the shipped file (``open_meteo.path``, 1 Jan 2023 - 15 Oct 2026), the
                         local cache of earlier fetches (``open_meteo.cache_path``) and, for days after both, the Open-
                         Meteo forecast endpoint (``past_days`` 92, refreshed every ``refresh_hours``; set
                         ``open_meteo.live: false`` or the environment variable COKE_ENERGY_WEATHER_LIVE=0 to disable).
                         Days without a value use the Open-Meteo day-of-year mean (31-day centred), then the monthly
                         table.
  rh_scaled              plant estimate from the Wardha daily weather record (2025-26), without step changes:
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

Source selection (``config/coke_energy.yml -> coke_energy.blast_moisture.source``): ``open_meteo`` | ``rh_scaled`` |
``psychrometric`` (alias ``weather_file``) | ``table`` (the monthly table HUMIDITY_G_NM3 for every date). Dates the
weather record does not cover fall back to the monthly table. Pure functions plus cached file reads and one optional
HTTP request per refresh; no Streamlit.
"""

from __future__ import annotations

import logging
import os
import time
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
SOURCES = ("open_meteo", "rh_scaled", "psychrometric", "weather_file", "table")
RH_DEFAULTS = {"smoothing_days": 31, "g_range": [7.0, 21.0], "rh_range": [26.7, 79.7]}
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_DEFAULTS = {
    "path": "src/assets/coke_energy/open_meteo_rh_wardha_daily.csv",
    "cache_path": "src/storage/coke_energy/open_meteo_rh_wardha_daily.csv",
    "latitude": 20.745, "longitude": 78.602,          # Wardha
    "live": True, "past_days": 92, "timeout_s": 8.0, "refresh_hours": 6.0, "retry_minutes": 60.0,
    "min_hours": 18,                                  # hourly values needed for a day mean
    "smoothing_days": 1, "rh_range": [26.06, 86.19],
}
LABELS = {"open_meteo": "Open-Meteo daily RH", "rh_scaled": "weather record (RH, smoothed)",
          "psychrometric": "weather record", "open_meteo_doy": "Open-Meteo day-of-year mean"}
log = logging.getLogger(__name__)


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


def _repo_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else SRC.parent / p)    # repo-relative, like the other data_sources paths


@lru_cache(maxsize=4)
def _settings(config_path: str = str(CONFIG_PATH)) -> dict[str, Any]:
    out = {"source": "open_meteo", "path": str(DEFAULT_PATH), "station_pressure_hpa": P_STATION_HPA, **RH_DEFAULTS}
    om = dict(OPEN_METEO_DEFAULTS)
    try:
        import yaml
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        bm = dict((cfg.get("coke_energy", cfg) or {}).get("blast_moisture") or {})
        om.update({k: v for k, v in (bm.pop("open_meteo", None) or {}).items() if v is not None})
        out.update({k: v for k, v in bm.items() if v is not None})
    except FileNotFoundError:
        pass
    out["path"] = _repo_path(out["path"])
    om["path"], om["cache_path"] = _repo_path(om["path"]), _repo_path(om["cache_path"])
    out["open_meteo"] = om
    if out["source"] not in SOURCES:
        raise ValueError(f"blast_moisture.source must be one of {SOURCES}, not {out['source']!r}")
    return out


# ---------------------------------------------------------------------------------------------- Open-Meteo RH
def _read_rh_csv(path: str) -> pd.Series:
    if not Path(path).exists():
        return pd.Series(dtype=float)
    w = pd.read_csv(path, parse_dates=["Date"]).dropna(subset=["Date", "RH_mean_pct"])
    s = pd.Series(w.RH_mean_pct.to_numpy(float), index=pd.DatetimeIndex(w.Date).normalize())
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_open_meteo_rh(latitude: float, longitude: float, past_days: int = 92, timeout_s: float = 8.0,
                        min_hours: int = 18) -> pd.Series:
    """Daily mean RH (%) at a point from the Open-Meteo forecast endpoint: hourly relative_humidity_2m in IST, days
    with at least ``min_hours`` values. Raises on any network or format error."""
    import requests
    r = requests.get(OPEN_METEO_URL, timeout=float(timeout_s), params=dict(
        latitude=float(latitude), longitude=float(longitude), hourly="relative_humidity_2m",
        past_days=int(past_days), forecast_days=1, timezone="Asia/Kolkata"))
    r.raise_for_status()
    h = r.json()["hourly"]
    x = pd.Series(pd.to_numeric(h["relative_humidity_2m"], errors="coerce"), index=pd.to_datetime(h["time"])).dropna()
    g = x.groupby(x.index.normalize())
    d = g.mean()[g.size() >= int(min_hours)]
    return d.astype(float)


_LIVE: dict[str, Any] = {"t_ok": 0.0, "t_fail": 0.0}


def _live_enabled(om: dict[str, Any]) -> bool:
    env = os.environ.get("COKE_ENERGY_WEATHER_LIVE")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(om.get("live", True))


def _refresh_live(om: dict[str, Any], known_last: pd.Timestamp | None) -> None:
    """Fetch recent days when they are missing from file and cache; write them to the cache. Failures are remembered
    for ``retry_minutes`` so that an offline plant does not wait on every call."""
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    if not _live_enabled(om) or (known_last is not None and known_last >= today):
        return
    now = time.time()
    if now - _LIVE["t_ok"] < 3600.0 * float(om["refresh_hours"]) or now - _LIVE["t_fail"] < 60.0 * float(om["retry_minutes"]):
        return
    try:
        d = fetch_open_meteo_rh(om["latitude"], om["longitude"], om["past_days"], om["timeout_s"], om["min_hours"])
    except Exception as exc:                                   # offline, proxy, format: keep the file values
        _LIVE["t_fail"] = now
        log.warning("Open-Meteo RH fetch failed (%s); using the stored days", exc)
        return
    _LIVE["t_ok"] = now
    if d.empty:
        return
    cache = _read_rh_csv(om["cache_path"])
    out = pd.concat([cache[~cache.index.isin(d.index)], d]).sort_index()
    try:
        Path(om["cache_path"]).parent.mkdir(parents=True, exist_ok=True)
        tmp = om["cache_path"] + ".tmp"
        pd.DataFrame({"Date": out.index.strftime("%Y-%m-%d"), "RH_mean_pct": out.round(3).to_numpy()}).to_csv(tmp, index=False)
        os.replace(tmp, om["cache_path"])
    except OSError as exc:
        log.warning("Open-Meteo RH cache not written (%s)", exc)
    _rh_open_meteo.cache_clear()
    daily_record.cache_clear()


@lru_cache(maxsize=2)
def _rh_open_meteo(path: str, cache_path: str) -> pd.Series:
    base, cache = _read_rh_csv(path), _read_rh_csv(cache_path)
    return pd.concat([base, cache[~cache.index.isin(base.index)]]).sort_index()   # shipped file wins on overlap


def open_meteo_rh(refresh: bool = True) -> pd.Series:
    """Daily mean RH (%) at Wardha: shipped file + cache, refreshed from the forecast endpoint when recent days are
    missing."""
    om = _settings()["open_meteo"]
    rh = _rh_open_meteo(om["path"], om["cache_path"])
    if refresh:
        _refresh_live(om, rh.index.max() if len(rh) else None)
        rh = _rh_open_meteo(om["path"], om["cache_path"])
    return rh


def open_meteo_record(refresh: bool = True) -> pd.Series:
    """Daily blast moisture (g/Nm3) from the Open-Meteo RH (``open_meteo`` source)."""
    om = _settings()["open_meteo"]
    rh = open_meteo_rh(refresh)
    if rh.empty:
        return pd.Series(dtype=float, name="humidity_g")
    k = int(om["smoothing_days"])
    rs = smoothed_rh(rh, k) if k > 1 else rh
    g = _settings()["g_range"]
    return pd.Series(rh_scaled(rs.to_numpy(), g, om["rh_range"]), index=rs.index, name="humidity_g").dropna()


@lru_cache(maxsize=2)
def _open_meteo_doy(path: str, cache_path: str) -> pd.Series:
    """Day-of-year mean of the Open-Meteo moisture (31-day centred on the circular calendar), index 1..366."""
    om = _settings()["open_meteo"]
    rh = _rh_open_meteo(path, cache_path)
    if rh.empty:
        return pd.Series(dtype=float)
    m = rh.groupby(rh.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([m.iloc[-15:], m, m.iloc[:15]])
    sm = ext.rolling(31, center=True).mean().iloc[15:-15]
    sm.index = range(1, 367)
    return pd.Series(rh_scaled(sm.to_numpy(), _settings()["g_range"], om["rh_range"]), index=sm.index)


@lru_cache(maxsize=8)
def daily_record(path: str | None = None, p_hpa: float | None = None, method: str | None = None) -> pd.Series:
    """Daily blast moisture (g/Nm3) indexed by date for the configured method; empty when the file is missing.
    ``open_meteo`` reads the stored days only (no fetch); ``blast_moisture`` refreshes them."""
    s = _settings()
    method = method or s["source"]
    if method == "open_meteo":
        return open_meteo_record(refresh=False)
    path = path or s["path"]
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
    """(g/Nm3, source label) for one date: the configured estimate, else (open_meteo) the Open-Meteo day-of-year mean,
    else the month's table value."""
    d = pd.Timestamp(day).normalize()
    src = source or _settings()["source"]
    if src == "weather_file":
        src = "psychrometric"
    if src == "open_meteo":
        open_meteo_rh(refresh=True)                            # recent days from the endpoint when missing (rate-limited)
        rec = daily_record(method="open_meteo")
        if d in rec.index and np.isfinite(rec.loc[d]):
            return float(rec.loc[d]), LABELS["open_meteo"]
        om = _settings()["open_meteo"]
        doy = _open_meteo_doy(om["path"], om["cache_path"])
        if len(doy) and np.isfinite(doy.get(d.dayofyear, np.nan)):
            return float(doy.loc[d.dayofyear]), LABELS["open_meteo_doy"]
    elif src != "table":
        rec = daily_record(method=src)
        if d in rec.index and np.isfinite(rec.loc[d]):
            return float(rec.loc[d]), LABELS[src]
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
    _rh_open_meteo.cache_clear()
    _open_meteo_doy.cache_clear()
    _LIVE.update(t_ok=0.0, t_fail=0.0)
