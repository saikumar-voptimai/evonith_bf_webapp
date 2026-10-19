"""Cached data loading for the Coke Energy Model page.

Every expensive step is cached on small tokens (file modification time, a clock rounded to the refresh interval and a
manual refresh counter), never on the large frames themselves, so a rerun costs nothing until the data can have
changed:

    hourly dataset    re-read when the CSV file changes
    10-min online     disk cache per month (data.coke_energy.sources); re-read at most every refresh interval
    Neon              HM/slag taps, DPR and the lab-sample times for the freshness notices
    pipeline          re-run when any of the above changed (per-shift results are memoised, so this is quick)
    Rist frames       per set of shifts on the slider
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from data.coke_energy import sources
from data.coke_energy.freshness import fetch_offline_samples, material_names
from utils.coke_energy.bf_balance import DEFAULTS, calibrate_loss
from utils.coke_energy.pipeline import PipelineResult, PipelineSettings, run_pipeline, shift_params
from utils.coke_energy.rist_diagram import RistShift, shift_rist
from utils.coke_energy.run_shifts import case_from_shift

PLANT_TZ = "Asia/Kolkata"
# How far back lab samples are read for the freshness check (a burden-distribution change can be months old).
SAMPLE_LOOKBACK = pd.Timedelta(days=60)


@dataclass(frozen=True)
class DataKeys:
    """Cache tokens for one load of the page's data."""

    hourly_path: str
    hourly_mtime_ns: int
    clock: str                 # now, rounded down to the refresh interval
    nonce: int                 # bumped by the Refresh button
    history_start: str
    fields_json: str
    cache_dir: str
    online_refresh_minutes: float


@dataclass
class PageData:
    """Everything the page draws from: the model results and what the freshness notices need."""

    result: PipelineResult
    fresh: sources.SourceFreshness
    provisional: list
    samples: pd.DataFrame          # source, material, t (IST): one row per valid record of every source
    charged: pd.DataFrame          # tonnes charged per IST hour, one column per material code (+ 'charges')
    names: dict = field(default_factory=dict)


def data_keys(cfg: Mapping, nonce: int) -> DataKeys:
    """Tokens for the current state of the sources."""

    ds = cfg.get("data_sources", {}) or {}
    status = sources.static_dataset_status(ds)
    minutes = int(ds.get("neon_cache_minutes", 30) or 30)
    clock = pd.Timestamp.now(tz=PLANT_TZ).floor(f"{minutes}min").isoformat()
    return DataKeys(
        hourly_path=str(status["path"]),
        hourly_mtime_ns=int(status["mtime_ns"]),
        clock=clock,
        nonce=int(nonce),
        history_start=str(cfg.get("history_start", "2025-08-01")),
        fields_json=json.dumps(cfg.get("influx_fields", {}), sort_keys=True),
        cache_dir=str(ds.get("influx_cache_dir", "src/storage/coke_energy/influx_10min")),
        online_refresh_minutes=float(ds.get("influx_current_month_refresh_minutes", 30) or 30),
    )


@st.cache_data(show_spinner="Reading the hourly furnace dataset…", max_entries=2)
def hourly_dataset(path: str, mtime_ns: int) -> pd.DataFrame:
    return sources.load_hourly_dataset(path)


@st.cache_data(show_spinner="Reading 10-min online data (the first run fetches about 15 months from InfluxDB, ~1 min)…",
               max_entries=2)
def online_data(start: str, clock: str, nonce: int, fields_json: str, cache_dir: str, refresh_minutes: float) -> pd.DataFrame:
    end = pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None)
    return sources.fetch_10min(pd.Timestamp(start), end, fields=json.loads(fields_json), cache_dir=cache_dir,
                               refresh_minutes=refresh_minutes)


@st.cache_data(show_spinner="Reading hot metal / slag analyses and the DPR…", max_entries=2)
def neon_data(start: str, clock: str, nonce: int) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    end = pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None)
    hm = sources.fetch_hm_slag(pd.Timestamp(start), end)
    dpr = sources.fetch_dpr_daily(pd.Timestamp(start) - pd.Timedelta(days=1), end)
    dpr, provisional = sources.extend_dpr_provisional(dpr, end)
    return hm, dpr, sorted(provisional)


@st.cache_data(show_spinner="Reading lab-sample times for the freshness check…", max_entries=2)
def lab_samples(start: str, clock: str, nonce: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_offline_samples(pd.Timestamp(start) - SAMPLE_LOOKBACK)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def names() -> dict:
    try:
        return material_names()
    except Exception:  # noqa: BLE001 - codes are shown instead of names
        return {}


def _source_times(fd: pd.DataFrame, m10: pd.DataFrame, hm: pd.DataFrame, dpr: pd.DataFrame, provisional: Sequence,
                  charged: pd.DataFrame) -> pd.DataFrame:
    """When each in-app source last had a valid value: end of the hour / 10-min bin / DPR date, tap sample time."""

    parts = []

    def add(source: str, times: pd.Series | pd.Index) -> None:
        t = pd.Series(pd.to_datetime(times)).dropna()
        parts.append(pd.DataFrame({"source": source, "material": "", "t": t.to_numpy()}))

    if not fd.empty:
        add("hourly_dataset", fd.loc[fd["COKE_CALC_MT"].notna(), "time"] + pd.Timedelta(hours=1))
    if not m10.empty:
        add("online_10min", m10.loc[m10["hot_blast_vol_nm3h"].notna(), "ts"] + pd.Timedelta(minutes=10))
    if not hm.empty:
        add("hm_slag", hm.loc[hm["chem_pct_si"].notna() | hm["slag_pct_al2o3"].notna(), "t"])
    if not dpr.empty:
        real = dpr.index[~dpr.index.isin(pd.to_datetime(list(provisional)))]
        add("dpr", real + pd.Timedelta(days=1))
    if not charged.empty and "charges" in charged:
        add("charge_data", charged.index[charged["charges"] > 0] + pd.Timedelta(hours=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["source", "material", "t"])


@st.cache_data(show_spinner="Balancing every shift and running both models…", max_entries=2)
def page_data(keys: DataKeys, settings: PipelineSettings) -> PageData:
    fd = hourly_dataset(keys.hourly_path, keys.hourly_mtime_ns)
    m10 = online_data(keys.history_start, keys.clock, keys.nonce, keys.fields_json, keys.cache_dir,
                      keys.online_refresh_minutes)
    hm, dpr, provisional = neon_data(keys.history_start, keys.clock, keys.nonce)
    result = run_pipeline(fd, m10, hm, dpr, settings, provisional_dates=provisional)
    real_dpr = dpr[~dpr.index.isin(pd.to_datetime(provisional))]
    try:
        lab, charged = lab_samples(keys.history_start, keys.clock, keys.nonce)
    except Exception:  # noqa: BLE001 - the page still works; the freshness table says the lab times are missing
        lab, charged = pd.DataFrame(columns=["source", "material", "t"]), pd.DataFrame()
    samples = pd.concat([lab, _source_times(fd, m10, hm, dpr, provisional, charged)], ignore_index=True)
    return PageData(result=result, fresh=sources.freshness(fd, m10, hm, real_dpr), provisional=provisional,
                    samples=samples, charged=charged, names=names())


@st.cache_data(show_spinner="Drawing the Rist diagram for each shift…", max_entries=4)
def rist_frames(keys: DataKeys, settings: PipelineSettings, skeys: Sequence[str]) -> list[RistShift]:
    """Rist geometry per shift: the line of what the blast burnt (bias-corrected burnt coke, with the blast that closes
    the oxygen balance at the analyser's eta CO) and the energy model's line with omega fixed.

    The web model calibrated each shift on charged coke; charged coke reaches the tuyeres 4-5 h later, which made the
    per-shift line jump (lower-zone loss changed by 18 GJ/h from shift to shift against 6 GJ/h on burnt coke)."""

    data = page_data(keys, settings)
    S, R, T = data.result.S, data.result.R, data.result.shifts
    params = shift_params(S, pd.Index(list(skeys)), settings)
    frames = []
    for k in skeys:
        rb, row, p = R.loc[k], T.loc[k], params[k]
        K = row.K_burnt
        if not np.isfinite(K):
            continue
        case = case_from_shift(S.loc[k])
        case["eta_H2"] = rb.eta_H2 / 100 if 10 < rb.eta_H2 < 70 else DEFAULTS["eta_H2"]
        eta = rb.eta_CO / 100
        try:
            _, V, _ = calibrate_loss(case, K, p, eta_CO=eta)
            frames.append(shift_rist(case, K, V, eta, prod_th=rb.prod_th, p=p, skey=k,
                                     omega_fixed=row.omega_fixed if pd.notna(row.omega_fixed) else None,
                                     Q_LZ_GJh_fixed=row.Q_LZ_GJh_fixed if pd.notna(row.Q_LZ_GJh_fixed) else None,
                                     Q_UZ_GJh_fixed=row.Q_UZ_GJh_fixed if pd.notna(row.Q_UZ_GJh_fixed) else None,
                                     r_H_fixed=row.r_H_fixed if pd.notna(row.r_H_fixed) else None))
        except Exception:  # noqa: BLE001 - a shift whose geometry does not solve is left off the slider
            continue
    return frames


def clear_caches() -> None:
    """Forget everything loaded (the Refresh button also bumps the nonce, so the disk cache is re-checked)."""

    for fn in (hourly_dataset, online_data, neon_data, lab_samples, page_data, rist_frames):
        fn.clear()
