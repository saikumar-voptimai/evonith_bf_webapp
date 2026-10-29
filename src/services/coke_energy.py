"""Coke Energy data for TRACE, FORGE and the Live Twin: the pipeline result and what the freshness notices need.

Moved from ``ui/coke_energy/loaders.py`` (#271) without changing the composition. Two ways in:

* ``page_data(keys, settings)`` / ``rist_frames(...)``: memoised here, for the API and the worker;
* ``build_page_data(keys, settings, loaders)`` / ``build_rist_frames(data, settings, skeys)``: uncached, for callers
  that cache each step themselves. The Streamlit page passes its ``st.cache_data`` loaders, so it keeps one spinner
  per step and runs this same composition.

Every expensive step is keyed on small tokens (file modification time, a clock rounded to the refresh interval and
a manual refresh counter), never on the large frames themselves:

    hourly dataset    re-read when the CSV file changes
    10-min online     disk cache per month (data.coke_energy.sources); re-read at most every refresh interval
    Neon              HM/slag taps, DPR and the lab-sample times for the freshness notices
    pipeline          re-run when any of the above changed (per-shift results are memoised, so this is quick)
    Rist frames       per set of shifts
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config.config_loader import load_config
from data.coke_energy import sources
from data.coke_energy.freshness import fetch_offline_samples, material_names
from utils.coke_energy.bf_balance import DEFAULTS, calibrate_loss
from utils.coke_energy.pipeline import PipelineResult, PipelineSettings, run_pipeline, shift_params
from utils.coke_energy.rist_diagram import RistShift, shift_rist
from utils.coke_energy.run_shifts import case_from_shift
from utils.memo import memo

PLANT_TZ = "Asia/Kolkata"
# How far back lab samples are read for the freshness check (a burden-distribution change can be months old).
SAMPLE_LOOKBACK = pd.Timedelta(days=60)
# Most shifts one Rist animation draws (page 10 shows the latest 360 of a longer period).
RIST_MAX_FRAMES = 360
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "coke_energy.yml"


@memo(max_entries=2)
def _read_config(mtime_ns: int) -> dict:
    return load_config("coke_energy.yml").get("coke_energy", {})


def config() -> dict:
    """The ``coke_energy`` block of ``config/coke_energy.yml``, re-read when the file changes. Treat as read-only."""
    return _read_config(CONFIG_PATH.stat().st_mtime_ns if CONFIG_PATH.exists() else 0)


@dataclass(frozen=True)
class DataKeys:
    """Cache tokens for one load of the page's data."""

    hourly_path: str
    hourly_mtime_ns: int
    clock: str  # now, rounded down to the refresh interval
    nonce: int  # bumped by the Refresh button
    history_start: str
    fields_json: str
    cache_dir: str
    online_refresh_minutes: float


@dataclass
class PageData:
    """Everything the pages draw from: the model results and what the freshness notices need."""

    result: PipelineResult
    fresh: sources.SourceFreshness
    provisional: list
    samples: pd.DataFrame  # source, material, t (IST): one row per valid record of every source
    charged: pd.DataFrame  # tonnes charged per IST hour, one column per material code (+ 'charges')
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


# ---------------------------------------------------------------------------------------------- loading steps
def load_hourly(path: str, mtime_ns: int) -> pd.DataFrame:
    """Hourly furnace dataset (``mtime_ns`` is the cache token only)."""
    return sources.load_hourly_dataset(path)


def load_online(
    start: str, clock: str, nonce: int, fields_json: str, cache_dir: str, refresh_minutes: float
) -> pd.DataFrame:
    """10-min online data from ``start`` to now (``clock`` and ``nonce`` are cache tokens only)."""
    end = pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None)
    return sources.fetch_10min(
        pd.Timestamp(start), end, fields=json.loads(fields_json), cache_dir=cache_dir, refresh_minutes=refresh_minutes
    )


def load_neon(start: str, clock: str, nonce: int) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """HM/slag taps and the DPR (extended provisionally to today), with the provisional dates."""
    end = pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None)
    hm = sources.fetch_hm_slag(pd.Timestamp(start), end)
    dpr = sources.fetch_dpr_daily(pd.Timestamp(start) - pd.Timedelta(days=1), end)
    dpr, provisional = sources.extend_dpr_provisional(dpr, end)
    return hm, dpr, sorted(provisional)


def load_lab_samples(start: str, clock: str, nonce: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lab-sample times and tonnes charged per hour, for the freshness check."""
    return fetch_offline_samples(pd.Timestamp(start) - SAMPLE_LOOKBACK)


def load_names() -> dict:
    """Material code -> name; empty when the database is unreachable (codes are shown instead)."""
    try:
        return material_names()
    except Exception:  # noqa: BLE001 - codes are shown instead of names
        return {}


@dataclass(frozen=True)
class Loaders:
    """The five loading steps; the Streamlit page passes its own cached versions."""

    hourly: Callable[[str, int], pd.DataFrame]
    online: Callable[[str, str, int, str, str, float], pd.DataFrame]
    neon: Callable[[str, str, int], tuple[pd.DataFrame, pd.DataFrame, list]]
    lab: Callable[[str, str, int], tuple[pd.DataFrame, pd.DataFrame]]
    names: Callable[[], dict]


# Memoised steps for the API and the worker (same lifetimes as the Streamlit page: two loads each, names one day).
CACHED_LOADERS = Loaders(
    hourly=memo(max_entries=2)(load_hourly),
    online=memo(max_entries=2)(load_online),
    neon=memo(max_entries=2)(load_neon),
    lab=memo(max_entries=2)(load_lab_samples),
    names=memo(ttl_seconds=24 * 3600)(load_names),
)


# ---------------------------------------------------------------------------------------------- composition
def source_times(
    fd: pd.DataFrame,
    m10: pd.DataFrame,
    hm: pd.DataFrame,
    dpr: pd.DataFrame,
    provisional: Sequence,
    charged: pd.DataFrame,
) -> pd.DataFrame:
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


def build_page_data(keys: DataKeys, settings: PipelineSettings, loaders: Loaders) -> PageData:
    """Load every source through ``loaders`` and run the pipeline (no caching of its own)."""

    fd = loaders.hourly(keys.hourly_path, keys.hourly_mtime_ns)
    m10 = loaders.online(
        keys.history_start, keys.clock, keys.nonce, keys.fields_json, keys.cache_dir, keys.online_refresh_minutes
    )
    hm, dpr, provisional = loaders.neon(keys.history_start, keys.clock, keys.nonce)
    result = run_pipeline(fd, m10, hm, dpr, settings, provisional_dates=provisional)
    real_dpr = dpr[~dpr.index.isin(pd.to_datetime(provisional))]
    try:
        lab, charged = loaders.lab(keys.history_start, keys.clock, keys.nonce)
    except Exception:  # noqa: BLE001 - the page still works; the freshness table says the lab times are missing
        lab, charged = pd.DataFrame(columns=["source", "material", "t"]), pd.DataFrame()
    samples = pd.concat([lab, source_times(fd, m10, hm, dpr, provisional, charged)], ignore_index=True)
    return PageData(
        result=result,
        fresh=sources.freshness(fd, m10, hm, real_dpr),
        provisional=provisional,
        samples=samples,
        charged=charged,
        names=loaders.names(),
    )


def rist_candidates(table: pd.DataFrame) -> pd.DataFrame:
    """Shifts the Rist diagram animates: stable, with burnt coke and a fixed-omega prediction (page 10, FORGE)."""
    return table[table.stable & table.K_energy.notna() & table.K_burnt.notna()]


def rist_keys(
    table: pd.DataFrame, start: date | None = None, end: date | None = None, limit: int | None = RIST_MAX_FRAMES
) -> list[str]:
    """Keys of the Rist shifts dated ``start`` to ``end`` (IST days, both included), oldest first; the latest
    ``limit`` of them (None: all)."""
    rows = rist_candidates(table)
    if start is not None:
        rows = rows[rows.sdate.dt.date >= start]
    if end is not None:
        rows = rows[rows.sdate.dt.date <= end]
    keys = [str(k) for k in rows.index]
    return keys[-limit:] if limit is not None and len(keys) > limit else keys


def build_rist_frames(data: PageData, settings: PipelineSettings, skeys: Sequence[str]) -> list[RistShift]:
    """Rist geometry per shift: the line of what the blast burnt (bias-corrected burnt coke, with the blast that closes
    the oxygen balance at the analyser's eta CO) and the energy model's line with omega fixed.

    The web model calibrated each shift on charged coke; charged coke reaches the tuyeres 4-5 h later, which made the
    per-shift line jump (lower-zone loss changed by 18 GJ/h from shift to shift against 6 GJ/h on burnt coke)."""

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
            frames.append(
                shift_rist(
                    case,
                    K,
                    V,
                    eta,
                    prod_th=rb.prod_th,
                    p=p,
                    skey=k,
                    omega_fixed=row.omega_fixed if pd.notna(row.omega_fixed) else None,
                    Q_LZ_GJh_fixed=row.Q_LZ_GJh_fixed if pd.notna(row.Q_LZ_GJh_fixed) else None,
                    Q_UZ_GJh_fixed=row.Q_UZ_GJh_fixed if pd.notna(row.Q_UZ_GJh_fixed) else None,
                    r_H_fixed=row.r_H_fixed if pd.notna(row.r_H_fixed) else None,
                )
            )
        except Exception:  # noqa: BLE001 - a shift whose geometry does not solve is left off the slider
            continue
    return frames


# ---------------------------------------------------------------------------------------------- memoised service API
@memo(max_entries=2)
def page_data(keys: DataKeys, settings: PipelineSettings) -> PageData:
    """Pipeline result and freshness inputs for the API and the worker. Treat the result as read-only."""
    return build_page_data(keys, settings, CACHED_LOADERS)


@memo(max_entries=4)
def rist_frames(keys: DataKeys, settings: PipelineSettings, skeys: tuple[str, ...]) -> list[RistShift]:
    """Rist frames for the given shifts, from the memoised page data."""
    return build_rist_frames(page_data(keys, settings), settings, skeys)


def clear() -> None:
    """Forget everything loaded through the service (the caller also bumps its nonce)."""
    for fn in (*vars(CACHED_LOADERS).values(), page_data, rist_frames):
        fn.clear()
