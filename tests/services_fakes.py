"""Synthetic Coke Energy sources for the service-layer tests (no InfluxDB, Neon or plant server)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.coke_energy.pipeline import PipelineResult, PipelineSettings

PLANT_TZ = "Asia/Kolkata"


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None).floor("h")


def hourly_frame(days: int = 10) -> pd.DataFrame:
    t = pd.date_range(now_ist() - pd.Timedelta(days=days), now_ist() - pd.Timedelta(hours=2), freq="h")
    coke = np.linspace(20.0, 24.0, len(t))
    coke[::17] = np.nan
    return pd.DataFrame({"time": t, "COKE_CALC_MT": coke, "SINTER_CALC_MT": coke * 3.1})


def online_frame(days: int = 10) -> pd.DataFrame:
    ts = pd.date_range(now_ist() - pd.Timedelta(days=days), now_ist() - pd.Timedelta(minutes=20), freq="10min")
    vol = np.linspace(180_000.0, 190_000.0, len(ts))
    vol[::29] = np.nan
    return pd.DataFrame({"ts": ts, "hot_blast_vol_nm3h": vol, "co_pct": 22.0})


def hm_frame(days: int = 10) -> pd.DataFrame:
    t = pd.date_range(now_ist() - pd.Timedelta(days=days), now_ist() - pd.Timedelta(hours=3), freq="3h")
    si = np.linspace(0.4, 0.6, len(t))
    al = np.linspace(17.0, 18.0, len(t))
    si[::5] = np.nan
    al[::5] = np.nan
    al[::10] = 17.5  # rows with Si missing but slag present stay valid
    return pd.DataFrame({"t": t, "chem_pct_si": si, "slag_pct_al2o3": al})


def dpr_frame(days: int = 10, lag_days: int = 2) -> pd.DataFrame:
    end = now_ist().normalize() - pd.Timedelta(days=lag_days)
    idx = pd.date_range(end - pd.Timedelta(days=days), end, freq="D", name="date")
    hm_t = np.linspace(2200.0, 2300.0, len(idx))
    return pd.DataFrame(
        {"hm_t": hm_t, "dc_dust_t": hm_t * 0.012, "gcp_dust_t": hm_t * 0.018, "etaco_dpr": np.nan, "delay_min": np.nan},
        index=idx,
    )


def lab_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = pd.DataFrame(
        {
            "source": ["ore_chemistry", "fuel_chemistry", "sinter_chemistry"],
            "material": ["ore_1", "coke_1", "sinter_1"],
            "t": [now_ist() - pd.Timedelta(hours=h) for h in (30, 12, 5)],
        }
    )
    idx = pd.date_range(now_ist() - pd.Timedelta(hours=48), now_ist() - pd.Timedelta(hours=1), freq="h")
    charges = np.where(np.arange(len(idx)) % 7 == 0, 0, 6)
    charged = pd.DataFrame({"ore_1": charges * 4.0, "coke_1": charges * 2.0, "charges": charges}, index=idx)
    return lab, charged


@dataclass
class PipelineRecorder:
    """Stand-in for ``run_pipeline``: records its inputs and returns a result derived from them."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, fd, m10, hm, dpr, settings: PipelineSettings, provisional_dates=None) -> PipelineResult:
        self.calls.append(dict(fd=fd, m10=m10, hm=hm, dpr=dpr, settings=settings, provisional=provisional_dates))
        shifts = pd.DataFrame(
            {
                "n_hourly": [len(fd)],
                "n_10min": [len(m10)],
                "n_taps": [len(hm)],
                "n_dpr": [len(dpr)],
                "dpr_hm_t_sum": [float(dpr["hm_t"].sum())],
                "provisional": [",".join(str(d) for d in (provisional_dates or []))],
            },
            index=pd.Index(["2026-10-26_A"], name="skey"),
        )
        empty = pd.DataFrame()
        return PipelineResult(S=empty, R=empty, P=empty, B=empty, Z=empty, O2=empty, shifts=shifts, settings=settings)


def patch_sources(monkeypatch, *modules) -> PipelineRecorder:
    """Patch the source functions in ``data.coke_energy.sources`` and the names imported into ``modules``."""

    from data.coke_energy import sources

    hourly, online, hm, dpr = hourly_frame(), online_frame(), hm_frame(), dpr_frame()
    lab, charged = lab_frames()
    monkeypatch.setattr(sources, "load_hourly_dataset", lambda path: hourly.copy())
    monkeypatch.setattr(sources, "fetch_10min", lambda start, end, fields, cache_dir, refresh_minutes: online.copy())
    monkeypatch.setattr(sources, "fetch_hm_slag", lambda start, end: hm.copy())
    monkeypatch.setattr(sources, "fetch_dpr_daily", lambda start, end: dpr.copy())
    recorder = PipelineRecorder()
    for module in modules:
        monkeypatch.setattr(module, "fetch_offline_samples", lambda start: (lab.copy(), charged.copy()))
        monkeypatch.setattr(module, "material_names", lambda: {"ore_1": "Lloyds CLO", "coke_1": "Coke"})
        monkeypatch.setattr(module, "run_pipeline", recorder)
    return recorder
