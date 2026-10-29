"""Burnt against charged coke, the admin analysis (SV-4.4, #307; method in docs/burnt_vs_charged_coke.md).

Every number is computed by ``utils.coke_energy.transit`` and the leak-detection model on the Coke Energy pipeline's
shift table (page 10's values, the table page 11's *Burnt against charged coke* tab reads):

    windows      burnt (calibrated) against charged coke per averaging window from one shift to one month: Lin's CCC
                 with r and C_b, the Bland-Altman bias and limits of agreement and the window pairs
                 (``transit.window_pairs`` + ``transit.concordance``, the rows of ``transit.window_agreement``)
    split        SD(W)^2 = A/W^2 + B/W + C over those windows (``transit.decompose_window_sd``)
    lag          correlation of the shift-to-shift movements by lag (``transit.lag_scan``, ``peak_lag_hours``)
    closure      raw burnt over charged coke in tonnes over the record and per month, and the cumulative charged - raw
                 burnt (``transit.closure`` on the raw burnt coke, before the 14-day correction)
    leak         step of (charged - raw burnt) and of top-gas H2 with RAFT unchanged (``utils.coke_energy.leak_check``
                 with ``utils.strata_flow.plant_effects.leak_for_gap``)

on two bases: every balanced shift (the TRACE page's basis) and the stable shifts only. One basis reads every window
of the record (about 4 s for 15 months of shifts on a laptop), so a request never computes it: ``twin.trace``
builds it with the TRACE view every hour at :20 (``services.trace.refresh``) into
``src/storage/coke_energy/burnt_vs_charged.json`` and the admin API reads that file (``load``). Plain JSON: coke in
kg/tHM dry, masses in t, H2 in % of the dry top gas, RAFT in deg C, blast moisture in g/Nm3, times UTC (ISO 8601),
months and days as IST plant dates.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from services.twin_views import number
from utils.coke_energy import leak_check
from utils.coke_energy import transit as tr
from utils.coke_energy.pipeline import PipelineResult
from utils.memo import memo

VERSION = 1
STORE_NAME = "burnt_vs_charged.json"
STORE_PATH = Path(__file__).resolve().parents[1] / "storage" / "coke_energy" / STORE_NAME
#: Built with the TRACE view (hourly); older than this it is stale (as the view).
STALE_AFTER_S = 2 * 3600.0
#: Averaging windows, label -> shifts: transit.WINDOWS (shift to week) and two longer ones.
WINDOWS: dict[str, int] = {**tr.WINDOWS, "2 weeks": 42, "1 month": 90}
#: Lags of the scan, shifts (burnt after charged is positive), and the detrending window of the movements, shifts.
LAGS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
DETREND_SHIFTS = 21
#: Bases -> stable shifts only.
BASES: dict[str, bool] = {"all": False, "stable": True}
#: Pipeline input frame (``result.S``) column -> shift-table column for the leak check.
SHIFT_INPUTS = {"h2_pct": "h2_pct", "body_raft": "raft"}
SHIFT = pd.Timedelta(hours=8)
PLANT_TZ = "Asia/Kolkata"


# --------------------------------------------------------------------------------------------- values
def utc_text(ts: Any) -> str | None:
    """An IST wall-clock time (tz-naive) or an aware time as ISO 8601 UTC; None for a missing time."""
    if ts is None:
        return None
    stamp = pd.Timestamp(ts)
    if pd.isna(stamp):
        return None
    stamp = stamp.tz_localize(PLANT_TZ) if stamp.tzinfo is None else stamp
    return stamp.tz_convert("UTC").isoformat()


def day_text(ts: Any) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return f"{pd.Timestamp(ts):%Y-%m-%d}"


# --------------------------------------------------------------------------------------------- shift table
def shift_table(result: PipelineResult) -> pd.DataFrame:
    """The pipeline's shift table with the top-gas H2 (% dry) and RAFT (deg C) of each shift from its inputs."""
    T = result.shifts
    if T is None or T.empty:
        return pd.DataFrame()
    T = T.copy()
    S = result.S if result.S is not None else pd.DataFrame()
    for src, col in SHIFT_INPUTS.items():
        T[col] = S[src].reindex(T.index).astype(float) if src in S else float("nan")
    return T


def basis_table(T: pd.DataFrame, stable_only: bool) -> pd.DataFrame:
    """The shifts of a basis: all balanced shifts, or the stable ones."""
    if not stable_only or T.empty or "stable" not in T:
        return T
    return T[T["stable"].eq(True)]


# --------------------------------------------------------------------------------------------- windows
def stats_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """One row of the window table (``transit.window_agreement``) as plain values."""
    return {
        "window": str(row["window"]),
        "shifts": int(row["shifts"]),
        "hours": float(row["hours"]),
        "n": int(row["n"]),
        "ccc": number(row["ccc"]),
        "r": number(row["r"]),
        "cb": number(row["cb"]),
        "bias": number(row["bias"]),
        "sd": number(row["sd"]),
        "loa_lo": number(row["loa_lo"]),
        "loa_hi": number(row["loa_hi"]),
        "mean_charged": number(row["mean_x"]),
        "mean_burnt": number(row["mean_y"]),
        "slope": number(row["slope"]),
    }


def pair_points(W: pd.DataFrame) -> list[dict[str, Any]]:
    """Window pairs (``transit.window_pairs``) with their Bland-Altman coordinates: the mean of burnt and charged and
    burnt - charged, kg/tHM."""
    return [
        {
            "t": utc_text(t),
            "n": int(r["n"]),
            "charged": number(r["K_charged"]),
            "burnt": number(r["K_burnt"]),
            "mean": number((r["K_burnt"] + r["K_charged"]) / 2),
            "diff": number(r["K_burnt"] - r["K_charged"]),
        }
        for t, r in W.iterrows()
    ]


def windows(T: pd.DataFrame, spec: Mapping[str, int] | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``transit.window_agreement`` over ``spec`` and the pairs behind each row: the pairs of every window once
    (``transit.window_pairs``), their concordance as window_agreement takes it (charged as x, burnt as y)."""
    spec = spec or WINDOWS
    rows, pairs = [], {}
    for label, w in spec.items():
        W = tr.window_pairs(T, w)
        pairs[label] = W
        rows.append(dict(window=label, shifts=w, hours=w * tr.SHIFT_H, **tr.concordance(W["K_charged"], W["K_burnt"])))
    return pd.DataFrame(rows), pairs


def split(table: pd.DataFrame) -> dict[str, Any]:
    """``transit.decompose_window_sd`` with each window's observed SD, fitted SD and the three parts at that window:
    transit sqrt(A)/W, random sqrt(B)/sqrt(W), floor sqrt(C) (their squares sum to the fitted SD squared)."""
    d = tr.decompose_window_sd(table)
    a, b, c = (number(d.get(k)) for k in ("transit_sd", "random_sd", "floor_sd"))
    fitted = d.get("fitted") or {}
    rows = []
    for r in table.to_dict("records"):
        w = float(r["shifts"])
        used = r["window"] in fitted
        rows.append(
            {
                "window": str(r["window"]),
                "shifts": int(r["shifts"]),
                "n": int(r["n"]),
                "observed_sd": number(r["sd"]),
                "fitted_sd": number(fitted.get(r["window"])) if used else None,
                "transit_sd": None if a is None else a / w,
                "random_sd": None if b is None else b / w**0.5,
                "floor_sd": c,
                "in_fit": used,
            }
        )
    return {"transit_sd": a, "random_sd": b, "floor_sd": c, "windows": rows}


def lag(T: pd.DataFrame) -> dict[str, Any]:
    """``transit.lag_scan`` over ``LAGS`` (movements against a centred ``DETREND_SHIFTS`` mean) and the peak."""
    scan = tr.lag_scan(T, LAGS, detrend_shifts=DETREND_SHIFTS)
    rows = [
        {
            "lag_shifts": int(r["lag_shifts"]),
            "lag_h": float(r["lag_h"]),
            "n": int(r["n"]),
            "r": number(r["r"]),
            "ccc": number(r["ccc"]),
        }
        for r in scan.to_dict("records")
    ]
    return {"detrend_shifts": DETREND_SHIFTS, "rows": rows, "peak_lag_h": number(tr.peak_lag_hours(scan))}


# --------------------------------------------------------------------------------------------- closure
def _closure_values(c: Mapping[str, Any]) -> dict[str, Any]:
    burnt, charged = number(c["burnt_t"]), number(c["charged_t"])
    return {
        "n": int(c["n"]),
        "ratio": number(c["ratio"]),
        "burnt_t": burnt,
        "charged_t": charged,
        "gap_t": None if burnt is None or charged is None else charged - burnt,
    }


def closure(T: pd.DataFrame) -> dict[str, Any] | None:
    """``transit.closure`` (raw burnt coke) over the record and per IST month of the shift dates, with the cumulative
    charged - raw burnt in t at the end of each shift; None without a raw burnt coke."""
    if T.empty or "K_burnt_raw" not in T or not T["K_burnt_raw"].notna().any():
        return None
    total = tr.closure(T)
    if not total["n"]:
        return None
    months = []
    for month, part in T.groupby(pd.to_datetime(T["sdate"]).dt.to_period("M")):
        c = tr.closure(part)
        if c["n"]:
            months.append({"month": str(month), **_closure_values(c)})
    t_mid = T["t_mid"]
    cumulative = [
        {"t": utc_text(pd.Timestamp(t_mid.loc[k]) + SHIFT / 2), "skey": str(k), "cumulative_t": number(v)}
        for k, v in total["cumulative"].items()
    ]
    return {**_closure_values(total), "months": months, "cumulative": cumulative}


# --------------------------------------------------------------------------------------------- leak water
def leak(T: pd.DataFrame, settings: leak_check.LeakCheckSettings) -> dict[str, Any]:
    """``leak_check.check`` as plain values: the periods, every signal's step, the model's leak and the shifts."""
    out = leak_check.check(T, settings)
    steps = []
    for key, (label, unit) in leak_check.SIGNALS.items():
        s = out["steps"][key]
        steps.append(
            {
                "key": key,
                "label": label,
                "unit": unit,
                "n_recent": s["n_recent"],
                "n_baseline": s["n_baseline"],
                "recent": number(s["recent"]),
                "baseline": number(s["baseline"]),
                "step": number(s["step"]),
                "se": number(s["se"]),
                "z": number(s["z"]),
            }
        )
    model = out["model"]
    frame = out["frame"]
    shifts = [
        {
            "skey": str(k),
            "start": utc_text(pd.Timestamp(r["t_mid"]) - SHIFT / 2),
            "end": utc_text(pd.Timestamp(r["t_mid"]) + SHIFT / 2),
            "period": str(r["period"]),
            **{key: number(r[key]) for key in leak_check.SIGNALS},
        }
        for k, r in frame.iterrows()
    ]
    return {
        "state": out["state"],
        "reason": out["reason"],
        "recent_first": day_text(out["recent_first"]),
        "recent_last": day_text(out["recent_last"]),
        "baseline_first": day_text(out["baseline_first"]),
        "steps": steps,
        "model": None
        if model is None
        else {
            "leak_kg_h": number(model["leak_kg_h"]),
            "water_kg_per_thm": number(model["water_kg_per_thm"]),
            "h2_rise_pt": number(model["top_gas_h2_rise_pt"]),
            "blast_moisture_equivalent_g_nm3": number(model["blast_moisture_equivalent_g_nm3"]),
            "forge_coke_missing_kg_thm": number(model["forge_coke_missing_kg_thm"]),
            "raw_burnt_low_by_kg_thm": number(model["trace_raw_burnt_low_by_kg_thm"]),
        },
        "shifts": shifts,
    }


# --------------------------------------------------------------------------------------------- the analysis
def settings_from(cfg: Mapping[str, Any] | None) -> leak_check.LeakCheckSettings:
    """Leak-check settings of ``coke_energy.yml → twin.burnt_vs_charged.leak_check``."""
    block = ((((cfg or {}).get("twin") or {}).get("burnt_vs_charged") or {}).get("leak_check")) or {}
    return leak_check.LeakCheckSettings.from_config(block)


def analysis(T: pd.DataFrame, settings: leak_check.LeakCheckSettings) -> dict[str, Any]:
    """The analysis of one basis (``T`` already holds only its shifts)."""
    table, pairs = windows(T)
    sdate = pd.to_datetime(T["sdate"]) if not T.empty else pd.Series(dtype="datetime64[ns]")
    return {
        "shifts": int(len(T)),
        "stable_shifts": int(T["stable"].eq(True).sum()) if "stable" in T else 0,
        "first_day": day_text(sdate.min()) if len(sdate) else None,
        "last_day": day_text(sdate.max()) if len(sdate) else None,
        "table": [stats_row(r) for r in table.to_dict("records")],
        "points": {label: pair_points(W) for label, W in pairs.items()},
        "split": split(table),
        "lag": lag(T),
        "closure": closure(T),
        "leak": leak(T, settings),
    }


def build(result: PipelineResult, cfg: Mapping[str, Any] | None = None, *, now: Any = None) -> dict[str, Any]:
    """Both bases of one pipeline result (``now``: build time, IST tz-naive or aware)."""
    now = pd.Timestamp.now(tz=PLANT_TZ) if now is None else pd.Timestamp(now)
    settings = settings_from(cfg)
    T = shift_table(result)
    return {
        "version": VERSION,
        "built_at": utc_text(now),
        "windows": [{"window": label, "shifts": w, "hours": w * tr.SHIFT_H} for label, w in WINDOWS.items()],
        "leak_settings": settings.as_dict(),
        "bases": {name: analysis(basis_table(T, stable), settings) for name, stable in BASES.items()},
    }


# --------------------------------------------------------------------------------------------- store
def path_beside(view_path: Path | None) -> Path:
    """The analysis file next to a TRACE view file (the default store without one)."""
    return STORE_PATH if view_path is None else Path(view_path).with_name(STORE_NAME)


def save(analysis_: Mapping[str, Any], path: Path | None = None) -> Path:
    """Write atomically (a reader never sees a partial file)."""
    target = Path(path or STORE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".burnt_vs_charged_", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(analysis_, fh, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


@memo(max_entries=2)
def _read(path: str, mtime_ns: int) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path: Path | None = None) -> dict[str, Any] | None:
    """The stored analysis (re-read when the file changes), or None before the first build. Treat as read-only."""
    target = Path(path or STORE_PATH)
    try:
        mtime = target.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    stored = _read(str(target), mtime)
    return stored if stored.get("version") == VERSION else None


def age_s(stored: Mapping[str, Any], now: datetime | None = None) -> float | None:
    built = stored.get("built_at")
    if built is None:
        return None
    now = now or datetime.now(UTC)
    return (now - pd.Timestamp(built).to_pydatetime()).total_seconds()
