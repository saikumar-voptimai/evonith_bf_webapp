"""Live two-zone calibration and operating regime for the Burden & Fuel Optimiser (#252).

Source: the Coke Energy pipeline result (``utils.coke_energy.pipeline.PipelineResult``), which the page gets from the
same cached loader as the Coke Energy Model page (``ui.coke_energy.loaders.page_data``), so the two pages calibrate on
the same shifts. Window: stable shifts in the ``window_days`` before ``as_of`` (default 14, the Coke Energy page's
rolling window). With fewer than ``min_shifts`` stable shifts the shipped Jul-Oct 2026 calibration is used and the
result says so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.burden_engine.inputs import repo_path
from utils.burden_engine.calibration import (CalibrationError, calibrate_from_shifts, operating_regime,
                                             select_shifts)
from utils.burden_engine.engine import DEFAULT_CAL_PATH
from utils.coke_energy.bf_balance import DEFAULTS
from utils.coke_energy.pipeline import slag_superheat_c

logger = logging.getLogger(__name__)


@dataclass
class CalibrationInfo:
    cal: dict[str, Any]
    regime: dict[str, Any] | None
    source: str                       # "live" or "fallback"
    window: tuple[str, str]
    n_shifts: int
    message: str = ""
    shifts: list[str] = field(default_factory=list)


def fallback_calibration(path: str | None = None) -> dict[str, Any]:
    with open(repo_path(path) if path else DEFAULT_CAL_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def network_context(S: pd.DataFrame, keys) -> dict[str, float]:
    """STRATA-Flow context of a calibration window (STRATA 1.3): median measured top-gas eta_CO, H2 and temperature and
    the hot-blast temperature of the window's shifts. The charging matrix stays the asset's (burden_history is not in
    the pipeline result)."""
    if S is None or S.empty or not len(keys):
        return {}
    w = S.loc[[k for k in keys if k in S.index]]
    num = lambda c: pd.to_numeric(w[c], errors="coerce") if c in w else pd.Series(dtype=float)
    out: dict[str, float] = {}
    co, co2 = num("co_pct"), num("co2_pct")
    eta = (co2 / (co + co2)).median()
    if np.isfinite(eta) and 0.3 < eta < 0.6:
        out["top_eta_co"] = float(eta)
    h2 = num("h2_pct").median()
    if np.isfinite(h2) and 0 < h2 < 10:
        out["top_h2"] = float(h2) / 100.0
    for col, key, lo, hi in (("top_temp_avg", "top_gas_T_c", 60.0, 400.0), ("hot_blast_temp", "hbt_c", 900.0, 1300.0)):
        v = num(col).median()
        if np.isfinite(v) and lo < v < hi:
            out[key] = float(v)
    return out


def sinter_strength(S: pd.DataFrame, end: pd.Timestamp | None = None) -> dict[str, Any]:
    """Sinter RI and RDI from the latest raw-material strength data (pipeline columns SINTER_HOT_STRENGTH_RI / _RDI,
    the routine sinter test carried forward per shift) up to ``end``: the value STRATA uses for the calibration state
    and the plans (user decision 23 Oct 2026; not the laboratory report)."""
    if S is None or S.empty or "SINTER_HOT_STRENGTH_RDI" not in S:
        return {}
    d = S.copy()
    if end is not None and "sdate" in d:
        d = d[pd.to_datetime(d["sdate"]) < end]
    ri = pd.to_numeric(d.get("SINTER_HOT_STRENGTH_RI"), errors="coerce")
    rdi = pd.to_numeric(d.get("SINTER_HOT_STRENGTH_RDI"), errors="coerce")
    ok = rdi.notna() & ri.notna() & rdi.between(5, 60) & ri.between(40, 95)
    if not ok.any():
        return {}
    k = ok[ok].index[-1]
    return dict(ri_pct=float(ri.loc[k]), rdi_pct=float(rdi.loc[k]),
                as_of=str(d.loc[k, "sdate"]) if "sdate" in d else str(k), source="RM strength (latest, pipeline)")


def live_calibration(result: Any, cfg: Mapping[str, Any], *, as_of: pd.Timestamp | None = None,
                     enrich_tag_per_o2_pct: float = 0.84) -> CalibrationInfo:
    """Calibration and regime over the stable shifts of the window before ``as_of`` (end exclusive; default: the day
    after the last balanced shift, so the latest shifts are included)."""
    ccfg = dict(cfg.get("calibration", {}))
    window_days = int(ccfg.get("window_days", 14))
    min_shifts = int(ccfg.get("min_shifts", 10))
    regime_days = int(ccfg.get("regime_window_days", window_days))
    S, R = getattr(result, "S", pd.DataFrame()), getattr(result, "R", pd.DataFrame())
    if R is None or R.empty:
        return CalibrationInfo(fallback_calibration(ccfg.get("fallback_path")), None, "fallback", ("", ""), 0,
                               "No balanced shifts available: using the shipped Jul-Oct 2026 calibration.")
    last = pd.to_datetime(R["sdate"]).max()
    end = (as_of if as_of is not None else last + pd.Timedelta(days=1)).normalize()
    start = end - pd.Timedelta(days=window_days)
    window = (str(start.date()), str((end - pd.Timedelta(days=1)).date()))
    keys = select_shifts(S, R, start=start, end=end)
    reg_keys = select_shifts(S, R, start=end - pd.Timedelta(days=regime_days), end=end)
    regime = operating_regime(S, R, reg_keys, enrich_tag_per_o2_pct) if len(reg_keys) else None
    settings = getattr(result, "settings", None)
    al2o3 = pd.to_numeric(S.loc[keys, "slag_pct_al2o3"], errors="coerce").median() if "slag_pct_al2o3" in S else np.nan
    p = {**DEFAULTS, "slag_T_above_HM": slag_superheat_c(al2o3, getattr(settings, "slag_superheat", None))}
    try:
        cal = calibrate_from_shifts(S, R, keys, p=p, min_shifts=min_shifts)
        cal["strata_flow"] = network_context(S, keys)
        cal["sinter_strength"] = sinter_strength(S, end) or cal.get("sinter_strength", {})
        return CalibrationInfo(cal, regime, "live", window, len(keys),
                               f"Calibrated on {len(keys)} stable shifts, {window[0]} to {window[1]}.", list(map(str, keys)))
    except CalibrationError as exc:
        msg = f"{exc} ({window[0]} to {window[1]}): using the shipped Jul-Oct 2026 calibration."
    except Exception as exc:  # noqa: BLE001 - a bad window must not take the page down
        logger.exception("burden engine live calibration failed")
        msg = f"Live calibration failed ({type(exc).__name__}: {exc}): using the shipped Jul-Oct 2026 calibration."
    return CalibrationInfo(fallback_calibration(ccfg.get("fallback_path")), regime, "fallback", window, len(keys), msg,
                           list(map(str, keys)))
