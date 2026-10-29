"""Live two-zone calibration and operating regime for the Burden & Fuel Optimiser (#252).

Source: the Coke Energy pipeline result (``utils.coke_energy.pipeline.PipelineResult``), assembled by
``services.coke_energy`` (the Streamlit pages share its cached ``page_data``; the twin-api and the worker use the
service's memoised one), so every view calibrates on the same shifts. Window: stable shifts in the ``window_days``
before ``as_of`` (default 14, the Coke Energy page's rolling window). With fewer than ``min_shifts`` stable shifts the
shipped Jul-Oct 2026 calibration is used and the result says so.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.burden_engine.inputs import repo_path
# network_context and sinter_strength moved to the pure calibration module (SV-1.1: the twin uses them); still
# importable from here.
from utils.burden_engine.calibration import (CalibrationError, calibrate_from_shifts, network_context,  # noqa: F401
                                             operating_regime, select_shifts, shift_mask, sinter_strength)
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
    # The window's balanced shifts with the stability filter's verdict (``shift_mask``; SV-4.6 calibration console)
    mask: list[dict[str, Any]] = field(default_factory=list)


def fallback_calibration(path: str | None = None) -> dict[str, Any]:
    with open(repo_path(path) if path else DEFAULT_CAL_PATH, encoding="utf-8") as fh:
        return json.load(fh)


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
    mask = shift_mask(R, keys, start=start, end=end)
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
                               f"Calibrated on {len(keys)} stable shifts, {window[0]} to {window[1]}.", list(map(str, keys)),
                               mask)
    except CalibrationError as exc:
        msg = f"{exc} ({window[0]} to {window[1]}): using the shipped Jul-Oct 2026 calibration."
    except Exception as exc:  # noqa: BLE001 - a bad window must not take the page down
        logger.exception("burden engine live calibration failed")
        msg = f"Live calibration failed ({type(exc).__name__}: {exc}): using the shipped Jul-Oct 2026 calibration."
    return CalibrationInfo(fallback_calibration(ccfg.get("fallback_path")), regime, "fallback", window, len(keys), msg,
                           list(map(str, keys)), mask)
