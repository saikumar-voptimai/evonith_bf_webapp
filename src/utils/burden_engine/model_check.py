"""Model check for the Burden & Fuel Optimiser: is the energy model right on today's furnace?

Before an operator trusts a blend the optimiser proposes, the page shows the model against what the furnace is doing:
for each recent shift, the coke the blast actually burnt next to what the energy balance (two-zone, shaft efficiency
and heat losses held at the 14 days before the shift: the optimiser's model and calibration rule) and the oxygen
balance (eta CO fixed at the analyser's value) predict for that shift's HMT, Si, blast and burden. The predictions are
out of sample. The numbers come from the Coke Energy pipeline's shift table (``PipelineResult.shifts``), the same
result the optimiser calibrates on, so this module only selects and scores; it does no physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.coke_energy.accuracy import accuracy

COLS = ("K_burnt", "K_energy", "K_obal", "K_charged")
CONDITIONS = ("HMT", "hm_si", "prod_th", "pci_kgthm_dry", "nut_kgthm_dry", "HBT")


@dataclass(frozen=True)
class Thresholds:
    """Window statistics under which the model is called on track (energy model minus burnt, stable shifts).

    Defaults: bias within 8 kg/tHM and SD within 12 kg/tHM, about 1.5 x the validated shift SD of 7.7 (register
    DATA-06); at least ``min_shifts`` scored stable shifts for a verdict."""

    bias_kgthm: float = 8.0
    sd_kgthm: float = 12.0
    min_shifts: int = 6

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "Thresholds":
        cfg = dict(cfg or {})
        return cls(float(cfg.get("ok_bias_kgthm", cls.bias_kgthm)), float(cfg.get("ok_sd_kgthm", cls.sd_kgthm)),
                   int(cfg.get("min_shifts", cls.min_shifts)))


def scored_shifts(T: pd.DataFrame | None) -> pd.DataFrame:
    """Shifts with burnt coke and at least one model prediction, oldest first."""
    if T is None or T.empty or "K_burnt" not in T:
        return pd.DataFrame(columns=["sdate", "t_mid", "stable", *COLS, *CONDITIONS])
    have_model = T[[c for c in ("K_energy", "K_obal") if c in T]].notna().any(axis=1)
    out = T[T["K_burnt"].notna() & have_model].copy()
    return out.sort_values("t_mid") if "t_mid" in out else out


def window(scored: pd.DataFrame, days: int, *, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """The scored shifts of the last ``days`` days up to ``end`` (default: the latest scored shift's date)."""
    if scored.empty:
        return scored
    last = pd.to_datetime(scored["sdate"]).max() if end is None else pd.Timestamp(end)
    first = last.normalize() - pd.Timedelta(days=int(days) - 1)
    return scored[pd.to_datetime(scored["sdate"]) >= first]


def latest(scored: pd.DataFrame) -> pd.Series | None:
    """The latest shift with an energy-model prediction (else the latest scored shift)."""
    if scored.empty:
        return None
    with_energy = scored[scored["K_energy"].notna()] if "K_energy" in scored else scored
    return (with_energy if not with_energy.empty else scored).iloc[-1]


def age_days(row: pd.Series | None, *, today: pd.Timestamp) -> int | None:
    """Whole days from the shift's date to ``today`` (both tz-naive dates), or None without a date."""
    if row is None or "sdate" not in row or pd.isna(row["sdate"]):
        return None
    return int((pd.Timestamp(today).normalize() - pd.Timestamp(row["sdate"]).normalize()).days)


def summary(frame: pd.DataFrame, model: str = "K_energy", *, stable_only: bool = True) -> dict[str, float]:
    """Accuracy of ``model`` against burnt coke over the frame (stable shifts by default)."""
    rows = frame[frame["stable"].astype(bool)] if stable_only and "stable" in frame else frame
    if model not in rows or rows.empty:
        return accuracy(pd.Series(dtype=float), pd.Series(dtype=float))
    return accuracy(rows[model], rows["K_burnt"])


def verdict(stats: Mapping[str, float], th: Thresholds) -> tuple[str, str]:
    """("ok" | "watch" | "insufficient", one-line reason) for the energy model's window statistics."""
    n = int(stats.get("n") or 0)
    if n < th.min_shifts or not np.isfinite(stats.get("bias", np.nan)):
        return "insufficient", f"only {n} stable shifts with a prediction in the window (at least {th.min_shifts} needed)"
    bias, sd = float(stats["bias"]), float(stats["sd"])
    problems = []
    if abs(bias) > th.bias_kgthm:
        problems.append(f"bias {bias:+.1f} kg/tHM beyond ±{th.bias_kgthm:.0f}")
    if sd > th.sd_kgthm:
        problems.append(f"shift-to-shift error ±{sd:.1f} kg/tHM beyond ±{th.sd_kgthm:.0f}")
    if problems:
        return "watch", "; ".join(problems)
    return "ok", f"bias {bias:+.1f} kg/tHM, typical error ±{sd:.1f} kg/tHM over {n} stable shifts"
