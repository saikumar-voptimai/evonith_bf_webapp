"""Hot-metal temperature one descent time ahead (SV-1.4, #284): model, persistence baseline and gate.

The coke charged now reaches the raceways one descent time later (tau_RW, ``burden_column``). If it is more than the
two-zone balance needs to make the present hot-metal temperature, the furnace heats up by the excess over the
balance's coke-temperature slope:

    HMT_f(t + tau_RW) = HMT(t) + [K_arr(t) - K_FORGE(HMT(t), x(t))] / (dK_FORGE/dHMT)

HMT(t)        the latest tap temperature before t (at most ``max_age_h`` old)
K_arr(t)      dry coke per tonne of hot metal of the charges dumped in [t - 1 h, t): their coke to ferrous ratio
              times the ferrous rate of the window (wet kg/tHM), so that neither the hourly charge counter nor the
              hourly production figure enters (both alternate from hour to hour, SV-1.3)
K_FORGE(HMT)  the FORGE balance of the window ending at t (omega, Q_LZ, Q_UZ, r_H at the 14-day medians) with the
              hot-metal temperature set to HMT(t)
dK/dHMT       its slope, kg/tHM per degC, by a 10 degC difference

Persistence: HMT_f = HMT(t). Target: the tap temperature at t + tau_RW (linear between the taps that bracket it, both
within ``tol_h``). Gate (#284): at the tau_RW horizon the model's RMSE at least 10 % below persistence and its
absolute bias at most 5 degC, in the 2026 expanding test and in the 2023-24 test with frozen parameters; until it
passes the Live Twin shows the measured HMT only. Pure numerics: no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

HMT_COL = "hmt_gt_1480c"
MIN_VALID_HMT = 1350.0  # as the pipeline's tap filter
DHMT = 10.0  # degC step of the slope


@dataclass(frozen=True)
class GateCriteria:
    rmse_ratio_max: float = 0.90  # model RMSE / persistence RMSE
    bias_max_c: float = 5.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> GateCriteria:
        c = dict(cfg or {})
        return cls(float(c.get("rmse_ratio_max", 0.90)), float(c.get("bias_max_c", 5.0)))


def taps(hm: pd.DataFrame) -> pd.Series:
    """Valid tap temperatures by tap time, sorted (``t``, ``hmt_gt_1480c``)."""
    if hm is None or hm.empty or HMT_COL not in hm:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(hm[HMT_COL], errors="coerce").to_numpy(), index=pd.to_datetime(hm["t"]))
    s = s[s >= MIN_VALID_HMT].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def latest(temps: pd.Series, at: Any, max_age_h: float = 3.0) -> tuple[float, float] | None:
    """(HMT, age in h) of the last tap before ``at``, None when there is none within ``max_age_h``."""
    at = pd.Timestamp(at)
    before = temps[temps.index < at]
    if before.empty:
        return None
    age = (at - before.index[-1]) / pd.Timedelta(hours=1)
    return (float(before.iloc[-1]), float(age)) if age <= max_age_h else None


def target(temps: pd.Series, t: Any, tol_h: float = 2.0) -> float | None:
    """Measured HMT at ``t``: linear between the bracketing taps when both lie within ``tol_h`` of ``t``; else the
    nearest tap within ``tol_h / 2``; else None."""
    t = pd.Timestamp(t)
    if temps.empty:
        return None
    i = int(temps.index.searchsorted(t))
    tol = pd.Timedelta(hours=tol_h)
    if 0 < i < len(temps):
        t0, t1 = temps.index[i - 1], temps.index[i]
        if t - t0 <= tol and t1 - t <= tol:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return float(temps.iloc[i - 1] + w * (temps.iloc[i] - temps.iloc[i - 1]))
    near = temps.index[np.argmin(np.abs((temps.index - t) / pd.Timedelta(hours=1)))]
    return float(temps.loc[near]) if abs(near - t) <= tol / 2 else None


def k_arr(record: pd.DataFrame, at: Any, ferrous_kg_thm: float) -> float | None:
    """Dry coke per tonne of hot metal of the charges dumped in [at - 1 h, at) (see the module notes)."""
    at = pd.Timestamp(at)
    h = record if isinstance(record.index, pd.DatetimeIndex) else record.set_index(pd.to_datetime(record["time"]))
    hour = at - pd.Timedelta(hours=1)
    if hour not in h.index:
        return None
    row = h.loc[hour]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    moisture = min(max(_num(row.get("COKE_MOIST%"), 0.0), 0.0), 15.0)  # as the pipeline's COKE_DRY
    coke = _num(row.get("COKE_CALC_MT")) * (1 - moisture / 100)
    fer = sum(_num(row.get(c), 0.0) for c in ("SINTER_CALC_MT", "ORE_CALC_MT", "TOTAL_PELLET_CALC_MT"))
    if not (math.isfinite(coke) and coke > 0 and fer > 0 and ferrous_kg_thm > 0):
        return None
    return float(coke / fer * ferrous_kg_thm)


def forecast(hmt_now: float, k_arr_kg: float, k_forge_now: float, dk_dhmt: float) -> float | None:
    """HMT_f = HMT(t) + (K_arr - K_FORGE(HMT(t))) / (dK/dHMT); None for a slope that is not positive."""
    if not (dk_dhmt > 0 and all(math.isfinite(v) for v in (hmt_now, k_arr_kg, k_forge_now))):
        return None
    return float(hmt_now + (k_arr_kg - k_forge_now) / dk_dhmt)


def _num(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass
class GateResult:
    n: int
    rmse_model: float
    rmse_persistence: float
    bias_model: float
    bias_persistence: float
    ratio: float
    passed: bool
    reason: str


def gate(rows: pd.DataFrame, criteria: GateCriteria | None = None) -> GateResult:
    """Model against persistence on evaluations with a forecast and a target (columns hmt_now, hmt_forecast,
    hmt_target): errors are target - forecast."""
    c = criteria or GateCriteria()
    d = rows.dropna(subset=["hmt_now", "hmt_forecast", "hmt_target"])
    n = len(d)
    if n < 10:
        return GateResult(n, *(math.nan,) * 5, False, f"{n} forecasts with a target (need 10)")
    em = d.hmt_target - d.hmt_forecast
    ep = d.hmt_target - d.hmt_now
    rm, rp = float(np.sqrt((em**2).mean())), float(np.sqrt((ep**2).mean()))
    bm, bp = float(em.mean()), float(ep.mean())
    ratio = rm / rp if rp > 0 else math.inf
    ok_r, ok_b = ratio <= c.rmse_ratio_max, abs(bm) <= c.bias_max_c
    reason = (
        f"RMSE {rm:.1f} against persistence {rp:.1f} degC (ratio {ratio:.2f}, limit {c.rmse_ratio_max:.2f}); "
        f"bias {bm:+.1f} degC (limit {c.bias_max_c:g})"
    )
    return GateResult(n, rm, rp, bm, bp, ratio, bool(ok_r and ok_b), reason)


def register_fact(
    result: GateResult, fact_id: str, test: str, period: str, criteria: GateCriteria, date: str, version: str
) -> dict[str, Any]:
    """The fact-register row of a gate run (group COKE): value = model RMSE, se = None, status Validated when the
    gate passes, else Disputed; the persistence RMSE, ratio and bias in ``params``."""
    return dict(
        id=fact_id,
        group="COKE",
        statement=f"HMT forecast one descent time ahead (SV-1.4), {test}: model against persistence",
        value=None if not math.isfinite(result.rmse_model) else round(result.rmse_model, 2),
        unit="degC RMSE",
        status="Validated" if result.passed else "Disputed",
        grade="B",
        basis=f"{period}; {result.n} hourly forecasts at the tau_RW horizon; {result.reason}",
        literature="",
        next_test="" if result.passed else "SV-1.4 revision: smoothed K_arr or a plant-data slope",
        version=version,
        date=date,
        note=(
            f"Gate: RMSE ratio <= {criteria.rmse_ratio_max:.2f} and |bias| <= {criteria.bias_max_c:g} degC. "
            + ("Passed." if result.passed else "Failed: the Live Twin shows the measured HMT only.")
        ),
        params=dict(
            n=result.n,
            rmse_model=result.rmse_model,
            rmse_persistence=result.rmse_persistence,
            ratio=result.ratio,
            bias_model=result.bias_model,
            bias_persistence=result.bias_persistence,
            passed=result.passed,
        ),
    )
