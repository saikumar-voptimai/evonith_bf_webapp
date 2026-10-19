"""Burnt coke against charged coke: the same coke, a few hours apart.

Coke tipped at the stockline reaches the raceway one burden-descent time later (6-7 h on BF-02: the lag at which the
thermal state answers a fuel change, docs/bmo_fuel_slag_si_findings.md). So in any window the coke the blast burnt and
the coke charged differ by the change in coke in transit over that window, plus measurement error. If they are the
same coke, three things must hold, and each gives a number:

1. **Agreement on the 45-degree line.** Lin's concordance correlation coefficient
   ``CCC = 2 s_xy / (s_x^2 + s_y^2 + (m_x - m_y)^2) = r * C_b`` is 1 only when the two series are identical: it is
   Pearson's r (scatter) times C_b (shift and scale). Bland-Altman bias and limits of agreement (bias +/- 1.96 SD)
   say the same in kg/tHM.
2. **The gap shrinks with the averaging window.** Coke in transit is a bounded inventory, so its contribution to the
   window difference falls as 1/W; random measurement error falls as 1/sqrt(W); only a real difference between the
   two quantities stays. Fitting ``SD(W)^2 = A/W^2 + B/W + C`` separates the three; C is the floor that would be
   a genuine difference (or a drifting instrument).
3. **The delay is the descent time.** The lag that maximises agreement should equal the descent time. 8-hour shifts
   only bound it (a fraction of a shift); hourly charge-dump times are needed to measure it.

Two traps, both in the data:
* Tonnes per hour cannot show the delay: charging follows the stockline, so the furnace pulls charges as fast as the
  blast burns (charged t/h trails burnt t/h by about an hour). The coke CONTENT of the burden (kg/tHM, coke per
  charge) is what travels down the shaft.
* Per-tHM figures of the same shift share one production denominator, which adds a lag-0 correlation of its own.

Pure functions on the Coke Energy pipeline's shift table (``K_burnt``, ``K_charged``, ``hm_t``, ``sdate``,
``t_mid``; ``K_burnt_raw`` when available). No physics here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from utils.coke_energy.accuracy import MIN_FRACTION, window_means

SHIFT_H = 8.0
# Label -> shifts per window (non-overlapping, on the full 8-hour grid).
WINDOWS = {"Shift (8 h)": 1, "1 day": 3, "2 days": 6, "3 days": 9, "5 days": 15, "1 week": 21}


def concordance(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float]:
    """Agreement of y with x (pairs with both present): n, CCC, r, C_b, bias (y - x), SD of y - x, limits of
    agreement, means and the Deming (orthogonal) slope of y on x."""

    d = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    n = len(d)
    out = dict(n=n, ccc=np.nan, r=np.nan, cb=np.nan, bias=np.nan, sd=np.nan, loa_lo=np.nan, loa_hi=np.nan,
               mean_x=np.nan, mean_y=np.nan, slope=np.nan)
    if n < 3:
        return out
    mx, my = float(d.x.mean()), float(d.y.mean())
    vx, vy = float(d.x.var(ddof=0)), float(d.y.var(ddof=0))
    sxy = float(((d.x - mx) * (d.y - my)).mean())
    e = d.y - d.x
    out.update(mean_x=mx, mean_y=my, bias=float(e.mean()), sd=float(e.std(ddof=1)))
    out.update(loa_lo=out["bias"] - 1.96 * out["sd"], loa_hi=out["bias"] + 1.96 * out["sd"])
    denom = vx + vy + (mx - my) ** 2
    if denom > 0:
        out["ccc"] = 2 * sxy / denom
    if vx > 0 and vy > 0:
        out["r"] = sxy / np.sqrt(vx * vy)
        out["cb"] = out["ccc"] / out["r"] if out["r"] else np.nan
    if sxy != 0:
        out["slope"] = (vy - vx + np.sqrt((vy - vx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    return out


def _pairs(T: pd.DataFrame, burnt: str, charged: str, stable_only: bool) -> pd.DataFrame:
    if T is None or T.empty or burnt not in T or charged not in T:
        return pd.DataFrame(columns=["sdate", "t_mid", "hm_t", burnt, charged])
    rows = T[T[burnt].notna() & T[charged].notna()]
    if stable_only and "stable" in rows:
        rows = rows[rows["stable"].eq(True)]
    cols = [c for c in ("sdate", "t_mid", "hm_t", "stable") if c in rows] + [burnt, charged]
    return rows[cols].copy()


def window_agreement(T: pd.DataFrame, windows: dict[str, int] | None = None, *, burnt: str = "K_burnt",
                     charged: str = "K_charged", weight: str = "hm_t", stable_only: bool = False,
                     min_fraction: float = MIN_FRACTION) -> pd.DataFrame:
    """Burnt against charged coke per averaging window: production-weighted means over non-overlapping windows of
    the full shift grid (``accuracy.window_means`` rules), then :func:`concordance` across the windows."""

    P = _pairs(T, burnt, charged, stable_only)
    rows = []
    for label, w in (windows or WINDOWS).items():
        if P.empty:
            rows.append(dict(window=label, shifts=w, hours=w * SHIFT_H, **concordance([], [])))
            continue
        W = window_means(P, w, [burnt, charged], weight=weight, min_fraction=min_fraction).dropna(subset=[burnt, charged])
        rows.append(dict(window=label, shifts=w, hours=w * SHIFT_H, **concordance(W[charged], W[burnt])))
    return pd.DataFrame(rows)


def decompose_window_sd(table: pd.DataFrame) -> dict[str, float]:
    """Split the window-difference variance into transit (1/W^2), random (1/W) and floor (constant) parts.

    Non-negative least squares of SD(W)^2 = A/W^2 + B/W + C over the windows of :func:`window_agreement` (W in
    shifts, weighted by the number of windows). Returns the per-shift SD of each part (sqrt A, sqrt B), the floor SD
    (sqrt C) and the fitted SD per window."""

    from scipy.optimize import nnls

    t = table.dropna(subset=["sd"])
    t = t[t["n"] >= 5]
    if len(t) < 3:
        return dict(transit_sd=np.nan, random_sd=np.nan, floor_sd=np.nan, fitted={})
    W = t["shifts"].to_numpy(float)
    wt = np.sqrt(t["n"].to_numpy(float))
    M = np.column_stack([1 / W ** 2, 1 / W, np.ones_like(W)]) * wt[:, None]
    coef, _ = nnls(M, (t["sd"].to_numpy(float) ** 2) * wt)
    A, B, C = (float(c) for c in coef)
    fitted = {lbl: float(np.sqrt(A / w ** 2 + B / w + C)) for lbl, w in zip(t["window"], W)}
    return dict(transit_sd=float(np.sqrt(A)), random_sd=float(np.sqrt(B)), floor_sd=float(np.sqrt(C)), fitted=fitted)


def lag_scan(T: pd.DataFrame, lags_shifts: Sequence[int] = (-2, -1, 0, 1, 2), *, burnt: str = "K_burnt",
             charged: str = "K_charged", detrend_shifts: int | None = 21) -> pd.DataFrame:
    """Agreement of burnt coke with charged coke moved later by whole shifts (positive lag = burnt after charged).

    With ``detrend_shifts`` both series are taken as deviations from their centred rolling mean over that many shifts
    first, so the slow common level (the 14-day offset, campaign drift) does not mask the shift-to-shift response.
    Uses the full 8-hour grid so a shift move is a real 8 hours."""

    P = _pairs(T, burnt, charged, stable_only=False)
    if P.empty:
        return pd.DataFrame(columns=["lag_shifts", "lag_h", "n", "r", "ccc"])
    from utils.coke_energy.accuracy import shift_timeline

    grid = shift_timeline(P.sdate.min(), P.sdate.max())
    X = P.reindex(grid.index)
    b, k = X[burnt].astype(float), X[charged].astype(float)
    if detrend_shifts:
        roll = dict(window=int(detrend_shifts), center=True, min_periods=max(2, int(detrend_shifts) // 2))
        b, k = b - b.rolling(**roll).mean(), k - k.rolling(**roll).mean()
    rows = []
    for L in lags_shifts:
        c = concordance(k.shift(int(L)), b)
        rows.append(dict(lag_shifts=int(L), lag_h=int(L) * SHIFT_H, n=c["n"], r=c["r"], ccc=c["ccc"]))
    return pd.DataFrame(rows)


def peak_lag_hours(scan: pd.DataFrame, col: str = "r") -> float:
    """Sub-shift lag of the correlation peak: a parabola through the best whole-shift lag and its neighbours."""

    s = scan.dropna(subset=[col]).set_index("lag_shifts")[col]
    if s.empty:
        return float("nan")
    L = int(s.idxmax())
    if L - 1 not in s or L + 1 not in s:
        return L * SHIFT_H
    a, b, c = s[L - 1], s[L], s[L + 1]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if den < 0 else 0.0
    return float((L + max(-0.5, min(0.5, off))) * SHIFT_H)


def closure(T: pd.DataFrame, *, burnt: str = "K_burnt_raw", charged: str = "K_charged",
            weight: str = "hm_t") -> dict[str, Any]:
    """Mass closure over the whole table: total burnt over total charged coke (tonnes), and the cumulative
    difference in tonnes (coke charged ahead of what the blast burnt) per shift, oldest first.

    Use the RAW burnt coke (before the 14-day charged - burnt offset): with the corrected figure the totals agree by
    construction."""

    P = _pairs(T, burnt, charged, stable_only=False)
    if P.empty or weight not in P:
        return dict(n=0, ratio=np.nan, burnt_t=np.nan, charged_t=np.nan, cumulative=pd.Series(dtype=float))
    P = P.sort_values("t_mid") if "t_mid" in P else P
    bt = P[burnt] * P[weight] / 1000.0
    ct = P[charged] * P[weight] / 1000.0
    return dict(n=int(len(P)), ratio=float(bt.sum() / ct.sum()), burnt_t=float(bt.sum()), charged_t=float(ct.sum()),
                cumulative=(ct - bt).cumsum())


def summary(T: pd.DataFrame, *, burnt: str = "K_burnt", charged: str = "K_charged",
            windows: dict[str, int] | None = None) -> dict[str, Any]:
    """Everything the page and the commentary show: window table, variance split, lag scan and peak, closure."""

    table = window_agreement(T, windows, burnt=burnt, charged=charged)
    scan = lag_scan(T, burnt=burnt, charged=charged)
    has_raw = T is not None and "K_burnt_raw" in T and T["K_burnt_raw"].notna().any()
    return dict(table=table, split=decompose_window_sd(table), scan=scan, peak_lag_h=peak_lag_hours(scan),
                closure=closure(T, charged=charged) if has_raw else None)
