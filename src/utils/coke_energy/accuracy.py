"""Window averages and accuracy statistics for the coke-rate timeline.

Same rules as the web model's ``burnt.window_stats``: non-overlapping windows of w consecutive shifts on the FULL shift
timeline (A, B, C of every date, present or not), production-weighted means over the shifts present, and a window is
kept only when ceil(0.67 w) of its shifts are present - all 3 for a day, 5 of 6 for two days, 15 of 21 for a week.
Charged coke catches up with burnt coke over 2 days or more, so the windows are how the models are judged against
charged coke.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from utils.coke_energy.pipeline import SHIFT_MID_HOURS

# Label -> shifts per window.
WINDOWS = {"Shift": 1, "1 day": 3, "2 days": 6, "1 week": 21}
MIN_FRACTION = 0.67


def shift_timeline(first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Every shift from ``first`` to ``last`` (dates), with its key and mid-time."""

    days = pd.date_range(pd.Timestamp(first).normalize(), pd.Timestamp(last).normalize(), freq="D")
    rows = [(f"{d:%Y-%m-%d}_{s}", d + pd.Timedelta(hours=SHIFT_MID_HOURS[s])) for d in days for s in "ABC"]
    return pd.DataFrame(rows, columns=["skey", "t_mid"]).set_index("skey")


def window_means(table: pd.DataFrame, w: int, cols: Sequence[str], weight: str = "hm_t",
                 min_fraction: float = MIN_FRACTION) -> pd.DataFrame:
    """Production-weighted means of ``cols`` per window of ``w`` shifts, indexed by the window's mid-time.

    ``table`` holds the shifts to use (index = shift key, with ``sdate``, ``t_mid`` and ``weight`` columns). With
    w = 1 the shifts come back unchanged, on the full 8-hour grid so that missing shifts are gaps.
    """

    if table.empty:
        return pd.DataFrame(columns=["n", *cols])
    grid = shift_timeline(table.sdate.min(), table.sdate.max())
    X = table.reindex(grid.index)
    X["t_grid"] = grid.t_mid
    gid = np.arange(len(X)) // w
    need = max(1, int(np.ceil(min_fraction * w)))
    rows = []
    for g, part in X.groupby(gid):
        mid = part.t_grid.iloc[0] + (part.t_grid.iloc[-1] - part.t_grid.iloc[0]) / 2
        present = part.dropna(subset=list(cols))
        if len(present) < need:
            rows.append({"t_mid": mid, "n": len(present), **{c: np.nan for c in cols}})
            continue
        wt = present[weight].fillna(present[weight].mean()) if weight in present else pd.Series(1.0, index=present.index)
        rows.append({"t_mid": mid, "n": len(present), **{c: float((present[c] * wt).sum() / wt.sum()) for c in cols}})
    return pd.DataFrame(rows).set_index("t_mid")


def rolling_mean(table: pd.DataFrame, shifts: int, cols: Sequence[str], weight: str = "hm_t") -> pd.DataFrame:
    """Display smoothing: centred production-weighted mean over ``shifts`` consecutive shifts of the full 8-hour grid.

    A point is drawn only when at least two-thirds of its window's shifts have a value, so long gaps stay gaps. With
    shifts = 1 the shifts come back unchanged on the grid. Indexed by each shift's mid-time. For the chart only: the
    accuracy statistics use non-overlapping windows (``window_means``)."""

    if table.empty:
        return pd.DataFrame(columns=list(cols))
    grid = shift_timeline(table.sdate.min(), table.sdate.max())
    X = table.reindex(grid.index)
    wt = X[weight].fillna(X[weight].mean()) if weight in X else pd.Series(1.0, index=X.index)
    need = int(np.ceil(2 * shifts / 3)) if shifts > 1 else 1
    out = pd.DataFrame(index=grid.index)
    for c in cols:
        have = X[c].notna()
        w = wt.where(have, 0.0)
        num = (X[c].fillna(0.0) * w).rolling(shifts, center=True, min_periods=1).sum()
        den = w.rolling(shifts, center=True, min_periods=1).sum()
        count = have.astype(float).rolling(shifts, center=True, min_periods=1).sum()
        out[c] = (num / den.replace(0.0, np.nan)).where(count >= need)
    out.index = grid.t_mid.to_numpy()
    out.index.name = "t_mid"
    return out


def accuracy(model: pd.Series, target: pd.Series) -> dict[str, float]:
    """Bias, spread and agreement of ``model - target`` over the rows where both exist."""

    e = (model - target).dropna()
    if len(e) < 2:
        return dict(n=len(e), bias=np.nan, sd=np.nan, mae=np.nan, r=np.nan, within_5=np.nan, within_10=np.nan)
    m, a = model.loc[e.index], target.loc[e.index]
    return dict(n=int(len(e)), bias=float(e.mean()), sd=float(e.std(ddof=1)), mae=float(e.abs().mean()),
                r=float(np.corrcoef(m, a)[0, 1]), within_5=float((e.abs() <= 5).mean()),
                within_10=float((e.abs() <= 10).mean()))
