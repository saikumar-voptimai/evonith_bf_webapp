"""Hot-metal temperature estimate and forecast band from the taps (user decision 12 Nov 2026).

A continuous-time Kalman filter on the tap temperatures, for taps at irregular instants:

    L    thermal level of the hearth: random walk, variance rate ``q_l`` (degC^2/h)
    d    deviation from the level: Ornstein-Uhlenbeck, time constant ``tau_h``, stationary SD ``s_d``
    y    a tap reads L + d + v, v ~ N(0, r^2): scatter of a single tap (sampling, position in the cast)

Between two taps ``dt`` hours apart, with a = exp(-dt / tau_h):

    L <- L;  d <- a d;  P_LL <- P_LL + q_l dt;  P_Ld <- a P_Ld;  P_dd <- a^2 P_dd + s_d^2 (1 - a^2)

At a tap, S = P_LL + 2 P_Ld + P_dd + r^2 and e = y - (L + d); the gains are (P_LL + P_Ld) / S for the level and
(P_Ld + P_dd) / S for the deviation. The forecast at lead ``h`` from a state is L + a d; the variance of the thermal
state is P_LL + q_l h + a^2 P_dd + s_d^2 (1 - a^2) + 2 a P_Ld, and a tap read then adds r^2. The band therefore widens
from the scatter of one tap to the long-run SD of the taps about their level within about three time constants.

Parameters: maximum likelihood on the taps of Jan-Jun 2026 (tau 4.06 h, s_d 14.65 degC, q_l 0.0116 degC^2/h,
r 6.53 degC). Out of sample with the taps as published (Jul-19 Oct 2026: RMSE 2 h ahead 14.6 degC against 18.7 for
the last tap held; 20 Oct-12 Nov 2026: 11.9 against 18.1; +-1 sigma held 74-80 % of the taps, +-2 sigma 94-99 %).
The earlier forecast of SV-1.4 (``hmt_forecast``) divides a coke surplus by the balance's coke-temperature slope
(0.09 kg/tHM per degC): 1 kg/tHM of noise in the surplus is 11 degC, hence its swings of +-100 degC from hour to hour.

Pure numerics: no I/O. Times enter as hours (floats) or as a tap series indexed by tap time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    tau_h: float = 4.06  # time constant of the deviation, h
    s_d_c: float = 14.65  # stationary SD of the deviation, degC
    q_l_c2_per_h: float = 0.0116  # variance rate of the level, degC^2/h (0.53 degC per sqrt(day))
    r_c: float = 6.53  # scatter of a single tap, degC
    level_var0_c2: float = 400.0  # variance of the level before the first tap, degC^2
    level_taps: int = 20  # taps whose mean starts the level
    min_taps: int = 8  # taps needed before an estimate is given

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> Params:
        c, d = dict(cfg or {}), cls()
        return cls(
            tau_h=float(c.get("tau_h", d.tau_h)),
            s_d_c=float(c.get("s_d_c", d.s_d_c)),
            q_l_c2_per_h=float(c.get("q_l_c2_per_h", d.q_l_c2_per_h)),
            r_c=float(c.get("r_c", d.r_c)),
            level_var0_c2=float(c.get("level_var0_c2", d.level_var0_c2)),
            level_taps=int(c.get("level_taps", d.level_taps)),
            min_taps=int(c.get("min_taps", d.min_taps)),
        )


class State(NamedTuple):
    """The filter after the tap at ``t_h`` (hours on the caller's time base)."""

    t_h: float
    level: float
    dev: float
    p_ll: float
    p_ld: float
    p_dd: float


class Step(NamedTuple):
    """One tap: what the filter expected before it (mean and SD of a tap), the tap, and the state after it."""

    expected: float
    sd: float
    value: float
    state: State

    @property
    def z(self) -> float:
        return (self.value - self.expected) / self.sd if self.sd > 0 else 0.0


class Estimate(NamedTuple):
    mean: float
    sd_state: float  # SD of the thermal state (level + deviation)
    sd_tap: float  # SD of a tap read at that instant (adds the scatter of a single tap)


def run(t_h: Sequence[float], y: Sequence[float], p: Params) -> list[Step]:
    """The filter over taps in time order (``t_h`` increasing, hours). One ``Step`` per tap."""
    n = len(t_h)
    if n == 0:
        return []
    level = float(np.mean(np.asarray(y[: max(1, p.level_taps)], dtype=float)))
    dev, p_ll, p_ld, p_dd = 0.0, p.level_var0_c2, 0.0, p.s_d_c**2
    var_d, var_r = p.s_d_c**2, p.r_c**2
    out: list[Step] = []
    prev = float(t_h[0])
    for i in range(n):
        t, obs = float(t_h[i]), float(y[i])
        dt = max(0.0, t - prev)
        a = math.exp(-dt / p.tau_h)
        dev *= a
        p_ll += p.q_l_c2_per_h * dt
        p_ld *= a
        p_dd = a * a * p_dd + var_d * (1.0 - a * a)
        s = p_ll + 2.0 * p_ld + p_dd + var_r
        expected = level + dev
        e = obs - expected
        k_l, k_d = (p_ll + p_ld) / s, (p_ld + p_dd) / s
        level += k_l * e
        dev += k_d * e
        p_ll, p_ld, p_dd = p_ll - k_l * (p_ll + p_ld), p_ld - k_l * (p_ld + p_dd), p_dd - k_d * (p_ld + p_dd)
        out.append(Step(expected, math.sqrt(s), obs, State(t, level, dev, p_ll, p_ld, p_dd)))
        prev = t
    return out


def predict(state: State, t_h: float, p: Params) -> Estimate:
    """Mean and SDs at ``t_h`` (at or after the state's tap) from ``state``."""
    lead = max(0.0, float(t_h) - state.t_h)
    a = math.exp(-lead / p.tau_h)
    var = state.p_ll + p.q_l_c2_per_h * lead + a * a * state.p_dd + p.s_d_c**2 * (1.0 - a * a) + 2.0 * a * state.p_ld
    var = max(var, 0.0)
    return Estimate(state.level + a * state.dev, math.sqrt(var), math.sqrt(var + p.r_c**2))


def p_below(mean: float, sd: float, limit: float) -> float:
    """Probability of a value below ``limit`` for a normal (mean, sd)."""
    if sd <= 0:
        return 1.0 if mean < limit else 0.0
    return 0.5 * (1.0 + math.erf((limit - mean) / (sd * math.sqrt(2.0))))


# ------------------------------------------------------------------------------------------------ tap series
def hours(index: Any, origin: pd.Timestamp) -> np.ndarray:
    """Hours of the instants of ``index`` after ``origin`` (NaN for NaT)."""
    idx = pd.DatetimeIndex(index)
    return np.asarray((idx - origin) / pd.Timedelta(hours=1), dtype=float)


@dataclass(frozen=True)
class Filtered:
    """The filter over a tap series: ``steps[i]`` belongs to the tap at ``times[i]``."""

    origin: pd.Timestamp
    times: pd.DatetimeIndex
    steps: list[Step]
    params: Params

    @property
    def ready(self) -> bool:
        return len(self.steps) >= self.params.min_taps

    def at(self, when: Any) -> Estimate | None:
        """The estimate at ``when`` from the last tap at or before it; None before ``min_taps`` taps."""
        t = float(hours([pd.Timestamp(when)], self.origin)[0])
        i = int(np.searchsorted(hours(self.times, self.origin), t, side="right")) - 1
        if i + 1 < self.params.min_taps:
            return None
        return predict(self.steps[i].state, t, self.params)

    def tap_interval_h(self, last: int = 20) -> float | None:
        """Median interval between the last taps, h."""
        t = hours(self.times[-(last + 1) :], self.origin)
        return float(np.median(np.diff(t))) if len(t) > 1 else None


def filter_taps(temps: pd.Series, p: Params) -> Filtered:
    """The filter over ``temps`` (degC by tap time, any order; duplicates of a time keep the last)."""
    s = temps.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    origin = pd.Timestamp(s.index[0]) if len(s) else pd.Timestamp(0)
    times = pd.DatetimeIndex(s.index)
    return Filtered(origin, times, run(hours(times, origin), s.to_numpy(dtype=float), p), p)


def tap_check(steps: Sequence[Step], *, z_single: float = 2.0, z_pair: float = 1.0) -> dict[str, Any]:
    """The newest tap against what the filter expected before it: ``watch`` when it lies beyond ``z_single`` SDs, or
    when it and the tap before it both lie beyond ``z_pair`` SDs on the same side; else ``ok``."""
    if not steps:
        return {"state": "unknown", "rule": None, "z": None, "expected": None, "sd": None, "value": None}
    last = steps[-1]
    z = last.z
    rule = None
    if abs(z) > z_single:
        rule = "single"
    elif len(steps) > 1:
        zp = steps[-2].z
        if (z > z_pair and zp > z_pair) or (z < -z_pair and zp < -z_pair):
            rule = "pair"
    return {
        "state": "watch" if rule else "ok",
        "rule": rule,
        "z": z,
        "expected": last.expected,
        "sd": last.sd,
        "value": last.value,
    }


def backtest(
    temps: pd.Series,
    received: pd.Series | None,
    p: Params,
    horizons_h: Sequence[float],
    *,
    since: Any = None,
    default_delay_h: float = 1.0,
    memory_h: float = 240.0,
) -> pd.DataFrame:
    """Every tap from ``since`` against the forecast made ``h`` hours before it with the taps received by then.

    ``received``: when each tap was published (same index as ``temps``); a tap without it counts as received
    ``default_delay_h`` after its time. Persistence is the last tap received by the forecast's origin. Columns: at,
    horizon_h, forecast, sd (of a tap), measured, persistence.
    """
    cols = ["at", "horizon_h", "forecast", "sd", "measured", "persistence"]
    s = temps.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return pd.DataFrame(columns=cols)
    origin = pd.Timestamp(s.index[0])
    t = hours(s.index, origin)
    y = s.to_numpy(dtype=float)
    rec = np.full(len(t), np.nan)
    if received is not None and len(received):
        r = received[~received.index.duplicated(keep="last")].reindex(s.index)
        rec = hours(pd.DatetimeIndex(pd.to_datetime(r.to_numpy())), origin)
    rec = np.where(np.isfinite(rec), np.maximum(rec, t), t + default_delay_h)
    full = run(t, y, p)
    latest_rec = np.maximum.accumulate(rec)
    start = -math.inf if since is None else float(hours([pd.Timestamp(since)], origin)[0])
    rows = []
    for k in range(len(t)):
        if t[k] < start:
            continue
        for h in horizons_h:
            o = t[k] - float(h)
            j = int(np.searchsorted(t, o, side="right")) - 1
            if j < 0:
                continue
            if latest_rec[j] <= o:  # every earlier tap was published by the origin: the state of the full run
                known, state, last = j + 1, full[j].state, j
            else:
                idx = [i for i in range(j + 1) if rec[i] <= o and t[i] >= o - memory_h]
                if not idx:
                    continue
                known, state, last = len(idx), run(t[idx], y[idx], p)[-1].state, idx[-1]
            if known < p.min_taps:
                continue
            est = predict(state, t[k], p)
            rows.append((s.index[k], float(h), est.mean, est.sd_tap, y[k], y[last]))
    return pd.DataFrame(rows, columns=cols)


def score(rows: pd.DataFrame) -> dict[str, Any]:
    """RMSE of the forecast and of persistence, bias, their ratio and the share of taps inside +-1 and +-2 SD."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "rmse_model_c": None, "rmse_persistence_c": None, "bias_model_c": None, "ratio": None,
                "within_1sd": None, "within_2sd": None}  # fmt: skip
    err = rows["forecast"].to_numpy(dtype=float) - rows["measured"].to_numpy(dtype=float)
    per = rows["persistence"].to_numpy(dtype=float) - rows["measured"].to_numpy(dtype=float)
    z = np.abs(err) / rows["sd"].to_numpy(dtype=float)
    rmse_m, rmse_p = float(np.sqrt(np.mean(err**2))), float(np.sqrt(np.mean(per**2)))
    return {
        "n": n,
        "rmse_model_c": rmse_m,
        "rmse_persistence_c": rmse_p,
        "bias_model_c": float(np.mean(err)),
        "ratio": rmse_m / rmse_p if rmse_p > 0 else None,
        "within_1sd": float(np.mean(z <= 1.0)),
        "within_2sd": float(np.mean(z <= 2.0)),
    }
