"""Implied-omega tracking and drift alarm (SV-1.2, #282).

Residual of each evaluation: r = omega_imp - omega_STRATA (window or day). In control, r follows the AR(1) process of
the STRATA 1.5 posterior (daily values: rho 0.682, sigma_e 0.0172, asset ``channel_scales``),

    r_t - mu = rho (r_{t-L} - mu) + e_t,      e_t ~ N(0, sigma_e^2),

and the standardised innovations z_t = e_t / sigma_e feed a two-sided tabular CUSUM

    S+_t = max(0, S+_{t-L} + z_t - k),   S-_t = max(0, S-_{t-L} - z_t - k),   signal when S+ > h or S- > h,

k = 0.5, h = 5 (a sustained shift of about one sigma_e in the innovations is detected after about 10 lags); after a
signal both sums of that chain restart at 0. S+ signals omega_imp above omega_STRATA (the furnace uses the gas better
than STRATA predicts for the charged blend), S- below.

Lag L. Hourly windows overlap by 7 h: an AR(1) fitted to the hourly series leaves innovations with autocorrelation 0.86
at one hour, and the CUSUM then signals about once a day in control (simulation of an AR(1) hourly residual whose daily
means have the posterior's rho). The innovations are therefore taken between non-overlapping windows, L = 8 h: eight
interleaved chains (windows ending at t, t - 8 h, t - 16 h, ...), each advanced every 8 h; an evaluation updates the
chain of its own window. Simulated in-control interval between false-alarm episodes: about 84 days (one chain alone,
shift-aligned: 113 days). For the daily series L = 1 day.

Parameters. rho and sigma_e are estimated on the series itself (pairs L apart, mu fixed; sigma_e = 1.4826 x median
absolute deviation of the innovations, robust to drift episodes) unless fixed values are given; re-estimated at every
evaluation. A missing evaluation is bridged up to ``max_gap`` lags (innovation e = (r_t - mu) - rho^g (r_{t-gL} - mu)
with variance sigma_e^2 (1 - rho^2g) / (1 - rho^2)); a longer gap restarts the chain from the stationary distribution.

Alarm. Raised at the first signal of a direction, kept while signals recur within ``clear_after`` (default 3 L) of the
previous one, cleared ``clear_after`` after the last signal. Pure numerics: no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DIRECTIONS = ("high", "low")  # S+ (omega_imp above omega_STRATA), S- (below)


@dataclass(frozen=True)
class DriftSettings:
    lag: pd.Timedelta = pd.Timedelta(hours=8)
    k: float = 0.5
    h: float = 5.0
    mu: float = 0.0
    clear_after: pd.Timedelta | None = None  # default 3 x lag
    max_gap: int = 3  # lags a chain bridges
    min_pairs: int = 72
    rho: float | None = None  # fixed parameters; None: estimated on the series
    sigma_e: float | None = None

    @property
    def clear(self) -> pd.Timedelta:
        return self.clear_after if self.clear_after is not None else 3 * self.lag

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> DriftSettings:
        c = dict(cfg or {})
        lag = pd.Timedelta(hours=float(c.get("lag_hours", 8)))
        clear = c.get("clear_after_hours")
        return cls(
            lag=lag,
            k=float(c.get("k", 0.5)),
            h=float(c.get("h", 5.0)),
            mu=float(c.get("mu", 0.0)),
            clear_after=None if clear is None else pd.Timedelta(hours=float(clear)),
            max_gap=int(c.get("max_gap_lags", 3)),
            min_pairs=int(c.get("min_pairs", 72)),
            rho=None if c.get("rho") is None else float(c["rho"]),
            sigma_e=None if c.get("sigma_e") is None else float(c["sigma_e"]),
        )


@dataclass(frozen=True)
class AR1:
    rho: float
    sigma_e: float
    mu: float  # in-control mean used (settings)
    mu_hat: float  # sample mean of r (bias diagnostic)
    n_pairs: int
    source: str  # "estimated" | "fixed"


@dataclass
class Episode:
    """One alarm from its first signal; ``cleared_at`` None while open."""

    direction: str
    raised_at: pd.Timestamp
    last_signal: pd.Timestamp
    cleared_at: pd.Timestamp | None = None
    signals: int = 1
    peak: float = 0.0  # largest CUSUM sum at a signal

    @property
    def open(self) -> bool:
        return self.cleared_at is None


@dataclass
class DriftReport:
    """The CUSUM over a residual series and the alarm episodes; ``points`` one row per evaluation."""

    points: pd.DataFrame  # index time; columns r, z, s_pos, s_neg, signal
    ar1: AR1 | None
    episodes: list[Episode] = field(default_factory=list)
    settings: DriftSettings = field(default_factory=DriftSettings)
    note: str = ""

    def open_episodes(self) -> dict[str, Episode]:
        return {e.direction: e for e in self.episodes if e.open}

    def latest(self) -> dict[str, Any]:
        if self.points.empty:
            return {}
        t = self.points.index[-1]
        p = self.points.iloc[-1]
        return dict(t=t, r=_f(p.r), z=_f(p.z), s_pos=_f(p.s_pos), s_neg=_f(p.s_neg))


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def residuals(times: Any, omega_imp: Any, omega_strata: Any) -> pd.Series:
    """r = omega_imp - omega_STRATA by evaluation time, sorted; evaluations without both values are dropped."""
    s = pd.Series(np.asarray(omega_imp, float) - np.asarray(omega_strata, float), index=pd.DatetimeIndex(times))
    s = s[np.isfinite(s.to_numpy())]
    return s[~s.index.duplicated(keep="last")].sort_index()


def _pairs(r: pd.Series, lag: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    prev = r.reindex(r.index - lag)
    ok = np.isfinite(prev.to_numpy())
    return r.to_numpy()[ok], prev.to_numpy()[ok]


def fit_ar1(r: pd.Series, settings: DriftSettings) -> AR1 | None:
    """rho and sigma_e from the pairs of evaluations ``lag`` apart (mu fixed at ``settings.mu``); None with fewer than
    ``min_pairs`` pairs and no fixed values."""
    mu = settings.mu
    mu_hat = float(r.mean()) if len(r) else float("nan")
    if settings.rho is not None and settings.sigma_e is not None:
        return AR1(settings.rho, settings.sigma_e, mu, mu_hat, int(len(_pairs(r, settings.lag)[0])), "fixed")
    a, b = _pairs(r, settings.lag)
    if len(a) < settings.min_pairs:
        return None
    den = float(np.sum((b - mu) ** 2))
    rho = settings.rho if settings.rho is not None else (float(np.sum((a - mu) * (b - mu)) / den) if den > 0 else 0.0)
    rho = min(max(rho, 0.0), 0.999)
    e = (a - mu) - rho * (b - mu)
    if settings.sigma_e is not None:
        sigma = settings.sigma_e
    else:
        sigma = 1.4826 * float(np.median(np.abs(e - np.median(e))))
        if not sigma > 0:
            sigma = float(np.std(e, ddof=1)) if len(e) > 1 else float("nan")
    if not (math.isfinite(sigma) and sigma > 0):
        return None
    return AR1(rho, sigma, mu, mu_hat, int(len(a)), "estimated")


def cusum(r: pd.Series, ar: AR1, settings: DriftSettings) -> pd.DataFrame:
    """Innovations and CUSUM sums of every evaluation (chains ``lag`` apart, gaps bridged up to ``max_gap`` lags)."""
    lag, k, h = settings.lag, settings.k, settings.h
    rho, sig, mu = ar.rho, ar.sigma_e, ar.mu
    value = dict(zip(r.index, r.to_numpy(), strict=True))
    state: dict[pd.Timestamp, tuple[float, float]] = {}
    rows = []
    for t, v in zip(r.index, r.to_numpy(), strict=True):
        z, prev_state = None, (0.0, 0.0)
        for g in range(1, settings.max_gap + 1):
            u = value.get(t - g * lag)
            if u is not None:
                var = (1 - rho ** (2 * g)) / (1 - rho**2) if rho < 1 else float(g)
                z = ((v - mu) - rho**g * (u - mu)) / (sig * math.sqrt(var))
                prev_state = state[t - g * lag]
                break
        if z is None:  # new chain: stationary distribution
            z = (v - mu) * math.sqrt(1 - rho**2) / sig
        sp = max(0.0, prev_state[0] + z - k)
        sn = max(0.0, prev_state[1] - z - k)
        signal = "high" if sp > h else ("low" if sn > h else None)
        rows.append((t, v, z, sp, sn, signal))
        state[t] = (0.0, 0.0) if signal else (sp, sn)
    out = pd.DataFrame(rows, columns=["t", "r", "z", "s_pos", "s_neg", "signal"]).set_index("t")
    return out


def episodes(points: pd.DataFrame, settings: DriftSettings, until: Any = None) -> list[Episode]:
    """Alarm episodes per direction: raised at a signal, cleared ``clear`` after the last signal when that time is
    reached by ``until`` (default: the last evaluation)."""
    end = pd.Timestamp(until) if until is not None else (points.index[-1] if len(points) else None)
    out: list[Episode] = []
    current: dict[str, Episode] = {}
    for t, p in points.iterrows():
        for d, ep in list(current.items()):  # close episodes whose clear time has passed
            if t >= ep.last_signal + settings.clear:
                ep.cleared_at = ep.last_signal + settings.clear
                del current[d]
        if p.signal in DIRECTIONS:
            s = float(p.s_pos if p.signal == "high" else p.s_neg)
            ep = current.get(p.signal)
            if ep is None:
                ep = Episode(p.signal, t, t, peak=s)
                current[p.signal] = ep
                out.append(ep)
            else:
                ep.last_signal, ep.signals, ep.peak = t, ep.signals + 1, max(ep.peak, s)
    if end is not None:
        for ep in current.values():
            if end >= ep.last_signal + settings.clear:
                ep.cleared_at = ep.last_signal + settings.clear
    return out


def evaluate(r: pd.Series, settings: DriftSettings | None = None, until: Any = None) -> DriftReport:
    """Fit, CUSUM and alarm episodes of a residual series (``residuals``)."""
    settings = settings or DriftSettings()
    ar = fit_ar1(r, settings)
    if ar is None:
        empty = pd.DataFrame(columns=["r", "z", "s_pos", "s_neg", "signal"])
        n = len(_pairs(r, settings.lag)[0])
        return DriftReport(empty, None, [], settings, f"{n} pairs {settings.lag} apart (need {settings.min_pairs})")
    points = cusum(r, ar, settings)
    return DriftReport(points, ar, episodes(points, settings, until), settings)


def run_lengths(points: pd.DataFrame, lag: pd.Timedelta) -> tuple[list[int], list[int]]:
    """Run lengths of the chains in evaluations: (completed runs, each ending at a signal; the censored run of each
    chain since its last signal). The first run of a chain counts from its first evaluation."""
    done: list[int] = []
    count: dict[pd.Timedelta, int] = {}
    if points.empty:
        return done, []
    t0 = points.index[0]
    for t, p in points.iterrows():
        phase = (t - t0) % lag
        count[phase] = count.get(phase, 0) + 1
        if p.signal in DIRECTIONS:
            done.append(count[phase])
            count[phase] = 0
    return done, [c for c in count.values() if c > 0]


def arl_estimate(done: list[int], censored: list[int]) -> float | None:
    """In-control average run length from a replay: evaluations per signal (exponential estimate with the censored
    runs); None without a signal (the ARL then exceeds the evaluations replayed)."""
    return (sum(done) + sum(censored)) / len(done) if done else None


def evidence_residuals(df: pd.DataFrame, channel_scales: Mapping[str, Any]) -> pd.Series:
    """Daily r = omega_imp - omega_STRATA of the STRATA-Learn evidence (``strata_evidence_seed.csv``): the day's change
    of the calibrated omega against its 14-day anchor (``dobs``) minus the posterior mean prediction from the day's
    channels (asset ``channel_scales``: names, mean)."""
    names = list(channel_scales["names"])
    pred = df[names].to_numpy(float) @ np.asarray(channel_scales["mean"], float)
    return residuals(pd.to_datetime(df["date"]), df["dobs"].to_numpy(float), pred)


def siegmund_arl(k: float = 0.5, h: float = 5.0, shift: float = 0.0) -> float:
    """Average run length of the two-sided CUSUM on N(shift, 1) innovations (Siegmund's approximation)."""

    def one(delta: float) -> float:
        b = h + 1.166
        if abs(delta) < 1e-9:
            return b * b
        return (math.exp(-2 * delta * b) + 2 * delta * b - 1) / (2 * delta * delta)

    up, down = one(shift - k), one(-shift - k)
    return 1.0 / (1.0 / up + 1.0 / down)
