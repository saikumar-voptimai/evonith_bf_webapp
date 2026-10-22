"""Random-walk Kalman filter on the STRATA-omega plant-data scales with AR(1) day errors.

State theta = (lam_pellet, lam_lump, lam_production, lam_other, a_moisture); theta_t = theta_{t-1} + w_t,
w ~ N(0, Q), Q = diag(rw_sd^2) per day. Observation on consecutive days after the Prais-Winsten transform
    dobs_t - rho dobs_{t-1} = (x_t - rho x_{t-1})' theta_t + e_t,   e ~ N(0, sigma_e^2)
and sqrt(1 - rho^2) (dobs_t = x_t' theta_t + e_t) after a gap. The one-step prediction of each day is made with
theta_{t-1} before the day updates the state, so the prediction history is out of sample by construction.
The prediction recorded is x_t' theta_{t-1} (no lagged residual): the quantity the optimiser uses for a plan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.burden_engine.learn.evidence import PARAMS

PRIOR_MEAN = (1.0, 1.0, 1.0, 1.0, 0.0)
PRIOR_SD = (0.5, 0.5, 0.5, 0.5, 0.005)
RW_SD = (0.01, 0.01, 0.01, 0.01, 0.00002)          # per day: about 0.19 (lam) over a year of random walk


@dataclass
class KalmanState:
    mean: np.ndarray
    cov: np.ndarray
    rho: float
    sigma_e: float
    last_date: Any = None
    last_x: np.ndarray | None = None
    last_y: float | None = None
    n: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class RandomWalkKalman:
    def __init__(self, mean, cov, rho: float, sigma_e: float, rw_sd=RW_SD):
        self.s = KalmanState(np.asarray(mean, float).copy(), np.asarray(cov, float).copy(), float(rho), float(sigma_e))
        self.Q = np.diag(np.asarray(rw_sd, float) ** 2)

    @classmethod
    def from_prior(cls, rho: float, sigma_e: float, rw_sd=RW_SD) -> RandomWalkKalman:
        """Start from the priors of the STRATA posterior (for an honest replay of the record)."""
        return cls(PRIOR_MEAN, np.diag(np.asarray(PRIOR_SD) ** 2), rho, sigma_e, rw_sd)

    @classmethod
    def from_asset(cls, spec: Mapping[str, Any], rw_sd=RW_SD) -> RandomWalkKalman:
        """Start from the deployed STRATA posterior (``channel_scales`` of the asset): the forward-running learner."""
        cs = spec["channel_scales"]
        return cls(cs["mean"], cs["cov"], cs["rho"], cs["sigma_e"], rw_sd)

    @property
    def mean(self) -> np.ndarray:
        return self.s.mean

    @property
    def cov(self) -> np.ndarray:
        return self.s.cov

    def predict(self, x) -> tuple[float, float]:
        """(prediction of dobs, 1-sigma of the parameter part) for regressors x with the current state."""
        x = np.asarray(x, float)
        return float(x @ self.s.mean), float(math.sqrt(max(x @ self.s.cov @ x, 0.0)))

    def step(self, date, x, y: float) -> dict[str, Any]:
        """Predict the day out of sample, then update with its observation."""
        s = self.s
        x = np.asarray(x, float)
        d = np.datetime64(date, "D")
        consecutive = s.last_date is not None and (d - s.last_date).astype(int) == 1
        s.cov = s.cov + self.Q                                         # time update (random walk)
        pred, psd = self.predict(x)
        if consecutive:
            h = x - s.rho * s.last_x
            yt = y - s.rho * s.last_y
        else:
            f = math.sqrt(1 - s.rho ** 2)
            h, yt = f * x, f * y
        S = float(h @ s.cov @ h + s.sigma_e ** 2)
        K = s.cov @ h / S
        innov = yt - float(h @ s.mean)
        s.mean = s.mean + K * innov
        s.cov = s.cov - np.outer(K, h @ s.cov)
        s.cov = 0.5 * (s.cov + s.cov.T)
        s.last_date, s.last_x, s.last_y = d, x, float(y)
        s.n += 1
        rec = dict(date=str(d), y=float(y), pred=pred, pred_sd=psd, z=innov / math.sqrt(S),
                   **{p: float(v) for p, v in zip(PARAMS, s.mean)},
                   **{p + "_sd": float(math.sqrt(max(v, 0.0))) for p, v in zip(PARAMS, np.diag(s.cov))})
        s.history.append(rec)
        return rec

    def run(self, dates, X, y) -> list[dict[str, Any]]:
        return [self.step(d, x, yy) for d, x, yy in zip(dates, X, y)]

    def summary(self) -> dict[str, Any]:
        sd = np.sqrt(np.clip(np.diag(self.s.cov), 0.0, None))
        return dict(n=self.s.n, last_date=str(self.s.last_date), mean=dict(zip(PARAMS, map(float, self.s.mean))),
                    sd=dict(zip(PARAMS, map(float, sd))), cov=self.s.cov.tolist())
