"""Champion-challenger gate and drift alarm of STRATA-Learn.

Gate: on the last ``window`` days after deployment, compare the one-step out-of-sample errors of the challenger (the
Kalman state of the day before) with those of the champion (the deployed asset). Promotion is proposed - never applied
- when the challenger's MSE is lower by at least ``min_gain`` and the Diebold-Mariano t (Newey-West, ``lags``) on the
squared-error difference exceeds ``t_min``, and no drift alarm is open.

Drift: two-sided Page CUSUM on the standardised AR(1) innovations of the champion's errors,
    u_t = (e_t - rho e_{t-1}) / sigma_e,  S+_t = max(0, S+_{t-1} + u_t - k),  S-_t = max(0, S-_{t-1} - u_t - k),
alarm when S+ or S- exceeds h (k 0.5, h 5: two-sided in-control run length about 470 days for independent normal
innovations).
An alarm marks the model 'recalibration needed' and widens sigma by ``sigma_inflation`` until it is cleared."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

WINDOW, MIN_GAIN, T_MIN, LAGS = 60, 0.05, 2.0, 5
CUSUM_K, CUSUM_H, SIGMA_INFLATION = 0.5, 5.0, 1.5


def dm_test(e_a: Sequence[float], e_b: Sequence[float], lags: int = LAGS) -> tuple[float, float]:
    """(mean loss difference, t) for d = e_a^2 - e_b^2 with a Newey-West variance. t > 0: b is better."""
    d = np.asarray(e_a, float) ** 2 - np.asarray(e_b, float) ** 2
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dm = d - d.mean()
    v = float(dm @ dm) / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * float(dm[L:] @ dm[:-L]) / n
    return float(d.mean()), float(d.mean() / math.sqrt(max(v, 1e-30) / n))


def gate(y, pred_champion, pred_challenger, window: int = WINDOW, min_gain: float = MIN_GAIN, t_min: float = T_MIN,
         drift_open: bool = False) -> dict[str, Any]:
    y, pc, pn = (np.asarray(a, float) for a in (y, pred_champion, pred_challenger))
    n = len(y)
    if n < window:
        return dict(status="collecting", n=n, window=window, promote=False,
                    reason=f"{n} of {window} days after deployment")
    e_c, e_n = (y - pc)[-window:], (y - pn)[-window:]
    mse_c, mse_n = float(np.mean(e_c ** 2)), float(np.mean(e_n ** 2))
    gain = 1 - mse_n / mse_c if mse_c > 0 else 0.0
    _, t = dm_test(e_c, e_n)
    promote = bool(gain >= min_gain and t >= t_min and not drift_open)
    reason = ("challenger better: proposal for approval" if promote else
              "drift alarm open" if drift_open else
              f"gain {100 * gain:.1f} % (need {100 * min_gain:.0f} %), DM t {t:.2f} (need {t_min:.1f})")
    return dict(status="evaluated", n=n, window=window, mse_champion=mse_c, mse_challenger=mse_n, gain=gain,
                dm_t=t, promote=promote, reason=reason)


def cusum(errors, rho: float, sigma_e: float, k: float = CUSUM_K, h: float = CUSUM_H) -> dict[str, Any]:
    e = np.asarray(errors, float)
    if len(e) < 2:
        return dict(alarm=False, s_plus=0.0, s_minus=0.0, first_alarm=None, path=[])
    u = (e[1:] - rho * e[:-1]) / sigma_e
    sp = sm = 0.0
    first, path = None, []
    for i, x in enumerate(u, start=1):
        sp, sm = max(0.0, sp + x - k), max(0.0, sm - x - k)
        path.append((sp, sm))
        if first is None and (sp > h or sm > h):
            first = i
    return dict(alarm=bool(sp > h or sm > h), s_plus=sp, s_minus=sm, first_alarm=first,
                ever_alarm=first is not None, path=path)
