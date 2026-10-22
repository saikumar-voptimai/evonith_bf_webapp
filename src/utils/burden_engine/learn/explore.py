"""Constrained exploration for STRATA-Learn (shadow mode: suggestions only).

For candidate plans j with expected cost C_j at the posterior mean and cost sensitivity H_j (Rs/tHM per unit of each
parameter), cost_j(theta) = C_j + H_j (theta - theta_hat) is exact for the blend and operating channels (they are
linear in the scales). Thompson sampling gives the probability that each plan is optimal under the posterior;
the expected regret of plan j is E[cost_j - min_i cost_i]. A plan is an exploration candidate only if it is not in the
Investigate tier and its expected regret against the posterior-mean choice is within the regret budget."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def thompson(C0, H, mean, cov, n: int = 20000, seed: int = 0) -> dict[str, np.ndarray]:
    C0, H = np.asarray(C0, float), np.asarray(H, float)
    rng = np.random.default_rng(seed)
    th = rng.multivariate_normal(np.asarray(mean, float), np.asarray(cov, float), n)
    cost = C0 + (th - np.asarray(mean, float)) @ H.T
    best = cost.argmin(axis=1)
    p_opt = np.bincount(best, minlength=len(C0)) / n
    regret = (cost - cost.min(axis=1, keepdims=True)).mean(axis=0)
    return dict(p_opt=p_opt, regret=regret, mean_cost=cost.mean(axis=0))


def suggest(labels: Sequence[str], C0, H, mean, cov, tiers: Sequence[str], regret_budget: float = 15.0,
            n: int = 20000) -> dict[str, Any]:
    """Posterior-mean choice, Thompson probabilities and the exploration suggestion within the regret budget."""
    t = thompson(C0, H, mean, cov, n)
    i_hat = int(np.argmin(C0))
    extra = np.asarray(C0, float) - float(C0[i_hat])
    ok = [j for j in range(len(C0)) if tiers[j] != "Investigate" and extra[j] <= regret_budget]
    j_exp = max(ok, key=lambda j: t["p_opt"][j]) if ok else i_hat
    rows = [dict(plan=labels[j], cost=float(C0[j]), extra_cost=float(extra[j]), p_optimal=float(t["p_opt"][j]),
                 expected_regret=float(t["regret"][j]), tier=tiers[j], eligible=j in ok) for j in range(len(C0))]
    return dict(choice=labels[i_hat], explore=labels[j_exp], rows=rows, regret_budget=regret_budget,
                note="Shadow mode: the suggestion is reported, not applied.")


def plan_sensitivity(om_terms: Mapping[str, Any], cls: Mapping[str, float], ref_cls: Mapping[str, float],
                     g_per_pt: Mapping[str, float], d_humidity: float, rs_per_omega: float) -> np.ndarray:
    """H of one plan (Rs/tHM per unit of lam_pellet, lam_lump, lam_production, lam_other, a_moisture)."""
    ds = {c: float(cls.get(c, 0.0)) - float(ref_cls.get(c, 0.0)) for c in ("pellet", "clo", "rom", "gomti", "oversize")}
    h = np.array([g_per_pt["sh_pellet"] * ds["pellet"],
                  sum(g_per_pt["sh_" + c] * ds[c] for c in ("clo", "rom", "gomti", "oversize")),
                  float(om_terms.get("d_prod_phys", 0.0)), float(om_terms.get("d_rest_phys", 0.0)), float(d_humidity)])
    return rs_per_omega * h
