"""Ergun pressure gradient and its linear/quadratic coefficients.

dp/dz = A mu u + B rho u^2,   A = 150 (1-eps)^2 / (phi^2 d^2 eps^3),  B = 1.75 (1-eps) / (phi d eps^3)

u is the superficial velocity at local T and p. For the network the gradient is written in terms
of the mass flow m (kg/s) through an area S over a length L:
    dp = L (A mu / (rho S)) m + L (B / (rho S^2)) m |m|
so each edge carries two coefficients (a, b) with dp = a m + b m |m|.

Ergun was fitted on uniform beds with 1 < Re_p < 3,000; BF beds sit inside that range in the
lumpy zone (Re_p ~ 100–1,000) and the inertial term dominates, which is why the plant law uses
an exponent of 1.8 rather than 2.
"""
from __future__ import annotations

import numpy as np


def ergun_AB(eps: float, d_m: float, phi: float) -> tuple[float, float]:
    e3 = eps ** 3
    A = 150.0 * (1 - eps) ** 2 / (phi ** 2 * d_m ** 2 * e3)
    B = 1.75 * (1 - eps) / (phi * d_m * e3)
    return A, B


def ergun_gradient(eps: float, d_m: float, phi: float, mu: float, rho: float, u: float) -> float:
    A, B = ergun_AB(eps, d_m, phi)
    return A * mu * u + B * rho * u * abs(u)


def edge_coefficients(eps: float, d_m: float, phi: float, mu: float, rho: float, S: float, L: float) -> tuple[float, float]:
    """(a, b) with dp = a m + b m |m| for mass flow m through area S over length L."""
    A, B = ergun_AB(eps, d_m, phi)
    a = L * A * mu / (rho * S)
    b = L * B / (rho * S * S)
    return a, b


def series(coeffs: list[tuple[float, float]]) -> tuple[float, float]:
    """Edges in series carry the same m: coefficients add."""
    return (sum(c[0] for c in coeffs), sum(c[1] for c in coeffs))


def particle_reynolds(d_m: float, rho: float, u: float, mu: float, eps: float) -> float:
    return rho * u * d_m / (mu * (1 - eps))
