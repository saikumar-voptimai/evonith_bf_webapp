"""Porosity of a layer from the size distributions of what is in it.

Linear packing model (Stovall, de Larrard & Buil 1986; Yu & Standish 1987, Powder Technol. 52):
components sorted by decreasing size d_1 > ... > d_n, each with the specific volume of its own
random packing V_i = 1/(1 - eps_i) and solid volume fraction X_i. The packing is controlled by
one component at a time:

    V_T(i) = V_i X_i + sum_{j<i} [V_j - (V_j - 1) g(r_ij)] X_j + sum_{j>i} V_j [1 - f(r_ij)] X_j
    V_T = max_i V_T(i),   eps = 1 - 1/V_T,   r_ij = d_small / d_large

with the empirical interaction functions
    f(r) = (1 - r)^3.3 + 2.8 r (1 - r)^2.7     (filling: small j fill the voids of i)
    g(r) = (1 - r)^2.0 + 0.4 r (1 - r)^3.7     (occupation: large j are embedded in i)

f = g = 1 at r = 0 (no interaction penalty) and 0 at r = 1 (same size). The model reproduces the
porosity minimum of binary mixtures within about ±0.02 for r < 0.5 and is linear in the fractions,
which is what makes it cheap enough to run per cell per shift.

Wall effect: a layer against the wall packs looser over about one particle diameter; the ring
next to the wall gets  eps_wall = eps + k_wall * d_sv / w_ring  with k_wall ~ 0.05 (small for
BF rings 0.5–1 m wide).
"""
from __future__ import annotations

import numpy as np

from .materials import Material, SIZE_MID_MM


def f_fill(r: np.ndarray) -> np.ndarray:
    return (1 - r) ** 3.3 + 2.8 * r * (1 - r) ** 2.7


def g_occ(r: np.ndarray) -> np.ndarray:
    return (1 - r) ** 2.0 + 0.4 * r * (1 - r) ** 3.7


def lpm_porosity(d: np.ndarray, x: np.ndarray, eps_mono: np.ndarray) -> float:
    """Porosity of a mixture of size classes.

    d: class sizes (any unit), x: solid volume fractions (sum 1), eps_mono: porosity of each class
    packed alone.
    """
    d, x, eps_mono = map(np.asarray, (d, x, eps_mono))
    keep = x > 1e-9
    d, x, eps_mono = d[keep], x[keep], eps_mono[keep]
    x = x / x.sum()
    order = np.argsort(-d)
    d, x, eps_mono = d[order], x[order], eps_mono[order]
    V = 1.0 / (1.0 - eps_mono)
    n = len(d)
    best = 0.0
    for i in range(n):
        vt = V[i] * x[i]
        for j in range(n):
            if j == i:
                continue
            r = min(d[i], d[j]) / max(d[i], d[j])
            if j < i:      # larger than i
                vt += (V[j] - (V[j] - 1.0) * g_occ(r)) * x[j]
            else:          # smaller than i
                vt += V[j] * (1.0 - f_fill(r)) * x[j]
        best = max(best, vt)
    return float(1.0 - 1.0 / best)


class LayerBed:
    """Packed-bed properties of one layer (ore or coke) built from its materials and masses."""

    def __init__(self, parts: list[tuple[Material, float]], extra_fines: list[tuple[float, float, float]] | None = None):
        """parts: (material, mass kg); extra_fines: (size mm, solid volume m3, eps_mono) percolated in."""
        d_all, vol_all, eps_all, phi_all = [], [], [], []
        self.solid_volume = 0.0
        self.mass = 0.0
        for m, mass in parts:
            if mass <= 0:
                continue
            f = m.size.mass_fractions()
            vol = mass / m.particle_density
            self.mass += mass
            self.solid_volume += vol
            d_all.append(SIZE_MID_MM)
            vol_all.append(f * vol)
            eps_all.append(np.full_like(f, m.mono_porosity))
            phi_all.append(np.full_like(f, m.shape_factor))
        for d_mm, vol, eps in (extra_fines or []):
            if vol <= 0:
                continue
            d_all.append(np.array([d_mm])); vol_all.append(np.array([vol]))
            eps_all.append(np.array([eps])); phi_all.append(np.array([0.7]))
            self.solid_volume += vol
        if not d_all:
            raise ValueError("empty layer")
        d = np.concatenate(d_all); vol = np.concatenate(vol_all)
        eps = np.concatenate(eps_all); phi = np.concatenate(phi_all)
        x = vol / vol.sum()
        self.classes = (d, x, eps, phi)
        self.porosity = lpm_porosity(d, x, eps)
        self.sauter_mm = float(1.0 / np.sum(x / d))
        self.shape_factor = float(np.sum(x * phi))
        self.fines_frac = float(x[d < 5.0].sum())
        self.bulk_volume = self.solid_volume / (1.0 - self.porosity)

    def with_wall_effect(self, ring_width_m: float, k_wall: float = 0.05) -> float:
        return min(0.75, self.porosity + k_wall * (self.sauter_mm / 1000.0) / max(ring_width_m, 0.05))

    def specific_surface(self) -> float:
        """Solid surface per bed volume, 1/m (6(1-eps)/(phi d))."""
        return 6.0 * (1.0 - self.porosity) / (self.shape_factor * self.sauter_mm / 1000.0)


def percolating_fines(ore_bed: LayerBed, coke_sauter_mm: float, ratio: float = 0.154, share: float = 0.5):
    """Ore fines that can percolate into the coke layer below.

    Fines smaller than `ratio` × coke size pass through the coke voids spontaneously (geometric
    percolation threshold, Bridgwater/Scott: 0.154 for monosize spheres). `share` of them are
    assumed to do so during charging and descent. Returns (list of (d_mm, solid vol, eps) for the
    coke layer, remaining fraction of ore solid volume).
    """
    d, x, eps, _ = ore_bed.classes
    thr = ratio * coke_sauter_mm
    move = (d < thr)
    if not move.any():
        return [], 1.0
    moved = []
    for dk, xk, ek in zip(d[move], x[move], eps[move]):
        moved.append((float(dk), float(share * xk * ore_bed.solid_volume), float(ek)))
    remaining = 1.0 - share * float(x[move].sum())
    return moved, remaining
