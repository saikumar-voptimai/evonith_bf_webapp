"""From the gas distribution to the shaft efficiency ω: a contact hypothesis, to be tested.

The two-zone model holds ω (how close the gas gets to wüstite equilibrium at the reserve zone) at
the 14-day calibration value for every blend, because the plant data show no measurable blend
effect (GAS-10, ASM-01). The momentum model gives what the plant data do not resolve: how the gas
is shared between the rings, and the gas residence time in the ore of each ring. The hypothesis:

    ω = Σ_i f_i ω_i,     ω_i = 1 − exp(−NTU_i),     NTU_i = k_c · a_i · τ_i

f_i   share of the gas passing through ring i in the reserve zone (from the network);
a_i   specific surface of the ore layers of ring i (6 (1−ε)/(φ d), from the packing model);
τ_i   gas residence time in the ore layers of ring i between the cohesive zone and the top of the
      reserve zone: τ_i = Σ_cells ε_ore · t_ore/pitch · dz / u_i, with u_i the local superficial
      velocity (from the network at the local T and p);
k_c   one rate constant (m/s) calibrated so that ω = 0.80 at the reference point. It stands for
      the reducibility of the ferrous at the reserve-zone temperature and is the same for all rings.

What it predicts and what the register can test:
* gas that goes up a coke-rich centre meets little ore: f_centre high → ω low (centre-working);
* finer or more porous ferrous (pellet, Gomti) → higher a_i → higher ω_i (GAS-06, GAS-07, GAS-09);
* larger nut coke in the ore layer → more permeable ore layers → less gas squeezed to the centre
  → ω up (the GAS-09/FLOW-07 evidence: +0.01 ω per mm);
* ω day to day moves with the gas distribution, which the charging programme and the wall
  condition set: the part of the ω swing (SD 0.028) that the lab properties could not explain.

The number this module returns is a hypothesis until the record of daily ω (coke-energy pipeline,
14-day medians) has been regressed on its prediction. Until then the optimiser keeps ω held.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .network import NetworkResult
from .charging import BurdenColumn


@dataclass
class OmegaResult:
    omega: float
    omega_ring: np.ndarray
    gas_share: np.ndarray
    ntu_ring: np.ndarray
    tau_ring_s: np.ndarray
    surface_ring: np.ndarray
    k_c: float


def ring_contact(res: NetworkResult, column: BurdenColumn, z_rz_top: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gas share, specific surface 1/m, ore-layer residence time s) per ring over the reserve zone."""
    grid = res.grid
    n_r, n_z = grid.n_r, grid.n_z
    share = res.split_at(float(res.z_cz_top.max()) + 0.5)
    a = np.zeros(n_r); tau = np.zeros(n_r)
    for i in range(n_r):
        ring = column.rings[i]
        ore = ring.ore_degraded
        a[i] = ore.specific_surface()
        pitch = ring.t_ore + ring.t_coke
        f_ore = ring.t_ore / pitch
        for k in range(n_z):
            z = grid.z_cells[k]
            if res.z_cz_top[i] < z < z_rz_top:
                u = max(abs(res.u_axial[i, k]), 1e-4)
                tau[i] += ore.porosity * f_ore * grid.dz[k] / u
    return share, a, tau


def omega_from_network(res: NetworkResult, column: BurdenColumn, k_c: float, z_rz_top: float) -> OmegaResult:
    share, a, tau = ring_contact(res, column, z_rz_top)
    ntu = k_c * a * tau
    om = 1 - np.exp(-ntu)
    return OmegaResult(float(np.sum(share * om)), om, share, ntu, tau, a, k_c)


def calibrate_kc(res: NetworkResult, column: BurdenColumn, z_rz_top: float, omega_target: float = 0.80) -> float:
    """k_c such that the network's ω equals the calibration value (bisection on log k_c)."""
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        om = omega_from_network(res, column, mid, z_rz_top).omega
        if om < omega_target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))
