"""Resistance-network solution of the gas flow through the burden column.

Nodes are the cells of the ring × level grid; edges join vertically and radially adjacent cells
(and the top cells to the stockline at the top pressure). Each edge carries dp = a m + b m|m|
from the Ergun coefficients of the bed between the two cell centres (half of each cell in
series), with the gas density and viscosity of the cell. The bosh gas enters at the raceway
cells; the indirect-reduction and carbonate CO2 are added as sources in the lumpy zone. The
nonlinear system is solved by successive linearisation (g = 1/(a + b|m|)) with under-relaxation;
the gas density is updated from the new pressures and the cohesive-zone shape from the new gas
distribution, and the whole is repeated to convergence.

Anisotropy of a layered bed: along the axis the ore and coke layers are in series (the ore
layers set the resistance); across the axis they are in parallel (the coke layers carry the
flow). This is what lets the gas redistribute towards the coke-rich centre and is the mechanism
behind the radial split the omega link uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Grid
from .charging import BurdenColumn
from .gas import GasSpec, interpolate_composition, mu_mix, density, molar_mass, temperature_profile
from .ergun import ergun_AB
from .cohesive import CZParams, cz_shape, slit_angles

BAR = 1e5


@dataclass
class FlowParams:
    tuyere_dp_bar_ref: float = 0.25      # tuyere + blowpipe loss at the reference blast volume, all 18 tuyeres at design bore
    tuyere_dp_ref_nm3_h: float = 111250.0
    tuyere_area_factor: float = 1.0      # open tuyere area / design (sleeves, blanking): dp ∝ (V / area)²
    raceway_porosity: float = 0.85       # raceway cavity cells
    z_rz_top_below_stockline: float = 5.5  # top of the thermal reserve zone, m below the stockline
    degradation_front_T_c: float = 600.0


@dataclass
class NetworkResult:
    grid: Grid
    p: np.ndarray                 # Pa, (n_r, n_z)
    T: np.ndarray                 # K, (n_r, n_z)
    rho: np.ndarray
    u_axial: np.ndarray           # superficial velocity at cell centres (upward +), m/s
    m_axial_faces: np.ndarray     # kg/s through each ring at each face (n_r, n_z+1)
    m_radial: np.ndarray          # kg/s from ring i to i+1 at each level (n_r-1, n_z), outward +
    zone: np.ndarray              # 0 lumpy fresh, 1 lumpy degraded, 2 CZ, 3 dripping, 4 deadman, 5 raceway
    z_cz_bot: np.ndarray
    z_cz_top: np.ndarray
    p_tuyere_pa: float
    p_bustle_pa: float
    p_top_pa: float
    m_in_kg_s: float
    m_top_kg_s: float
    iterations: int
    converged: bool
    geometry_assumed: bool
    extras: dict = field(default_factory=dict)

    # ---- derived
    @property
    def dp_total_bar(self) -> float:
        return (self.p_bustle_pa - self.p_top_pa) / BAR

    @property
    def dp_bed_bar(self) -> float:
        return (self.p_tuyere_pa - self.p_top_pa) / BAR

    def top_split(self) -> np.ndarray:
        """Share of the gas leaving through each ring at the stockline."""
        m = self.m_axial_faces[:, -1]
        return m / m.sum()

    def split_at(self, z: float) -> np.ndarray:
        k = int(np.clip(np.searchsorted(self.grid.z_nodes, z) - 1, 0, self.grid.n_z - 1))
        m = np.clip(self.m_axial_faces[:, k + 1], 0, None)
        return m / max(m.sum(), 1e-12)

    def level_mean_pressure(self) -> np.ndarray:
        w = self.grid.ring_area_frac
        return (self.p * w[:, None]).sum(axis=0)

    def zone_dp_bar(self) -> dict[str, float]:
        """Pressure drop by zone along the cross-section-mean pressure profile."""
        pm = self.level_mean_pressure()
        zc = self.grid.z_cells
        def p_at(z):
            return float(np.interp(z, zc, pm))
        zb, zt = float(self.z_cz_bot.mean()), float(self.z_cz_top.mean())
        p_tuy = self.p_tuyere_pa
        out = {
            "tuyeres_and_blowpipe": (self.p_bustle_pa - p_tuy) / BAR,
            "raceway_dripping_deadman": (p_tuy - p_at(zb)) / BAR,
            "cohesive_zone": (p_at(zb) - p_at(zt)) / BAR,
            "lumpy_zone": (p_at(zt) - self.p_top_pa) / BAR,
        }
        return out


def _lumpy_coeffs(ring, degraded: bool, mixed_frac: float):
    """(A_ax, B_ax, A_rad, B_rad) per unit length for a lumpy-zone cell of one ring."""
    ore = ring.ore_degraded if degraded else ring.ore_fresh
    coke = ring.coke_degraded if degraded else ring.coke_fresh
    mixed = ring.mixed
    pitch = ring.t_ore + ring.t_coke
    fo, fc = ring.t_ore / pitch, ring.t_coke / pitch
    fm = mixed_frac
    fo, fc = fo * (1 - fm), fc * (1 - fm)
    Ao, Bo = ergun_AB(ore.porosity, ore.sauter_mm / 1000, ore.shape_factor)
    Ac, Bc = ergun_AB(coke.porosity, coke.sauter_mm / 1000, coke.shape_factor)
    Am, Bm = ergun_AB(mixed.porosity, mixed.sauter_mm / 1000, mixed.shape_factor)
    A_ax = fo * Ao + fc * Ac + fm * Am
    B_ax = fo * Bo + fc * Bc + fm * Bm
    A_rad = 1.0 / (fo / Ao + fc / Ac + fm / Am)
    B_rad = 1.0 / (fo / Bo + fc / Bc + fm / Bm)
    return A_ax, B_ax, A_rad, B_rad


def _cz_coeffs(ring, alpha: float, cz: CZParams):
    coke = ring.coke_degraded
    eps = max(0.15, coke.porosity - cz.coke_void_loss)
    d = coke.sauter_mm / 1000 * cz.coke_size_factor
    A, B = ergun_AB(eps, d, coke.shape_factor)
    pitch = ring.t_ore + ring.t_coke
    s = ring.t_coke / pitch
    ax = cz.axial_block_factor / (s * (np.sin(alpha) ** 2 + cz.slit_floor))
    rad = 1.0 / (s * max(np.cos(alpha) ** 2, 0.2))
    return A * ax, B * ax, A * rad, B * rad


def _lower_coeffs(ring, lower, deadman: bool):
    coke = ring.coke_degraded
    eps = coke.porosity - lower.liquid_holdup - (lower.deadman_porosity_drop if deadman else 0.0)
    eps = max(0.15, eps)
    d = coke.sauter_mm / 1000 * lower.coke_size_factor_lower
    A, B = ergun_AB(eps, d, coke.shape_factor)
    return A, B, A, B


def solve_network(grid: Grid, column: BurdenColumn, gas: GasSpec, cz: CZParams = CZParams(),
                  flow: FlowParams = FlowParams(), z_cz_top_init: np.ndarray | None = None,
                  max_outer: int = 60, tol: float = 5e-4, verbose: bool = False) -> NetworkResult:
    n_r, n_z = grid.n_r, grid.n_z
    zc, zf, dz = grid.z_cells, grid.z_nodes, grid.dz
    z_top = grid.profile.stockline_z
    p_top = (gas.top_pressure_bar_g + 1.01325) * BAR
    R_prof = grid.profile.radius(zc)                              # (n_z,)
    areas = np.array([grid.ring_area(z) for z in zc]).T           # (n_r, n_z)
    face_areas = np.array([grid.ring_area(z) for z in zf]).T      # (n_r, n_z+1)
    widths = np.array([grid.ring_width(z) for z in zc]).T         # (n_r, n_z)
    rmid = np.array([grid.ring_mid_radius(z) for z in zc]).T      # (n_r, n_z)
    rb = np.array([grid.ring_boundary_radius(z) for z in zc]).T   # (n_r-1, n_z)

    # gas: bosh and top
    n_bosh, y_bosh = gas.bosh_gas()
    n_top, y_top = gas.top_gas()
    m_in = n_bosh * 1000 * molar_mass(y_bosh) / 3600.0           # kg/s
    m_top = n_top * 1000 * molar_mass(y_top) / 3600.0
    dm = m_top - m_in

    # inlet distribution: raceway annulus R - depth .. R at tuyere level
    Rt = grid.profile.radius(0.0)
    edges_t = grid.ring_edges_frac * Rt
    r_in = Rt - grid.profile.raceway_depth_m
    overlap = np.clip(np.minimum(edges_t[1:], Rt) ** 2 - np.maximum(edges_t[:-1], r_in) ** 2, 0, None)
    inlet_share = overlap / overlap.sum()
    raceway_rings = inlet_share > 0

    # ore mass flux per ring (kg/m²/s), evaluated at the CZ reference height
    cph = column.charge.charges_per_hour
    ore_mass = (sum(column.charge.ore_kg.values()) + column.charge.nut_coke_kg)
    A_cz = grid.ring_area(cz.z_top_ref)
    G_ore = column.dist.ore_share * ore_mass * cph / 3600.0 / A_cz

    z_cz_top = np.full(n_r, cz.z_top_ref) if z_cz_top_init is None else z_cz_top_init.copy()
    z_cz_bot = z_cz_top - cz.thickness_m
    p = p_top + (1.3 * BAR) * (1 - zc / z_top)[None, :].repeat(n_r, 0)
    m_ax_prev = None
    converged = False
    it = 0
    for it in range(1, max_outer + 1):
        # ---- temperature, composition, properties per cell
        z_rz_top = z_top - flow.z_rz_top_below_stockline
        T = np.zeros((n_r, n_z)); rho = np.zeros_like(T); mu = np.zeros_like(T); zone = np.zeros((n_r, n_z), int)
        A_ax = np.zeros_like(T); B_ax = np.zeros_like(T); A_rad = np.zeros_like(T); B_rad = np.zeros_like(T)
        alpha = slit_angles(z_cz_top, rmid[:, 0])
        deadman_top = cz.z_top_ref * column.lower.deadman_top_frac_of_cz
        def overlap(lo, hi, a, b):
            return max(0.0, min(hi, b) - max(lo, a))

        for i in range(n_r):
            T[i] = temperature_profile(zc, z_cz_bot[i], z_cz_top[i], z_rz_top, z_top, gas.raft_c, gas.reserve_T_c, gas.top_gas_T_c)
            ring = column.rings[i]
            c_lum_f = _lumpy_coeffs(ring, False, column.shaft.mixed_layer_frac)
            c_lum_d = _lumpy_coeffs(ring, True, column.shaft.mixed_layer_frac)
            c_cz = _cz_coeffs(ring, alpha[i], cz)
            c_low = _lower_coeffs(ring, column.lower, False)
            c_dm = _lower_coeffs(ring, column.lower, True)
            A_r, B_r = ergun_AB(flow.raceway_porosity, 0.05, 0.75)
            c_race = (A_r, B_r, A_r, B_r)
            for k in range(n_z):
                z = zc[k]
                s = float(np.clip((z - z_cz_top[i]) / max(z_top - z_cz_top[i], 1e-6), 0, 1))
                y = interpolate_composition(y_bosh, y_top, s)
                rho[i, k] = density(y, T[i, k], p[i, k]); mu[i, k] = mu_mix(y, T[i, k])
                lo, hi = zf[k], zf[k + 1]
                f_cz = overlap(lo, hi, z_cz_bot[i], z_cz_top[i]) / dz[k]
                f_lum = overlap(lo, hi, z_cz_top[i], 1e9) / dz[k]
                f_low = max(0.0, 1.0 - f_cz - f_lum)
                degraded = T[i, k] - 273.15 > flow.degradation_front_T_c
                dm_cell = (rmid[i, k] < column.lower.deadman_radius_frac * R_prof[k]) and (z < deadman_top)
                race = (k == 0) and raceway_rings[i]
                parts = []
                if f_lum > 0:
                    parts.append((f_lum, c_lum_d if degraded else c_lum_f))
                if f_cz > 0:
                    parts.append((f_cz, c_cz))
                if f_low > 0:
                    parts.append((f_low, c_race if race else (c_dm if dm_cell else c_low)))
                # series along the axis, parallel across it
                A_ax[i, k] = sum(f * c[0] for f, c in parts)
                B_ax[i, k] = sum(f * c[1] for f, c in parts)
                A_rad[i, k] = 1.0 / sum(f / c[2] for f, c in parts)
                B_rad[i, k] = 1.0 / sum(f / c[3] for f, c in parts)
                fmax = max(parts, key=lambda t: t[0])[0]
                if f_cz == fmax:
                    zone[i, k] = 2
                elif f_lum == fmax:
                    zone[i, k] = 1 if degraded else 0
                else:
                    zone[i, k] = 5 if race else (4 if dm_cell else 3)

        # ---- edges
        # axial: cell (i,k) -> (i,k+1), k = 0..n_z-2 ; top: (i,n_z-1) -> boundary
        a_ax = np.zeros((n_r, n_z)); b_ax = np.zeros((n_r, n_z))
        for i in range(n_r):
            for k in range(n_z):
                S = face_areas[i, k + 1]
                if k < n_z - 1:
                    L1, L2 = 0.5 * dz[k], 0.5 * dz[k + 1]
                    a_ax[i, k] = (L1 * A_ax[i, k] * mu[i, k] / rho[i, k] + L2 * A_ax[i, k + 1] * mu[i, k + 1] / rho[i, k + 1]) / S
                    b_ax[i, k] = (L1 * B_ax[i, k] / rho[i, k] + L2 * B_ax[i, k + 1] / rho[i, k + 1]) / S ** 2
                else:
                    L1 = 0.5 * dz[k]
                    a_ax[i, k] = L1 * A_ax[i, k] * mu[i, k] / rho[i, k] / S
                    b_ax[i, k] = L1 * B_ax[i, k] / rho[i, k] / S ** 2
        # radial: (i,k) -> (i+1,k)
        a_rad = np.zeros((n_r - 1, n_z)); b_rad = np.zeros((n_r - 1, n_z))
        for i in range(n_r - 1):
            for k in range(n_z):
                S = 2 * np.pi * rb[i, k] * dz[k]
                L1, L2 = 0.5 * widths[i, k], 0.5 * widths[i + 1, k]
                a_rad[i, k] = (L1 * A_rad[i, k] * mu[i, k] / rho[i, k] + L2 * A_rad[i + 1, k] * mu[i + 1, k] / rho[i + 1, k]) / S
                b_rad[i, k] = (L1 * B_rad[i, k] / rho[i, k] + L2 * B_rad[i + 1, k] / rho[i + 1, k]) / S ** 2

        # ---- sources
        src = np.zeros((n_r, n_z))
        src[:, 0] += m_in * inlet_share
        w = np.zeros((n_r, n_z))
        for i in range(n_r):
            lum = zc > z_cz_top[i]
            w[i, lum] = column.dist.ore_share[i] * dz[lum]
        src += dm * w / w.sum()

        # ---- nonlinear solve: successive linearisation
        idx = lambda i, k: i * n_z + k
        N = n_r * n_z
        m_ax = np.full((n_r, n_z), m_in / n_r) if m_ax_prev is None else m_ax_prev.copy()
        m_rd = np.zeros((n_r - 1, n_z))
        for inner in range(200):
            g_ax = 1.0 / (a_ax + b_ax * np.abs(m_ax))
            g_rd = 1.0 / (a_rad + b_rad * np.abs(m_rd))
            rows, cols, vals = [], [], []
            rhs = src.reshape(-1).copy()
            diag = np.zeros(N)
            for i in range(n_r):
                for k in range(n_z):
                    n0 = idx(i, k)
                    g = g_ax[i, k]
                    diag[n0] += g
                    if k < n_z - 1:
                        n1 = idx(i, k + 1)
                        diag[n1] += g
                        rows += [n0, n1]; cols += [n1, n0]; vals += [-g, -g]
                    else:
                        rhs[n0] += g * p_top
            for i in range(n_r - 1):
                for k in range(n_z):
                    n0, n1 = idx(i, k), idx(i + 1, k)
                    g = g_rd[i, k]
                    diag[n0] += g; diag[n1] += g
                    rows += [n0, n1]; cols += [n1, n0]; vals += [-g, -g]
            rows += list(range(N)); cols += list(range(N)); vals += diag.tolist()
            G = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
            p_new = spla.spsolve(G, rhs).reshape(n_r, n_z)
            m_ax_new = np.zeros_like(m_ax); m_rd_new = np.zeros_like(m_rd)
            for i in range(n_r):
                m_ax_new[i, :-1] = g_ax[i, :-1] * (p_new[i, :-1] - p_new[i, 1:])
                m_ax_new[i, -1] = g_ax[i, -1] * (p_new[i, -1] - p_top)
            for i in range(n_r - 1):
                m_rd_new[i] = g_rd[i] * (p_new[i] - p_new[i + 1])
            relax = 0.5
            change = np.max(np.abs(m_ax_new - m_ax)) / max(m_in, 1e-9)
            m_ax = relax * m_ax_new + (1 - relax) * m_ax
            m_rd = relax * m_rd_new + (1 - relax) * m_rd
            if change < 1e-5:
                break
        p_old = p
        p = 0.6 * p_new + 0.4 * p
        m_ax_prev = m_ax

        # ---- cohesive zone from the gas flux just below the zone
        # gas distribution in the lumpy zone above the whole cohesive zone (the heat exchange that
        # sets where each ring's solids reach softening happens there)
        band = (zc > z_cz_top.max() + 0.3) & (zc < z_rz_top)
        if not band.any():
            band = zc > z_cz_top.max()
        G_gas = np.array([np.mean(np.clip(m_ax[i, band], 1e-9, None) / face_areas[i, 1:][band]) for i in range(n_r)])
        zb_new, zt_new = cz_shape(G_gas, G_ore, cz, z_top, z_cz_top, relax=0.3, r_mid=rmid[:, 0])
        dcz = float(np.max(np.abs(zt_new - z_cz_top)))
        dpc = float(np.max(np.abs(p - p_old)) / BAR)
        z_cz_bot, z_cz_top = zb_new, zt_new
        if verbose:
            print(f"outer {it}: dp_bed {(p[:,0].mean()-p_top)/BAR:.3f} bar, max dz_cz {dcz:.3f} m, max dp {dpc:.4f} bar")
        if dcz < 0.02 and dpc < tol:
            converged = True
            break

    # ---- outputs
    m_faces = np.zeros((n_r, n_z + 1))
    m_faces[:, 1:] = m_ax
    m_faces[:, 0] = 0.0
    u = m_ax / (rho * areas)   # upward superficial velocity at the cell's top face, approx at centre
    p_tuy = float(np.average(p[raceway_rings, 0], weights=inlet_share[raceway_rings]))
    dp_tuy = flow.tuyere_dp_bar_ref * BAR * (gas.blast_nm3_h / flow.tuyere_dp_ref_nm3_h / flow.tuyere_area_factor) ** 2
    return NetworkResult(grid, p, T, rho, u, m_faces, m_rd, zone, z_cz_bot, z_cz_top, p_tuy, p_tuy + dp_tuy, p_top,
                         m_in, m_top, it, converged, grid.profile.assumed,
                         extras={"inlet_share": inlet_share, "G_ore": G_ore, "alpha_deg": np.degrees(alpha),
                                 "gas_ore_ratio_mean": getattr(cz_shape, "ratio_mean", float("nan"))})
