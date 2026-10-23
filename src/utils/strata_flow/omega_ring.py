"""Shaft efficiency ω from the layered gas-flow model: ring-wise counter-current reduction of wüstite (v0.4).

Definition (FORGE, ``utils.coke_energy.rist`` E2): ω = oxygen removed from wüstite by the gas in the thermal reserve
zone (TRZ) / oxygen the same gas could remove at wüstite equilibrium (η_W at T_R).

Each of the six rings of the gas-flow model is a counter-current gas–solid exchanger over the TRZ:

    C_g,i = R · f_i            oxygen the ring's gas can take at equilibrium     f_i: share of the gas (network)
    C_s,i = o_i                wüstite oxygen the ring's ore brings                o_i: share of the Fe (burden profile)
    R     = η_W n_gas / (Y_W n_Fe)  furnace capacity ratio (two-zone balance; 0.704 at the Jul–Oct 2026 point)

The rate of wüstite reduction in the TRZ is controlled inside the particles (pore diffusion and interface reaction at
900–1000 °C), which is what the ISO 7215 reducibility index measures. The transfer units are therefore counted on the
solid:

    N_s,i = κ · k_i · t_TRZ,i                  k_i = Σ_m x_m,i r_m (d_test / d_s,m)^n
    UA_i  = N_s,i · C_s,i,   NTU_i = UA_i / C_min,i,   C_r,i = C_min,i / C_max,i
    ε_i   = (1 − e^{−N(1−C_r)}) / (1 − C_r e^{−N(1−C_r)})       (N/(1+N) at C_r = 1)
    ω     = Σ ε_i C_min,i / Σ C_g,i

x_m,i   share of ring i's Fe brought by ferrous material m;
r_m     reducibility relative to sinter from the ISO 7215 index on 10–12.5 mm pieces (d_test 11.25 mm):
        r_m = ln(1 − RI_m) / ln(1 − RI_sinter); the index already contains the material's porosity and shape;
d_s,m   as-charged Sauter mean size (plant sieve analyses); RDI and decrepitation fines act on the bed only;
n       1.5, size exponent of the reduction rate (chemical control d^−1, pore diffusion d^−2);
t_TRZ,i time the burden spends between the top of the TRZ and the top of the cohesive zone in ring i (descent at
        constant solid volume; the cohesive zone is higher at the centre, so the centre's TRZ is shorter);
κ       one constant, calibrated so that ω equals the plant calibration at the reference point.

What the form represents: (1) each ferrous reduces at its own rate per tonne of Fe (reducibility, size); (2) a ring
whose gas exceeds its ore's wüstite oxygen (the coke-rich centre, or the wall) cannot use the surplus, so gas sent up
the centre or the wall without matching ore lowers ω; (3) a faster descent (more production, a lighter burden per
tonne) or a lower cohesive zone shortens the TRZ residence and lowers ω.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

Y_W = 1.0 / 0.947          # O/Fe of wüstite at the W point (as rist.Y_W)
RI_SINTER_REF = 68.7       # % (plant mean, SP-02)
D_TEST_MM = 11.25          # ISO 7215 test pieces 10–12.5 mm
N_SIZE = 1.5               # size exponent of the reduction rate
CAP_RATIO_REF = 0.704      # η_W n_gas / (Y_W n_Fe) at the Jul–Oct 2026 calibration (burden engine, case 2)
FE_PCT = {"sinter": 54.5, "pellet": 63.2, "lloyds_clo": 60.6, "geomin": 49.0, "nmdc_rom": 64.6, "donimalai": 55.3,
          "lloyds_oversize": 61.1, "gomti": 55.3}        # % Fe, wet basis (UAT yard, deck 04 §4); default 58


def relative_reducibility(ri_pct: float | None, ri_ref: float = RI_SINTER_REF) -> float:
    if ri_pct is None:
        return 0.0
    return float(np.log(1.0 - ri_pct / 100.0) / np.log(1.0 - ri_ref / 100.0))


def material_rate(mat, degraded: bool = False) -> float:
    """Fractional wüstite-reduction rate of a ferrous material relative to sinter at the test size.

    The size is the as-charged Sauter mean: fines from low-temperature disintegration (RDI) or decrepitation act on
    the bed (voidage, gas distribution), not as extra reaction surface; the reducibility index already contains the
    pore structure the material develops on heating. ``degraded=True`` gives the alternative for sensitivity runs."""
    if mat.ri_pct is None:
        return 0.0
    size = mat.size
    if degraded and mat.rdi_pct > 0:
        size = replace(size, fines_added_frac=min(0.6, size.fines_added_frac + mat.deg_frac_in_furnace * mat.rdi_pct / 100.0))
    return relative_reducibility(mat.ri_pct) * (D_TEST_MM / size.sauter_mm()) ** N_SIZE


def counter_current_eff(ntu, cr) -> np.ndarray:
    ntu = np.asarray(ntu, float); cr = np.clip(np.asarray(cr, float), 0.0, 1.0)
    out = np.empty_like(ntu)
    one = np.isclose(cr, 1.0, atol=1e-6)
    out[one] = ntu[one] / (1.0 + ntu[one])
    k = ~one
    e = np.exp(-ntu[k] * (1.0 - cr[k]))
    out[k] = (1.0 - e) / (1.0 - cr[k] * e)
    return out


@dataclass
class OmegaRingResult:
    omega: float
    eff: np.ndarray            # ε per ring
    ntu: np.ndarray            # NTU on C_min per ring
    cr: np.ndarray             # C_min / C_max
    gas_share: np.ndarray      # f_i at the reserve zone
    fe_share: np.ndarray       # o_i
    gas_limited: np.ndarray    # True where C_g < C_s
    rate: np.ndarray           # k_i (relative to sinter at the test size)
    t_trz_h: np.ndarray
    kappa: float
    cap_ratio: float
    extras: dict = field(default_factory=dict)

    @property
    def gas_in_excess(self) -> float:
        """Share of the gas in rings where the gas exceeds the ore's wüstite oxygen."""
        return float(self.gas_share[~self.gas_limited].sum())


def blend_rate(column, degraded: bool = False) -> float:
    """Fe-weighted fractional reduction rate of the charge's ferrous (relative to sinter at the test size)."""
    num = den = 0.0
    for k, m in column.charge.ore_kg.items():
        mat = column.cat[k]
        if m <= 0 or mat.ri_pct is None:
            continue
        fe = m * FE_PCT.get(k, 58.0) / 100.0
        num += fe * material_rate(mat, degraded); den += fe
    return num / den if den > 0 else 0.0


def trz_residence_h(res, column, z_rz_top: float) -> np.ndarray:
    """Hours the burden spends between the TRZ top and the cohesive-zone top of each ring."""
    t_top = column.descent_time_h(z_bottom=z_rz_top)
    return np.array([max(column.descent_time_h(z_bottom=min(float(zc), z_rz_top)) - t_top, 0.0) for zc in res.z_cz_top])


def gas_share_trz(res, z: float | None = None) -> np.ndarray:
    z = float(res.z_cz_top.max()) + 0.5 if z is None else z
    f = np.clip(res.split_at(z), 0, None)
    return f / f.sum()


def omega_ring(res, column, kappa: float, z_rz_top: float, cap_ratio: float = CAP_RATIO_REF) -> OmegaRingResult:
    f = gas_share_trz(res)
    o = np.asarray(column.dist.ore_share, float); o = o / o.sum()     # ore composition is the same in every ring
    k = np.full(len(o), blend_rate(column))
    t = trz_residence_h(res, column, z_rz_top)
    cg = cap_ratio * f
    cs = o
    cmin = np.minimum(cg, cs); cmax = np.maximum(cg, cs)
    cr = np.where(cmax > 0, cmin / np.maximum(cmax, 1e-12), 0.0)
    ua = kappa * k * t * cs
    ntu = ua / np.maximum(cmin, 1e-12)
    eff = counter_current_eff(ntu, cr)
    om = float(np.sum(eff * cmin) / np.sum(cg))
    return OmegaRingResult(om, eff, ntu, cr, f, o, cg < cs, k, t, kappa, cap_ratio)


def calibrate_kappa(res, column, z_rz_top: float, omega_target: float = 0.80, cap_ratio: float = CAP_RATIO_REF) -> float:
    lo, hi = 1e-4, 1e3
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if omega_ring(res, column, mid, z_rz_top, cap_ratio).omega < omega_target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))
