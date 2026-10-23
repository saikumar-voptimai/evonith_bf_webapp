"""Evaluation of the ring ω model for an operating state, and the quantities a surrogate needs (v0.4).

``evaluate(state)`` runs the burden-profile, gas-network and cohesive-zone models for one state and returns the ring
quantities that enter the counter-current ω model:

    f_i   gas share of ring i at the top of the thermal reserve zone
    o_i   share of the Fe in ring i (burden profile)
    t_i   hours the burden spends in the thermal reserve zone of ring i
    x_m   share of the charge's Fe brought by ferrous material m (for the blend reduction rate k̄)

ω itself is recomputed from these with ``omega_from_parts`` for any capacity ratio R and constant κ, so the
calibration and the uncertainty runs do not repeat the network solution.

State keys: any ``reference.REF`` key (blend_wet, coke_kg_thm, nut_coke_kg_thm, pci_kg_thm, production_t_h,
blast_nm3_h, enrichment_true, hbt_c, blast_moisture, charges_per_hour, noncoke_t_per_charge, limestone_kg_thm,
top_eta_co, top_h2, top_gas_T_c), plus ``matrix`` ({coke: [(angle, rotations)], ore: [...]}), ``sinter_rdi_pct`` and the
size shifts ``coke_d_mm_delta``, ``nut_coke_d_mm_delta`` (mm) and ``material_size`` ({material: SizeSpec}).
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from .reference import make_builder, REF, reference_distribution
from .network import solve_network, FlowParams
from .run_case import calibrated_cz
from .run_burden_history import floating_cz, wall_flow
from .kequiv import k_equivalent
from .plant import ROW_BY_NO
from . import omega_ring as OR

FERROUS = ("sinter", "pellet", "lloyds_clo", "geomin", "nmdc_rom", "donimalai", "lloyds_oversize", "gomti")


@lru_cache(maxsize=1)
def context():
    from .run_case import CZ_VOID_LOSS_CALIBRATED, OMEGA_RING_KAPPA
    c = dict(cz_void_loss=CZ_VOID_LOSS_CALIBRATED, kappa_ring=OMEGA_RING_KAPPA)
    b = make_builder(); flow = FlowParams()
    g0 = b()[0]
    cz = floating_cz(calibrated_cz(coke_void_loss=c["cz_void_loss"]), b, reference_distribution(g0), flow)
    zr = g0.profile.stockline_z - flow.z_rz_top_below_stockline
    return dict(b=b, flow=flow, cz=cz, zr=zr, calib=c)


def fe_fractions(column) -> dict:
    fe = {k: m * OR.FE_PCT.get(k, 58.0) / 100.0 for k, m in column.charge.ore_kg.items()
          if m > 0 and column.cat[k].ri_pct is not None}
    s = sum(fe.values())
    return {k: v / s for k, v in fe.items()}


def evaluate(state: dict | None = None) -> dict:
    ctx = context()
    st = dict(state or {})
    ov = {k: v for k, v in st.items() if k in REF}
    extra = {k: st[k] for k in ("coke_d_mm_delta", "nut_coke_d_mm_delta", "material_size") if k in st}
    g, col, gas = ctx["b"](dist="auto", matrix=st.get("matrix"), sinter_rdi_pct=st.get("sinter_rdi_pct"), **ov, **extra)
    r = solve_network(g, col, gas, ctx["cz"], ctx["flow"])
    f = OR.gas_share_trz(r)
    o = np.asarray(col.dist.ore_share, float); o = o / o.sum()
    t = OR.trz_residence_h(r, col, ctx["zr"])
    return dict(f=f.tolist(), o=o.tolist(), t=t.tolist(), x=fe_fractions(col), k=OR.blend_rate(col),
                wall7=wall_flow(r, ROW_BY_NO[7].z_centre), K=k_equivalent(r, gas.blast_nm3_h), dp=r.dp_total_bar,
                z_cz_top=np.asarray(r.z_cz_top).tolist(), converged=bool(r.converged),
                oc=[round(x.oc_ratio, 3) for x in col.rings])


def material_rates(ri: dict | None = None, n_size: float = OR.N_SIZE, sauter: dict | None = None) -> dict:
    """Relative rate r_m (d_test/d_m)^n of each ferrous material; ri / sauter override the catalogue values."""
    from .materials import default_catalogue
    cat = default_catalogue()
    out = {}
    for k in FERROUS:
        m = cat[k]
        ri_k = (ri or {}).get(k, m.ri_pct)
        d = (sauter or {}).get(k, m.size.sauter_mm())
        out[k] = OR.relative_reducibility(ri_k) * (OR.D_TEST_MM / d) ** n_size
    return out


def blend_rate_from_x(x: dict, rates: dict) -> float:
    return float(sum(v * rates[k] for k, v in x.items()))


def omega_from_parts(f, o, t, k, kappa, R=OR.CAP_RATIO_REF) -> float:
    f = np.asarray(f, float); o = np.asarray(o, float); t = np.asarray(t, float)
    cg = R * f; cs = o
    cmin = np.minimum(cg, cs); cmax = np.maximum(cg, cs)
    cr = cmin / np.maximum(cmax, 1e-12)
    eff = OR.counter_current_eff(kappa * k * t * cs / np.maximum(cmin, 1e-12), cr)
    return float(np.sum(eff * cmin) / np.sum(cg))


def calibrate_kappa_parts(f, o, t, k, omega_target=0.80, R=OR.CAP_RATIO_REF) -> float:
    lo, hi = 1e-3, 1e3
    for _ in range(70):
        mid = np.sqrt(lo * hi)
        if omega_from_parts(f, o, t, k, mid, R) < omega_target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))
