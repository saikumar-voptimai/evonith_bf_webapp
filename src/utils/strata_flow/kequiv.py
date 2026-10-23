"""The plant's packed-bed law from the network, and the register's rules from the model.

K_eq = (Pb² − Pt²) / V^1.8   with Pb, Pt in bar absolute and V the blast volume in Nm³/h: the same
definition as FLOW-01 on the optimiser page, so a day's K from the plant and the model's K_eq are
directly comparable (FLOW-14 is the residual the model has to explain).

`register_sensitivities` perturbs the model the way the register's rules were fitted (pellet
+1 pt replacing sinter, Gomti +1 pt, O2 +1 pt of blast, coke +10 kg/tHM, nut-coke size +1 mm)
and reports the model's % change of K next to the register's value and its standard error.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .network import solve_network, NetworkResult, FlowParams, BAR
from .cohesive import CZParams
from .materials import SizeSpec


def k_equivalent(res: NetworkResult, blast_nm3_h: float) -> float:
    pb = res.p_bustle_pa / BAR
    pt = res.p_top_pa / BAR
    return (pb ** 2 - pt ** 2) / blast_nm3_h ** 1.8


def max_blast_at_pressure(k_eq: float, pb_bar_g: float, pt_bar_g: float) -> float:
    pb, pt = pb_bar_g + 1.01325, pt_bar_g + 1.01325
    return ((pb ** 2 - pt ** 2) / k_eq) ** (1 / 1.8)


REGISTER = {   # register v1.2 FLOW rules: % change of K per unit, ± 1 SE (or 95 % CI where noted)
    "pellet_+1pt": (-0.37, 0.13, "FLOW-02 (95 % CI −0.11 to −0.63)"),
    "gomti_+1pt": (+0.38, 0.17, "FLOW-03"),
    "o2_+1pt_blast": (+2.1, 0.65, "FLOW-05 (≈1.1 % is more bosh gas per Nm³)"),
    "coke_+10kg": (-0.5, 0.12, "FLOW-06"),
    "nut_size_+1mm": (-1.4, 0.55, "FLOW-07"),
}


def _run(grid, column, gas, cz, flow, z_init=None):
    return solve_network(grid, column, gas, cz, flow, z_cz_top_init=z_init)


def register_sensitivities(ref_builder, cz: CZParams = CZParams(), flow: FlowParams = FlowParams()) -> dict:
    """ref_builder(**overrides) -> (grid, column, gas, ref_dict). Returns {rule: (model %, register %, se, note)}."""
    from .reference import REF
    grid, column, gas = ref_builder()
    base = _run(grid, column, gas, cz, flow)
    zi = base.z_cz_top
    # the base K is taken from a run started where the perturbed runs start, so that the
    # fixed-point tolerance cancels in the differences
    base = _run(grid, column, gas, cz, flow, zi)
    k0 = k_equivalent(base, gas.blast_nm3_h)
    out = {}

    def k_of(**kw):
        g, c, s = ref_builder(**kw)
        r = _run(g, c, s, cz, flow, zi)
        return k_equivalent(r, s.blast_nm3_h)

    # pellet +1 pt replacing sinter
    b = dict(REF["blend_wet"]); b["pellet"] += 0.01; b["sinter"] -= 0.01
    out["pellet_+1pt"] = 100 * (k_of(blend_wet=b) / k0 - 1)
    # gomti +1 pt replacing sinter
    b = dict(REF["blend_wet"]); b["gomti"] = b.get("gomti", 0) + 0.01; b["sinter"] -= 0.01
    out["gomti_+1pt"] = 100 * (k_of(blend_wet=b) / k0 - 1)
    # O2 +1 pt of blast at the same blast volume
    out["o2_+1pt_blast"] = 100 * (k_of(enrichment_true=REF["enrichment_true"] + 0.01) / k0 - 1)
    # coke +10 kg/tHM (more coke per charge, same ore)
    out["coke_+10kg"] = 100 * (k_of(coke_kg_thm=REF["coke_kg_thm"] + 10) / k0 - 1)
    # nut-coke size +1 mm
    out["nut_size_+1mm"] = 100 * (k_of(nut_coke_d_mm_delta=1.0) / k0 - 1)
    return {k: (v, *REGISTER[k]) for k, v in out.items()}, k0, base


def calibrate_cz(ref_builder, target_dp_bar: float, cz: CZParams = CZParams(), flow: FlowParams = FlowParams(),
                 lo: float = 0.0, hi: float = 0.30) -> tuple[float, NetworkResult]:
    """Bisection on the cohesive-zone coke void loss so the model's bustle-to-top ΔP matches the plant.

    This is the single scalar calibration of the model: everything else is geometry, charge
    records, size distributions and literature constants. The identifiable content of the model is
    in its responses, not in this level.
    """
    grid, column, gas = ref_builder()
    def dp(v):
        r = solve_network(grid, column, gas, replace(cz, coke_void_loss=v), flow)
        return r.dp_total_bar, r
    d_lo, r_lo = dp(lo); d_hi, r_hi = dp(hi)
    if not (d_lo <= target_dp_bar <= d_hi):
        raise ValueError(f"target {target_dp_bar:.3f} bar outside [{d_lo:.3f}, {d_hi:.3f}] reachable by the CZ void loss; "
                         "adjust the layer angle, lower-zone or tuyere parameters")
    mid, r = 0.5 * (lo + hi), r_lo
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        d_mid, r = dp(mid)
        if d_mid < target_dp_bar:
            lo = mid
        else:
            hi = mid
        if abs(d_mid - target_dp_bar) < 0.002:
            break
    return mid, r
