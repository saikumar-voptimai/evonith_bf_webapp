"""What the plant layout and the stave history change in the energy model (V-FORGE) and the burnt-coke balance (V-TRACE).

These are small, self-contained calculations meant to be called from the FORGE/TRACE code in the application
repository (not in this package); each returns plain numbers with the assumptions stated.

1. Leak water from failed staves (the 2023–24 log: 21 events, rows 6–8). Water entering the bosh/belly is
   evaporated, superheated and gasified on coke (C + H₂O → CO + H₂):
   * FORGE: needs coke for it, but has no input for it → under-predicts by ≈ 1.05 kg coke per kg of water
     (0.67 from FORGE's own blast-moisture coefficient COKE-06, 0.8 kg/tHM per g/Nm³ at 1,190 Nm³/tHM, plus 0.38 for
     heating liquid water at 35 °C to steam at blast temperature, 4.9 MJ/kg × 0.0776 kg coke per MJ of lower-zone heat,
     COKE-13).
   * TRACE: the oxygen balance has no term for the water's oxygen, so the carbon it gasifies is under-counted by
     1/(1+ηCO) of it → raw burnt coke reads ≈ 0.52 kg coke per kg of water low.
   * Top gas: +1.24 Nm³ H₂ per kg of water, of which (1 − r_H) stays as H₂ → about +0.5 pt of dry H₂ per m³/h of leak.
   A steady leak is absorbed by the 14-day calibrations (FORGE's loss allowance, TRACE's offset); a new one shows as a
   step in (charged − raw burnt) and in top-gas H₂ with no change in RAFT (blast moisture would also lower RAFT).
2. Cooling water: 2,968 m³/h on five circuits (present flows): every 1 K of mean rise is 12.4 GJ/h.
3. The two-zone loss split by stave row, at the lower edge of the thermal reserve zone from the momentum model.
4. Burden and coke transit from the drawn geometry (transit.py).
"""
from __future__ import annotations

from .plant import COOLING_CIRCUITS, GJ_PER_H_PER_K_PER_M3H

NM3_PER_KMOL = 22.414
M_H2O = 18.015
M_C = 12.011


def leak_water(leak_kg_h: float, production_t_h: float = 94.6, blast_nm3_h: float = 111250.0, eta_co: float = 0.47,
               coke_c: float = 0.868, top_gas_dry_nm3_h: float = 153_400.0, r_h: float = 0.35,
               coke_per_kg_moisture: float = 0.672, extra_heat_mj_per_kg: float = 4.9,
               coke_per_mj_lower: float = 0.0776) -> dict:
    """Effects of `leak_kg_h` of cooling water reaching the burden in the lower zone. Defaults: BF-02 reference
    point (Jul–Oct 2026); r_H assumed 0.35 (hydrogen utilisation; replace with FORGE's calibrated value)."""
    w_thm = leak_kg_h / production_t_h                       # kg water per tHM
    forge_extra = w_thm * (coke_per_kg_moisture + extra_heat_mj_per_kg * coke_per_mj_lower)
    trace_under = w_thm * (M_C / M_H2O) / (1.0 + eta_co) / coke_c
    h2_nm3_h = leak_kg_h / M_H2O * NM3_PER_KMOL
    h2_dry = h2_nm3_h * (1.0 - r_h)
    return dict(
        water_kg_per_thm=w_thm,
        blast_moisture_equivalent_g_nm3=leak_kg_h * 1000.0 / blast_nm3_h,
        forge_coke_missing_kg_thm=forge_extra,
        trace_raw_burnt_low_by_kg_thm=trace_under,
        charged_minus_raw_burnt_step_kg_thm=trace_under,   # charged coke follows the true need; raw burnt misses this part
        top_gas_h2_rise_pt=100.0 * h2_dry / (top_gas_dry_nm3_h + h2_dry),
    )


def leak_for_gap(gap_kg_thm: float, **kw) -> dict:
    """Leak rate that would explain a (charged − raw burnt) gap of `gap_kg_thm`, and the H₂ rise it would show."""
    one = leak_water(1000.0, **kw)
    leak = gap_kg_thm / one["trace_raw_burnt_low_by_kg_thm"] * 1000.0
    out = leak_water(leak, **kw)
    out["leak_kg_h"] = leak
    return out


def cooling_water(loss_allowance_gj_h: float = 69.0, measured_gj_h: float = 4.9) -> dict:
    """Total flow, GJ/h per K, the mean rise the FORGE loss allowance implies, and what the stave tags imply."""
    flow = sum(now for _, now in COOLING_CIRCUITS.values())
    per_k = flow * GJ_PER_H_PER_K_PER_M3H
    return dict(flow_m3h=flow, gj_h_per_K=per_k, rise_K_for_allowance=loss_allowance_gj_h / per_k,
                rise_K_for_tags=measured_gj_h / per_k,
                rows_6_10_hot_rise_K_if_tags_are_that_circuit=measured_gj_h / (COOLING_CIRCUITS["rows 6–10 hot surface"][1] * GJ_PER_H_PER_K_PER_M3H))
