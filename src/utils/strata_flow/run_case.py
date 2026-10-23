"""One shift (or one plan) through the momentum model.

run_case(inputs) takes a flat dict and returns a flat dict, so it can be called from the
coke-energy pipeline's shift table or from the optimiser's plan. Inputs not given fall back to
the reference values (reference.REF), the drawn BF-02 profile and the default catalogue.

inputs (all optional):
  blast_nm3_h, enrichment_true (fraction), hbt_c, pb_bar_g, pt_bar_g, production_t_h,
  coke_kg_thm (dry), nut_coke_kg_thm, pci_kg_thm, charges_per_hour, noncoke_t_per_charge,
  limestone_kg_thm, blend_wet {material: wet fraction of ferrous}, blast_moisture (g/Nm3),
  top_eta_co, top_h2, top_gas_T_c,
  material_size {material: SizeSpec}           lab sieve fractions / mean sizes per material
  sinter_rdi_pct, nut_coke_d_mm_delta, coke_d_mm_delta
  charging_matrix {"coke": [(angle, rotations), …], "ore": [...]}   bell-less top programme → burden profile → ring
                                                shares; the zone then floats (anchored once on the reference matrix)
                                                unless heat_load_tags / cz_root_wall_m are given
  stream_sd_m                                   stream width at the stockline for the burden-profile model (0.30)
  centre_coke_bias, wall_ore_bias               legacy parametric programme (or `dist`: RadialDistribution)
  heat_load_tags {"heat_load_r7": GJ/h, "heat_load_r7_q2": …}   stave heat loads → cohesive-zone root anchor
  cz_root_wall_m                                explicit anchor (m above the tuyeres), overrides the tags
  stockline_m                                   operating stockline below the zero line (default 1.0)
  tuyere_area_factor                            open tuyere area / design (sleeves, blanking)
  leak_water_kg_h                               cooling water into the furnace (FORGE / TRACE effects)
  cz_void_loss, cz_beta, layer_angle_deg, omega_kc (calibrated)
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .reference import make_builder, REF
from .geometry import bf02_profile, elev_from_z
from .charging import RadialDistribution
from .cohesive import CZParams
from .network import solve_network, FlowParams
from .kequiv import k_equivalent, max_blast_at_pressure
from .omega import omega_from_network
from .heatload import parse_heat_load_tags, cz_root_from_rows, quadrant_index
from .transit import transit
from .plant import row_at_elev
from .plant_effects import leak_water

# Calibrated 21 Oct 2026 (v0.3) on the drawn BF-02 profile (stockline 1.0 m, typical bulk densities, zone root anchored
# in stave row 7) with the charging matrix in force since 15 May 2026 (reference.REF_MATRIX, stream SD 0.30 m):
# bustle-to-top ΔP 1.37 bar at 111,250 Nm³/h, 2.79 / 1.42 bar g; ω = 0.80 at the same point. The zone-shape exponent
# and slit floor are set from 11 matrix changes Jan–May 2026 (measured ΔlnK SD 2.2 %; model 2.0 %, r = 0.35).
# v0.2 values (placeholder centre-coke distribution, beta 0.35, 20°): void loss 0.1043, k_c 0.0075.
CZ_VOID_LOSS_CALIBRATED = 0.0507   # v0.4: measured plant sieves (materials.MEASURED_SIEVE); v0.3 was 0.0903
OMEGA_KC_CALIBRATED = 0.00594     # m/s, gas-side contact form (omega.py); v0.3 was 0.00706
OMEGA_RING_KAPPA = 1.935          # 1/h, ring counter-current form (omega_ring.py), omega 0.80 at the reference
CZ_BETA_CALIBRATED = 0.10
CZ_LAYER_ANGLE_CALIBRATED = 30.0


def calibrated_cz(**kw) -> CZParams:
    """CZParams with the calibrated void loss, shape exponent and slit floor (override any field by keyword)."""
    base = dict(coke_void_loss=CZ_VOID_LOSS_CALIBRATED, beta=CZ_BETA_CALIBRATED, layer_angle_deg=CZ_LAYER_ANGLE_CALIBRATED)
    base.update(kw)
    return CZParams(**base)


def run_case(inputs: dict | None = None, profile=None, n_r: int = 6, n_z: int = 60) -> dict:
    inp = dict(inputs or {})
    notes = []
    cz = CZParams(coke_void_loss=inp.pop("cz_void_loss", CZ_VOID_LOSS_CALIBRATED),
                  layer_angle_deg=inp.pop("layer_angle_deg", CZ_LAYER_ANGLE_CALIBRATED),
                  beta=inp.pop("cz_beta", CZ_BETA_CALIBRATED))
    flow = FlowParams(tuyere_area_factor=inp.pop("tuyere_area_factor", 1.0))
    if profile is None:
        profile = bf02_profile(inp.pop("stockline_m", 1.0))
    tags = inp.pop("heat_load_tags", None)
    quads = None
    if "cz_root_wall_m" in inp:
        cz = replace(cz, z_root_wall=float(inp.pop("cz_root_wall_m")))
        notes.append(f"cohesive-zone root set to {cz.z_root_wall:.2f} m above the tuyeres")
    elif tags:
        rows, quads = parse_heat_load_tags(tags)
        z_root = cz_root_from_rows(rows)
        if z_root is not None:
            cz = replace(cz, z_root_wall=z_root)
            notes.append(f"cohesive-zone root anchored at {z_root:.2f} m (+{elev_from_z(z_root):,.0f}, stave row "
                         f"{row_at_elev(float(elev_from_z(z_root)))}) from the stave heat-load rows")
    else:
        notes.append("cohesive-zone root at the default anchor (stave row 7, from the 2023–24 failure log)")
    leak = inp.pop("leak_water_kg_h", None)
    kc = inp.pop("omega_kc", OMEGA_KC_CALIBRATED)
    bias = inp.pop("centre_coke_bias", None); wbias = inp.pop("wall_ore_bias", None)
    matrix = inp.pop("charging_matrix", None); sd = inp.pop("stream_sd_m", None)
    builder = make_builder(profile=profile, n_r=n_r, n_z=n_z)
    if matrix is not None and "dist" not in inp:
        from .reference import reference_distribution, STREAM_SD_M
        g0 = builder()[0]
        inp["dist"] = reference_distribution(g0, matrix=matrix, stream_sd_m=STREAM_SD_M if sd is None else sd)
        if not tags and "cz_root_wall_m" not in (inputs or {}):
            from .run_burden_history import floating_cz
            cz = floating_cz(cz, builder, reference_distribution(g0, stream_sd_m=STREAM_SD_M if sd is None else sd))
            notes = [n for n in notes if not n.startswith("cohesive-zone root at the default anchor")]
            notes.append("cohesive zone floats with each ring's gas/ore ratio (anchored once on the reference matrix)")
    if bias is not None and "dist" not in inp:
        g0 = builder()[0]
        inp["dist"] = RadialDistribution.centre_coke(g0, centre_coke_bias=bias, wall_ore_bias=wbias if wbias is not None else 0.625 * bias)
    grid, column, gas = builder(**inp)
    res = solve_network(grid, column, gas, cz, flow)
    z_rz = grid.profile.stockline_z - flow.z_rz_top_below_stockline
    om = omega_from_network(res, column, kc, z_rz)
    k = k_equivalent(res, gas.blast_nm3_h)
    pb_lim = inp.get("pb_bar_g", REF["pb_bar_g"]); pt = inp.get("pt_bar_g", REF["pt_bar_g"])
    tr = transit(res, column)
    out = {
        "converged": res.converged, "iterations": res.iterations, "geometry": grid.profile.name,
        "stockline_assumed": grid.profile.stockline_assumed,
        "dp_total_bar": res.dp_total_bar, "dp_bed_bar": res.dp_bed_bar,
        "pb_model_bar_g": res.p_bustle_pa / 1e5 - 1.01325,
        "zone_dp_bar": res.zone_dp_bar(),
        "k_eq": k, "max_blast_nm3_h_at_pb_limit": max_blast_at_pressure(k, pb_lim, pt),
        "z_cz_bottom_m": res.z_cz_bot.tolist(), "z_cz_top_m": res.z_cz_top.tolist(),
        "cz_root_wall_elev_mm": float(elev_from_z(res.z_cz_bot[-1])),
        "cz_root_wall_stave_row": row_at_elev(float(elev_from_z(res.z_cz_bot[-1]))),
        "top_gas_share_by_ring": res.top_split().tolist(),
        "omega_model": om.omega, "omega_by_ring": om.omega_ring.tolist(), "gas_share_reserve_zone": om.gas_share.tolist(),
        "hours_stockline_to_raceway": tr["hours_stockline_to_raceway"], "hours_to_cz_top": tr["hours_to_cz_top"],
        "coke_in_transit_t": tr["coke_in_transit_t"], "transit": tr,
        "column": column.summary(),
        "notes": notes,
    }
    if quads:
        out["heat_load_quadrant_share_rows_6_8"] = quadrant_index(quads)
    if leak:
        out["leak_water"] = leak_water(leak, production_t_h=inp.get("production_t_h", REF["production_t_h"]),
                                       blast_nm3_h=gas.blast_nm3_h)
    return out


if __name__ == "__main__":
    import json, sys
    inp = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(run_case(inp), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
