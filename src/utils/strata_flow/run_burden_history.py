"""Every charging matrix in ops_config.burden_history through the burden-profile model and the layered gas-flow model
at the reference operating point (distribution effect alone), with the measured furnace response per matrix period.

python -m bf2_momentum.run_burden_history <burden_history.parquet> [<10-min parquet>] [out.json]
"""
from __future__ import annotations

import json, sys
import numpy as np
import pandas as pd

from .reference import make_builder, REF, build_charge
from .materials import default_catalogue
from .burden_profile import load_burden_history, radial_distribution, matrix_label
from .network import solve_network, FlowParams
from .cohesive import CZParams
from .kequiv import k_equivalent
from .omega import omega_from_network
from .run_case import CZ_VOID_LOSS_CALIBRATED, OMEGA_KC_CALIBRATED, calibrated_cz
from .plant import ROW_BY_NO


def charge_volumes(ref: dict = REF):
    cat = default_catalogue()
    ch = build_charge(ref, cat)
    coke_m3 = ch.coke_kg / cat["coke"].bulk_density
    ore_m3 = sum(m / cat[k].bulk_density for k, m in ch.ore_kg.items()) + ch.nut_coke_kg / cat["nut_coke"].bulk_density
    ore_t = sum(ch.ore_kg.values()) + ch.nut_coke_kg
    return coke_m3, ore_m3, ch.coke_kg, ore_t


def wall_flow(res, z):
    k = int(np.clip(np.searchsorted(res.grid.z_nodes, z) - 1, 0, res.grid.n_z - 1))
    return float(np.clip(res.m_axial_faces[-1, k + 1], 0, None)) / float(np.clip(res.m_axial_faces[:, k + 1], 0, None).sum())


REF_MATRIX = ([(12, 2), (20, 1), (24, 2), (27, 2), (30, 2), (32, 2), (35, 3)], [(24, 2), (27, 2), (30, 2), (32, 2)])   # 15 May 2026 →


def floating_cz(cz: CZParams, builder, dist_ref, flow: FlowParams = FlowParams()) -> CZParams:
    """Anchor the zone once on the reference matrix (wall root at cz.z_root_wall), then return parameters with which
    the whole zone, wall root included, follows each ring's absolute gas/ore ratio (as in sectors.solve_sectors)."""
    from dataclasses import replace
    g, col, gas = builder(dist=dist_ref)
    r0 = solve_network(g, col, gas, cz, flow)
    cz_s = replace(cz, z_root_wall=None, ratio_ref=r0.extras["gas_ore_ratio_mean"], z_top_ref=float(np.mean(r0.z_cz_top)))
    for _ in range(6):
        r1 = solve_network(g, col, gas, cz_s, flow, z_cz_top_init=r0.z_cz_top)
        cz_s = replace(cz_s, z_top_ref=cz_s.z_top_ref * (r0.z_cz_top[-1] / r1.z_cz_top[-1]))
        if abs(r1.z_cz_top[-1] - r0.z_cz_top[-1]) < 0.01:
            break
    return cz_s


def model_matrix(coke, ore, cz_void_loss=None, kc=None, builder=None, ref=REF, cz: CZParams | None = None, stream_sd_m=0.30):
    """`cz`: zone parameters to use (e.g. floating_cz(...)); default: anchored at the row-7 wall root."""
    cz_void_loss = CZ_VOID_LOSS_CALIBRATED if cz_void_loss is None else cz_void_loss
    kc = OMEGA_KC_CALIBRATED if kc is None else kc
    builder = builder or make_builder()
    grid = builder()[0]
    coke_m3, ore_m3, coke_kg, ore_kg = charge_volumes(ref)
    dist, prof = radial_distribution(grid, coke, ore, coke_m3, ore_m3, stream_sd_m=stream_sd_m)
    g, col, gas = builder(dist=dist)
    flow = FlowParams()
    res = solve_network(g, col, gas, cz or calibrated_cz(coke_void_loss=cz_void_loss), flow)
    om = omega_from_network(res, col, kc, g.profile.stockline_z - flow.z_rz_top_below_stockline)
    oc_ring = (dist.ore_share * ore_kg) / (dist.coke_share * coke_kg)
    return dict(oc_ring=oc_ring.round(2).tolist(), ore_share=dist.ore_share.round(4).tolist(), coke_share=dist.coke_share.round(4).tolist(),
                dp_bar=res.dp_total_bar, k_eq=k_equivalent(res, gas.blast_nm3_h), omega=om.omega,
                top_share=res.top_split().round(4).tolist(), wall_row7=wall_flow(res, ROW_BY_NO[7].z_centre),
                wall_row8=wall_flow(res, ROW_BY_NO[8].z_centre), wall_row10=wall_flow(res, ROW_BY_NO[10].z_centre),
                cz_root_z=float(res.z_cz_bot[-1]), cz_apex_z=float(res.z_cz_top[0]), converged=res.converged,
                landing=prof.landing, r=prof.r[::5].round(3).tolist(), t_coke=prof.t_coke[::5].round(4).tolist(),
                t_ore=prof.t_ore[::5].round(4).tolist())


def measured_by_period(hist: pd.DataFrame, m10: pd.DataFrame) -> pd.DataFrame:
    P = lambda c: m10["process_params__" + c]
    ok = P("hot_blast_vol_nm3h").between(95000, 125000) & P("hot_blast_press").between(2.0, 3.2) & P("top_press_avg").between(0.9, 1.7)
    x = m10[ok]
    pb = x["process_params__hot_blast_press"] + 1.01325; pt = x["process_params__top_press_avg"] + 1.01325
    v = x["process_params__hot_blast_vol_nm3h"]
    d = pd.DataFrame({"k_meas": (pb ** 2 - pt ** 2) / v ** 1.8 * 1e9, "blast": v, "pb": pb - 1.01325, "pt": pt - 1.01325,
                      "dp_bottom": x["process_params__body_dp_bottom"], "dp_top": x["process_params__body_dp_top"],
                      "eta_co": x["process_params__co2_pct"] / (x["process_params__co_pct"] + x["process_params__co2_pct"]),
                      "top_T": x["process_params__top_temp_avg"], "coke_rate": x["process_params__coke_rate"],
                      "o2": x["process_params__oxygen_flow"]}, index=x.index)
    for r in (6, 7, 8, 9, 10):
        c = f"heatload_delta_t__heat_load_row_{r}"
        if c in x:
            d[f"hl_r{r}"] = x[c].where(x[c].between(0, 30))
    rows = []
    for h in hist.itertuples():
        end = h.valid_to if pd.notna(h.valid_to) else d.index.max()
        s = d[(d.index >= h.valid_from + pd.Timedelta("12h")) & (d.index < end)]      # skip the first 12 h (transition)
        rec = dict(valid_from=h.valid_from, n_10min=len(s))
        if len(s) >= 36:
            rec.update(s.median().round(4).to_dict())
        rows.append(rec)
    return pd.DataFrame(rows)


def main(hist_path: str, m10_path: str | None = None, out: str | None = None, cz_void_loss=None, kc=None,
         float_zone: bool = True, stream_sd_m: float = 0.30):
    hist = load_burden_history(hist_path)
    builder = make_builder()
    cz = None
    if float_zone:
        grid = builder()[0]
        cm3, om3, _, _ = charge_volumes()
        d_ref, _ = radial_distribution(grid, *REF_MATRIX, cm3, om3, stream_sd_m=stream_sd_m)
        cz = floating_cz(calibrated_cz(**({} if cz_void_loss is None else {"coke_void_loss": cz_void_loss})), builder, d_ref)
    cache = {}
    recs = []
    for h in hist.itertuples():
        key = (tuple(h.coke), tuple(h.ore))
        if key not in cache:
            cache[key] = model_matrix(h.coke, h.ore, cz_void_loss=cz_void_loss, kc=kc, builder=builder, cz=cz,
                                      stream_sd_m=stream_sd_m)
        m = cache[key]
        recs.append(dict(valid_from=h.valid_from, valid_to=h.valid_to, coke=matrix_label(h.coke), ore=matrix_label(h.ore),
                         coke_mean_angle=h.coke_mean_angle, ore_mean_angle=h.ore_mean_angle, purpose=h.purpose,
                         **{k: m[k] for k in ("oc_ring", "dp_bar", "k_eq", "omega", "wall_row7", "wall_row8", "wall_row10",
                                              "cz_root_z", "cz_apex_z", "top_share", "converged")}))
    df = pd.DataFrame(recs)
    if m10_path:
        meas = measured_by_period(hist, pd.read_parquet(m10_path))
        df = df.merge(meas, on="valid_from", how="left")
    if out:
        df.to_json(out, orient="records", date_format="iso", indent=1, default_handler=str)
        with open(out.replace(".json", "_profiles.json"), "w") as f:
            json.dump({f"{matrix_label(k[0])} | {matrix_label(k[1])}": v for k, v in cache.items()}, f, default=str)
    return df, cache


if __name__ == "__main__":
    a = sys.argv[1:]
    df, _ = main(a[0], a[1] if len(a) > 1 else None, a[2] if len(a) > 2 else None)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["top_share"]).to_string())
