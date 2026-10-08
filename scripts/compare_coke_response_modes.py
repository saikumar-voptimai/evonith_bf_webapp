"""How the coke-response mode changes coke, total fuel, cost and blend ranking.

Before the optimiser's default response mode is chosen for the charged-coke
engine, this scores one set of feasible candidate blends three ways, all from
one common live coke anchor F:

    physics      today's BMO: F + physics coke correction (slag heat
                 0.22 kg/kg vs observed DPR slag, flux calcination, saturation)
    fitted       F + 0.0205 x (slag - slag of current blend)   frozen model
    engineering  F + 0.22   x (slag - slag of current blend)   plant assumption

PCI and nut coke are the operator's current rates for every blend, so their
terms are equal across candidates and only the slag response separates them.
Candidates come from the production LP on a saved BMO snapshot, solved at a
range of slag caps, for the operator's selection and for the whole yard.

Usage:
    python scripts/compare_coke_response_modes.py [snapshot.json] [anchor_kg_thm]
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from ui.bmo.editor_inputs import (  # noqa: E402
    dust_inputs_from_editor,
    flux_inputs_from_editor,
    fuel_ash_inputs_from_editor,
    slag_balance_settings_from_editor,
)
from utils.bmo.charged_coke.model import ChargedCokeModel  # noqa: E402
from utils.bmo.charged_coke.scenario import ENGINEERING_BETA  # noqa: E402
from utils.bmo.coke_correction import (  # noqa: E402
    build_drivers,
    build_reference,
    compute_coke_correction,
    load_coke_correction_settings,
)
from utils.bmo.lp_solver import run_lp_baseline  # noqa: E402
from utils.bmo.snapshot import decode  # noqa: E402

DEFAULT_SNAPSHOT = ROOT / "src/storage/bmo_snapshots/20260925_225537_blend_mix_optimiser.json"
SLAG_CAPS = range(290, 390, 10)


def _ores(snapshot: dict, *, whole_yard: bool):
    calls = decode(snapshot["frozen"]["provider_calls"])
    ores = calls["build_ore_inputs(mode='latest', window_days=30)"][0]
    editor = decode(snapshot["inputs"])["applied_ore_editor_df"].set_index("ore_id")
    out = []
    for ore in ores:
        if ore.ore_id not in editor.index:
            continue
        row = editor.loc[ore.ore_id]
        if not whole_yard and not bool(row["selected"]):
            continue
        if whole_yard and float(row["stock_mt"]) <= 0:
            continue
        out.append(replace(ore, price_rs_per_mt=float(row["price_rs_per_mt"]), stock_mt=float(row["stock_mt"]),
                           min_share_pct=float(row["min_share_pct"]) if bool(row["selected"]) else 0.0,
                           max_share_pct=float(row["max_share_pct"])))
    return out


def main(snapshot_path: Path, anchor: float | None) -> pd.DataFrame:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    inp, ctx, res = decode(snapshot["inputs"]), decode(snapshot["context"]), decode(snapshot["results"])
    bmo_cfg = yaml.safe_load((ROOT / "src/config/setting_bmo.yml").read_text(encoding="utf-8"))["bmo"]
    fuel_ash = fuel_ash_inputs_from_editor(inp["applied_fuel_ash_editor_df"])
    fluxes = flux_inputs_from_editor(inp["applied_flux_editor_df"])
    dust = dust_inputs_from_editor(inp["applied_dust_editor_df"])
    slag_settings = slag_balance_settings_from_editor(ctx["slag_settings_values"], ctx["hm_chem_values"], ctx["hm_snapshot"])
    production = float(inp["target_production_mt"])
    rates = ctx["recent_fuel_rates"]
    pci, nut = float(rates["pci_rate_kg_thm"]), float(rates["nut_coke_rate_kg_thm"])
    price = {"coke": inp["fuel_price_coke_rs_per_mt"] / 1000, "pci": inp["fuel_price_pci_rs_per_mt"] / 1000,
             "nut": inp["fuel_price_nut_coke_rs_per_mt"] / 1000}
    anchor = float(anchor if anchor is not None else 318.0)

    manual = res["manual_blend"]
    candidates = {"Manual (current)": (manual, None, fluxes)}
    candidates["Balanced (snapshot LP)"] = (res["lp_result"], None, fluxes)
    for family, whole in (("selection", False), ("whole yard", True)):
        ores = _ores(snapshot, whole_yard=whole)
        for cap in SLAG_CAPS:
            blend, _errors = run_lp_baseline(
                ores, target_production_mt=float(ctx["target_fe_mt"]),
                target_slag_qty_mt=cap * production / 1000 / float(ctx["model_to_plant_slag_factor"]),
                feo_in_slag_pct=float(ctx["feo_in_slag_pct"]),
                target_slag_basicity_min=ctx["target_slag_basicity_min"], target_slag_basicity_max=ctx["target_slag_basicity_max"],
                target_slag_t_basicity_min=ctx.get("target_slag_t_basicity_min") or None,
                target_slag_t_basicity_max=ctx.get("target_slag_t_basicity_max") or None,
                target_slag_al2o3_max_pct=ctx["target_slag_al2o3_max_pct"], target_slag_mgo_min_pct=ctx["target_slag_mgo_min_pct"],
                target_slag_mgo_al2o3_ratio_min=ctx["target_slag_mgo_al2o3_ratio_min"],
                max_burden_qty_mt=ctx["max_burden_qty_mt"], fuel_ash_inputs=fuel_ash, flux_inputs=fluxes,
                dust_inputs=dust, slag_balance_settings=slag_settings, hot_metal_target_mt=production, _explain=False,
            )
            if blend is not None:
                solved = [replace(f, wet_qty_mt=blend.diagnostics.get("lp_flux_quantities_mt", {}).get(f.flux_id, f.wet_qty_mt)) for f in fluxes]
                candidates[f"LP {family}, slag cap {cap}"] = (blend, ores, solved)

    settings = load_coke_correction_settings(bmo_cfg)
    manual_ores = _ores(snapshot, whole_yard=False)
    manual_drivers = build_drivers(blend=manual, ores=manual_ores, quantities_mt=manual.quantities_mt, flux_inputs=fluxes)
    reference = build_reference(settings=settings, observed_slag_rate_kg_per_thm=ctx["observed_slag_rate"],
                                current_drivers=manual_drivers)
    fitted_slag = float(ChargedCokeModel().beta[2])
    slag_ref = float(manual.slag_rate_kg_per_thm)

    rows = []
    seen = set()
    for name, (blend, ores, solved_fluxes) in candidates.items():
        key = tuple(round(v, 1) for v in sorted(blend.quantities_mt.values()))
        if key in seen:
            continue
        seen.add(key)
        ores = ores or manual_ores
        drivers = build_drivers(blend=blend, ores=ores, quantities_mt=blend.quantities_mt, flux_inputs=solved_fluxes)
        physics = compute_coke_correction(anchor_coke_rate_kg_thm=anchor, anchor_nut_coke_rate_kg_thm=nut,
                                          anchor_pci_rate_kg_thm=pci, drivers=drivers, reference=reference, settings=settings)
        slag = float(blend.slag_rate_kg_per_thm)
        coke = {"physics": float(physics.corrected_coke_rate_kg_thm),
                "fitted": anchor + fitted_slag * (slag - slag_ref),
                "engineering": anchor + float(ENGINEERING_BETA[2]) * (slag - slag_ref)}
        ore_flux = float(blend.ore_cost_per_thm_rs) + float(blend.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0)
        row = {"blend": name, "slag_kg_thm": slag, "ore_flux_rs_thm": ore_flux}
        for mode, c in coke.items():
            row[f"coke_{mode}"] = c
            row[f"total_fuel_{mode}"] = c + pci + nut
            row[f"cost_{mode}"] = ore_flux + c * price["coke"] + pci * price["pci"] + nut * price["nut"]
        rows.append(row)
    table = pd.DataFrame(rows)
    for mode in ("physics", "fitted", "engineering"):
        table[f"rank_{mode}"] = table[f"cost_{mode}"].rank(method="min").astype(int)
    return table.sort_values("cost_physics").reset_index(drop=True)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SNAPSHOT
    table = main(path, float(sys.argv[2]) if len(sys.argv) > 2 else None)
    pd.set_option("display.width", 260)
    show = ["blend", "slag_kg_thm", "ore_flux_rs_thm", "coke_physics", "coke_fitted", "coke_engineering",
            "cost_physics", "cost_fitted", "cost_engineering", "rank_physics", "rank_fitted", "rank_engineering"]
    print(table[show].round(1).to_string(index=False))
    out = ROOT / "tmp" / "coke_response_mode_comparison.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    for mode in ("physics", "fitted", "engineering"):
        best = table.loc[table[f"cost_{mode}"].idxmin()]
        print(f"{mode:12s} picks: {best['blend']} (slag {best['slag_kg_thm']:.0f}, ore+flux {best['ore_flux_rs_thm']:,.0f}, "
              f"coke {best[f'coke_{mode}']:.1f}, total {best[f'cost_{mode}']:,.0f} Rs/THM)")
    rho = table[["rank_physics", "rank_fitted", "rank_engineering"]].corr(method="spearman")
    print("Spearman rank agreement:\n", rho.round(2).to_string())
    print("written", out)
