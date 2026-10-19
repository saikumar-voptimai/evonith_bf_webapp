"""Blend Mix Optimiser LP (page 9) against the burden engine with fuel left out (#258).

Both solve "cheapest ore + flux blend at a fixed production" on the same yard, prices, share bounds and slag limits.
Differences come from the slag models: the engine's slag is the two-zone balance's own (fuel ash at the solved coke
rate, dust, Si / Mn / Ti to hot metal, FeO to slag), the LP's is ``utils.bmo.slag_balance`` with fixed fuel rates.

    python scripts/compare_bmo_vs_burden_engine.py                 # reference yard of 16 Oct 2026 (offline)
    python scripts/compare_bmo_vs_burden_engine.py --live          # live yard (Neon) via the BMO context provider
    python scripts/compare_bmo_vs_burden_engine.py --target 2300 --out docs/bfo_vs_bmo.md

Writes a Markdown table (stdout, or --out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "furnace_data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import yaml  # noqa: E402

from data.burden_engine.inputs import (  # noqa: E402
    build_engine_inputs, catalogue_fluxes, flux_rows, fuel_prices, yard_rows)
from data.bmo.dust_inputs import dust_cfg_with_live  # noqa: E402
from utils.bmo.lp_solver import run_lp_baseline  # noqa: E402
from utils.bmo.types import DustInput, FluxInput, FuelAshInput, OreChemistry, OreInput, SlagBalanceSettings  # noqa: E402
from utils.burden_engine import Engine, Limits, Settings  # noqa: E402
from utils.burden_engine.engine import DEFAULT_CAL_PATH, REFERENCE_INPUTS_PATH  # noqa: E402

FE2O3_PER_FE = 159.69 / (2 * 55.845)


def reference_records(prefs: dict) -> tuple[list[OreInput], list[FluxInput]]:
    ref = json.load(open(REFERENCE_INPUTS_PATH, encoding="utf-8"))
    ores = []
    for m in ref["materials"]:
        c = m["chem"]
        chem = OreChemistry(fe_t_pct=c["Fe"], moisture_pct=m["moist"], feo_pct=c.get("FeO", 0.0), sio2_pct=c["SiO2"],
                            al2o3_pct=c["Al2O3"], cao_pct=c["CaO"], mgo_pct=c["MgO"], mno_pct=c["MnO"],
                            tio2_pct=c["TiO2"], p_pct=c["P"], loi_pct=c.get("LOI", 0.0))
        ores.append(OreInput(m["id"], m["name"], 1e6, m["price"], m["lo"], m["hi"], chem,
                             metadata=dict(material_key=m["code"], chemistry_source="reference")))
    fluxes = []
    for f in ref["fluxes"]:
        c = f["chem"]
        fluxes.append(FluxInput(f["id"], f["name"], moisture_pct=f["moist"], sio2_pct=c["SiO2"], al2o3_pct=c["Al2O3"],
                                cao_pct=c["CaO"], mgo_pct=c["MgO"], fe2o3_pct=c["Fe"] * FE2O3_PER_FE, loi_pct=c["LOI"],
                                price_rs_per_mt=f["price"], stock_mt=1e6, max_qty_mt=f["max_t_day"], optimizable=True))
    return ores, fluxes


def live_fluxes(prefs: dict) -> list[FluxInput]:
    """Page 9's flux catalogue (setting_bmo.yml) with the operator's saved price / stock / limit, as page 9 offers it."""
    bmo = yaml.safe_load(open(ROOT / "src/config/setting_bmo.yml", encoding="utf-8"))["bmo"]
    saved = (prefs.get("flux_editor", {}) or {}).get("rows", {}) or {}
    out = []
    for r in catalogue_fluxes(bmo.get("flux_inputs", [])):
        sv = saved.get(r.flux_id, {}) or {}
        mx = sv.get("max_qty_mt", r.max_qty_mt)
        out.append(FluxInput(r.flux_id, r.display_name, enabled=r.enabled, moisture_pct=r.moisture_pct,
                             sio2_pct=r.sio2_pct, al2o3_pct=r.al2o3_pct, cao_pct=r.cao_pct, mgo_pct=r.mgo_pct,
                             fe2o3_pct=r.fe2o3_pct, mno_pct=r.mno_pct, tio2_pct=r.tio2_pct, p_pct=r.p_pct,
                             loi_pct=r.loi_pct, price_rs_per_mt=float(sv.get("price_rs_per_mt", r.price_rs_per_mt)),
                             stock_mt=float(sv.get("stock_mt", r.stock_mt)), max_qty_mt=None if mx is None else float(mx),
                             optimizable=r.enabled))
    return out


def fuel_ash(cal: dict, K_wet: float, s: Settings) -> list[FuelAshInput]:
    """Fuel rows as page 9 builds them: ash oxides from the calibration's analyses (the engine's), sulphur,
    phosphorus and alkali from page 9's configured fuel rows (setting_bmo.yml), which the engine has no field for."""
    bmo = yaml.safe_load(open(ROOT / "src/config/setting_bmo.yml", encoding="utf-8"))["bmo"]
    cfg = {r["fuel_id"]: r for r in bmo.get("fuel_ash_inputs", [])}
    out = []
    for fid, name, rate, key in (("coke", "Coke", K_wet, "coke"), ("nut_coke", "Nut coke", s.nut_plant, "nut"),
                                 ("pci", "PCI", s.pci_plant, "pci")):
        f = cal["case"][key]; a = f["ash_comp"]; c = cfg.get(fid, {})
        out.append(FuelAshInput(fid, name, rate_kg_per_thm=rate, rate_basis="wet", moisture_pct=f["moist_pct"],
                                vm_pct=f.get("VM", 0.0), ash_pct=f["ash"], sio2_pct=a.get("SiO2", 0.0),
                                al2o3_pct=a.get("Al2O3", 0.0), cao_pct=a.get("CaO", 0.0), mgo_pct=a.get("MgO", 0.0),
                                fe2o3_pct=a.get("Fe2O3", 0.0), tio2_pct=a.get("TiO2", 0.0),
                                alkali_pct=float(c.get("alkali_pct", 0.0)), s_pct=float(c.get("s_pct", 0.0)),
                                p_pct=float(c.get("p_pct", 0.0))))
    return out


def page9_dust(prefs: dict, rates: dict | None = None) -> list[DustInput]:
    """Page 9's top-gas dust rows: setting_bmo.yml dust_inputs, the DPR quantity laid over them when given (--live),
    then any saved operator values, as the page builds them."""
    bmo = yaml.safe_load(open(ROOT / "src/config/setting_bmo.yml", encoding="utf-8"))["bmo"]
    saved = (prefs.get("dust_editor", {}) or {}).get("rows", {}) or {}
    out = []
    for row in dust_cfg_with_live(bmo.get("dust_inputs", []), rates, None):
        row = {**row, **(saved.get(row["dust_id"]) or {})}
        out.append(DustInput(**{k: v for k, v in row.items() if k in DustInput.__dataclass_fields__}))
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=2270.0, help="hot metal, t/d")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    prefs = yaml.safe_load(open(ROOT / "src/config/bmo_operator_inputs.yml", encoding="utf-8"))
    cal = json.load(open(DEFAULT_CAL_PATH, encoding="utf-8"))
    if a.live:
        from data.bmo.context_provider import EvonithBmoContextProvider
        prov = EvonithBmoContextProvider()
        ores, _ = prov.build_ore_inputs(mode="avg", window_days=30)
        dust_rows = page9_dust(prefs, prov.get_dpr_dust_rates(window_days=30)[0])
        fluxes = live_fluxes(prefs)
    else:
        ores, fluxes = reference_records(prefs)
        dust_rows = page9_dust(prefs)
    mats = yard_rows(ores, prefs, k_rules=dict(gomti_clo="gomti", lloyds_oversize_clo="oversize"))
    fl = flux_rows(fluxes, prefs)
    inputs = build_engine_inputs(mats, fl, fuel_prices(prefs), use_stock=a.live)
    eng = Engine(inputs, cal)
    s = Settings()
    L = Limits()
    r = eng.min_cost(prod_t_day=a.target, s=s, L=L, fuel=False)

    sel = {m["id"] for m in inputs["materials"] if m["selected"]}
    ores_sel = []
    for o in ores:
        if o.ore_id not in sel:
            continue
        m = next(x for x in mats if x["id"] == o.ore_id)
        o.price_rs_per_mt, o.min_share_pct, o.max_share_pct = m["price"], m["lo"], m["hi"]
        ores_sel.append(o)
    hm = cal["case"]["hm"]
    lp, errors = run_lp_baseline(
        ores_sel, target_production_mt=a.target * hm["Fe"] / 100.0, target_slag_qty_mt=L.slag_max * a.target / 1000.0,
        feo_in_slag_pct=cal["case"].get("slag_FeO_pct", 0.4), target_slag_basicity_min=L.B2_min,
        target_slag_basicity_max=L.B2_max, target_slag_al2o3_max_pct=L.Al2O3_max, target_slag_mgo_min_pct=L.MgO_min,
        target_slag_mgo_al2o3_ratio_min=L.MgO_Al2O3_min,
        max_burden_qty_mt=L.charge_t * L.charges_per_h * 24 - s.nut_plant * a.target / 1000.0,
        fuel_ash_inputs=fuel_ash(cal, r["out"]["K_wet"], s), flux_inputs=fluxes, dust_inputs=dust_rows,
        slag_balance_settings=SlagBalanceSettings(enabled=True, carbon_pct=hm["C"], silicon_pct=s.hm_Si,
                                                  sulphur_pct=hm["S"], mn_pct=hm["Mn"], ti_pct=hm["Ti"]),
        hot_metal_target_mt=a.target, charge_mass_mt=L.charge_t, _explain=True)
    lines = [f"# Blend Mix Optimiser LP vs burden engine (ore + flux objective), {a.target:,.0f} t/d",
             "", f"Yard: {'live (Neon)' if a.live else 'reference, 16 Oct 2026'}; prices and bounds from "
             "bmo_operator_inputs.yml; slag <= %.0f kg/tHM, CaO/SiO2 %.2f-%.2f." % (L.slag_max, L.B2_min, L.B2_max), ""]
    if lp is None:
        lines += ["The page-9 LP returned no blend:", *[f"- {e}" for e in errors]]
    else:
        qty = {k: float(v) for k, v in (getattr(lp, "quantities_mt", None) or getattr(lp, "ore_qty_mt", {}) or {}).items()}
        tot = sum(qty.values()) or 1.0
        lines += ["| Material | Engine, wet % | Page-9 LP, wet % |", "|---|---:|---:|"]
        for v in eng.fer:
            lines.append(f"| {v.name} | {r['share_wet'][v.id]:.1f} | {100 * qty.get(v.id, 0.0) / tot:.1f} |")
        flux_ids = [v.id for v in eng.flux]
        lines += ["", "| Flux | Engine, kg/tHM | Page-9 LP, kg/tHM |", "|---|---:|---:|"]
        for fid in flux_ids:
            lp_flux = (getattr(lp, "diagnostics", {}) or {}).get("lp_flux_quantities_mt", {}) or {}
            lp_kg = 1000 * float(lp_flux.get(fid, qty.get(fid, 0.0))) / a.target
            lines.append(f"| {fid} | {r['wet_kg'][fid]:.1f} | {lp_kg:.1f} |")
        lines += ["", "| | Engine | Page-9 LP |", "|---|---:|---:|",
                  f"| Slag, kg/tHM | {r['out']['slag']:.0f} | {float(lp.slag_rate_kg_per_thm):.0f} |",
                  f"| CaO/SiO2 | {r['out']['B2']:.3f} | {float(lp.slag_basicity):.3f} |",
                  f"| Al2O3, % | {r['out']['Al2O3_pct']:.2f} | {float(lp.slag_al2o3_pct):.2f} |",
                  f"| MgO, % | {r['out']['MgO_pct']:.2f} | {float(lp.slag_mgo_pct):.2f} |",
                  f"| MgO/Al2O3 | {r['out']['MgO'] / r['out']['Al2O3']:.3f} | {float(lp.slag_mgo_al2o3_ratio):.3f} |",
                  f"| Ferrous charged, kg/tHM wet | {sum(r['wet_kg'][v.id] for v in eng.fer):,.0f} | "
                  f"{1000 * sum(qty.get(v.id, 0.0) for v in eng.fer) / a.target:,.0f} |",
                  f"| Ore cost, Rs/tHM | {r['cost']['ore']:,.0f} | {float(lp.ore_cost_per_thm_rs):,.0f} |",
                  ""]
        binds_lp = [k for k, ok in (("Al2O3 max", float(lp.slag_al2o3_pct) >= L.Al2O3_max - 0.01),
                                    ("MgO min", float(lp.slag_mgo_pct) <= L.MgO_min + 0.01),
                                    ("MgO/Al2O3 min", float(lp.slag_mgo_al2o3_ratio) <= L.MgO_Al2O3_min + 0.001)) if ok]
        binds_en = [b for b in r["binding"] if not b.startswith(("min share", "max share"))]
        lines += [f"Binding in the engine: {', '.join(binds_en) or '-'}. Slag-quality limits at their bound in the "
                  f"page-9 LP: {', '.join(binds_lp) or 'none'}.", "",
                  "Same blend through both slag calculations: see docs/burden_fuel_optimiser.md. Both count TiO2, "
                  "alkalis and sulphur in the slag total (page 9 from its analyses, the engine as a 1.2 % allowance), "
                  "and both take the top-gas dust from the DPR with the plant's dust analyses (page 9 per stream, the "
                  "engine pooled and without MgO). What remains is the Fe sizing and the dust MgO (page 9 removes "
                  "about 0.2 kg MgO/tHM more, so its MgO/Al2O3 limit binds a little earlier)."]
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
