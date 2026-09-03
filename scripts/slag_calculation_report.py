"""Step-by-step slag calculation report for BMO blends.

Traces every stage of ``calculate_full_slag_balance`` for a named blend, prints
the intermediate masses, and compares the resulting slag chemistry against the
plant's own slag analysis so a discrepancy can be attributed to a component.

Run from the repo root:

    python scripts/slag_calculation_report.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from utils.bmo.calculations import evaluate_blend  # noqa: E402
from utils.bmo.slag_balance import (  # noqa: E402
    FE_FROM_FE2O3_FACTOR,
    MN_FROM_MNO_FACTOR,
    TI_FROM_TIO2_FACTOR,
    calculate_full_slag_balance,
)
from utils.bmo.types import (  # noqa: E402
    FluxInput,
    FuelAshInput,
    OreChemistry,
    OreInput,
    SlagBalanceSettings,
)

CFG = yaml.safe_load((ROOT / "src/config/setting_bmo.yml").read_text(encoding="utf-8"))["bmo"]
HM_MT = 2350.0

# 3-month means from Neon: ore_chemistry, sinter_chemistry, flux_chemistry.
CHEM = {
    "sinter": dict(fe_t_pct=54.58, sio2_pct=5.37, al2o3_pct=2.53, cao_pct=10.79,
                   mgo_pct=2.22, feo_pct=9.10, mno_pct=0.21, tio2_pct=0.16,
                   k2o_pct=0.05, na2o_pct=0.03, p_pct=0.06, moisture_pct=6.0),
    "pellet": dict(fe_t_pct=63.93, sio2_pct=4.08, al2o3_pct=2.43, cao_pct=1.02,
                   mgo_pct=0.05, mno_pct=0.04, tio2_pct=0.16, k2o_pct=0.03,
                   na2o_pct=0.01, p_pct=0.05, moisture_pct=1.14),
    "clo_hi": dict(fe_t_pct=63.45, sio2_pct=3.26, al2o3_pct=2.65, cao_pct=0.07,
                   mgo_pct=0.05, mno_pct=0.05, tio2_pct=0.14, k2o_pct=0.02,
                   na2o_pct=0.02, p_pct=0.06, moisture_pct=2.68),
    "ibrm_lean": dict(fe_t_pct=56.38, sio2_pct=14.68, al2o3_pct=2.03, cao_pct=0.07,
                      mgo_pct=0.06, mno_pct=0.12, tio2_pct=0.13, k2o_pct=0.03,
                      na2o_pct=0.02, p_pct=0.04, moisture_pct=1.04),
}
PRICES = {"sinter": 5200.0, "pellet": 7000.0, "clo_hi": 6800.0, "ibrm_lean": 4900.0}

# Plant slag analysis, dataset means (SLAG_PCT_* in furnace_dataset.csv).
PLANT_SLAG_PCT = {"sio2": 34.02, "cao": 36.63, "mgo": 7.84, "al2o3": 18.71,
                  "feo": 0.44, "mno": 0.30, "tio2": 0.52, "s": 0.75,
                  "na2o": 0.10, "k2o": 0.14}


def ores() -> list[OreInput]:
    return [OreInput(ore_id=k, display_name=k, stock_mt=1e6, price_rs_per_mt=PRICES[k],
                     min_share_pct=0.0, max_share_pct=100.0, chemistry=OreChemistry(**v))
            for k, v in CHEM.items()]


def fuel_ash() -> list[FuelAshInput]:
    return [FuelAshInput(**row) for row in CFG["fuel_ash_inputs"]]


def slag_settings(si_pct: float, mn_pct: float, ti_pct: float) -> SlagBalanceSettings:
    cfg = dict(CFG["slag_balance"])
    valid = SlagBalanceSettings.__dataclass_fields__.keys()
    s = {k: v for k, v in cfg.items() if k in valid}
    # Live HM chemistry overrides the yml fallbacks in the app; mirror that here.
    s.update(silicon_pct=si_pct, mn_pct=mn_pct, ti_pct=ti_pct)
    return SlagBalanceSettings(**s)


def fluxes(dolomite_mt=0.0, limestone_mt=0.0, quartz_mt=0.0) -> list[FluxInput]:
    return [
        FluxInput(flux_id="dolomite", display_name="Dolomite", enabled=True,
                  wet_qty_mt=dolomite_mt, moisture_pct=0.23, cao_pct=30.27,
                  mgo_pct=22.42, sio2_pct=1.45, al2o3_pct=0.21, fe2o3_pct=0.16,
                  loi_pct=45.16, price_rs_per_mt=3000.0),
        FluxInput(flux_id="limestone", display_name="Limestone", enabled=True,
                  wet_qty_mt=limestone_mt, moisture_pct=0.21, cao_pct=50.18,
                  mgo_pct=3.87, sio2_pct=2.87, al2o3_pct=0.90, fe2o3_pct=0.54,
                  loi_pct=41.10, price_rs_per_mt=1800.0),
        FluxInput(flux_id="quartz", display_name="Quartz", enabled=True,
                  wet_qty_mt=quartz_mt, moisture_pct=0.30, sio2_pct=96.07,
                  al2o3_pct=0.62, fe2o3_pct=1.12, loi_pct=1.10,
                  price_rs_per_mt=2000.0),
    ]


def quantities(shares_pct: dict[str, float], target_fe_mt: float) -> dict[str, float]:
    fe_per_wet = sum((shares_pct.get(o.ore_id, 0.0) / 100.0)
                     * (100.0 - o.chemistry.moisture_pct) / 100.0
                     * o.chemistry.fe_t_pct / 100.0 for o in ores())
    total = target_fe_mt / fe_per_wet
    return {o.ore_id: total * shares_pct.get(o.ore_id, 0.0) / 100.0 for o in ores()}


def line(char="-", n=86):
    print(char * n)


def report(name: str, shares_pct: dict[str, float], flux_rows: list[FluxInput],
           si_pct=0.45, mn_pct=0.25, ti_pct=0.06, hm_fe_pct=94.5) -> None:
    O = ores()
    qty = quantities(shares_pct, HM_MT * hm_fe_pct / 100.0)
    FA = fuel_ash()
    SET = slag_settings(si_pct, mn_pct, ti_pct)

    line("=")
    print(f"BLEND: {name}")
    line("=")
    print(f"Target hot metal {HM_MT:,.0f} MT/day at {hm_fe_pct:.1f}% Fe "
          f"=> Fe required {HM_MT*hm_fe_pct/100:,.1f} MT")
    print(f"HM chemistry used: Si {si_pct:.2f}%  Mn {mn_pct:.2f}%  Ti {ti_pct:.2f}%  "
          f"C {SET.carbon_pct:.2f}%  S {SET.sulphur_pct:.3f}%")
    print()

    # --- STEP 1: wet -> dry ------------------------------------------------
    print("STEP 1  Wet quantity -> dry quantity      dry = wet x (100 - moisture)/100")
    print(f"  {'material':<12}{'wet MT':>10}{'moist %':>9}{'dry MT':>10}{'Fe %':>7}{'Fe MT':>10}")
    total_dry = total_fe = 0.0
    for o in O:
        w = qty[o.ore_id]
        dry = w * (100.0 - o.chemistry.moisture_pct) / 100.0
        fe = dry * o.chemistry.fe_t_pct / 100.0
        total_dry += dry; total_fe += fe
        print(f"  {o.ore_id:<12}{w:>10,.1f}{o.chemistry.moisture_pct:>9.2f}"
              f"{dry:>10,.1f}{o.chemistry.fe_t_pct:>7.2f}{fe:>10,.1f}")
    print(f"  {'TOTAL':<12}{sum(qty.values()):>10,.1f}{'':>9}{total_dry:>10,.1f}"
          f"{total_fe/total_dry*100:>7.2f}{total_fe:>10,.1f}")
    print()

    # --- STEP 2: components into the furnace -------------------------------
    bal = calculate_full_slag_balance(
        ores=O, quantities_mt=qty, hot_metal_mt=HM_MT, settings=SET,
        fuel_ash_inputs=FA, flux_inputs=flux_rows, dust_inputs=None)
    d = bal.diagnostics
    keys = ["sio2", "al2o3", "cao", "mgo", "fe", "mn", "ti", "s", "p", "alkali"]
    print("STEP 2  Component masses in, MT      component = dry MT x component% / 100")
    print(f"  {'component':<10}{'ore':>12}{'flux':>10}{'fuel ash':>11}{'TOTAL IN':>12}")
    for k in keys:
        print(f"  {k:<10}{bal.ore_components_mt[k]:>12,.2f}{bal.flux_components_mt[k]:>10,.2f}"
              f"{bal.fuel_ash_components_mt[k]:>11,.2f}{bal.total_into_bf_mt[k]:>12,.2f}")
    print(f"\n  Mn and Ti are converted from oxide to element on the way in:")
    print(f"    MnO -> Mn  x {MN_FROM_MNO_FACTOR:.4f}     TiO2 -> Ti x {TI_FROM_TIO2_FACTOR:.4f}"
          f"     Fe2O3 -> Fe x {FE_FROM_FE2O3_FACTOR:.4f}")
    print(f"  Dust deduction: none configured, so net in = total in")
    print()

    # --- STEP 3: pig iron ---------------------------------------------------
    print("STEP 3  Pig iron")
    print(f"  metal % in PI = 100 - C - Si - S - other "
          f"= 100 - {SET.carbon_pct} - {SET.silicon_pct} - {SET.sulphur_pct} - {SET.other_pct}"
          f" = {d['pig_iron_metal_pct']:.3f}%")
    print(f"  Fe to PI      = Fe in x {SET.fe_to_pig_iron_fraction} = {d['fe_to_pig_iron_mt']:,.2f} MT"
          f"   (Fe left for slag {d['fe_remaining_mt']:,.3f} MT)")
    print(f"  Mn to PI      = {d['mn_to_pig_iron_mt']:,.3f} MT  (left {d['mn_remaining_mt']:,.3f})")
    print(f"  Ti to PI      = {d['ti_to_pig_iron_mt']:,.3f} MT  (left {d['ti_remaining_mt']:,.3f})")
    print(f"  metallic mass = Fe + Mn + Ti + P + Zn = {d['metallic_mass_mt']:,.2f} MT")
    print(f"  theoretical PI= metallic x 100 / {d['pig_iron_metal_pct']:.3f} = {bal.theoretical_pig_iron_mt:,.2f} MT")
    print(f"  actual PI     = theoretical x (100 - {SET.pi_loss_pct})/100 = {bal.actual_pig_iron_mt:,.2f} MT")
    print()

    # --- STEP 4: what leaves the slag --------------------------------------
    print("STEP 4  Reductions out of the slag")
    print(f"  SiO2 consumed by Si reduction = PI x Si% x {SET.si_to_sio2_factor}"
          f" = {bal.actual_pig_iron_mt:,.1f} x {SET.silicon_pct/100:.4f} x {SET.si_to_sio2_factor}"
          f" = {d['sio2_consumed_by_si_mt']:,.2f} MT")
    print(f"  S to PI  = PI x {SET.sulphur_pct}% = {d['sulphur_to_pig_iron_mt']:,.3f} MT")
    print(f"  S to gas = S in x {SET.sulphur_gas_loss_pct}% = {d['sulphur_to_gas_mt']:,.3f} MT")
    print(f"  alkali to slag = {SET.alkali_to_slag_fraction*100:.0f}% "
          f"(remaining {d['hm_reduction_alkali_mt']:,.3f} MT leaves as vapour)")
    print()

    # --- STEP 5: slag components -------------------------------------------
    print("STEP 5  Final slag components, MT")
    raw = d["raw_slag_components_mt"]
    print(f"  {'component':<10}{'MT':>12}{'% of slag':>12}   formula")
    formulas = {
        "sio2": "SiO2 in  - SiO2 consumed by Si",
        "al2o3": "Al2O3 in, all of it",
        "cao": "CaO in, all of it",
        "mgo": "MgO in, all of it",
        "feo": f"Fe left x {SET.fe_to_feo_factor:.4f}",
        "mno": f"Mn left x {SET.mn_to_mno_factor}",
        "s": "S in - S to PI - S to gas",
        "alkali": f"alkali in x {SET.alkali_to_slag_fraction}",
        "caf2": "CaF2 in, all of it",
    }
    tot = d["raw_total_slag_mt"]
    for k, v in raw.items():
        print(f"  {k:<10}{v:>12,.2f}{v/tot*100:>12.2f}   {formulas.get(k,'')}")
    print(f"  {'TOTAL':<10}{tot:>12,.2f}{100.0:>12.2f}")
    print(f"\n  x correction factor {d['slag_correction_factor']:.3f} "
          f"=> final slag {bal.total_slag_mt:,.2f} MT")
    print(f"  slag rate = {bal.total_slag_mt:,.2f} / {HM_MT:,.0f} x 1000 "
          f"= {bal.total_slag_mt/HM_MT*1000:,.1f} kg/THM")
    print()

    # --- STEP 6: chemistry vs the plant ------------------------------------
    print("STEP 6  Calculated slag chemistry vs the plant's slag analysis")
    print(f"  {'component':<10}{'calc %':>10}{'plant %':>10}{'diff':>9}"
          f"{'calc MT':>11}{'implied MT':>13}")
    implied_total = {}
    for k in ["sio2", "cao", "mgo", "al2o3", "feo", "mno", "s"]:
        calc_pct = raw.get(k, 0.0) / tot * 100.0
        plant_pct = PLANT_SLAG_PCT.get(k, 0.0)
        implied = raw.get(k, 0.0) / plant_pct * 100.0 if plant_pct else float("nan")
        implied_total[k] = implied
        print(f"  {k:<10}{calc_pct:>10.2f}{plant_pct:>10.2f}{calc_pct-plant_pct:>+9.2f}"
              f"{raw.get(k,0.0):>11,.1f}{implied:>13,.0f}")
    alk_pct = raw["alkali"] / tot * 100.0
    plant_alk = PLANT_SLAG_PCT["na2o"] + PLANT_SLAG_PCT["k2o"]
    print(f"  {'alkali':<10}{alk_pct:>10.2f}{plant_alk:>10.2f}{alk_pct-plant_alk:>+9.2f}"
          f"{raw['alkali']:>11,.1f}{raw['alkali']/plant_alk*100:>13,.0f}")
    print(f"  {'tio2':<10}{0.0:>10.2f}{PLANT_SLAG_PCT['tio2']:>10.2f}"
          f"{-PLANT_SLAG_PCT['tio2']:>+9.2f}{0.0:>11,.1f}{'n/a':>13}")
    print()
    print("  'implied MT' = calculated mass of that component / the plant's % for it.")
    print("  If the model and the plant agreed, every row would imply the same total.")
    print(f"  Al2O3 tracer implies {implied_total['al2o3']:,.0f} MT; "
          f"the model reports {bal.total_slag_mt:,.0f} MT "
          f"({bal.total_slag_mt/implied_total['al2o3']-1:+.1%}).")
    print()
    print(f"  basicity B2 CaO/SiO2  calc {raw['cao']/raw['sio2']:.3f}   "
          f"plant {PLANT_SLAG_PCT['cao']/PLANT_SLAG_PCT['sio2']:.3f}")
    print(f"  basicity B4 (CaO+MgO)/SiO2  calc {(raw['cao']+raw['mgo'])/raw['sio2']:.3f}   "
          f"plant {(PLANT_SLAG_PCT['cao']+PLANT_SLAG_PCT['mgo'])/PLANT_SLAG_PCT['sio2']:.3f}")
    print()

    # --- STEP 7: what the model does not put into slag ----------------------
    print("STEP 7  Components the model leaves out of the slag")
    ti_missing = d["tio2_unaccounted_mt"]
    mno_expected = tot * PLANT_SLAG_PCT["mno"] / 100.0
    tio2_expected = tot * PLANT_SLAG_PCT["tio2"] / 100.0
    print(f"  TiO2   model puts 0.00 MT into slag. calculate_full_slag_balance builds")
    print(f"         raw_slag_components with no 'tio2' key, so Ti that did not go to")
    print(f"         hot metal is dropped: {ti_missing:,.2f} MT of TiO2 unaccounted.")
    print(f"         At the plant's {PLANT_SLAG_PCT['tio2']:.2f}% this slag should carry "
          f"~{tio2_expected:,.1f} MT.")
    print(f"  MnO    model puts {raw['mno']:,.2f} MT into slag, because all Mn was assigned")
    print(f"         to pig iron (HM Mn {SET.mn_pct:.2f}% x {bal.actual_pig_iron_mt:,.0f} MT PI"
          f" = {bal.actual_pig_iron_mt*SET.mn_pct/100:,.2f} MT capacity vs "
          f"{bal.net_into_bf_mt['mn']:,.2f} MT available).")
    print(f"         At the plant's {PLANT_SLAG_PCT['mno']:.2f}% this slag should carry "
          f"~{mno_expected:,.1f} MT.")
    understated = (tio2_expected - 0.0) + (mno_expected - raw["mno"])
    print(f"  Net effect: slag understated by ~{understated:,.1f} MT "
          f"({understated/tot*100:.1f}% of the total).")
    print()


if __name__ == "__main__":
    # Blend A uses the flux rate the plant actually charges: FLUX_CALC_MT averages
    # 0.415 MT/hr = 10 MT/day over the last 90 days, i.e. 0.27% of the burden. The
    # sinter (basicity 2.01) carries essentially all the CaO.
    report("A — current operation, PLANT flux rate (10 MT/day)",
           {"sinter": 68.0, "pellet": 10.0, "clo_hi": 14.0, "ibrm_lean": 8.0},
           fluxes(dolomite_mt=4.0, limestone_mt=6.0))
    # Blend B is what an optimizer reaches for when it buys lean, acidic ore and
    # then has to flux the basicity back up. This is the case that diverges.
    report("B — lean burden with heavy LP-added flux (260 MT/day)",
           {"sinter": 60.0, "pellet": 6.0, "clo_hi": 9.0, "ibrm_lean": 25.0},
           fluxes(dolomite_mt=60.0, limestone_mt=200.0))
