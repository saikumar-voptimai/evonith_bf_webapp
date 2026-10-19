"""Daily replay of both slag calculations on the furnace's actual charge, against the lab's slag analysis (#258).

For every day with a full charge record, the day's charged masses (``charge_data``, by material code), the lab
chemistry of each material (trailing window), the day's hot metal (DPR), PCI (DPR), dust (DPR) and the day's mean
hot-metal analysis go through

* page 9's slag balance (``utils.bmo.slag_balance.calculate_full_slag_balance``), TiO2 in the slag total (default)
  and on the plant-workbook basis (``tio2_to_slag=False``), with page 9's saved dust rows and fuel S / alkali;
* the burden engine's slag (``utils.coke_energy.bf_balance.slag_and_flux``, the two-zone model's oxide balance),
  with the DPR dust masses and the calibration's dust composition.

Both get the same masses, the same chemistry and the same measured hot-metal analysis (page 9 normally holds its
HM chemistry at configured values; here it gets the day's, so only the balances differ). Fuel ash oxides are not in
Neon: both use the calibration's coke / nut / PCI ash analysis (Jul-Oct 2026 hourly dataset).

The lab's daily mean slag analysis is the reference for composition. The slag mass is not measured (DPR slag
generation is a fixed ~0.33 t/tHM), so the slag total is checked with the Al2O3 tracer: model Al2O3 mass / lab
Al2O3 %.

Run:  python scripts/replay_slag_vs_lab.py [--start 2026-06-01] [--end 2026-10-16] [--out docs/slag_replay_vs_lab.md]
      [--csv slag_replay_daily.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "furnace_data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from furnace_data.offline import fetch_offline_data  # noqa: E402
from utils.bmo.calculations import resolve_dust_inputs  # noqa: E402
from utils.bmo.slag_balance import calculate_full_slag_balance  # noqa: E402
from utils.bmo.types import DustInput, FluxInput, FuelAshInput, OreChemistry, OreInput, SlagBalanceSettings  # noqa: E402
from utils.burden_engine.engine import DEFAULT_CAL_PATH  # noqa: E402
from utils.coke_energy.bf_balance import slag_and_flux  # noqa: E402

IST = "Asia/Kolkata"
FE_IN_FE2O3 = 2 * 55.845 / 159.69
FERROUS_PREFIX = ("ore_", "sinter_", "pellet_")
LAB = dict(sio2="slag_pct_sio2", al2o3="slag_pct_al2o3", cao="slag_pct_cao", mgo="slag_pct_mgo", tio2="slag_pct_tio2",
           mno="slag_pct_mno", feo="slag_pct_feo", s="slag_pct_s", k2o="slag_pct_k2o", na2o="slag_pct_na2o")
HM = dict(Fe="chem_pct_fe", C="chem_pct_c", Si="chem_pct_si", Mn="chem_pct_mn", P="chem_pct_p", S="chem_pct_s",
          Ti="chem_pct_ti")


def _ist_date(idx) -> pd.Series:
    return pd.Series(pd.to_datetime(idx, utc=True).tz_convert(IST).date, index=idx)


def load(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    rng = (start - pd.Timedelta(days=45), end + pd.Timedelta(days=1))
    names = ["charge_data", "dpr_data", "hot_metal_slag_analysis", "ore_chemistry", "sinter_chemistry",
             "fuel_chemistry", "flux_chemistry"]
    out = {}
    for n in names:
        df = fetch_offline_data(n, rng, query_type="raw")
        df = df[~df.index.isna()]
        df["d"] = _ist_date(df.index).values
        out[n] = df
    return out


def _avg_non_zero(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    for c in cols:
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce")
            v = v[v.notna() & (v != 0)]
            out[c] = float(v.mean()) if len(v) else 0.0
    return out


def chemistry_for(tab: pd.DataFrame, code: str, day, window: int, cols: list[str]) -> tuple[dict | None, str]:
    """Mean of the material's lab rows in the ``window`` days to ``day``; else its last analysis within 45 days."""
    t = tab[tab["material_code"] == code]
    t = t[t["d"] <= day]
    w = t[t["d"] > day - pd.Timedelta(days=window)]
    if len(w):
        return _avg_non_zero(w, cols), f"{len(w)} rows, {window} d"
    w = t[t["d"] > day - pd.Timedelta(days=45)]
    if len(w):
        last = w["d"].max()
        return _avg_non_zero(w[w["d"] == last], cols), f"carried from {last}"
    return None, "none"


ORE_COLS = ["fe_t", "feo", "sio2", "al2o3", "cao", "mgo", "mno", "tio2", "p", "loi", "k2o", "na2o", "tm"]
FLUX_COLS = ["sio2", "al2o3", "cao", "mgo", "fe2o3", "loi", "tm"]
FUEL_COLS = ["ash", "vm", "fc", "tm", "moisture"]


def day_inputs(T: dict, day, window: int) -> dict | None:
    cd = T["charge_data"]; cd = cd[cd["d"] == day]
    dpr = T["dpr_data"]; dpr = dpr[dpr["d"] == day]
    lab = T["hot_metal_slag_analysis"]; lab = lab[lab["d"] == day]
    if cd.empty or dpr.empty or lab.empty:
        return None
    hm_t = float(pd.to_numeric(dpr["total_hot_metal_mt"], errors="coerce").sum())
    mass = {c[:-3]: float(pd.to_numeric(cd[c], errors="coerce").fillna(0).sum()) for c in cd.columns if c.endswith("_mt")}
    ferrous = {k: v for k, v in mass.items() if k.startswith(FERROUS_PREFIX) and v > 0}
    flux = {k: v for k, v in mass.items() if k.startswith("flux_") and v > 0}
    fer_t = sum(ferrous.values())
    chem, missing = {}, []
    for code, q in ferrous.items():
        tab = T["sinter_chemistry"] if code.startswith("sinter_") else T["ore_chemistry"]
        c, how = chemistry_for(tab, code, day, window, ORE_COLS)
        if c is None or c.get("fe_t", 0) <= 0:
            missing.append((code, q / fer_t))
        else:
            chem[code] = (c, how)
    for code in flux:
        c, how = chemistry_for(T["flux_chemistry"], code, day, window, FLUX_COLS)
        if c is None:
            missing.append((code, 0.0))
        else:
            chem[code] = (c, how)
    fuel = {}
    for code in ("coke_1", "nut_coke_1", "pci_1"):
        c, _ = chemistry_for(T["fuel_chemistry"], code, day, window, FUEL_COLS)
        fuel[code] = c
    hm_chem = {k: float(pd.to_numeric(lab[v], errors="coerce").mean()) for k, v in HM.items()}
    slag_lab = {k: float(pd.to_numeric(lab[v], errors="coerce").mean()) for k, v in LAB.items()}
    pci_t = float(pd.to_numeric(dpr.get("pci_mt"), errors="coerce").sum())
    return dict(day=day, hm_t=hm_t, charges=int(len(cd)), ferrous=ferrous, flux=flux, chem=chem, fuel=fuel,
                missing=missing, coke_t=mass.get("coke_1", 0) + mass.get("coke_2", 0),
                nut_t=mass.get("nut_coke_1", 0) + mass.get("nut_coke_2", 0), pci_t=pci_t, hm=hm_chem, lab=slag_lab,
                lab_n=int(lab[LAB["al2o3"]].notna().sum()),
                dc_t=float(pd.to_numeric(dpr.get("flue_dust_mt"), errors="coerce").sum()),
                gcp_t=float(pd.to_numeric(dpr.get("gcp_dust_mt"), errors="coerce").sum()),
                slag_dpr_t=float(pd.to_numeric(dpr.get("slag_generation_mt"), errors="coerce").sum()))


def _fuel_props(c: dict | None, fallback: dict) -> dict:
    """Dry FC / ash / VM summing to 100 and moisture, from the day's analysis or the calibration's."""
    if not c or c.get("ash", 0) <= 0:
        return dict(FC=fallback["FC"], ash=fallback["ash"], VM=fallback["VM"], moist_pct=fallback["moist_pct"])
    fc, ash, vm = c.get("fc", 0.0), c.get("ash", 0.0), c.get("vm", 0.0)
    moist = c.get("tm") or c.get("moisture") or fallback["moist_pct"]
    if fc <= 0:  # nut coke rows carry ash only: VM from the calibration, FC by difference
        vm = vm if vm > 0 else fallback["VM"]
        return dict(FC=100 - ash - vm, ash=ash, VM=vm, moist_pct=moist)
    s = fc + ash + vm
    return dict(FC=100 * fc / s, ash=100 * ash / s, VM=100 * vm / s, moist_pct=moist)


def page9(D: dict, cal: dict, bmo: dict, prefs: dict, tio2_to_slag: bool, dust_basis: str = "saved") -> dict:
    """Page 9's slag balance on the day. ``dust_basis``: "saved" = page 9's saved dust rows; "engine" = the day's DPR
    dust mass with the engine's dust composition (calibration case, SiO2 7.6 / Al2O3 5.1 / CaO 2.0 / MgO 0 %)."""
    hm_t = D["hm_t"]
    ores, qty = [], {}
    for code, q in D["ferrous"].items():
        if code not in D["chem"]:
            continue
        c = D["chem"][code][0]
        chem = OreChemistry(fe_t_pct=c.get("fe_t", 0), moisture_pct=c.get("tm", 0), feo_pct=c.get("feo", 0),
                            sio2_pct=c.get("sio2", 0), al2o3_pct=c.get("al2o3", 0), cao_pct=c.get("cao", 0),
                            mgo_pct=c.get("mgo", 0), mno_pct=c.get("mno", 0), tio2_pct=c.get("tio2", 0),
                            p_pct=c.get("p", 0), na2o_pct=c.get("na2o", 0), k2o_pct=c.get("k2o", 0),
                            loi_pct=c.get("loi", 0))
        ores.append(OreInput(code, code, 1e6, 0.0, 0.0, 100.0, chem, metadata={}))
        qty[code] = q
    fluxes = []
    for code, q in D["flux"].items():
        if code not in D["chem"]:
            continue
        c = D["chem"][code][0]
        fluxes.append(FluxInput(code, code, wet_qty_mt=q, moisture_pct=c.get("tm", 0), sio2_pct=c.get("sio2", 0),
                                al2o3_pct=c.get("al2o3", 0), cao_pct=c.get("cao", 0), mgo_pct=c.get("mgo", 0),
                                fe2o3_pct=c.get("fe2o3", 0), loi_pct=c.get("loi", 0)))
    cfg_fuel = {r["fuel_id"]: r for r in bmo.get("fuel_ash_inputs", [])}
    fa = []
    for fid, key, code, t in (("coke", "coke", "coke_1", D["coke_t"]), ("nut_coke", "nut", "nut_coke_1", D["nut_t"]),
                              ("pci", "pci", "pci_1", D["pci_t"])):
        f = _fuel_props(D["fuel"].get(code), cal["case"][key]); a = cal["case"][key]["ash_comp"]; c = cfg_fuel.get(fid, {})
        fa.append(FuelAshInput(fid, fid, rate_kg_per_thm=1000 * t / hm_t, rate_basis="wet", moisture_pct=f["moist_pct"],
                               vm_pct=f["VM"], ash_pct=f["ash"], sio2_pct=a.get("SiO2", 0), al2o3_pct=a.get("Al2O3", 0),
                               cao_pct=a.get("CaO", 0), mgo_pct=a.get("MgO", 0), fe2o3_pct=a.get("Fe2O3", 0),
                               tio2_pct=a.get("TiO2", 0), alkali_pct=float(c.get("alkali_pct", 0)),
                               s_pct=float(c.get("s_pct", 0)), p_pct=float(c.get("p_pct", 0))))
    dust = []
    for did, d in (prefs.get("dust_editor", {}).get("rows", {}) or {}).items():
        dust.append(DustInput(did, did, enabled=bool(d.get("enabled", True)), rate_basis=d.get("rate_basis", "mt_per_day"),
                              wet_qty_mt=float(d.get("wet_qty_mt", 0)), quantity_kg_per_charge=float(d.get("quantity_kg_per_charge", 0)),
                              moisture_pct=float(d.get("moisture_pct", 0)), sio2_pct=float(d.get("sio2_pct", 0)),
                              al2o3_pct=float(d.get("al2o3_pct", 0)), cao_pct=float(d.get("cao_pct", 0)),
                              mgo_pct=float(d.get("mgo_pct", 0)), fe_pct=float(d.get("fe_pct", 0)),
                              p_pct=float(d.get("p_pct", 0)), s_pct=float(d.get("s_pct", 0))))
    if dust_basis == "engine":
        du = cal["case"]["dust"]
        fe = (du["dc_kg"] * du["dc_Fe_pct"] + du["gcp_kg"] * du["gcp_Fe_pct"]) / (du["dc_kg"] + du["gcp_kg"])
        dust = [DustInput("dpr_dust", "DPR dust", wet_qty_mt=D["dc_t"] + D["gcp_t"], rate_basis="mt_per_day",
                          sio2_pct=du.get("SiO2_pct", 7.6), al2o3_pct=du.get("Al2O3_pct", 5.1),
                          cao_pct=du.get("CaO_pct", 2.0), mgo_pct=du.get("MgO_pct", 0.0), fe_pct=fe)]
    dust_res, _ = resolve_dust_inputs(dust, charge_count=D["charges"])
    sb = {k: v for k, v in bmo["slag_balance"].items() if k in SlagBalanceSettings.__dataclass_fields__}
    h = D["hm"]
    sb.update(enabled=True, carbon_pct=h["C"], silicon_pct=h["Si"], sulphur_pct=h["S"], mn_pct=h["Mn"], ti_pct=h["Ti"],
              tio2_to_slag=tio2_to_slag)
    b = calculate_full_slag_balance(ores=ores, quantities_mt=qty, hot_metal_mt=hm_t, settings=SlagBalanceSettings(**sb),
                                    fuel_ash_inputs=fa, flux_inputs=fluxes, dust_inputs=dust_res)
    c = b.slag_components_mt; tot = b.total_slag_mt
    return dict(slag_kg=1000 * tot / hm_t, al2o3_kg=1000 * c["al2o3"] / hm_t, al2o3=100 * c["al2o3"] / tot,
                mgo=100 * c["mgo"] / tot, sio2=100 * c["sio2"] / tot, cao=100 * c["cao"] / tot,
                tio2=100 * c.get("tio2", 0.0) / tot, b2=c["cao"] / c["sio2"])


def engine(D: dict, cal: dict) -> dict:
    hm_t = D["hm_t"]; k = 1000 / hm_t
    case = json.loads(json.dumps(cal["case"]))
    burden = []
    for code, q in D["ferrous"].items():
        if code not in D["chem"]:
            continue
        c = D["chem"][code][0]; mo = c.get("tm", 0.0)
        burden.append(dict(name=code, dry_kg=q * k * (1 - mo / 100), moist_pct=mo, LOI_type="H2O",
                           comp=dict(Fe=c.get("fe_t", 0), FeO=c.get("feo", 0), SiO2=c.get("sio2", 0), CaO=c.get("cao", 0),
                                     MgO=c.get("mgo", 0), Al2O3=c.get("al2o3", 0), MnO=c.get("mno", 0),
                                     TiO2=c.get("tio2", 0), P=c.get("p", 0), LOI=c.get("loi", 0))))
    for code, q in D["flux"].items():
        if code not in D["chem"]:
            continue
        c = D["chem"][code][0]; mo = c.get("tm", 0.0)
        burden.append(dict(name=code, dry_kg=q * k * (1 - mo / 100), moist_pct=mo,
                           LOI_type="CO2" if c.get("cao", 0) + c.get("mgo", 0) > 20 else "H2O",
                           comp=dict(Fe=c.get("fe2o3", 0) * FE_IN_FE2O3, SiO2=c.get("sio2", 0), CaO=c.get("cao", 0),
                                     MgO=c.get("mgo", 0), Al2O3=c.get("al2o3", 0), LOI=c.get("loi", 0))))
    case["burden"] = burden
    for key, code in (("coke", "coke_1"), ("nut", "nut_coke_1"), ("pci", "pci_1")):
        case[key].update(_fuel_props(D["fuel"].get(code), cal["case"][key]))
    K = D["coke_t"] * k * (1 - case["coke"]["moist_pct"] / 100)
    case["nut_kg"] = D["nut_t"] * k * (1 - case["nut"]["moist_pct"] / 100)
    case["pci_kg"] = D["pci_t"] * k * (1 - case["pci"]["moist_pct"] / 100)
    case["hm"] = {kk: (v if np.isfinite(v) else cal["case"]["hm"][kk]) for kk, v in D["hm"].items()}
    feo = D["lab"]["feo"]
    case["slag_FeO_pct"] = feo if np.isfinite(feo) else cal["case"]["slag_FeO_pct"]
    case["dust"].update(dc_kg=D["dc_t"] * k, gcp_kg=D["gcp_t"] * k)
    s = slag_and_flux(K, case, cal["p"])
    return dict(slag_kg=s["slag_kg"], al2o3_kg=s["Al2O3"], al2o3=s["Al2O3_pct"], mgo=s["MgO_pct"],
                sio2=100 * s["SiO2"] / s["slag_kg"], cao=100 * s["CaO"] / s["slag_kg"],
                tio2=100 * s["TiO2"] / s["slag_kg"], b2=s["B2"], K=K)


def replay(start: str, end: str, window: int = 7, min_hm: float = 1800.0, min_charges: int = 120) -> pd.DataFrame:
    s, e = pd.Timestamp(start, tz=IST), pd.Timestamp(end, tz=IST)
    T = load(s, e)
    cal = json.load(open(DEFAULT_CAL_PATH, encoding="utf-8"))
    bmo = yaml.safe_load(open(ROOT / "src/config/setting_bmo.yml", encoding="utf-8"))["bmo"]
    prefs = yaml.safe_load(open(ROOT / "src/config/bmo_operator_inputs.yml", encoding="utf-8"))
    rows = []
    for day in pd.date_range(s.date(), e.date(), freq="D").date:
        D = day_inputs(T, day, window)
        if D is None:
            continue
        miss = sum(f for _, f in D["missing"])
        row = dict(day=day, hm_t=D["hm_t"], charges=D["charges"], lab_n=D["lab_n"], missing_ferrous_pct=100 * miss,
                   slag_dpr_kg=1000 * D["slag_dpr_t"] / D["hm_t"] if D["hm_t"] else np.nan,
                   dust_kg=1000 * (D["dc_t"] + D["gcp_t"]) / D["hm_t"] if D["hm_t"] else np.nan,
                   ok=D["hm_t"] >= min_hm and D["charges"] >= min_charges and D["lab_n"] >= 3 and miss <= 0.02)
        for k2, v in D["lab"].items():
            row[f"lab_{k2}"] = v
        row["lab_b2"] = D["lab"]["cao"] / D["lab"]["sio2"]
        row["lab_sum"] = sum(v for v in D["lab"].values() if np.isfinite(v))
        if row["ok"]:
            for tag, res in (("eng", engine(D, cal)), ("p9", page9(D, cal, bmo, prefs, True)),
                             ("p9wb", page9(D, cal, bmo, prefs, False)),
                             ("p9ed", page9(D, cal, bmo, prefs, True, dust_basis="engine"))):
                for k2, v in res.items():
                    row[f"{tag}_{k2}"] = v
                row[f"{tag}_tracer_kg"] = 100 * res["al2o3_kg"] / D["lab"]["al2o3"]
        rows.append(row)
    return pd.DataFrame(rows)


MODELS = (("eng", "Engine (two-zone)"), ("p9", "Page 9, TiO2 in slag"), ("p9wb", "Page 9, workbook basis"),
          ("p9ed", "Page 9, TiO2 in, engine's dust"))


def summary(df: pd.DataFrame) -> str:
    ok = df[df["ok"]]
    lines = [f"# Slag replay against the lab, {ok['day'].min()} to {ok['day'].max()}", "",
             f"{len(ok)} days with a full charge record (hot metal >= 1,800 t, >= 120 charges, >= 3 slag analyses, "
             f"lab chemistry for >= 98 % of the ferrous charge) out of {len(df)}. Model minus lab, daily means.", "",
             "| | " + " | ".join(n for _, n in MODELS) + " |", "|---|" + "---:|" * len(MODELS)]

    def stat(col_m, col_l, f="{:+.2f}"):
        out = []
        for t, _ in MODELS:
            d = ok[f"{t}_{col_m}"] - ok[col_l]
            out.append(f"{f.format(d.mean())} (MAE {abs(d).mean():.2f})")
        return out
    for lab_col, m_col, name in (("lab_al2o3", "al2o3", "Al2O3, % pt"), ("lab_mgo", "mgo", "MgO, % pt"),
                                 ("lab_sio2", "sio2", "SiO2, % pt"), ("lab_cao", "cao", "CaO, % pt"),
                                 ("lab_tio2", "tio2", "TiO2, % pt")):
        lines.append(f"| {name} | " + " | ".join(stat(m_col, lab_col)) + " |")
    out = []
    for t, _ in MODELS:
        d = ok[f"{t}_b2"] - ok["lab_b2"]
        out.append(f"{d.mean():+.3f} (MAE {abs(d).mean():.3f})")
    lines.append("| CaO/SiO2 | " + " | ".join(out) + " |")
    lines.append("| Slag, kg/tHM (mean) | " + " | ".join(f"{ok[f'{t}_slag_kg'].mean():.0f}" for t, _ in MODELS) + " |")
    lines.append("| Al2O3-tracer slag, kg/tHM (mean) | " + " | ".join(f"{ok[f'{t}_tracer_kg'].mean():.0f}"
                                                                   for t, _ in MODELS) + " |")
    lines += ["", f"Lab means: Al2O3 {ok['lab_al2o3'].mean():.2f} %, MgO {ok['lab_mgo'].mean():.2f} %, CaO/SiO2 "
              f"{ok['lab_b2'].mean():.3f}, TiO2 {ok['lab_tio2'].mean():.2f} %, oxide sum {ok['lab_sum'].mean():.1f} %. "
              f"DPR slag generation {ok['slag_dpr_kg'].mean():.0f} kg/tHM (not measured), DPR dust "
              f"{ok['dust_kg'].mean():.0f} kg/tHM.", "",
              "Basis: charged masses by material from `charge_data`, each material's lab chemistry averaged over the "
              "7 days to the day, PCI / dust / hot metal from the DPR, the day's mean hot-metal analysis for both "
              "models (page 9 normally holds its configured HM chemistry). Fuel ash oxides are not in Neon: both use "
              "the calibration's coke / nut / PCI ash analysis. Page 9's dust: its saved row (530 kg per charge, CaO "
              "5.63 %, MgO 3.57 %, from the plant BURDEN workbook); the engine's: the DPR dust mass with SiO2 7.6, "
              "Al2O3 5.1, CaO 2.0, MgO 0 %. The Al2O3-tracer slag is each model's Al2O3 mass over the lab's Al2O3 %.",
              "", "`python scripts/replay_slag_vs_lab.py --start 2025-09-01 --out docs/slag_replay_vs_lab.md`", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default=str(pd.Timestamp.now(tz=IST).date()))
    ap.add_argument("--window", type=int, default=7, help="days of lab chemistry averaged per material")
    ap.add_argument("--out")
    ap.add_argument("--csv")
    a = ap.parse_args()
    df = replay(a.start, a.end, a.window)
    text = summary(df)
    if a.csv:
        df.to_csv(a.csv, index=False)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
