"""BMO fuel-cost model V5: coke strength + physics-derived energy features.

Run:  python scripts/train_bmo_fuel_v5.py

Rebuilds the V4 notebook pipeline as a runnable script and adds three things:

  1. COKE STRENGTH - M-40, M-10, CRI, CSR. Absent from furnace_dataset.csv;
     pulled from offline_feed.raw_material_strength_analysis (325 daily rows)
     and merged as-of, so each hour carries the most recent lab sample.
  2. SLAG RATE, CALCULATED from burden gangue rather than measured, so it is
     available for a candidate blend that has never been charged.
  3. ENERGY TERMS - blast sensible heat, burden moisture and calcination loads,
     iron reduction demand, and the energy deficit that carbon must close.

The last of these ends in ``IMPLIED_COKE_KG_THM``: the coke rate the energy
balance says this burden needs. It is built from burden quantities, blast
settings and PCI ONLY - no contemporaneous coke rate, no total fuel, no top-gas
analysis. That makes it a physics prior the tree can lean on or ignore.

TWO DEFECTS IN THE V4 SPLIT, FOUND WHILE PORTING.

First, cell 35 reads ``shuffle=True`` under a comment that says "CRITICAL FIX:
Chronological Split (shuffle=False)". On hourly autocorrelated data a random
split puts neighbouring hours either side of the fence, so the reported score is
not reachable forward. Both splits are run below so the size of that gap is
visible rather than argued about.

Second, the target is worth expanding:

    unitcost = 28*coke + 24*(total_fuel - coke - pci) + 18*pci
             = 4*coke + 24*total_fuel - 6*pci

It is dominated by TOTAL FUEL at a coefficient of 24, not by coke at 4. Total
fuel and coke rate are both correctly excluded from the features, but it is
worth knowing what the model is really being asked to predict: any feature
carrying total fuel is six times more dangerous than one carrying coke.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

STATIC_CSV = REPO / "src" / "assets" / "data" / "furnace_dataset.csv"
RANDOM_STATE = 42
TEST_FRACTION = 0.2

# Physics constants, matching utils/energy_balance.
FE_REDUCTION_MJ_PER_KG = 7.38
SLAG_MJ_PER_KG = 1.80
HOT_METAL_MJ_PER_T = 1378.0
BURDEN_MOISTURE_MJ_PER_KG = 2.70
CALCINATION_MJ_PER_KG_CO2 = 4.05
BLAST_CP_KJ_PER_NM3_K = 1.40
C_FULL_MJ_PER_KG = 32.8
C_FRAC_COKE = 0.87
C_FRAC_PCI = 0.75
T_REF_C = 25.0


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


# --------------------------------------------------------------------------- #
# data                                                                          #
# --------------------------------------------------------------------------- #


def load_base() -> pd.DataFrame:
    df = pd.read_csv(STATIC_CSV)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    coke = num(df, "COKE RATE KG/THM")
    pci = num(df, "PCI_KG/THM")
    total = num(df, "ACT. FUEL RATEKG/THM.")
    nut = total - (coke + pci)
    df["unitcost_new"] = coke * 28.0 + nut * 24.0 + pci * 18.0
    df = df[df["unitcost_new"] > 0].reset_index(drop=True)

    # V4 cruising filters: stable operation only.
    mask = (
        (num(df, "HOT BLAST VOLUMENM3/HR.") >= 90_000)
        & (pci.reindex(df.index).fillna(0) >= 100)
        & (num(df, "FURNACETOPGASANALYSISCO2ETACO").between(38, 47))
        & (num(df, "PRODUCTIONTONNESPERHR") >= 75)
        & (num(df, "ACT. FUEL RATEKG/THM.").between(500, 600))
    )
    return df[mask].reset_index(drop=True)


def add_coke_strength(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge M-40, M-10, CRI, CSR as-of, so no row sees a future lab result."""

    try:
        from furnace_data.offline import fetch_offline_data

        raw = fetch_offline_data(
            "raw_material_strength_analysis", time_range="full", query_type="raw"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  coke strength unavailable ({str(exc)[:80]}); continuing without")
        return df, []

    coke = raw[raw["material_code"].astype(str).str.startswith("coke")].copy()
    if coke.empty:
        print("  no coke rows in the strength table; continuing without")
        return df, []

    tidy = pd.DataFrame(index=coke.index)
    for slot in (1, 2, 3, 4):
        names = coke[f"property_{slot}_name"].dropna().astype(str)
        if names.empty:
            continue
        label = names.iloc[0].strip().upper().replace("-", "")
        tidy[f"COKE_{label}"] = pd.to_numeric(coke[f"property_{slot}"], errors="coerce")
    tidy.index = pd.to_datetime(tidy.index).tz_convert(None)
    tidy = tidy.sort_index().groupby(level=0).mean()

    added = list(tidy.columns)
    merged = pd.merge_asof(
        df.sort_values("time"),
        tidy.reset_index().rename(columns={"index": "time", "time": "time"}),
        on="time", direction="backward", tolerance=pd.Timedelta("21D"),
    )
    for col in added:
        merged[col] = merged[col].ffill()
    covered = merged[added].notna().all(axis=1).mean() if added else 0.0
    print(f"  coke strength merged: {added}  coverage {covered:.0%}")
    return merged, added


def add_engineered(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Slag rate and energy terms, from burden and blast only.

    Nothing here touches coke rate, total fuel or the top-gas analysis. The
    point is a physics prior the model can use for a blend nobody has charged.
    """

    out = df.copy()
    prod = num(out, "PRODUCTIONTONNESPERHR").replace(0, np.nan)
    prod = prod.fillna(prod.median())

    def per_thm(col: str) -> pd.Series:
        return num(out, col) / prod * 1000.0  # t/hr -> kg/tHM

    sinter = per_thm("SINTER_CALC_MT")
    pellet = per_thm("TOTAL_PELLET_CALC_MT")
    flux = per_thm("FLUX_CALC_MT")
    ore = per_thm("ORE_CALC_MT") if "ORE_CALC_MT" in out.columns else per_thm(
        "TOTAL_ORE_MT"
    )
    pci_rate = num(out, "PCI_KG/THM")

    # --- slag, from gangue rather than from the slag report -----------------
    gangue = pd.Series(0.0, index=out.index)
    for qty, prefix, oxides in (
        (ore, "ORE_", ("SIO2%", "AL2O3%", "CAO%", "MGO%")),
        (sinter, "SINTER_", ("SIO2%", "AL2O3%", "CAO%", "MGO%")),
        (pellet, "PELLET_PCT_", ("SIO2", "AL2O3", "CAO", "MGO")),
        (flux, "FLUX_", ("SIO2%", "AL2O3%", "CAO%", "MGO%")),
    ):
        for oxide in oxides:
            gangue = gangue + qty * num(out, f"{prefix}{oxide}") / 100.0
    out["SLAG_RATE_CALC_KG_THM"] = gangue
    out["BURDEN_TOTAL_KG_THM"] = ore + sinter + pellet + flux

    # --- iron in, from burden composition ------------------------------------
    fe_in = (
        ore * num(out, "ORE_FE(T)%") / 100.0
        + sinter * num(out, "SINTER_FE(T)%") / 100.0
        + pellet * num(out, "PELLET_PCT_FE2O3") / 100.0 * 0.6994
    )
    out["FE_INPUT_KG_THM"] = fe_in

    # --- moisture and LOI loads ----------------------------------------------
    moisture = (
        ore * num(out, "ORE_TM%") + pellet * num(out, "PELLET_PCT_TM")
        + flux * num(out, "FLUX_TM%")
    ) / 100.0
    out["BURDEN_MOISTURE_KG_THM"] = moisture
    out["FLUX_LOI_KG_THM"] = flux * num(out, "FLUX_LOI%") / 100.0

    # --- energy demand and blast supply --------------------------------------
    demand = (
        fe_in * FE_REDUCTION_MJ_PER_KG
        + gangue * SLAG_MJ_PER_KG
        + HOT_METAL_MJ_PER_T
        + moisture * BURDEN_MOISTURE_MJ_PER_KG
        + out["FLUX_LOI_KG_THM"] * CALCINATION_MJ_PER_KG_CO2
    )
    blast_per_thm = num(out, "HOT BLAST VOLUMENM3/HR.") / prod
    supply = blast_per_thm * BLAST_CP_KJ_PER_NM3_K * (
        num(out, "HOT BLAST TEMP.OC") - T_REF_C
    ) / 1000.0

    out["ENERGY_DEMAND_MJ_THM"] = demand
    out["BLAST_SENSIBLE_MJ_THM"] = supply
    out["BLAST_NM3_THM"] = blast_per_thm
    out["OXYGEN_NM3_THM"] = num(out, "OXYGENFLOWNM3/HR.") / prod
    out["STEAM_KG_THM"] = num(out, "STEAMKGS/HR.") / prod

    # THE physics prior: what carbon, and hence what coke, this burden needs.
    deficit = demand - supply
    out["ENERGY_DEFICIT_MJ_THM"] = deficit
    implied_carbon = deficit / C_FULL_MJ_PER_KG
    out["IMPLIED_CARBON_KG_THM"] = implied_carbon
    out["IMPLIED_COKE_KG_THM"] = (
        implied_carbon - pci_rate * C_FRAC_PCI
    ) / C_FRAC_COKE
    # Energy per unit of blast: how hard the blast is being asked to work.
    out["DEMAND_PER_NM3_BLAST"] = demand / blast_per_thm.replace(0, np.nan)

    added = [
        "SLAG_RATE_CALC_KG_THM", "BURDEN_TOTAL_KG_THM", "FE_INPUT_KG_THM",
        "BURDEN_MOISTURE_KG_THM", "FLUX_LOI_KG_THM", "ENERGY_DEMAND_MJ_THM",
        "BLAST_SENSIBLE_MJ_THM", "BLAST_NM3_THM", "OXYGEN_NM3_THM",
        "STEAM_KG_THM", "ENERGY_DEFICIT_MJ_THM", "IMPLIED_CARBON_KG_THM",
        "IMPLIED_COKE_KG_THM", "DEMAND_PER_NM3_BLAST",
    ]
    out[added] = out[added].replace([np.inf, -np.inf], np.nan)
    return out, added


# --------------------------------------------------------------------------- #
# features                                                                      #
# --------------------------------------------------------------------------- #

# Anything carrying the target. Total fuel is the dangerous one: it enters the
# target at a coefficient of 24 against coke's 4.
LEAK = {
    "COKE RATE KG/THM", "ACT. FUEL RATEKG/THM.", "UNITCOST LAKHS/THM",
    "unitcost_new", "PRODUCTIONTONNESPERHR", "TOTAL OXYGENNM3/HR.",
    "FURNACETOPGASANALYSISCO2ETACO", "FURNACE TOP GAS ANALYSISCO2%",
    "FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%",
    "COKE_CALC_MT", "NUTCOKE_CALC_MT", "COKE_CALC_THM", "NUTCOKE_CALC_THM",
}
LEAK_PREFIX = ("SLAG_PCT", "CHEM_", "RAFTOC", "HMT_GT")


def base_features(df: pd.DataFrame) -> list[str]:
    keep = []
    for col in df.columns:
        if col in LEAK or col == "time":
            continue
        if col.startswith(LEAK_PREFIX):
            continue
        if col in ("SLAG_BASICITY", "SLAG_T_BASICITY"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        keep.append(col)
    return keep


def evaluate(
    df: pd.DataFrame, features: list[str], *, chronological: bool, label: str
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    frame = df[["time", "unitcost_new", *features]].replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna(subset=["unitcost_new"]).sort_values("time")
    x = frame[features].fillna(frame[features].median(numeric_only=True)).fillna(0.0)
    y = frame["unitcost_new"]

    if chronological:
        cut = int(len(frame) * (1 - TEST_FRACTION))
        tr, te = slice(0, cut), slice(cut, None)
        x_tr, x_te, y_tr, y_te = x.iloc[tr], x.iloc[te], y.iloc[tr], y.iloc[te]
    else:
        from sklearn.model_selection import train_test_split

        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=TEST_FRACTION, shuffle=True, random_state=RANDOM_STATE
        )

    scaler = StandardScaler().fit(x_tr)
    model = XGBRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=4, subsample=0.75,
        colsample_bytree=0.6, min_child_weight=8, reg_alpha=3.0, reg_lambda=7.0,
        random_state=RANDOM_STATE, eval_metric="rmse", early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(
        scaler.transform(x_tr), y_tr,
        eval_set=[(scaler.transform(x_te), y_te)], verbose=False,
    )
    pred = model.predict(scaler.transform(x_te))
    result = {
        "R2": float(r2_score(y_te, pred)),
        "MAE": float(mean_absolute_error(y_te, pred)),
        "n_features": len(features),
        "n_test": len(y_te),
    }
    print(f"  {label:38s} feats {result['n_features']:4d}  "
          f"MAE {result['MAE']:7.1f}  R2 {result['R2']:+7.4f}")
    return result | {"model": model, "features": features, "x": x, "y": y}


def main() -> None:
    print("loading and filtering...")
    df = load_base()
    print(f"  rows after cruising filters: {len(df)}   "
          f"{df['time'].min().date()} -> {df['time'].max().date()}")

    df, strength_cols = add_coke_strength(df)
    df, energy_cols = add_engineered(df)

    base = [c for c in base_features(df) if c not in strength_cols + energy_cols]
    print(f"  base features {len(base)}, coke strength {len(strength_cols)}, "
          f"energy {len(energy_cols)}")

    banner("1. THE V4 SPLIT BUG - same features, two splits")
    print("  Cell 35 says 'Chronological Split (shuffle=False)' and does the")
    print("  opposite. On hourly data neighbouring rows straddle the fence.")
    shuffled = evaluate(df, base, chronological=False, label="baseline, RANDOM split")
    chrono = evaluate(df, base, chronological=True, label="baseline, chronological")
    print(f"\n  The random split reports R2 {shuffled['R2']:+.4f}; held to a real "
          f"forward test it is {chrono['R2']:+.4f}.")
    print(f"  Gap: {shuffled['R2'] - chrono['R2']:+.4f} R2. That is the leakage,")
    print("  and every V4 number should be read with it in mind.")

    banner("2. WHAT THE NEW FEATURES ADD (all chronological)")
    runs = {"baseline": chrono}
    if strength_cols:
        runs["+ coke strength"] = evaluate(
            df, base + strength_cols, chronological=True, label="+ coke strength")
    runs["+ energy terms"] = evaluate(
        df, base + energy_cols, chronological=True, label="+ energy terms")
    runs["+ both"] = evaluate(
        df, base + strength_cols + energy_cols, chronological=True, label="+ both")

    banner("3. SUMMARY")
    best_name = max(runs, key=lambda k: runs[k]["R2"])
    print(f"  {'variant':22s} {'R2':>9s} {'MAE':>9s} {'vs baseline':>13s}")
    for name, r in runs.items():
        print(f"  {name:22s} {r['R2']:+9.4f} {r['MAE']:9.1f} "
              f"{r['R2'] - chrono['R2']:+13.4f}")
    print(f"\n  best: {best_name}")

    banner("4. DID THE PHYSICS PRIOR GET USED?")
    best = runs[best_name]
    importances = pd.Series(
        best["model"].feature_importances_, index=best["features"]
    ).sort_values(ascending=False)
    print("  top 20 features by gain:")
    for rank, (name, gain) in enumerate(importances.head(20).items(), 1):
        tag = ""
        if name in energy_cols:
            tag = "  <- energy"
        elif name in strength_cols:
            tag = "  <- coke strength"
        print(f"    {rank:2d}. {name:44s} {gain:.4f}{tag}")
    new_share = importances[
        [c for c in strength_cols + energy_cols if c in importances.index]
    ].sum()
    print(f"\n  new features carry {new_share:.1%} of total gain")

    banner("5. WHY R2 IS ZERO, AND WHAT ACTUALLY HELPS")
    frame = df[["time", "unitcost_new"]].dropna().sort_values("time")
    cut = int(len(frame) * (1 - TEST_FRACTION))
    tr_y, te_y = frame["unitcost_new"].iloc[:cut], frame["unitcost_new"].iloc[cut:]
    print(f"  target level  train {tr_y.mean():8.0f}  test {te_y.mean():8.0f}"
          f"   shift {te_y.mean() - tr_y.mean():+.0f} Rs/tHM")
    print(f"  target spread train sd {tr_y.std():6.0f}  test sd {te_y.std():6.0f}")
    print(f"  MAE {chrono['MAE']:.0f} on a level of {te_y.mean():,.0f} is "
          f"{chrono['MAE']/te_y.mean():.1%} - the LEVEL is roughly right.")
    print("  R2 near zero means none of the test period's VARIANCE is explained.")
    print("  That is the same finding as the earlier fuel work: the plant holds")
    print("  fuel nearly constant, so there is little real variance to predict.")

    print("\n  Four reframings, all chronological:")
    # (a) physics only - does the prior carry signal without 125 competitors?
    evaluate(df, energy_cols, chronological=True, label="(a) energy features ONLY")
    # (b) physics + coke strength only
    evaluate(df, energy_cols + strength_cols, chronological=True,
             label="(b) energy + coke strength only")
    # (c) top-30 by gain - fewer features shift less between regimes
    top30 = [c for c in importances.head(30).index]
    evaluate(df, top30, chronological=True, label="(c) top 30 features by gain")
    # (d) linear, which extrapolates where trees cannot
    _ridge(df, base + strength_cols + energy_cols, label="(d) Ridge, all features")

    print("\n  A tree cannot extrapolate past its training range, and with a")
    print(f"  {te_y.mean() - tr_y.mean():+.0f} Rs/tHM regime shift it is being asked to.")
    print("  Ridge is included because a linear model can.")


def _ridge(df: pd.DataFrame, features: list[str], *, label: str) -> None:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    frame = df[["time", "unitcost_new", *features]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["unitcost_new"]).sort_values("time")
    x = frame[features].fillna(frame[features].median(numeric_only=True)).fillna(0.0)
    y = frame["unitcost_new"]
    cut = int(len(frame) * (1 - TEST_FRACTION))
    scaler = StandardScaler().fit(x.iloc[:cut])
    model = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(
        scaler.transform(x.iloc[:cut]), y.iloc[:cut]
    )
    pred = model.predict(scaler.transform(x.iloc[cut:]))
    print(f"  {label:38s} feats {len(features):4d}  "
          f"MAE {mean_absolute_error(y.iloc[cut:], pred):7.1f}  "
          f"R2 {r2_score(y.iloc[cut:], pred):+7.4f}")


if __name__ == "__main__":
    main()
