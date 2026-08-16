"""All features, plus a diagnosis of why slag rate carries no weight.

Run:  python scripts/fuel_model_slag_diagnosis.py

The physics is not in doubt: more slag needs more heat, so more coke. Roughly
+22 kg coke per 100 kg slag (findings section 2). The question this answers is
why a model fitted on the plant record does not recover it.

Four candidate explanations are tested separately, because they call for
different responses:

  A. LEAKAGE dressed up as skill - "all features" includes columns that ARE the
     target. COKE_CALC_MT / HM is the coke rate. Including it gives R2 ~ 1.0 and
     means nothing. Measured, then excluded.

  B. REDUNDANCY - slag rate may already be implied by the other 140 features
     (burden masses, ore chemistry, ash). A feature that is 95% predictable from
     its neighbours cannot add anything, however real its physics.

  C. OVER-ADJUSTMENT - findings section 7.2: PCI is the operator's RESPONSE to
     slag, not an independent confounder. Controlling for it removes the slag
     effect by construction.

  D. NO RAW SIGNAL AT ALL - before blaming any model, look at the unconditional
     relationship: bin the record by slag rate and read off mean coke rate.
     If that is flat or wrong-signed, no feature set will rescue it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "assets" / "data" / "furnace_dataset.csv"
SEED = 42
ASH_AL2O3_PCT = {"COKE": 26.38, "NUTCOKE": 26.81, "PCI": 28.27}

TARGET = "COKE RATE KG/THM"

# Columns that ARE the answer. coke_rate = COKE_CALC_MT/HM*1000, unit cost is a
# fixed function of coke + PCI rate, total fuel contains coke. Leaving any of
# these in produces a beautiful, worthless model.
LEAKY = {
    "COKE RATE KG/THM",
    "ACT. FUEL RATEKG/THM.",
    "UNITCOST LAKHS/THM",
    "COKE_CALC_MT",
    "NUTCOKE_CALC_MT",
    "PCI_CALC_MT",
    "slag_mt",
    "hot_metal_mt",
}
# The operator's co-decision. Not leakage, but see explanation C.
OPERATOR_RESPONSE = {"PCI_KG/THM"}
SLAG_FEATURES = ["slag_rate_kg_per_thm", "SLAG_BASICITY", "SLAG_T_BASICITY"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    burden = (
        df["SINTER_CALC_MT"].fillna(0) * df["SINTER_AL2O3%"].fillna(0)
        + df["ORE_CALC_MT"].fillna(0) * df["ORE_AL2O3%"].fillna(0)
        + df["TOTAL_PELLET_CALC_MT"].fillna(0) * df["PELLET_PCT_AL2O3"].fillna(0)
        + df["FLUX_CALC_MT"].fillna(0) * df["FLUX_AL2O3%"].fillna(0)
    ) / 100.0
    ash = sum(
        df[f"{f}_CALC_MT"].fillna(0) * df[f"{f}_ASH%"].fillna(0) / 100.0 * p / 100.0
        for f, p in ASH_AL2O3_PCT.items()
    )
    df["slag_mt"] = (burden + ash) / (df["SLAG_PCT_AL2O3"].replace(0, np.nan) / 100.0)
    df["hot_metal_mt"] = df["PRODUCTIONTONNESPERHR"]
    df["slag_rate_kg_per_thm"] = df["slag_mt"] / df["hot_metal_mt"] * 1000.0
    return df


def numeric_features(df: pd.DataFrame, *, drop: set[str]) -> list[str]:
    feats = []
    for col in df.columns:
        if col == "time" or col in drop:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < len(df) * 0.6 or s.std(skipna=True) in (0.0, np.nan):
            continue
        feats.append(col)
    return feats


def frame(df: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    cols = list(dict.fromkeys([*feats, target]))
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    if "slag_rate_kg_per_thm" in out.columns:
        out = out[out["slag_rate_kg_per_thm"].between(150, 700)]
    return out.dropna()


def holdout_r2(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    cut = int(len(X) * 0.75)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    ridge.fit(Xtr, ytr)
    gbm = HistGradientBoostingRegressor(random_state=SEED, max_iter=300)
    gbm.fit(Xtr, ytr)
    return (
        float(r2_score(yte, ridge.predict(Xte))),
        float(r2_score(yte, gbm.predict(Xte))),
    )


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    df = load()
    all_feats = numeric_features(df, drop=LEAKY)
    print(f"usable numeric features after removing leakage: {len(all_feats)}")

    # --- A. what leakage looks like -----------------------------------------
    banner("A. LEAKAGE CHECK - why 'all columns' cannot be taken literally")
    leaky_feats = numeric_features(df, drop={TARGET})
    d = frame(df, leaky_feats, TARGET)
    r_ridge, r_gbm = holdout_r2(d[leaky_feats], d[TARGET])
    print(f"  every column incl. COKE_CALC_MT : ridge {r_ridge:6.3f}  gbm {r_gbm:6.3f}")
    d = frame(df, all_feats, TARGET)
    r_ridge, r_gbm = holdout_r2(d[all_feats], d[TARGET])
    print(f"  leakage columns removed         : ridge {r_ridge:6.3f}  gbm {r_gbm:6.3f}")
    print("  The first number is coke mass divided by HM. It is not a prediction.")

    # --- all features, with and without slag --------------------------------
    banner("ALL FEATURES - does slag add anything on top?")
    rows = []
    for name, feats in {
        "all (no slag)": [f for f in all_feats if f not in SLAG_FEATURES],
        "all + slag": list(dict.fromkeys(all_feats + SLAG_FEATURES)),
        "all + slag, no PCI": [
            f
            for f in dict.fromkeys(all_feats + SLAG_FEATURES)
            if f not in OPERATOR_RESPONSE
        ],
    }.items():
        d = frame(df, feats, TARGET)
        r_ridge, r_gbm = holdout_r2(d[feats], d[TARGET])
        rows.append(
            {"features": name, "n_feat": len(feats), "n": len(d),
             "ridge_r2": r_ridge, "gbm_r2": r_gbm}
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    # --- D. the raw, unconditional relationship ------------------------------
    banner("D. RAW SIGNAL - mean coke rate by slag-rate decile")
    print("  No model, no controls. If the physics is visible anywhere, here.")
    d = frame(df, SLAG_FEATURES + ["PCI_KG/THM"], TARGET)
    d["decile"] = pd.qcut(d["slag_rate_kg_per_thm"], 10, labels=False, duplicates="drop")
    g = d.groupby("decile").agg(
        slag_rate=("slag_rate_kg_per_thm", "mean"),
        coke_rate=(TARGET, "mean"),
        pci_rate=("PCI_KG/THM", "mean"),
        n=(TARGET, "size"),
    )
    g["coke_plus_pci"] = g["coke_rate"] + 0.53 * g["pci_rate"]
    print(g.to_string(float_format=lambda v: f"{v:9.2f}"))
    lo, hi = g.iloc[0], g.iloc[-1]
    d_slag = hi["slag_rate"] - lo["slag_rate"]
    print(
        f"\n  across the range: slag {d_slag:+.0f} kg/THM  ->  "
        f"coke {hi['coke_rate'] - lo['coke_rate']:+.1f}, "
        f"PCI {hi['pci_rate'] - lo['pci_rate']:+.1f}, "
        f"coke+0.53*PCI {hi['coke_plus_pci'] - lo['coke_plus_pci']:+.1f} kg/THM"
    )
    if d_slag:
        print(
            f"  implied: {(hi['coke_rate'] - lo['coke_rate']) / d_slag * 100:+.1f} kg coke "
            f"and {(hi['coke_plus_pci'] - lo['coke_plus_pci']) / d_slag * 100:+.1f} kg "
            "total fuel per 100 kg slag   (physics: +22)"
        )

    # --- B. redundancy --------------------------------------------------------
    banner("B. REDUNDANCY - is slag rate already implied by the other features?")
    others = [f for f in all_feats if f not in SLAG_FEATURES]
    d = frame(df, others + ["slag_rate_kg_per_thm"], TARGET)
    cut = int(len(d) * 0.75)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    ridge.fit(d[others][:cut], d["slag_rate_kg_per_thm"][:cut])
    red = r2_score(
        d["slag_rate_kg_per_thm"][cut:], ridge.predict(d[others][cut:])
    )
    print(f"  R2 of slag rate predicted from the other {len(others)} features: {red:.3f}")
    print("  The closer to 1.0, the less independent information it can carry.")

    # --- C. over-adjustment ---------------------------------------------------
    banner("C. OVER-ADJUSTMENT - PCI is the operator's response to slag")
    d = frame(df, ["slag_rate_kg_per_thm", "PCI_KG/THM"], TARGET)
    raw = np.polyfit(d["slag_rate_kg_per_thm"], d[TARGET], 1)[0] * 100.0
    resid_y = d[TARGET] - LinearRegression().fit(
        d[["PCI_KG/THM"]], d[TARGET]
    ).predict(d[["PCI_KG/THM"]])
    resid_x = d["slag_rate_kg_per_thm"] - LinearRegression().fit(
        d[["PCI_KG/THM"]], d["slag_rate_kg_per_thm"]
    ).predict(d[["PCI_KG/THM"]])
    adj = np.polyfit(resid_x, resid_y, 1)[0] * 100.0
    print(f"  slag -> coke, raw                  : {raw:+7.1f} kg per 100 kg slag")
    print(f"  slag -> coke, controlling for PCI  : {adj:+7.1f} kg per 100 kg slag")
    print(f"  corr(slag rate, PCI rate)          : "
          f"{d['slag_rate_kg_per_thm'].corr(d['PCI_KG/THM']):+.3f}")
    print("  physics anchor                     :   +22.0")


if __name__ == "__main__":
    main()
