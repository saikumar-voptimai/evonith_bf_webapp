"""Does adding slag rate / basicity / T-basicity improve the fuel model?

Run:  python scripts/fuel_model_with_slag_features.py

The question is whether the slag quantities BMO can now compute from a candidate
blend - slag rate, CaO/SiO2, (CaO+MgO)/SiO2 - let a model predict fuel cost or
coke rate better than the current burden-blind one.

This script does not just report an R2, because on this dataset a naive R2 would
be badly misleading in three separate ways. Each is measured explicitly:

1. LEAKAGE FROM RANDOM CV.  The data is hourly and strongly autocorrelated -
   adjacent rows are nearly the same furnace state. Random K-fold puts hour t-1
   in train and hour t in test, so the model can score well by near-memorisation.
   Every model is therefore scored BOTH ways: random K-fold and a time-ordered
   holdout. The gap between them is the leakage.

2. SHARED-DENOMINATOR ARTIFACT.  slag_rate = slag/HM and coke_rate = coke/HM and
   unit cost is also per THM. Regressing one on the other partly measures 1/HM
   against itself. Section 7.1 of docs/bmo_fuel_slag_si_findings.md says plainly:
   work in masses. So the same comparison is run on a mass basis, and a PLACEBO
   feature - a shuffled slag mass divided by the real HM - is included to size
   how much "explanatory power" the denominator alone contributes.

3. CLOSED-LOOP BIAS.  Section 7: operators cut fuel when the furnace runs hot,
   so contemporaneous fuel-vs-thermal correlations carry the control law, not the
   physics, and come out wrong-signed. A model fitted on that will predict the
   historical record well and still give BMO the wrong answer when it asks "what
   if I change the burden". The fitted slag coefficient is therefore printed with
   its sign and compared against the physics anchor of +22 kg coke / 100 kg slag.

Read the three diagnostics before reading the headline R2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "assets" / "data" / "furnace_dataset.csv"
SEED = 42

# Ash oxide assumptions, from bmo.fuel_ash_inputs in setting_bmo.yml. Only the
# Al2O3 fraction of the ash matters for the tracer.
ASH_AL2O3_PCT = {"COKE": 26.38, "NUTCOKE": 26.81, "PCI": 28.27}

TARGETS = {
    "unit_cost_lakhs_per_thm": "UNITCOST LAKHS/THM",
    "coke_rate_kg_per_thm": "COKE RATE KG/THM",
    "total_fuel_rate_kg_per_thm": "ACT. FUEL RATEKG/THM.",
}

# Burden-blind-ish baseline: blast, top gas, burden distribution, raw material
# quality. Deliberately excludes anything derived from slag.
BASE_FEATURES = [
    "HOT BLAST VOLUMENM3/HR.",
    "HOT BLAST TEMP.OC",
    "HOT BLAST PRESSUREBAR",
    "O2 ENRICHMENT %",
    "STEAMKGS/HR.",
    "TOPPRESSUREBAR",
    "FURNACETOPGASANALYSISCO2ETACO",
    "PRODUCTIONTONNESPERHR",
    "PCI_KG/THM",
    "COKE_ASH%",
    "PCI_ASH%",
    "SINTER_AL2O3%",
    "ORE_AL2O3%",
    "TOTAL_COKE_PORTIONS",
    "TOTAL_NON_COKE_PORTIONS",
    "WEIGHTED_COKE_ANGLE",
    "WEIGHTED_NON_COKE_ANGLE",
]

SLAG_FEATURES = ["slag_rate_kg_per_thm", "SLAG_BASICITY", "SLAG_T_BASICITY"]


def load() -> pd.DataFrame:
    if not DATA.exists():
        sys.exit(f"dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)


def add_slag_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Slag tonnage per THM via the Al2O3 tracer.

    Al2O3 is inert: all of it charged reports to slag. So
        slag_MT = (Al2O3 charged) / (SLAG_PCT_AL2O3 / 100)
    which is the method used in section 0.4 of the findings doc. It is preferred
    over DPR slag_generation_mt there, but note the two only correlate +0.59, so
    the absolute level is better trusted than the day-to-day movement.
    """

    out = df.copy()
    burden_al2o3 = (
        out["SINTER_CALC_MT"].fillna(0) * out["SINTER_AL2O3%"].fillna(0)
        + out["ORE_CALC_MT"].fillna(0) * out["ORE_AL2O3%"].fillna(0)
        + out["TOTAL_PELLET_CALC_MT"].fillna(0) * out["PELLET_PCT_AL2O3"].fillna(0)
        + out["FLUX_CALC_MT"].fillna(0) * out["FLUX_AL2O3%"].fillna(0)
    ) / 100.0

    ash_al2o3 = sum(
        out[f"{fuel}_CALC_MT"].fillna(0)
        * out[f"{fuel}_ASH%"].fillna(0)
        / 100.0
        * pct
        / 100.0
        for fuel, pct in ASH_AL2O3_PCT.items()
    )

    slag_pct_al2o3 = out["SLAG_PCT_AL2O3"].replace(0, np.nan)
    out["slag_mt"] = (burden_al2o3 + ash_al2o3) / (slag_pct_al2o3 / 100.0)
    out["hot_metal_mt"] = out["PRODUCTIONTONNESPERHR"]
    out["slag_rate_kg_per_thm"] = out["slag_mt"] / out["hot_metal_mt"] * 1000.0

    # Placebo: same denominator, physically meaningless numerator. Any R2 this
    # buys is pure shared-denominator artifact.
    rng = np.random.default_rng(SEED)
    out["placebo_rate"] = (
        rng.permutation(out["slag_mt"].to_numpy()) / out["hot_metal_mt"] * 1000.0
    )
    return out


def clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[["time", *dict.fromkeys(cols)]].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name != "time" else s
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    # Drop slag rates outside anything the furnace has plausibly done. Section 8
    # puts the observed envelope at ~324-386 kg/THM; allow a wide margin and cut
    # only the arithmetic blow-ups from a near-zero SLAG_PCT_AL2O3 or HM.
    if "slag_rate_kg_per_thm" in out.columns:
        out = out[out["slag_rate_kg_per_thm"].between(150.0, 700.0)]
    return out


def _models() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "gbm": HistGradientBoostingRegressor(random_state=SEED, max_iter=300),
    }


def score(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, float]]:
    """R2 two ways: random K-fold, and a time-ordered 75/25 holdout."""

    results: dict[str, dict[str, float]] = {}
    split = int(len(X) * 0.75)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    for name, model in _models().items():
        kf_scores = []
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
            m = _models()[name]
            m.fit(X.iloc[tr], y.iloc[tr])
            kf_scores.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
        model.fit(Xtr, ytr)
        results[name] = {
            "random_kfold_r2": float(np.mean(kf_scores)),
            "time_holdout_r2": float(r2_score(yte, model.predict(Xte))),
        }
    return results


def ridge_slag_coefficient(X: pd.DataFrame, y: pd.Series) -> float | None:
    """Standardised slag-rate coefficient, for a sign check against physics."""

    if "slag_rate_kg_per_thm" not in X.columns:
        return None
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    pipe.fit(X, y)
    idx = list(X.columns).index("slag_rate_kg_per_thm")
    coef = pipe[-1].coef_[idx]
    return float(coef / X["slag_rate_kg_per_thm"].std())


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    df = add_slag_rate(load())

    banner("0. SLAG RATE FROM THE Al2O3 TRACER")
    s = df["slag_rate_kg_per_thm"].replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s.between(150, 700)]
    print(f"  n = {len(s)}")
    print(
        f"  mean {s.mean():.1f}   p5 {s.quantile(.05):.1f}   "
        f"p50 {s.median():.1f}   p95 {s.quantile(.95):.1f} kg/THM"
    )
    print("  findings doc section 8 observed envelope: ~324 - 386 kg/THM")

    for label, target in TARGETS.items():
        banner(f"TARGET: {label}   ({target})")
        feature_sets = {
            "base": BASE_FEATURES,
            "base + slag": BASE_FEATURES + SLAG_FEATURES,
            "base + placebo": BASE_FEATURES + ["placebo_rate"],
            "slag only": SLAG_FEATURES,
        }
        rows = []
        for name, feats in feature_sets.items():
            data = clean(df, feats + [target])
            if len(data) < 300:
                print(f"  {name}: only {len(data)} usable rows, skipped")
                continue
            X, y = data[feats], data[target]
            res = score(X, y)
            for model, sc in res.items():
                rows.append(
                    {
                        "features": name,
                        "model": model,
                        "n": len(data),
                        "random_kfold_r2": sc["random_kfold_r2"],
                        "time_holdout_r2": sc["time_holdout_r2"],
                        "leakage": sc["random_kfold_r2"] - sc["time_holdout_r2"],
                    }
                )
        table = pd.DataFrame(rows)
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        data = clean(df, BASE_FEATURES + SLAG_FEATURES + [target])
        coef = ridge_slag_coefficient(data[BASE_FEATURES + SLAG_FEATURES], data[target])
        if coef is not None:
            print(f"\n  ridge slag-rate coefficient: {coef:+.4f} {label} per kg/THM slag")
            if label == "coke_rate_kg_per_thm":
                print(
                    f"    -> {coef * 100:+.1f} kg coke per 100 kg slag "
                    "(physics anchor: +22, findings section 2)"
                )

    banner("MASS BASIS - no shared denominator")
    print("  Section 7.1: fuel_rate and slag_rate share HM. Repeat on masses.")
    mass = df.copy()
    mass["coke_mt"] = mass["COKE RATE KG/THM"] * mass["hot_metal_mt"] / 1000.0
    mass["fuel_mt"] = mass["ACT. FUEL RATEKG/THM."] * mass["hot_metal_mt"] / 1000.0
    for target in ("coke_mt", "fuel_mt"):
        base = ["hot_metal_mt"] + BASE_FEATURES
        rows = []
        for name, feats in {
            "HM only": ["hot_metal_mt"],
            "HM + controls": base,
            "HM + controls + slag MT": base + ["slag_mt"],
            "HM + controls + placebo MT": base + ["placebo_rate"],
        }.items():
            data = clean(mass, feats + [target])
            if len(data) < 300:
                continue
            res = score(data[feats], data[target])
            rows.append(
                {
                    "target": target,
                    "features": name,
                    "n": len(data),
                    "gbm_time_holdout_r2": res["gbm"]["time_holdout_r2"],
                    "ridge_time_holdout_r2": res["ridge"]["time_holdout_r2"],
                }
            )
        print(
            pd.DataFrame(rows).to_string(
                index=False, float_format=lambda v: f"{v:8.3f}"
            )
        )

    banner("LAG SWEEP - does the signal live at burden-descent time?")
    print(
        "  Section 7: contemporaneous fuel-vs-thermal correlations carry the\n"
        "  operator's control law and come out wrong-signed; the causal sign\n"
        "  reappears at 4-9 h, peaking at 6-7 h, which is burden descent time.\n"
        "  If slag features are ever going to help, it is at a lag.\n"
    )
    target = TARGETS["coke_rate_kg_per_thm"]
    rows = []
    for lag in (0, 2, 4, 6, 7, 9, 12):
        lagged = df.copy()
        for col in SLAG_FEATURES:
            lagged[col] = lagged[col].shift(lag)
        data = clean(lagged, BASE_FEATURES + SLAG_FEATURES + [target])
        if len(data) < 300:
            continue
        X, y = data[BASE_FEATURES + SLAG_FEATURES], data[target]
        res = score(X, y)
        rows.append(
            {
                "lag_h": lag,
                "n": len(data),
                "ridge_time_holdout_r2": res["ridge"]["time_holdout_r2"],
                "kg_coke_per_100kg_slag": (ridge_slag_coefficient(X, y) or 0.0) * 100.0,
            }
        )
    print(
        pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.3f}")
    )
    print("\n  physics anchor: +22 kg coke per 100 kg slag (findings section 2)")

    banner("HOW TO READ THIS")
    print(
        "  * Compare 'base' with 'base + slag' on the TIME-HOLDOUT column only.\n"
        "    The random-K-fold column is inflated by autocorrelation; the\n"
        "    'leakage' column shows by how much.\n"
        "  * 'base + placebo' has a physically meaningless feature that shares the\n"
        "    HM denominator. Whatever it gains over 'base' is artifact, and the\n"
        "    real slag features have to beat THAT bar, not the 'base' bar.\n"
        "  * Check the sign of the slag coefficient. Physics says more slag costs\n"
        "    more coke (+22 per 100 kg). A negative fit is the control law\n"
        "    (section 7), and a model carrying it will tell BMO that leaner\n"
        "    burden is cheaper - the exact error the coke correction exists to\n"
        "    prevent.\n"
        "  * A model good at predicting the historical record is not necessarily\n"
        "    usable inside an optimiser that asks counterfactual questions."
    )


if __name__ == "__main__":
    main()
