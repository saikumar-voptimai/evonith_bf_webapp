"""Slag -> fuel dependence from charge reports, daily masses.

Run:  python scripts/slag_fuel_dependence_from_charge_reports.py

This reproduces the analysis behind the shipped coke-rate correction on a longer
window, using the one fuel source the findings doc trusts.

Source choice, from the doc:
  * COKE_CALC_MT in the static CSV correlates +0.16 with the coke actually
    dumped (section 0.1). Unusable.
  * DPR coke under-reports ~13% and correlates only +0.53 day to day
    (section 0.2). Measured here: its implied coke rate spans p5 226 to p95 407
    kg/THM against a real ~305-324 band, and HM + slag explain 1% of its
    variance. Unusable for a slope.
  * offline_feed.charge_data is one row per charge and is the reference the
    other two were judged against. Used here.
  * DPR slag_generation_mt IS usable (section 0.2) and is one of the two slag
    measures. The other is the Al2O3 tracer (section 0.4). Both are run,
    because they agree on level (ratio 0.98) but correlate only +0.59.

Everything is in MASSES per day. Section 7.1: slag_rate and fuel_rate share the
HM denominator, so regressing rates on rates manufactures a relationship.

Benchmarks: physics anchor +22 kg coke per 100 kg slag; section 2 got +20.6 to
+26.9 across specifications on 222 days, IV estimate +20.0 +/- 7.8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from furnace_data.offline import fetch_offline_data  # noqa: E402

SEED = 42
IST = "Asia/Kolkata"
STATIC = REPO / "src" / "assets" / "data" / "furnace_dataset.csv"
ASH_AL2O3_PCT = {"COKE": 26.38, "NUTCOKE": 26.81, "PCI": 28.27}
ASH_PCT_COL = {"COKE": "COKE_ASH%", "NUTCOKE": "NUTCOKE_ASH%", "PCI": "PCI_ASH%"}


def _to_ist_date(index: pd.DatetimeIndex) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.Series(idx.tz_convert(IST).date, index=index)


def daily_charge_masses() -> pd.DataFrame:
    """One row per IST day of charge-report tonnes."""

    df = fetch_offline_data("charge_data", time_range="full", query_type="raw")
    num = lambda cols: sum(  # noqa: E731
        pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        for c in cols
        if c in df.columns
    )
    out = pd.DataFrame(index=df.index)
    out["coke_mt"] = num(["coke_1_mt", "coke_2_mt"])
    out["nut_coke_mt"] = num(["nut_coke_1_mt", "nut_coke_2_mt"])
    # charge_data.pci_mt exists but is identically zero across all 44,728 rows:
    # PCI is injected at the tuyeres, so it never travels through the skips and
    # a charge report cannot see it. PCI has to come from DPR, where section 0.2
    # measured the ratio against the tags at 1.044 - the one DPR fuel column
    # that is usable.
    out["sinter_mt"] = num([f"sinter_{i}_mt" for i in range(1, 5)])
    out["ore_mt"] = num([f"ore_{i}_mt" for i in range(1, 13)])
    out["pellet_mt"] = num([f"pellet_{i}_mt" for i in range(1, 3)])
    out["flux_mt"] = num([f"flux_{i}_mt" for i in range(1, 4)])
    out["date"] = _to_ist_date(df.index)

    daily = out.groupby("date").sum(numeric_only=True)
    daily["charges"] = out.groupby("date").size()
    daily["total_coke_mt"] = daily["coke_mt"] + daily["nut_coke_mt"]
    daily["burden_mt"] = (
        daily["sinter_mt"] + daily["ore_mt"] + daily["pellet_mt"] + daily["flux_mt"]
    )
    daily.index = pd.to_datetime(daily.index)
    # Partial first/last days and reporting gaps: a real day is ~6.4 charges/hr.
    return daily[daily["charges"].between(120, 190)]


def daily_dpr() -> pd.DataFrame:
    df = fetch_offline_data("dpr_data", time_range="full", query_type="raw")
    out = pd.DataFrame(index=df.index)
    out["dpr_slag_mt"] = pd.to_numeric(df["slag_generation_mt"], errors="coerce")
    out["hm_mt"] = pd.to_numeric(df["total_hot_metal_mt"], errors="coerce")
    out["pci_mt"] = pd.to_numeric(df["pci_mt"], errors="coerce")
    out["date"] = _to_ist_date(df.index)
    daily = out.groupby("date").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index)
    return daily[daily["hm_mt"].between(1200, 3200)]


def daily_static() -> pd.DataFrame:
    """Daily means of chemistry and process controls from the hourly dataset."""

    df = pd.read_csv(STATIC)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    keep = {
        "SLAG_PCT_AL2O3": "slag_pct_al2o3",
        "SINTER_AL2O3%": "sinter_al2o3_pct",
        "ORE_AL2O3%": "ore_al2o3_pct",
        "PELLET_PCT_AL2O3": "pellet_al2o3_pct",
        "FLUX_AL2O3%": "flux_al2o3_pct",
        "COKE_ASH%": "coke_ash_pct",
        "NUTCOKE_ASH%": "nutcoke_ash_pct",
        "PCI_ASH%": "pci_ash_pct",
        "HOT BLAST VOLUMENM3/HR.": "blast_vol",
        "HOT BLAST TEMP.OC": "blast_temp",
        "O2 ENRICHMENT %": "o2_enrich",
        "STEAMKGS/HR.": "steam",
        "FURNACETOPGASANALYSISCO2ETACO": "eta_co",
        "TOTAL_COKE_PORTIONS": "coke_portions",
        "TOTAL_NON_COKE_PORTIONS": "non_coke_portions",
        "WEIGHTED_COKE_ANGLE": "coke_angle",
        "WEIGHTED_NON_COKE_ANGLE": "non_coke_angle",
        "CHEM_PCT_SI": "hm_si",
    }
    have = {k: v for k, v in keep.items() if k in df.columns}
    sub = df[["time", *have]].rename(columns=have)
    for col in have.values():
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    daily = sub.set_index("time").resample("1D").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index.date)
    return daily


def build() -> pd.DataFrame:
    df = daily_charge_masses().join(daily_dpr(), how="inner").join(
        daily_static(), how="inner"
    )
    # Coke + nut coke from the charge reports, PCI from DPR.
    df["total_fuel_mt"] = df["total_coke_mt"] + df["pci_mt"].fillna(0.0)

    # Al2O3 tracer slag: Al2O3 is inert, so everything charged reports to slag.
    burden_al2o3 = (
        df["sinter_mt"] * df["sinter_al2o3_pct"]
        + df["ore_mt"] * df["ore_al2o3_pct"]
        + df["pellet_mt"] * df["pellet_al2o3_pct"]
        + df["flux_mt"] * df["flux_al2o3_pct"]
    ) / 100.0
    ash_al2o3 = (
        df["coke_mt"] * df["coke_ash_pct"] / 100.0 * ASH_AL2O3_PCT["COKE"] / 100.0
        + df["nut_coke_mt"]
        * df["nutcoke_ash_pct"]
        / 100.0
        * ASH_AL2O3_PCT["NUTCOKE"]
        / 100.0
        + df["pci_mt"] * df["pci_ash_pct"] / 100.0 * ASH_AL2O3_PCT["PCI"] / 100.0
    )
    frac = df["slag_pct_al2o3"].replace(0, np.nan) / 100.0
    df["tracer_slag_mt"] = (burden_al2o3 + ash_al2o3) / frac
    # Burden-only tracer shares NO input with the fuel masses, so it is the
    # instrument for the 2SLS in section 2's shared-error check.
    df["burden_only_slag_mt"] = burden_al2o3 / frac

    df["coke_rate"] = df["coke_mt"] / df["hm_mt"] * 1000.0
    df["fuel_rate"] = df["total_fuel_mt"] / df["hm_mt"] * 1000.0
    df["dpr_slag_rate"] = df["dpr_slag_mt"] / df["hm_mt"] * 1000.0
    df["tracer_slag_rate"] = df["tracer_slag_mt"] / df["hm_mt"] * 1000.0
    return df.replace([np.inf, -np.inf], np.nan)


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    yv = y.to_numpy(float)
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    dof = max(1, len(yv) - A.shape[1])
    sigma2 = float(resid @ resid) / dof
    var = np.diag(sigma2 * np.linalg.pinv(A.T @ A))
    se = np.sqrt(np.clip(var, 0.0, None))
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return {
        "names": ["const", *X.columns],
        "beta": beta,
        "se": se,
        "t": np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0),
        "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot else np.nan,
        "n": len(yv),
    }


def slag_coef(y: pd.Series, X: pd.DataFrame, slag_col: str) -> tuple[float, float, float]:
    res = ols(y, X)
    i = res["names"].index(slag_col)
    return res["beta"][i] * 100.0, res["t"][i], res["r2"]


def bootstrap(y: pd.Series, X: pd.DataFrame, slag_col: str, draws: int = 2000):
    rng = np.random.default_rng(SEED)
    i = list(X.columns).index(slag_col) + 1
    n = len(X)
    vals = [
        ols(y.iloc[p], X.iloc[p])["beta"][i]
        for p in (rng.integers(0, n, n) for _ in range(draws))
    ]
    return float(np.percentile(vals, 5)) * 100.0, float(np.percentile(vals, 95)) * 100.0


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


CONTROLS = {
    "1. hot metal only": [],
    "2. + blast": ["blast_vol", "blast_temp", "o2_enrich", "steam"],
    "3. + burden distribution": [
        "blast_vol", "blast_temp", "o2_enrich", "steam",
        "coke_portions", "non_coke_portions", "coke_angle", "non_coke_angle",
    ],
    "4. + Si": [
        "blast_vol", "blast_temp", "o2_enrich", "steam",
        "coke_portions", "non_coke_portions", "coke_angle", "non_coke_angle",
        "hm_si",
    ],
    "5. + eta CO": [
        "blast_vol", "blast_temp", "o2_enrich", "steam",
        "coke_portions", "non_coke_portions", "coke_angle", "non_coke_angle",
        "hm_si", "eta_co",
    ],
}


def main() -> None:
    df = build()
    banner("0. SAMPLE")
    print(f"  days after joining charge reports + DPR + static: {len(df)}")
    print(f"  {df.index.min().date()} -> {df.index.max().date()}")
    cols = ["charges", "hm_mt", "coke_mt", "total_fuel_mt", "dpr_slag_mt",
            "tracer_slag_mt", "coke_rate", "fuel_rate", "dpr_slag_rate",
            "tracer_slag_rate"]
    print(
        df[cols].describe(percentiles=[.05, .5, .95])
        .T[["mean", "5%", "50%", "95%"]]
        .to_string(float_format=lambda v: f"{v:9.1f}")
    )
    print(
        "\n  sanity vs the doc: coke rate should sit near 305-324 (section 0.3),\n"
        "  slag rate 324-386 (section 8), fuel rate 530-563 (section 8)."
    )
    print(
        f"  corr(DPR slag, tracer slag) = "
        f"{df['dpr_slag_mt'].corr(df['tracer_slag_mt']):+.3f}   "
        f"ratio mean = {(df['tracer_slag_mt'] / df['dpr_slag_mt']).mean():.3f}"
        "   (doc: +0.59, 0.980)"
    )

    for slag_col, label in (
        ("tracer_slag_mt", "Al2O3 TRACER SLAG"),
        ("dpr_slag_mt", "DPR MEASURED SLAG"),
    ):
        for target, tname in (
            ("total_coke_mt", "total coke (coke + nut coke, charge reports)"),
            ("total_fuel_mt", "total fuel (+ PCI from DPR)"),
        ):
            banner(f"{label}  ->  {tname} MT/day")
            rows = []
            for name, controls in CONTROLS.items():
                regressors = ["hm_mt", slag_col, *controls]
                data = df[[target, *regressors]].dropna()
                if len(data) < 60:
                    continue
                coef, t, r2 = slag_coef(data[target], data[regressors], slag_col)
                lo, hi = bootstrap(data[target], data[regressors], slag_col)
                rows.append(
                    {
                        "specification": name,
                        "n": len(data),
                        "kg_per_100kg": coef,
                        "t": t,
                        "boot_p5": lo,
                        "boot_p95": hi,
                        "r2": r2,
                    }
                )
            print(
                pd.DataFrame(rows).to_string(
                    index=False, float_format=lambda v: f"{v:9.2f}"
                )
            )

    banner("2SLS - shared-error check (section 2)")
    print(
        "  The tracer includes coke ash, which comes from the same charge masses\n"
        "  as the fuel figure, so a mis-recorded coke day moves both sides.\n"
        "  Instrument total slag with BURDEN-ONLY slag, which shares no input."
    )
    controls = CONTROLS["5. + eta CO"]
    cols = ["total_fuel_mt", "hm_mt", "tracer_slag_mt", "burden_only_slag_mt", *controls]
    data = df[cols].dropna()
    first = ols(data["tracer_slag_mt"], data[["burden_only_slag_mt", "hm_mt", *controls]])
    fitted = np.column_stack(
        [np.ones(len(data)), data[["burden_only_slag_mt", "hm_mt", *controls]].to_numpy(float)]
    ) @ first["beta"]
    iv_frame = data[["hm_mt", *controls]].copy()
    iv_frame.insert(1, "slag_hat", fitted)
    iv = ols(data["total_fuel_mt"], iv_frame)
    j = iv["names"].index("slag_hat")
    k = first["names"].index("burden_only_slag_mt")
    ols_coef, ols_t, _ = slag_coef(
        data["total_fuel_mt"], data[["hm_mt", "tracer_slag_mt", *controls]], "tracer_slag_mt"
    )
    print(f"\n  n = {len(data)}")
    print(f"  stage 1  d(total slag)/d(burden-only slag) = {first['beta'][k]:+.4f} "
          f"(t={first['t'][k]:+.1f})     doc: 0.9923, t=+51.5")
    print(f"  IV estimate                                = {iv['beta'][j] * 100:+.1f} "
          f"kg per 100 kg slag (t={iv['t'][j]:+.2f})   doc: +20.0")
    print(f"  OLS for comparison                         = {ols_coef:+.1f} "
          f"(t={ols_t:+.2f})                        doc: +21.8")

    banner("SPLIT-HALF STABILITY")
    half = len(df) // 2
    for slag_col in ("tracer_slag_mt", "dpr_slag_mt"):
        line = []
        for lbl, part in (("H1", df.iloc[:half]), ("H2", df.iloc[half:])):
            cols = ["total_fuel_mt", "hm_mt", slag_col, *CONTROLS["2. + blast"]]
            data = part[cols].dropna()
            coef, t, _ = slag_coef(
                data["total_fuel_mt"], data[cols[1:]], slag_col
            )
            line.append(f"{lbl} {coef:+7.1f} (t={t:+5.2f}, n={len(data)})")
        print(f"  {slag_col:20s}  " + "   ".join(line))

    banner("VERDICT INPUTS")
    print(
        "  Shipped coefficient: 22 kg coke per 100 kg slag\n"
        "  (bmo.coke_rate_correction.terms.slag_heat, pinned by\n"
        "   tests/test_bmo_coke_correction.py). Compare against the table above:\n"
        "  agreement across specifications AND across both slag measures AND\n"
        "  across both halves is what would justify keeping it. A coefficient\n"
        "  that only appears in one cell is noise."
    )


if __name__ == "__main__":
    main()
