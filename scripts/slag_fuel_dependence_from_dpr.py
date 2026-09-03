"""Slag -> fuel dependence from DPR daily data alone.

Run:  python scripts/slag_fuel_dependence_from_dpr.py

Why daily, and why masses. The hourly static dataset cannot answer this: coke
there is an operator SETPOINT that barely moves (306-311 kg/THM across eight
slag deciles) while the plant absorbs slag swings with PCI, so total fuel is
flat and there is nothing to regress. Daily DPR reports actual dispatched
tonnes, and section 7.1 of the findings doc is explicit that this has to be done
on MASSES - slag_rate and fuel_rate share the HM denominator, so regressing one
on the other manufactures a relationship.

    fuel MT/day  ~  hot metal t/day  +  slag MT/day  [+ controls]

The known problem with this source, from section 0.2: DPR under-reports coke by
~13% and nut coke by ~47% against the plant tags, and only correlates +0.53 with
them day to day. The doc therefore lists "DPR coke ~ slag, raw: +34.8" as an
estimate NOT to use.

That warning is about the LEVEL. Whether it also biases the SLOPE depends on the
structure of the error, which this script tests rather than assumes:

  * If the shortfall is proportional and unrelated to slag, it rescales the
    coefficient by ~0.87 and widens the interval. Recoverable.
  * If the shortfall correlates with slag or production, the slope is biased and
    no amount of care rescues it. Measured directly below.

Physics anchor: +22 kg coke per 100 kg slag. Charge-report benchmark from
section 2: +20.6 to +26.9 depending on specification, IV estimate +20.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from furnace_data.offline import fetch_offline_data  # noqa: E402

SEED = 42
PHYSICS_ANCHOR = 22.0


def load_dpr() -> pd.DataFrame:
    df = fetch_offline_data("dpr_data", time_range="full", query_type="raw")
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("date_time", "date", "time"):
            if col in df.columns:
                df = df.set_index(pd.to_datetime(df[col], errors="coerce"))
                break
    df = df.sort_index()

    num = lambda c: pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["coke_mt"] = num("coke_1_mt") + num("coke_2_mt")
    out["nut_coke_mt"] = num("nut_coke_1_mt") + num("nut_coke_2_mt")
    out["pci_mt"] = num("pci_mt")
    out["slag_mt"] = num("slag_generation_mt")
    out["hm_mt"] = num("total_hot_metal_mt")
    out["total_coke_mt"] = out["coke_mt"] + out["nut_coke_mt"]
    out["total_fuel_mt"] = out["total_coke_mt"] + out["pci_mt"]

    # A DPR day is only usable if it reports a plausible full day of operation.
    out = out[
        out["hm_mt"].between(1200, 3200)
        & out["slag_mt"].between(300, 1400)
        & out["coke_mt"].between(150, 1400)
        & out["pci_mt"].between(50, 900)
    ]
    out["coke_rate"] = out["coke_mt"] / out["hm_mt"] * 1000.0
    out["slag_rate"] = out["slag_mt"] / out["hm_mt"] * 1000.0
    out["fuel_rate"] = out["total_fuel_mt"] / out["hm_mt"] * 1000.0
    return out


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """Plain OLS with an intercept, returning coefficients and t-stats."""

    A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    yv = y.to_numpy(float)
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    dof = max(1, len(yv) - A.shape[1])
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return {
        "names": ["const", *X.columns],
        "beta": beta,
        "se": se,
        "t": beta / se,
        "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot else np.nan,
        "n": len(yv),
    }


def bootstrap_slag(y: pd.Series, X: pd.DataFrame, draws: int = 2000) -> tuple:
    rng = np.random.default_rng(SEED)
    idx = list(X.columns).index("slag_mt") + 1
    vals = []
    n = len(X)
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        vals.append(ols(y.iloc[pick], X.iloc[pick])["beta"][idx])
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    df = load_dpr()
    banner("0. DPR DAILY SAMPLE")
    print(f"  usable days: {len(df)}   {df.index.min().date()} -> {df.index.max().date()}")
    desc = df[["hm_mt", "slag_mt", "coke_mt", "pci_mt", "coke_rate", "slag_rate",
               "fuel_rate"]].describe(percentiles=[.05, .5, .95]).T
    print(desc[["mean", "5%", "50%", "95%"]].to_string(float_format=lambda v: f"{v:9.1f}"))
    print(
        "\n  sanity: section 0.2 puts DPR coke at 274.9 kg/THM against a tag 309.3,\n"
        "  and section 8 puts the real slag rate at 324-386 kg/THM."
    )

    banner("1. THE REGRESSION - masses, daily")
    specs = {
        "coke ~ HM + slag": ("coke_mt", ["hm_mt", "slag_mt"]),
        "total coke ~ HM + slag": ("total_coke_mt", ["hm_mt", "slag_mt"]),
        "total fuel ~ HM + slag": ("total_fuel_mt", ["hm_mt", "slag_mt"]),
        "coke ~ HM + slag + PCI": ("coke_mt", ["hm_mt", "slag_mt", "pci_mt"]),
    }
    rows = []
    for name, (target, regressors) in specs.items():
        data = df[[target, *regressors]].dropna()
        res = ols(data[target], data[regressors])
        i = res["names"].index("slag_mt")
        lo, hi = bootstrap_slag(data[target], data[regressors])
        rows.append(
            {
                "specification": name,
                "n": res["n"],
                "kg_per_100kg_slag": res["beta"][i] * 100.0,
                "t": res["t"][i],
                "boot_p5": lo * 100.0,
                "boot_p95": hi * 100.0,
                "r2": res["r2"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:9.2f}"))
    print(f"\n  physics anchor {PHYSICS_ANCHOR:+.0f}; charge-report benchmark +20.6 to +26.9 "
          "(IV +20.0)")
    print("  NOTE the PCI row: section 7.2 warns PCI is the operator's RESPONSE to")
    print("  slag, so controlling for it is over-adjustment, not a robustness check.")

    banner("2. IS THE DPR SHORTFALL CORRELATED WITH SLAG?")
    print(
        "  A proportional shortfall unrelated to slag only rescales the slope.\n"
        "  A shortfall that tracks slag or production biases it. Proxy the\n"
        "  shortfall by the implied coke RATE, which section 0.2 says DPR puts\n"
        "  ~11%% low, and see what it moves with.\n"
    )
    for col in ("slag_mt", "slag_rate", "hm_mt"):
        print(f"  corr(implied DPR coke rate, {col:10s}) = "
              f"{df['coke_rate'].corr(df[col]):+.3f}")
    print(
        "\n  If corr with slag is near zero the slope survives (attenuated ~13%);\n"
        "  if it is strongly negative the +34.8 in section 2 is that artifact."
    )

    banner("3. SPLIT-HALF STABILITY")
    print("  A coefficient that will not reproduce on two halves is not a finding.")
    half = len(df) // 2
    for label, part in (("first half", df.iloc[:half]), ("second half", df.iloc[half:])):
        data = part[["total_fuel_mt", "hm_mt", "slag_mt"]].dropna()
        res = ols(data["total_fuel_mt"], data[["hm_mt", "slag_mt"]])
        i = res["names"].index("slag_mt")
        print(f"  {label:12s} n={res['n']:4d}  total fuel: "
              f"{res['beta'][i] * 100:+7.2f} kg per 100 kg slag  (t={res['t'][i]:+.2f})")

    banner("4. DOES DAILY AGGREGATION RECOVER WHAT HOURLY LOST?")
    print("  Hourly static data gave -5.5 kg coke per 100 kg slag on raw deciles,")
    print("  because coke there is a near-constant setpoint. Same decile view here:")
    d = df.copy()
    d["decile"] = pd.qcut(d["slag_rate"], 5, labels=False, duplicates="drop")
    g = d.groupby("decile").agg(
        slag_rate=("slag_rate", "mean"),
        coke_rate=("coke_rate", "mean"),
        fuel_rate=("fuel_rate", "mean"),
        n=("coke_rate", "size"),
    )
    print(g.to_string(float_format=lambda v: f"{v:9.2f}"))
    lo, hi = g.iloc[0], g.iloc[-1]
    d_slag = hi["slag_rate"] - lo["slag_rate"]
    if d_slag:
        print(
            f"\n  across quintiles: slag {d_slag:+.0f} -> coke "
            f"{(hi['coke_rate'] - lo['coke_rate']) / d_slag * 100:+.1f}, total fuel "
            f"{(hi['fuel_rate'] - lo['fuel_rate']) / d_slag * 100:+.1f} kg per 100 kg slag"
        )


if __name__ == "__main__":
    main()
