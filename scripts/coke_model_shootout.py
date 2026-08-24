"""Which coke-rate model is better? Head to head against plant actuals.

Run:  python scripts/coke_model_shootout.py

WHY THIS EXISTS.

The Blend Optimizer currently shows TWO coke rates that disagree - one from the
ML fuel-cost model plus its physics correction, one from the energy balance.
Arguing from first principles about which deserves the operator's screen was
getting nowhere. This settles it on the plant's own record.

WHAT IS COMPARED.

  persistence     recent actual coke rate, carried forward. Not a model - the
                  baseline any model must beat to have earned its place.
  ml_cost         the shipped XGBoost fuel-cost model, scored on historical
                  rows, with coke backed out of the predicted unit cost at the
                  model's own baseline prices and the plant's actual PCI.
  energy_balance  the physics solve, with the rolling bias offset applied.

READ THE PERSISTENCE COLUMN FIRST. The LP path anchors on the recent ACTUAL
coke rate and adds a physics delta that is zero at current conditions. On a
backtest of days the plant actually ran, that is close to persistence by
construction, and persistence scores extremely well on an autocorrelated series.
A model that merely matches persistence has demonstrated nothing about blends it
has never seen - which is the one thing a blend optimiser needs.

RECENCY WEIGHTING. Metrics are reported three ways: whole record, last 30 days,
and exponentially weighted with a 30-day half-life. Operations drift, so the
recent window is the one that should carry the decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import energy_balance_phase0 as eb  # noqa: E402
from coke_rate_backtest import build_inputs, scores  # noqa: E402
from energy_balance_day_audit import daily_dust  # noqa: E402
from utils.energy_balance.constants import load_config  # noqa: E402
from utils.energy_balance.solve import solve_coke_rate_kg_per_thm  # noqa: E402

MODEL_DIR = REPO / "src" / "assets" / "models"
STATIC_CSV = REPO / "src" / "assets" / "data" / "furnace_dataset.csv"
PRICES = {"coke": 28.0, "nut_coke": 24.0, "pci": 18.0}
NUT_COKE_KG_THM = 70.0        # the app holds this fixed
HALF_LIFE_DAYS = 30.0
CALIBRATION_WINDOW = 90


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def weighted_scores(actual: pd.Series, predicted: pd.Series, weights: pd.Series):
    """Same metrics as scores(), but each day carries a weight."""

    err = predicted - actual
    w = weights / weights.sum()
    bias = float((w * err).sum())
    mae = float((w * err.abs()).sum())
    mape = float((w * (err.abs() / actual)).sum() * 100.0)
    mean = float((w * actual).sum())
    ss_res = float((w * err**2).sum())
    ss_tot = float((w * (actual - mean) ** 2).sum())
    return {"n": float(len(actual)), "bias": bias, "MAE": mae,
            "RMSE": float(np.sqrt(ss_res)), "MAPE%": mape,
            "R2": 1.0 - ss_res / ss_tot if ss_tot else float("nan")}


def show(label: str, s: dict) -> None:
    print(f"  {label:20s} n={s['n']:4.0f}  bias {s['bias']:+7.1f}  MAE {s['MAE']:6.1f}"
          f"  RMSE {s['RMSE']:6.1f}  MAPE {s['MAPE%']:5.2f}%  R2 {s['R2']:+6.3f}")


def ml_coke_from_static(index: pd.DatetimeIndex) -> pd.Series:
    """Score the shipped fuel-cost model and back a coke rate out of it.

    The model's target is unit cost, not coke. The app derives coke by holding
    nut coke at 70 kg/tHM and PCI at its current value, then dividing what is
    left by the coke price - so the same inversion is applied here.
    """

    try:
        import joblib
        import xgboost as xgb
    except ImportError as exc:  # noqa: BLE001
        print(f"  ML model unavailable ({exc}); skipping that column")
        return pd.Series(dtype=float)

    cols = json.loads(
        (MODEL_DIR / "unitcost_fuel_feature_columns.json").read_text(encoding="utf-8")
    )
    raw = pd.read_csv(STATIC_CSV)
    time_col = next((c for c in raw.columns if c.lower() in ("time", "date_time")), None)
    if time_col is None:
        print("  static dataset has no time column; skipping ML column")
        return pd.Series(dtype=float)
    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).set_index(time_col)

    missing = [c for c in cols if c not in raw.columns]
    if missing:
        print(f"  {len(missing)}/{len(cols)} model features absent from the static "
              f"dataset (e.g. {missing[:3]}); filling with column medians")
    frame = pd.DataFrame(index=raw.index)
    for c in cols:
        frame[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else np.nan
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)

    scaler = joblib.load(MODEL_DIR / "unitcost_fuel_scaler.joblib")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "unitcost_fuel_model.json"))
    scaled = scaler.transform(frame.to_numpy(dtype=float))
    unit_cost = booster.predict(xgb.DMatrix(scaled, feature_names=list(cols)))

    pci = pd.to_numeric(raw.get("PCI_KG/THM"), errors="coerce")
    out = pd.Series(unit_cost, index=raw.index, name="ml_unit_cost")
    # Some bundles carry cost in lakhs/tHM; rescale if the magnitude says so.
    if float(np.nanmedian(out)) < 100.0:
        out = out * 1000.0
    coke = (out - NUT_COKE_KG_THM * PRICES["nut_coke"] - pci * PRICES["pci"]) / PRICES[
        "coke"
    ]
    daily = coke.resample("1D").mean()
    daily.index = pd.to_datetime(daily.index.date)
    return daily.reindex(index)


def main() -> None:
    print("building daily history and solving the energy balance...")
    df = eb.build().join(daily_dust(), how="left").sort_index()
    cfg = load_config()

    solved = []
    for _, row in df.iterrows():
        try:
            solved.append(solve_coke_rate_kg_per_thm(build_inputs(row, "stave"), cfg))
        except Exception:  # noqa: BLE001
            solved.append(np.nan)
    df["eb_raw"] = solved

    # Rolling, strictly causal offset - the shipped correction.
    offsets = (df["eb_raw"] - df["coke_rate"]).rolling(
        CALIBRATION_WINDOW, min_periods=20
    ).mean().shift(1)
    df["energy_balance"] = df["eb_raw"] - offsets

    # Persistence: the previous 7 days of ACTUAL, which is what the LP path
    # anchors on before adding its physics delta.
    df["persistence"] = df["coke_rate"].shift(1).rolling(7, min_periods=3).mean()

    print("scoring the ML fuel-cost model on historical rows...")
    df["ml_cost"] = ml_coke_from_static(df.index)

    models = [m for m in ("persistence", "ml_cost", "energy_balance")
              if m in df and df[m].notna().sum() > 20]
    d = df[["coke_rate", *models]].replace([np.inf, -np.inf], np.nan).dropna()

    banner("0. SAMPLE")
    print(f"  days {len(d)}   {d.index.min().date()} -> {d.index.max().date()}")
    a = d["coke_rate"]
    print(f"  actual coke rate: mean {a.mean():.1f}  sd {a.std():.1f}  "
          f"last value {a.iloc[-1]:.1f} kg/tHM")

    banner("1. WHOLE RECORD")
    for m in models:
        show(m, scores(a, d[m]))

    banner("2. LAST 30 DAYS - the window that should decide")
    recent = d.tail(30)
    for m in models:
        show(m, scores(recent["coke_rate"], recent[m]))

    banner("3. RECENCY-WEIGHTED (30-day half-life over the whole record)")
    age = (d.index.max() - d.index).days.to_numpy(dtype=float)
    w = pd.Series(0.5 ** (age / HALF_LIFE_DAYS), index=d.index)
    for m in models:
        show(m, weighted_scores(a, d[m], w))

    banner("4. HOW MUCH OF EACH MODEL IS JUST PERSISTENCE?")
    print("  Correlation of each model's ERROR with persistence's error. High")
    print("  means the model is making the same mistakes as carrying yesterday")
    print("  forward, i.e. it is adding little of its own.")
    base_err = d["persistence"] - a
    for m in models:
        if m == "persistence":
            continue
        print(f"    {m:16s} corr(err, persistence err) = "
              f"{(d[m] - a).corr(base_err):+.3f}")

    banner("5. TREND PLOT")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(d.index, a, color="black", lw=2.0, label="actual", zorder=5)
        for m, colour in zip(models, ("tab:gray", "tab:orange", "tab:blue")):
            axes[0].plot(d.index, d[m], lw=1.2, alpha=0.85, label=m, color=colour)
        axes[0].set_ylabel("coke rate, kg/tHM")
        axes[0].legend(loc="upper right", ncol=4)
        axes[0].set_title("Coke rate: predicted vs actual")
        axes[0].grid(alpha=0.3)

        for m, colour in zip(models, ("tab:gray", "tab:orange", "tab:blue")):
            axes[1].plot(d.index, d[m] - a, lw=1.0, alpha=0.85, label=m, color=colour)
        axes[1].axhline(0.0, color="black", lw=1.0)
        axes[1].set_ylabel("error, kg/tHM")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="upper right", ncol=3)

        out = REPO / "docs" / "coke_model_shootout.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"  written: {out}")
    except ImportError:
        print("  matplotlib unavailable; skipped the plot")

    banner("VERDICT")
    ranked = sorted(models, key=lambda m: scores(recent["coke_rate"], recent[m])["MAE"])
    print(f"  By MAE over the last 30 days: {' < '.join(ranked)}")
    print("\n  But read section 4 before choosing. A model that only matches")
    print("  persistence cannot answer what happens when the BLEND changes,")
    print("  which is the question the optimiser exists to answer, and no")
    print("  backtest of days the plant actually ran can test it.")


if __name__ == "__main__":
    main()
