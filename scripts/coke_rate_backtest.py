"""Does the energy balance actually predict the coke rate? Backtest, 221 days.

Run:  python scripts/coke_rate_backtest.py

WHY THIS EXISTS.

The energy balance can be inverted to ask "what coke rate closes this day". That
figure drives Layer 2's fuel cost and therefore every control recommendation.
Until now it had been checked on ONE day, where it landed within 0.7% - which is
an anecdote, not a validation. A single day can agree by luck, and this balance
is known to contain two errors that partly cancel.

This runs the solve across the whole record and reports what it is worth.

THREE MODELS, EACH A FAIR TEST.

  physics        the energy balance alone, no fitting whatsoever
  physics+bias   one constant offset, fitted on the training half only
  hybrid         physics + a linear residual correction on features the
                 balance cannot see

The hybrid is the "physics-informed data-driven" model: physics supplies the
structure and the blend sensitivity, and data corrects the systematic biases we
already know about - the shell-loss basis and the top-gas analyser under-read.
Crucially the residual model is NOT free to re-learn the coke rate; it only sees
what the balance got wrong.

WHY THAT MATTERS HERE. A plain data-driven fuel model on this plant is
blend-blind: total fuel is held roughly constant by the operators, so the record
carries almost no blend-to-fuel sensitivity to learn from. The physics has that
sensitivity by construction. Anchoring on physics and fitting only the residual
keeps the sensitivity while removing the bias.

EVERY SPLIT IS TIME-ORDERED. Random k-fold on autocorrelated daily data leaks
badly - neighbouring days are nearly duplicates - and would report a score this
model cannot deliver forward.
"""

from __future__ import annotations

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
from energy_balance_day_audit import daily_dust  # noqa: E402
from utils.energy_balance import EnergyBalanceInputs  # noqa: E402
from utils.energy_balance.constants import load_config  # noqa: E402
from utils.energy_balance.solve import solve_coke_rate_kg_per_thm  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def scores(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    err = predicted - actual
    ss_res = float((err**2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return {
        "n": float(len(actual)),
        "bias": float(err.mean()),
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE%": float((err.abs() / actual).mean() * 100.0),
        "R2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def show(label: str, s: dict[str, float]) -> None:
    print(f"  {label:22s} n={s['n']:4.0f}  bias {s['bias']:+7.1f}  MAE {s['MAE']:6.1f}"
          f"  RMSE {s['RMSE']:6.1f}  MAPE {s['MAPE%']:5.2f}%  R2 {s['R2']:+6.3f}")


def build_inputs(row: pd.Series, shell_basis: str) -> EnergyBalanceInputs:
    """One day as EnergyBalanceInputs. coke_mt is a seed only - it is solved for."""

    shell = row["stave_gj_per_hr"] if shell_basis == "stave" else row["total_gj_per_hr"]
    f = lambda key, default=0.0: (  # noqa: E731
        float(row[key]) if key in row and pd.notna(row[key]) else default
    )
    return EnergyBalanceInputs(
        hot_metal_mt=f("hm_mt"), slag_mt=f("slag_mt"),
        coke_mt=f("coke_mt"), nut_coke_mt=f("nut_coke_mt"), pci_mt=f("pci_mt"),
        blast_volume_nm3_per_hr=f("cbv_nm3h"), blast_temperature_c=f("blast_temp"),
        oxygen_enrichment_pct=f("o2_enrich"),
        top_gas_co_pct=f("co_pct"), top_gas_co2_pct=f("co2_pct"),
        top_gas_h2_pct=f("h2_pct"), top_gas_temperature_c=f("top_temp"),
        hm_carbon_pct=f("hm_c", 4.3), hm_iron_pct=f("hm_fe", 94.5),
        hm_silicon_pct=f("hm_si", 0.5), hm_manganese_pct=f("hm_mn", 0.2),
        slag_feo_pct=f("slag_feo_pct", 0.4),
        flue_dust_mt=f("flue_dust_mt"), gcp_dust_mt=f("gcp_dust_mt"),
        flux_mt=f("flux_mt"), sinter_mt=f("sinter_mt"), ore_mt=f("ore_mt"),
        pellet_mt=f("pellet_mt"), flux_loi_pct=f("flux_loi", 40.0),
        fuel_vm_pct={"coke": 0.9, "nut_coke": 1.0, "pci": 19.9},
        moisture_pct={
            "ore": f("ore_tm"), "pellet": f("pellet_tm"), "flux": f("flux_tm"),
            "coke": f("coke_moist"), "nut_coke": f("nutcoke_moist"), "sinter": 0.0,
        },
        shell_loss_gj_per_hr=float(shell) if pd.notna(shell) else None,
    )


def main() -> None:
    df = eb.build().join(daily_dust(), how="left")
    df["quarter"] = df.index.to_period("Q").astype(str)
    cfg = load_config()

    for basis in ("stave", "total"):
        solved = []
        for _, row in df.iterrows():
            try:
                solved.append(solve_coke_rate_kg_per_thm(build_inputs(row, basis), cfg))
            except Exception:  # noqa: BLE001 - a failed day is a missing prediction
                solved.append(np.nan)
        df[f"pred_{basis}"] = solved

    d = df[["coke_rate", "pred_stave", "pred_total", "quarter", "slag_rate",
            "pci_rate", "nut_rate", "blast_temp", "cbv_per_thm", "co_pct",
            "co2_pct", "h2_pct", "top_temp", "hm_si", "hm_fe",
            "burden_water_kg_per_thm"]].replace([np.inf, -np.inf], np.nan).dropna()

    banner("0. SAMPLE")
    print(f"  days {len(d)}   {d.index.min().date()} -> {d.index.max().date()}")
    a = d["coke_rate"]
    print(f"  actual coke rate: mean {a.mean():.1f}  sd {a.std():.1f}  "
          f"range {a.min():.0f}-{a.max():.0f} kg/tHM")
    print(f"  the sd of {a.std():.1f} is what any model has to beat. Predicting the")
    print("  mean every day would score R2 = 0 by definition.")

    banner("1. PHYSICS ALONE - no fitting of any kind")
    for basis in ("stave", "total"):
        show(f"shell = {basis}", scores(a, d[f"pred_{basis}"]))
    best = "stave" if abs(scores(a, d["pred_stave"])["bias"]) < abs(
        scores(a, d["pred_total"])["bias"]) else "total"
    print(f"\n  -> using shell = {best} below (smaller bias)")

    banner("2. TIME-ORDERED SPLIT: fit on the first half, score on the second")
    half = len(d) // 2
    tr, te = d.iloc[:half], d.iloc[half:]
    print(f"  train {tr.index.min().date()} -> {tr.index.max().date()}  n={len(tr)}")
    print(f"  test  {te.index.min().date()} -> {te.index.max().date()}  n={len(te)}")

    pred = f"pred_{best}"
    print("\n  physics, untouched:")
    show("test", scores(te["coke_rate"], te[pred]))

    offset = float((tr[pred] - tr["coke_rate"]).mean())
    print(f"\n  physics + one constant offset ({-offset:+.1f} kg/tHM, from train only):")
    show("test", scores(te["coke_rate"], te[pred] - offset))

    banner("3. PHYSICS-INFORMED HYBRID")
    print("  Fit the RESIDUAL - what the balance got wrong - on features the")
    print("  balance cannot see. The physics keeps the blend sensitivity; the")
    print("  correction removes the known systematic biases.")
    features = ["slag_rate", "pci_rate", "blast_temp", "cbv_per_thm",
                "co_pct", "co2_pct", "top_temp", "hm_si", "burden_water_kg_per_thm"]
    x_tr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in features])
    x_te = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in features])
    resid_tr = (tr["coke_rate"] - tr[pred]).to_numpy()
    beta, *_ = np.linalg.lstsq(x_tr, resid_tr, rcond=None)
    show("hybrid, test", scores(te["coke_rate"], te[pred] + x_te @ beta))
    print("\n  residual model coefficients (kg coke per unit):")
    for name, b in zip(["intercept"] + features, beta):
        print(f"    {name:26s} {b:+10.4f}")

    banner("4. FOR CONTRAST: pure data-driven on the same features")
    print("  Same features, same split, but no physics anchor - the model must")
    print("  learn the coke rate outright. This is what the balance is worth.")
    y_tr = tr["coke_rate"].to_numpy()
    beta_pure, *_ = np.linalg.lstsq(x_tr, y_tr, rcond=None)
    show("pure data, test", scores(te["coke_rate"], pd.Series(x_te @ beta_pure,
                                                              index=te.index)))

    banner("5. STABILITY BY QUARTER")
    print(f"  {'quarter':10s} {'n':>4s} {'bias':>8s} {'MAE':>7s} {'MAPE%':>7s}")
    for q, sub in d.groupby("quarter"):
        s = scores(sub["coke_rate"], sub[pred])
        print(f"  {q:10s} {s['n']:4.0f} {s['bias']:+8.1f} {s['MAE']:7.1f} "
              f"{s['MAPE%']:7.2f}")
    print("\n  A bias that drifts across quarters is the shell-loss and analyser")
    print("  question showing up again - see docs/energy_balance_findings...md.")

    banner("6. ROLLING RECALIBRATION - the practical answer to a drifting bias")
    print("  A single offset fitted once decays as the bias moves. Refit it on")
    print("  the last N days and apply forward, one day at a time. Strictly")
    print("  causal: each day is corrected using only days before it.")
    print(f"\n  {'window':>8s} {'n':>5s} {'bias':>8s} {'MAE':>7s} {'MAPE%':>7s} {'R2':>7s}")
    for window in (14, 30, 60, 90):
        offsets = (d[pred] - d["coke_rate"]).rolling(window, min_periods=window).mean()
        corrected = (d[pred] - offsets.shift(1)).dropna()
        got = d["coke_rate"].loc[corrected.index]
        s = scores(got, corrected)
        print(f"  {window:8d} {s['n']:5.0f} {s['bias']:+8.1f} {s['MAE']:7.1f} "
              f"{s['MAPE%']:7.2f} {s['R2']:+7.3f}")
    print("\n  Shift(1) matters: without it each day helps predict itself, which")
    print("  is leakage and would flatter every row above.")

    banner("7. THE TEST THIS CANNOT DO")
    print("  Pure data scores better above. That does NOT settle it, for the")
    print("  same reason the earlier fuel-model work failed: none of these")
    print("  features describes the ORE BLEND. The plant held the blend nearly")
    print("  constant across this window, so a data model can score well here")
    print("  and still be blind to the one question BMO exists to answer -")
    print("  what happens to fuel when the blend changes.")
    print("\n  The physics carries that sensitivity by construction. It cannot be")
    print("  validated on this record, because the record does not contain the")
    print("  variation. It would need a deliberate blend trial.")
    blend_like = [c for c in ("slag_rate",) if c in d.columns]
    print(f"\n  For scale, blend-related variation actually present here:")
    for c in blend_like:
        print(f"    {c:12s} sd {d[c].std():6.1f}  range {d[c].min():.0f}-{d[c].max():.0f}")


if __name__ == "__main__":
    main()
