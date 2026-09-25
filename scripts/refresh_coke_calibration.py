"""Refit the energy balance's coke-rate offset from recent plant history.

Run:  python scripts/refresh_coke_calibration.py [window_days]

The target is the same measured plant rate used by the BMO accuracy panel:
1,000 times hourly COKE_CALC_MT divided by paired hourly hot-metal tonnes,
mass-weighted over each complete day.

WHAT TO WATCH. The offset is a standing measure of how much the balance is still
missing. When the analyser and shell-loss questions are resolved it should
SHRINK. If it grows instead, something new has broken and the calibration is
quietly hiding it - which is why this prints the trend rather than just the
number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from utils.bmo.coke_calibration import (  # noqa: E402
    DEFAULT_WINDOW_DAYS,
    fit_offset,
    save_calibration,
)
from utils.bmo.coke_history import build_daily_history  # noqa: E402


def main() -> None:
    window = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WINDOW_DAYS

    print(f"building measured daily history (window {window} days)...")
    history = build_daily_history(max(120, window + 30))
    for warning in history.warnings:
        print(f"  WARNING         {warning}")
    df = history.frame.rename(columns={
        "predicted_coke": "predicted",
        "actual_coke": "coke_rate",
    })

    usable = df[["predicted", "coke_rate"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if usable.empty:
        sys.exit("no usable days - cannot calibrate")

    recent = usable.tail(window)
    calibration = fit_offset(
        recent["predicted"].tolist(),
        recent["coke_rate"].tolist(),
        window_days=window,
        days=[d.date().isoformat() for d in recent.index],
    )

    print(f"\n  window          {calibration.first_day} -> {calibration.last_day}")
    print(f"  usable days     {calibration.sample_days}"
          f"   (dropped {calibration.outliers_dropped} outliers)")
    print(f"  OFFSET          {calibration.offset_kg_per_thm:+.1f} kg/tHM"
          "   <- subtracted from the raw balance figure")
    print(f"  residual sd     {calibration.residual_sd_kg_per_thm:.1f} kg/tHM")
    print(f"  target basis    {calibration.target_basis}")
    print(f"  usable          {calibration.is_usable}")
    for note in calibration.notes:
        print(f"  NOTE            {note}")

    corrected = recent["predicted"] - calibration.offset_kg_per_thm
    err = corrected - recent["coke_rate"]
    raw_err = recent["predicted"] - recent["coke_rate"]
    print(f"\n  in-window MAPE  {(raw_err.abs()/recent['coke_rate']).mean()*100:5.2f}%"
          f"  ->  {(err.abs()/recent['coke_rate']).mean()*100:5.2f}%")
    print("  (in-window, so descriptive rather than a forward validation score)")

    print("\n  TREND - is the balance getting better or worse?")
    usable = usable.copy()
    usable["quarter"] = usable.index.to_period("Q").astype(str)
    trend = (usable["predicted"] - usable["coke_rate"]).groupby(
        usable["quarter"]
    ).agg(["count", "mean"])
    for quarter, row in trend.iterrows():
        print(f"    {quarter:8s} n={row['count']:4.0f}  offset {row['mean']:+7.1f}")
    if len(trend) > 1:
        direction = trend["mean"].iloc[-1] - trend["mean"].iloc[0]
        print(f"    -> {'GROWING' if direction > 0 else 'shrinking'} by "
              f"{abs(direction):.1f} kg/tHM across the record")
        if direction > 0:
            print("       A growing offset means the balance is drifting further")
            print("       from the plant, not converging. Re-read the findings doc")
            print("       before trusting the correction to keep holding.")

    path = save_calibration(calibration)
    print(f"\n  written: {path}")


if __name__ == "__main__":
    main()
