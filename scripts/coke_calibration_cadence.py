"""How often should the coke-rate offset be refitted? Measured, not assumed.

Run:  python scripts/coke_calibration_cadence.py

WHY THIS EXISTS.

The energy balance predicts the coke rate with the right shape and the wrong
level, so a bias offset is fitted from recent plant history. The offset drifts -
+16.0, +16.9, +22.4, +23.8 kg/tHM across four quarters - so it has to be refitted
periodically. The open question is how often, and it decides how the retrain
button in the Blend Optimizer should behave.

Four cadences, all STRICTLY CAUSAL - every prediction uses only days before it:

    never       fit once on the first 90 days, then never again
    quarterly   refit every 90 days, hold in between
    monthly     refit every 30 days, hold in between
    daily       refit every day on the trailing 90 days

"Daily" is not a different model, only a different refresh rate on the same one.
If it does no better than monthly, the button can be a monthly chore rather than
something anyone has to think about.

WHAT WOULD INVALIDATE IT.

Any cadence that peeks at the day it is predicting would look wonderful and be
useless. Every offset here is shifted by one day before use, and the "never"
row exists as the floor: if elaborate refitting cannot beat fitting once, the
drift does not matter and the simplest option wins.
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
from coke_rate_backtest import build_inputs, scores  # noqa: E402
from energy_balance_day_audit import daily_dust  # noqa: E402
from utils.energy_balance.constants import load_config  # noqa: E402
from utils.energy_balance.solve import solve_coke_rate_kg_per_thm  # noqa: E402


def use_reachable_database() -> str:
    """Point DATABASE_URL at whichever configured server actually answers.

    The plant PostgreSQL sits behind a firewall that admits only whitelisted
    addresses, so analysis run from a developer machine frequently cannot reach
    it while the Neon read replica is fine. This PROBES rather than assumes: it
    opens a TCP connection to the primary and only falls back if that fails, so
    a working primary is always preferred and an outage is reported rather than
    silently papered over.

    Returns:
         - return str - Name of the environment variable in use.
    """

    import os

    from sqlalchemy import create_engine, text

    def works(url: str) -> bool:
        # A TCP probe is not enough. The plant firewall now admits this host, so
        # port 5432 opens, and PostgreSQL then refuses on pg_hba.conf. Only an
        # actual query proves the connection is usable.
        try:
            engine = create_engine(url, connect_args={"connect_timeout": 8})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  unusable: {type(exc).__name__}: {str(exc)[:90]}")
            return False

    primary = os.getenv("DATABASE_URL")
    if primary and works(primary):
        return "DATABASE_URL"

    for name in ("NEON_DATABASE_URL", "NEON_STR"):
        replica = os.getenv(name)
        if not replica:
            continue
        if "sslmode" not in replica:
            replica += ("&" if "?" in replica else "?") + "sslmode=require"
        os.environ["DATABASE_URL"] = replica
        return name
    return "DATABASE_URL"

WINDOW_DAYS = 90
WARMUP_DAYS = 90
SHELL_BASIS = "stave"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def offsets_for_cadence(
    residual: pd.Series, cadence: str, window: int = WINDOW_DAYS
) -> pd.Series:
    """Offset in force on each day, under one refresh policy.

    ``residual`` is predicted minus actual. Every value is shifted one day
    before use, so no day can inform its own correction.
    """

    causal = residual.shift(1)
    if cadence == "daily":
        return causal.rolling(window, min_periods=20).mean()

    if cadence == "never":
        first = causal.iloc[:window].mean()
        return pd.Series(first, index=residual.index)

    step = 30 if cadence == "monthly" else 90
    rolling = causal.rolling(window, min_periods=20).mean()
    # Refit only on refresh days, then hold that value until the next one.
    marks = pd.Series(np.nan, index=residual.index)
    for i in range(0, len(residual), step):
        marks.iloc[i] = rolling.iloc[i]
    return marks.ffill()


def main() -> None:
    source = use_reachable_database()
    print(f"  database: using {source}")
    print("building daily history and solving the balance...")
    df = eb.build().join(daily_dust(), how="left").sort_index()
    cfg = load_config()

    predicted = []
    for _, row in df.iterrows():
        try:
            predicted.append(
                solve_coke_rate_kg_per_thm(build_inputs(row, SHELL_BASIS), cfg)
            )
        except Exception:  # noqa: BLE001
            predicted.append(np.nan)
    df["predicted"] = predicted

    d = df[["predicted", "coke_rate"]].replace([np.inf, -np.inf], np.nan).dropna()
    residual = d["predicted"] - d["coke_rate"]

    banner("0. SAMPLE")
    print(f"  {len(d)} days   {d.index.min().date()} -> {d.index.max().date()}")
    print(f"  raw bias {residual.mean():+.1f} kg/tHM, sd {residual.std():.1f}")
    print(f"  scoring starts after a {WARMUP_DAYS}-day warm-up, so every cadence")
    print("  is judged on the same days.")

    banner("1. CADENCE COMPARISON")
    print(f"  {'cadence':12s} {'refits':>7s} {'bias':>8s} {'MAE':>7s} "
          f"{'MAPE%':>7s} {'R2':>8s}")
    results: dict[str, dict] = {}
    for cadence in ("never", "quarterly", "monthly", "daily"):
        offset = offsets_for_cadence(residual, cadence)
        corrected = (d["predicted"] - offset).iloc[WARMUP_DAYS:]
        actual = d["coke_rate"].iloc[WARMUP_DAYS:]
        both = pd.concat([corrected, actual], axis=1).dropna()
        if both.empty:
            continue
        s = scores(both.iloc[:, 1], both.iloc[:, 0])
        refits = {"never": 1, "quarterly": max(1, len(d) // 90),
                  "monthly": max(1, len(d) // 30), "daily": len(d)}[cadence]
        results[cadence] = s
        print(f"  {cadence:12s} {refits:7d} {s['bias']:+8.1f} {s['MAE']:7.1f} "
              f"{s['MAPE%']:7.2f} {s['R2']:+8.3f}")

    banner("2. DOES DAILY BEAT MONTHLY?")
    if "daily" in results and "monthly" in results:
        dd, mm = results["daily"], results["monthly"]
        gain_mae = mm["MAE"] - dd["MAE"]
        gain_r2 = dd["R2"] - mm["R2"]
        print(f"  daily vs monthly:  MAE {gain_mae:+.2f} kg/tHM, "
              f"R2 {gain_r2:+.3f}")
        if gain_mae < 0.5 and gain_r2 < 0.02:
            print("\n  NO MEANINGFUL GAIN. Monthly refitting captures the drift")
            print("  just as well, so the retrain button can be a monthly chore")
            print("  rather than something that has to run unattended.")
        else:
            print("\n  Daily refitting is materially better. The offset moves fast")
            print("  enough that holding it for a month costs real accuracy.")

    banner("3. HOW FAST DOES THE OFFSET ACTUALLY MOVE?")
    daily_offset = offsets_for_cadence(residual, "daily").dropna()
    if len(daily_offset) > 40:
        month = daily_offset.diff(30).abs().median()
        week = daily_offset.diff(7).abs().median()
        print(f"  median change over 7 days:  {week:5.2f} kg/tHM")
        print(f"  median change over 30 days: {month:5.2f} kg/tHM")
        print(f"  full range across the record: {daily_offset.min():.1f} to "
              f"{daily_offset.max():.1f} kg/tHM")
        print("\n  Compare against the residual spread of "
              f"{residual.std():.1f} kg/tHM. If a month of drift is small next to")
        print("  the day-to-day scatter, refitting more often is chasing noise.")

    banner("4. STALENESS - what does letting it rot actually cost?")
    print(f"  {'held for':>10s} {'MAE':>7s} {'MAPE%':>7s} {'R2':>8s}")
    for hold in (0, 30, 60, 90, 180):
        offset = offsets_for_cadence(residual, "daily")
        stale = offset.shift(hold) if hold else offset
        corrected = (d["predicted"] - stale).iloc[WARMUP_DAYS:]
        actual = d["coke_rate"].iloc[WARMUP_DAYS:]
        both = pd.concat([corrected, actual], axis=1).dropna()
        if both.empty:
            continue
        s = scores(both.iloc[:, 1], both.iloc[:, 0])
        print(f"  {hold:8d} d {s['MAE']:7.1f} {s['MAPE%']:7.2f} {s['R2']:+8.3f}")
    print("\n  This is the number the staleness warning should be set from.")


if __name__ == "__main__":
    main()
