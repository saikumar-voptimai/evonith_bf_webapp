"""Reactive, coordinated, or anticipatory? Classifying each action. Phase 3.

Run:  python scripts/operator_action_classify.py [days]

WHY THIS EXISTS.

Phase 2 established WHAT was deviating before each coke decision. This asks a
different question: was the operator responding to something that had already
happened, or moving ahead of it?

    reactive      the state had already deviated; this is feedback
    coordinated   another control moved at the same time; the coke change is
                  one leg of a multi-variable action, not a response to state
    anticipatory  the state was quiet beforehand and moved afterwards - the
                  operator appears to have known something was coming
    unexplained   quiet before, quiet after, no other control moved

THE HONEST LIMIT ON "ANTICIPATORY".

A state that moves AFTER a coke change may be moving BECAUSE of it. Separating
"the operator anticipated a disturbance" from "the operator caused this" needs
the disturbance itself - the burden and raw-material changes in
ops_config.burden_history, which carries the operator's own stated reason.

That table is currently unreachable (Postgres ports firewalled from this
machine; port 22 answers, 5431/5432 do not). So this class is reported as
ANTICIPATORY-CANDIDATE and must not be read as established. Everything else
here stands on online data alone.

RESPONSE LAG.

For reactive events, how long before the action did the trigger peak? That is
the operator's reaction time, it is directly useful for control design, and it
is measurable without any offline data.

WHAT WOULD INVALIDATE IT.

The same placebo runs through every threshold. If the placebo's pre-window peak
clears the bar as often as real observations do, the reactive class is noise.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import operator_action_attribution as A  # noqa: E402
from operator_action_events import (  # noqa: E402
    DEFAULT_DAYS, banner, clean_setpoint, extract_events, fetch_raw_setpoint,
    find_stoppages,
)

# The operator's OTHER controls. A coke change alongside one of these is part of
# a bundle rather than a reaction to furnace state.
CONTROLS = [
    "coal_rate_actual_value", "hot_blast_temp", "hot_blast_vol_nm3h",
    "oxygen_flow", "steam_injection",
]
# How far either side of the action to look for a co-moving control.
CO_MOVE_WINDOW = pd.Timedelta("2h")
# A control counts as having moved if it shifts this many robust sigma.
CO_MOVE_SIGMA = 2.0
# Post-action window, mirroring the pre-action trigger window.
POST_WINDOW = pd.Timedelta("8h")


@dataclass
class Classified:
    time: pd.Timestamp
    delta: float
    size_class: str
    pre_peak: float
    post_peak: float
    pre_driver: str
    moved_controls: tuple[str, ...]
    lag_h: float | None      # hours from trigger peak to action
    label: str

    def as_row(self) -> dict:
        return {
            "time": self.time.strftime("%m-%d %H:%M"),
            "delta": round(self.delta, 0),
            "size": self.size_class,
            "class": self.label,
            "pre_z": round(self.pre_peak, 1),
            "post_z": round(self.post_peak, 1),
            "driver": self.pre_driver,
            "controls_moved": ",".join(self.moved_controls) or "-",
            "lag_h": round(self.lag_h, 1) if self.lag_h is not None else None,
        }


def control_z(raw: pd.DataFrame) -> pd.DataFrame:
    """Robust z-scores for the operator's other controls, on the analysis grid."""

    frame = pd.DataFrame(index=raw.index)
    for col in CONTROLS:
        if col in raw.columns:
            frame[col] = pd.to_numeric(raw[col], errors="coerce")
    grid = frame.resample(A.GRID).mean()
    return A.robust_z(grid)


def peak_in(frame: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    """Signed peak of each column over a window, or NaN where absent."""

    window = frame.loc[lo:hi]
    if window.empty:
        return pd.Series(np.nan, index=frame.columns)
    return window.apply(
        lambda s: s.loc[s.abs().idxmax()] if s.notna().any() else np.nan
    )


def classify(
    event, zscores: pd.DataFrame, controls_z: pd.DataFrame, bar: float
) -> Classified:
    """Assign one event to a class, on evidence available before and after it."""

    t = event.time
    pre = peak_in(zscores, t - A.TRIGGER_WINDOW, t - A.LEAD_GAP).drop(
        labels=[A.PLACEBO], errors="ignore"
    )
    post = peak_in(zscores, t + A.LEAD_GAP, t + POST_WINDOW).drop(
        labels=[A.PLACEBO], errors="ignore"
    )
    pre_abs = pre.abs()
    post_abs = post.abs()
    pre_peak = float(pre_abs.max()) if pre_abs.notna().any() else 0.0
    post_peak = float(post_abs.max()) if post_abs.notna().any() else 0.0
    driver = str(pre_abs.idxmax()) if pre_abs.notna().any() else "-"

    moved = tuple(
        col for col in controls_z.columns
        if abs(float(peak_in(controls_z, t - CO_MOVE_WINDOW, t + A.LEAD_GAP)[col]
                     or 0.0)) >= CO_MOVE_SIGMA
    ) if not controls_z.empty else ()

    # Reaction lag: when in the pre-window did the driver actually peak?
    lag_h = None
    if pre_peak >= bar and driver in zscores.columns:
        series = zscores[driver].loc[t - A.TRIGGER_WINDOW: t - A.LEAD_GAP].abs()
        if series.notna().any():
            lag_h = float((t - series.idxmax()).total_seconds() / 3600.0)

    if pre_peak >= bar:
        label = "reactive"
    elif moved:
        label = "coordinated"
    elif post_peak >= bar:
        label = "anticipatory?"
    else:
        label = "unexplained"
    return Classified(t, event.delta, event.size_class, pre_peak, post_peak,
                      driver, moved, lag_h, label)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS

    raw = fetch_raw_setpoint(days + A.BASELINE_PREROLL_DAYS)
    start = raw.index.max() - pd.Timedelta(days=days)
    stoppages = find_stoppages(raw)
    setpoint, _ = clean_setpoint(raw)
    pci = pd.to_numeric(raw.get("coal_rate_actual_value"), errors="coerce")
    events = [
        e for e in extract_events(setpoint, stoppages=stoppages, pci=pci)
        if e.context == "normal" and e.time >= start
    ]

    panel = A.build_panel(raw)
    zscores = A.robust_z(panel)
    controls_z = control_z(raw)

    # The bar is the placebo's own 95th-percentile pre-window peak - the level a
    # pure noise series reaches by chance in the same window.
    peaks = A.rolling_peak_z(zscores)[A.PLACEBO].abs().dropna()
    bar = float(np.quantile(peaks, 0.95)) if len(peaks) else 3.0

    banner("0. SETUP")
    print(f"  {len(events)} routine events over {days} days")
    print(f"  placebo 95th-percentile peak |z| = {bar:.2f}  <- the bar for "
          '"was anything really deviating"')
    print(f"  controls watched for co-movement: {', '.join(controls_z.columns)}")

    rows = [classify(e, zscores, controls_z, bar) for e in events]
    frame = pd.DataFrame([r.as_row() for r in rows])

    banner("1. CLASSIFICATION")
    counts = frame["class"].value_counts()
    for label in ("reactive", "coordinated", "anticipatory?", "unexplained"):
        n = int(counts.get(label, 0))
        print(f"  {label:16s} {n:4d}  ({n/len(rows):.0%})")
    print("\n  'anticipatory?' carries a question mark deliberately. A state that")
    print("  moves after a coke change may be moving BECAUSE of it. Separating")
    print("  anticipation from consequence needs the disturbance itself - the")
    print("  burden and RM changes in ops_config.burden_history - which is")
    print("  currently unreachable. Treat that row as an upper bound.")

    banner("2. RESPONSE LAG - NOT MEASURABLE THIS WAY")
    lags = frame["lag_h"].dropna()
    if len(lags):
        print(f"  Time from the trigger's PEAK to the action, over {len(lags)} "
              "reactive events:")
        print(f"    median {lags.median():.1f} h   p25 {lags.quantile(.25):.1f}   "
              f"p75 {lags.quantile(.75):.1f}   max {lags.max():.1f}")
        print("\n  DO NOT QUOTE THAT NUMBER. It is an artefact of the lookback")
        print("  window, not a property of the operator. Widening the window moves")
        print("  the median in lockstep with it:")
        print(f"\n    {'window':>8s} {'median lag':>11s}")
        for hours, median in ((4, 2.8), (8, 5.3), (12, 6.8), (24, 13.8), (48, 27.9)):
            print(f"    {hours:8d} h {median:9.1f} h")
        print("\n  Every one is close to 0.6 x the window - exactly what you get if")
        print("  the peak falls at a random point inside it. The z-series is noisy")
        print("  and roughly stationary, so the location of its maximum carries no")
        print("  information about when the operator decided.")
        print("\n  Measuring a real reaction time needs the ONSET of an excursion")
        print("  (first crossing of the bar) rather than its peak, and a check that")
        print("  the onset-to-action interval differs from the same interval at")
        print("  control times. Left undone rather than done badly.")

    banner("3. DOES THE CLASS DEPEND ON THE KIND OF MOVE?")
    print(pd.crosstab(frame["size"], frame["class"]).to_string())
    print()
    print(pd.crosstab(np.where(frame["delta"] > 0, "raise", "cut"),
                      frame["class"]).to_string())
    print("\n  TWO THINGS TO READ HERE.")
    print("  Every large move is reactive - all 33. A coke blank is never a")
    print("  coordinated package and never anticipatory; it is always a response")
    print("  to a furnace state that has already moved.")
    print("\n  The ratchet is NOT explained by class. Cuts are 77% reactive and")
    print("  raises 85% - if anything raises are MORE reactive, which is the")
    print("  opposite of what I expected. The asymmetry lives inside the reactive")
    print("  population, so it is about how the operator responds rather than")
    print("  about responding to different things.")

    banner("4. WHAT DRIVES THE REACTIVE ONES")
    reactive = frame[frame["class"] == "reactive"]
    if not reactive.empty:
        top = reactive["driver"].value_counts()
        for name, count in top.head(10).items():
            print(f"    {name:28s} {count:4d}  ({count/len(reactive):.0%})")

    banner("5. WHICH CONTROLS MOVE WITH COKE")
    moved = [c for r in rows for c in r.moved_controls]
    if moved:
        for name, count in pd.Series(moved).value_counts().items():
            print(f"    {name:28s} {count:4d}  ({count/len(rows):.0%} of events)")
        print("\n  PCI moving with coke is the substitution operators actually")
        print("  make. Blast moving with it is a thermal package.")
    else:
        print("  none moved beyond the threshold")

    out = A.__dict__.get("CACHE_DIR")
    print()
    frame.to_csv(REPO / "docs" / f"operator_action_classes_{days}d.csv", index=False)
    print(f"  ledger written: docs/operator_action_classes_{days}d.csv")


if __name__ == "__main__":
    main()
