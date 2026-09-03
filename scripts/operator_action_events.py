"""Extract every coke-rate setpoint change the operator made. Phase 1.

Run:  python scripts/operator_action_events.py [days]

WHY THIS EXISTS.

``coke_rate`` in InfluxDB is not a measurement - it is the OPERATOR SETPOINT.
Raw, it is piecewise constant: 29 distinct levels across 30 days, sitting at
exactly 305.00 or 300.00 for hours at a time. Every change in it is a deliberate,
timestamped decision.

That matters because every fuel model in this project has hit the same wall.
Total fuel is held nearly flat and the operator closes the loop, so passive
correlation cannot separate what coke DOES from what the operator does about it -
the documented contemporaneous fuel/thermal correlation is -0.45, the wrong sign,
because it is measuring the controller rather than the furnace.

Setpoint changes are interventions. This script recovers them so later phases can
ask what prompted each one.

THE METHOD.

Raw pull at the native 10-second rate. This is not optional: hourly averaging
smears the steps into a ramp, turning 29 clean levels into 335 smeared ones and
destroying the exact change times the whole analysis depends on.

A change is recorded only if the new level HOLDS for at least ``min_hold``. A
setpoint that flickers and returns is a keying correction, not a decision.

FILTER ON OPERATING STATE, NOT ON VALUE.

The first version of this script rejected setpoints outside [280, 420] - the
guardrail band in setting_bmo.yml - and that was wrong twice over. That band is a
constraint on what the BMO optimiser may RECOMMEND, not a limit on what operators
actually enter. Worse, clipping mid-excursion invented events: a genuine
297 -> 560 escalation became a fake "+113" where it crossed 420, and another fake
"-40" on the way back.

What those high values actually mark is the furnace being DOWN. During the
08-25 period the setpoint sits at 560 while production is 0.0, blast is 0,
PCI is 0 and RAFT reads -3000 because the calculation divides by a blast that
is not there. It is a parked value, not a control action.

So the filter is: the furnace must be RUNNING - blast above 40,000 Nm3/h and
production above 20 t/h. Stoppages are reported as a first-class output rather
than quietly dropped, and events within ``post_stoppage_buffer`` of a restart are
flagged: a blow-in is not steady-state control and its coke moves have a
different cause.

WHAT WOULD INVALIDATE IT.

If reconstructing the setpoint series from the extracted events does not
reproduce the raw series over the running periods, the extraction is wrong and
nothing downstream is worth reading. That check runs every time and is the first
thing printed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

SETPOINT_COL = "coke_rate"
DEFAULT_DAYS = 30
# The furnace is RUNNING when it is taking blast and making iron. Below either
# threshold it is down, and the setpoint tag holds a parked value that is not a
# control decision. Measured: 51.5 h down out of 693 h (7.4%) in the last 30 days.
RUNNING_BLAST_MIN_NM3H = 40_000.0
RUNNING_PROD_MIN_TPH = 20.0
# A stoppage shorter than this is a blip, not an outage.
MIN_STOPPAGE = pd.Timedelta("30min")
# After a restart the furnace is being blown in, not steadily controlled. Events
# here are kept but FLAGGED - their cause is the restart, not furnace state.
#
# 12 h rather than 6: burden descent is 6-8 h, so the material charged during a
# blow-in is still working through the furnace for at least one full turnover
# afterwards, and the operator is still walking the setpoint back down. Measured
# on the 2026-08-27 restart, coke was still being unwound 7.7 h after blast
# returned. Set from the physics, not tuned to that event.
POST_STOPPAGE_BUFFER = pd.Timedelta("12h")
# And BEFORE a stoppage the operator is preparing to bank or blow down, which
# also moves coke for reasons that have nothing to do with routine control.
PRE_STOPPAGE_BUFFER = pd.Timedelta("6h")
# PCI BELOW THIS, WITH THE BLAST STILL ON, IS AN ABNORMAL MODE.
#
# Normal running injects 150-210 kg/tHM. Zero PCI while still blowing is either
# a PCI system trip or deliberate banking, and in both cases coke is raised to
# replace the lost fuel. Measured on 2026-08-24: PCI fell 210 -> 112 -> 2 -> 0
# while blast held at ~105,000, and the coke setpoint jumped 297 -> 515. The
# blow-down did not begin until nine hours later, so no time buffer catches it -
# but the PCI signal does, immediately and unambiguously.
#
# These are CLASSIFIED, not discarded. "Coke raised because PCI was lost" is a
# genuine control action and one of the most interpretable in the record.
PCI_MIN_KG_THM = 20.0
# A level must survive this long to count as a decision rather than a keying slip.
MIN_HOLD = pd.Timedelta("10min")
# Steps below this are noise in a tag that is otherwise exactly constant.
MIN_STEP_KG = 0.5
# Above this a change is not a trim. Measured: routine moves cluster at 3-10
# kg/tHM, with a separate population at 30-50 that unwinds within hours - the
# signature of a coke blank charged against a chill.
LARGE_STEP_KG = 20.0
CACHE_DIR = Path(
    "C:/Users/sairi/AppData/Local/Temp/claude/"
    "e--Personal-MarketResearch-EvonithSteel-BlastFurnaceProject-PythonBlastFurnace-evonith-webapp/"
    "ef13d38c-6ac2-4dd2-964a-111b3c164734/scratchpad"
)

# Shift windows are fixed at this plant (CLAUDE.md): A 06-14, B 14-22, C 22-06.
SHIFTS = {"A": (6, 14), "B": (14, 22), "C": (22, 6)}


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


@dataclass
class ActionEvent:
    """One deliberate change of the coke-rate setpoint."""

    time: pd.Timestamp
    level_from: float
    level_to: float
    delta: float
    direction: str          # "raise" | "cut"
    held_for: pd.Timedelta  # how long the NEW level lasted
    since_previous: pd.Timedelta | None
    shift: str
    # "trim" for routine control, "large" for a step big enough to be a coke
    # blank against a chill. The two have different causes and must not be pooled.
    size_class: str = "trim"
    # Operating context, which decides whether the event belongs in the routine
    # control population at all:
    #   "normal"   blast on, production on, PCI injecting - a control decision
    #   "pci_off"  PCI lost or deliberately cut while still blowing - coke is
    #              replacing lost fuel, a real action with an obvious cause
    #   "restart"  inside a stoppage window or its buffers - a blow-in, not control
    context: str = "normal"

    @property
    def restart_related(self) -> bool:
        return self.context != "normal"

    def as_row(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "from": round(self.level_from, 1),
            "to": round(self.level_to, 1),
            "delta": round(self.delta, 1),
            "direction": self.direction,
            "size": self.size_class,
            "context": self.context,
            "held_h": round(self.held_for.total_seconds() / 3600.0, 1),
            "since_prev_h": (
                round(self.since_previous.total_seconds() / 3600.0, 1)
                if self.since_previous is not None else None
            ),
            "shift": self.shift,
        }


def shift_of(ts: pd.Timestamp) -> str:
    hour = ts.hour
    if 6 <= hour < 14:
        return "A"
    if 14 <= hour < 22:
        return "B"
    return "C"


def fetch_raw_setpoint(days: int, use_cache: bool = True) -> pd.DataFrame:
    """Raw 10-second process_params. Cached - the pull takes minutes.

    Args:
         - days: int - Lookback in days.
         - use_cache: bool - Reuse a previous pull for the same window length.

    Returns:
         - return pd.DataFrame - IST-indexed raw frame.
    """

    cache = CACHE_DIR / f"operator_action_raw_{days}d.pkl"
    if use_cache and cache.exists():
        print(f"  using cached raw pull: {cache.name}")
        return pd.read_pickle(cache)

    from furnace_data.influx.online import fetch_online_df

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"  pulling {days} days of RAW process_params (10 s sampling)...")
    df = fetch_online_df(
        selected_measurements=["process_params"],
        time_range="last 1 week",          # ignored when overrides are supplied
        request_type="ts", window_by=None,
        start_time_override=start, end_time_override=end,
        column_naming="field",
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def find_stoppages(raw: pd.DataFrame) -> pd.DataFrame:
    """Periods when the furnace was not running.

    Reported rather than silently dropped: an outage is an operational event in
    its own right, and the coke moves that bracket one are restart decisions
    with a different cause from routine control.

    Args:
         - raw: pd.DataFrame - Raw online frame.

    Returns:
         - return pd.DataFrame - One row per stoppage: start, end, hours,
           and the setpoint value parked during it.
    """

    blast = pd.to_numeric(raw.get("hot_blast_vol_nm3h"), errors="coerce")
    prod = pd.to_numeric(raw.get("production_per_hour"), errors="coerce")
    setpoint = pd.to_numeric(raw[SETPOINT_COL], errors="coerce").replace(0.0, np.nan)

    running = (blast > RUNNING_BLAST_MIN_NM3H) & (prod > RUNNING_PROD_MIN_TPH)
    blocks = (running != running.shift()).cumsum()
    rows = []
    for _, block in running.groupby(blocks):
        if bool(block.iloc[0]):
            continue
        span = block.index[-1] - block.index[0]
        if span < MIN_STOPPAGE:
            continue
        parked = setpoint.loc[block.index].dropna()
        rows.append({
            "start": block.index[0],
            "end": block.index[-1],
            "hours": round(span.total_seconds() / 3600.0, 1),
            "setpoint_parked": round(float(parked.median()), 0) if len(parked) else np.nan,
        })
    return pd.DataFrame(rows)


def clean_setpoint(raw: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Setpoint restricted to samples where the furnace was actually running.

    NOT filtered on value. See the module docstring: a value band clipped a real
    297 -> 560 escalation mid-flight and manufactured two events that never
    happened.
    """

    series = pd.to_numeric(raw[SETPOINT_COL], errors="coerce")
    blast = pd.to_numeric(raw.get("hot_blast_vol_nm3h"), errors="coerce")
    prod = pd.to_numeric(raw.get("production_per_hour"), errors="coerce")

    zero = int((series == 0.0).sum())
    series = series.replace(0.0, np.nan)        # 0 is a dropout, not a setpoint
    running = (blast > RUNNING_BLAST_MIN_NM3H) & (prod > RUNNING_PROD_MIN_TPH)
    kept = series.where(running).dropna()

    notes = {
        "samples": len(series),
        "zero_dropouts": zero,
        "not_running": int((~running).sum()),
        "kept": len(kept),
        "levels": int(kept.nunique()),
        "range": (float(kept.min()), float(kept.max())) if len(kept) else (np.nan, np.nan),
    }
    return kept, notes


def extract_events(
    setpoint: pd.Series,
    *,
    stoppages: pd.DataFrame | None = None,
    pci: pd.Series | None = None,
    min_hold: pd.Timedelta = MIN_HOLD,
    min_step: float = MIN_STEP_KG,
) -> list[ActionEvent]:
    """Debounced step changes in a piecewise-constant series.

    A change counts only when the NEW level survives ``min_hold``. Without that,
    a two-sample overshoot while the operator types becomes two decisions.

    Args:
         - setpoint: pd.Series - Cleaned setpoint, time-indexed, sorted.
         - min_hold: pd.Timedelta - How long a level must last to be real.
         - min_step: float - Smallest change treated as deliberate.

    Returns:
         - return list[ActionEvent] - Events in time order.
    """

    if setpoint.empty:
        return []
    s = setpoint.sort_index()

    # Collapse to runs of constant value first; this is what makes the debounce
    # cheap and makes "how long did it hold" fall out directly.
    changed = s.ne(s.shift())
    run_id = changed.cumsum()
    runs = pd.DataFrame({
        "value": s.groupby(run_id).first(),
        "start": s.groupby(run_id).apply(lambda g: g.index[0]),
        "end": s.groupby(run_id).apply(lambda g: g.index[-1]),
    })
    runs["duration"] = runs["end"] - runs["start"]

    # A run that does not survive min_hold is a transient. Drop it and re-collapse,
    # so the surrounding runs merge if they carried the same value.
    keep = runs[runs["duration"] >= min_hold].copy()
    if keep.empty:
        return []
    merged = keep[keep["value"].ne(keep["value"].shift())].copy()
    merged["next_start"] = merged["start"].shift(-1)
    merged["held"] = merged["next_start"].fillna(keep["end"].iloc[-1]) - merged["start"]

    # Restart windows: the stoppage itself plus a buffer after it comes back.
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if stoppages is not None and not stoppages.empty:
        windows = [
            (row["start"] - PRE_STOPPAGE_BUFFER, row["end"] + POST_STOPPAGE_BUFFER)
            for _, row in stoppages.iterrows()
        ]

    def pci_is_off(t: pd.Timestamp) -> bool:
        """Is PCI absent either side of this action?

        BEFORE and AFTER are tested SEPARATELY, not as one spanning window. The
        two cases are different decisions:

            PCI already off, then coke raised  -> replacing fuel after a trip
            coke raised, then PCI cut          -> banking ahead of a stop

        A single window straddling the event dilutes both. Measured on the
        2026-08-24 banking sequence: the spanning median is 33.5 kg/tHM - above
        any sane threshold - while the post-event median is 0.0.
        """

        if pci is None or pci.empty:
            return False
        before = pci.loc[t - pd.Timedelta("1h"): t]
        after = pci.loc[t: t + pd.Timedelta("2h")]
        for window in (before, after):
            if len(window) and float(window.median()) < PCI_MIN_KG_THM:
                return True
        return False

    def near_restart(t: pd.Timestamp, prev_t: pd.Timestamp) -> bool:
        # Either the event sits in a restart window, or the run it ends spans a
        # stoppage - in which case the apparent step is really two decisions with
        # an outage between them.
        return any(
            (start <= t <= end) or (prev_t < start and t > start)
            for start, end in windows
        )

    events: list[ActionEvent] = []
    previous_time: pd.Timestamp | None = None
    values = merged["value"].tolist()
    starts = merged["start"].tolist()
    ends = keep["end"].reindex(merged.index).tolist()
    helds = merged["held"].tolist()
    for i in range(1, len(values)):
        delta = float(values[i]) - float(values[i - 1])
        if abs(delta) < min_step:
            continue
        t = starts[i]
        events.append(ActionEvent(
            time=t,
            level_from=float(values[i - 1]),
            level_to=float(values[i]),
            delta=delta,
            direction="raise" if delta > 0 else "cut",
            held_for=helds[i],
            since_previous=(t - previous_time) if previous_time is not None else None,
            shift=shift_of(t),
            size_class="large" if abs(delta) >= LARGE_STEP_KG else "trim",
            context=("restart" if near_restart(t, ends[i - 1])
                     else "pci_off" if pci_is_off(t) else "normal"),
        ))
        previous_time = t
    return events


def reconstruct(events: list[ActionEvent], setpoint: pd.Series) -> dict[str, float]:
    """Rebuild the series from the events and measure the disagreement.

    The validity check for the whole exercise. If the events do not reproduce the
    setpoint the operator actually held, nothing downstream means anything.
    """

    if not events:
        return {"coverage": 0.0, "max_abs_error": float("nan")}
    steps = pd.Series(
        [e.level_to for e in events], index=[e.time for e in events]
    ).sort_index()
    # Prepend the level in force before the first event.
    first = pd.Series([events[0].level_from], index=[setpoint.index[0]])
    ladder = pd.concat([first, steps]).sort_index()
    rebuilt = ladder.reindex(setpoint.index, method="ffill")
    err = (rebuilt - setpoint).abs()
    return {
        "coverage": float((err <= MIN_STEP_KG).mean()),
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
    }


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS

    raw = fetch_raw_setpoint(days)
    stoppages = find_stoppages(raw)
    setpoint, notes = clean_setpoint(raw)

    banner("0. RAW SIGNAL")
    interval = raw.index.to_series().diff().dt.total_seconds().median()
    print(f"  {notes['samples']:,} samples at {interval:.0f} s   "
          f"{raw.index.min()} -> {raw.index.max()}")
    print(f"  kept {notes['kept']:,} while RUNNING   dropped "
          f"{notes['zero_dropouts']:,} zero-dropouts, "
          f"{notes['not_running']:,} not-running")
    print(f"  levels held while running: {notes['levels']}   "
          f"range {notes['range'][0]:.0f}-{notes['range'][1]:.0f} kg/tHM")

    banner("1. STOPPAGES - reported, not silently dropped")
    if stoppages.empty:
        print("  none")
    else:
        print(stoppages.to_string(index=False))
        down = stoppages["hours"].sum()
        span = (raw.index.max() - raw.index.min()).total_seconds() / 3600.0
        print(f"\n  total down {down:.1f} h of {span:.0f} h = {down/span:.1%}")
        print("  The setpoint parked at 480-560 during the long outages. That is a")
        print("  held value with production, blast and PCI all at zero - NOT a")
        print("  control action, and the reason this filters on operating state")
        print("  rather than on setpoint value.")

    pci_series = pd.to_numeric(raw.get("coal_rate_actual_value"), errors="coerce")
    events = extract_events(setpoint, stoppages=stoppages, pci=pci_series)

    banner("2. VALIDITY - do the events reproduce the setpoint?")
    check = reconstruct(events, setpoint)
    print(f"  reconstruction agrees with raw on {check['coverage']:.3%} of samples")
    print(f"  max abs error {check['max_abs_error']:.2f}   "
          f"mean {check['mean_abs_error']:.3f} kg/tHM")
    if check["coverage"] < 0.99:
        print("  FAILED. The event list does not describe the setpoint that was")
        print("  actually held. Do not use anything below.")
    else:
        print("  PASSED.")

    banner(f"3. THE EVENTS - {len(events)} over {days} days")
    if not events:
        print("  none found; nothing further to report")
        return
    frame = pd.DataFrame([e.as_row() for e in events])
    print(frame.to_string(index=False))

    banner("4. CHARACTER OF THE ACTIONS")
    clean = frame[frame["context"] == "normal"]
    counts = frame["context"].value_counts().to_dict()
    print(f"  {len(frame)} events total, by operating context: "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"  {len(clean)} are routine control on a normally running furnace;")
    print("  the rest are classified, not discarded - 'coke raised because PCI")
    print("  was lost' is a real action with an unusually clear cause.")
    print(f"  rate: {len(clean)/days:.2f} clean events per day")

    ups = clean[clean["direction"] == "raise"]
    downs = clean[clean["direction"] == "cut"]
    print(f"\n  raise {len(ups):3d}   cut {len(downs):3d}"
          f"   ratio {len(downs)/max(1,len(ups)):.2f} cuts per raise")
    for name, sub in (("raise", ups), ("cut", downs)):
        if sub.empty:
            continue
        d = sub["delta"].abs()
        print(f"  |step| {name:5s}: median {d.median():5.1f}  p90 {d.quantile(.9):5.1f}"
              f"  max {d.max():5.1f} kg/tHM")
    print("\n  THE RATCHET: if cuts outnumber raises at equal median size but the")
    print("  raises carry the fatter tail, the operator trims down in many small")
    print("  steps and adds back in fewer, larger ones. That is a policy, and it")
    print("  is the first thing the attribution has to explain.")

    banner("5. TWO POPULATIONS, NOT ONE")
    by_size = clean.groupby(["size", "direction"]).agg(
        n=("delta", "size"),
        median_abs=("delta", lambda s: s.abs().median()),
        median_held_h=("held_h", "median"),
    )
    print(by_size.to_string())
    large = clean[clean["size"] == "large"]
    if not large.empty:
        print(f"\n  {len(large)} large moves (>= {LARGE_STEP_KG:.0f} kg/tHM), held a")
        print(f"  median {large['held_h'].median():.1f} h before being unwound.")
        print("  A big raise that is stepped back down within hours is a COKE BLANK")
        print("  charged against a chill - a different decision from a trim, with a")
        print("  different cause. Phases 2-3 must attribute them separately.")

    banner("6. TIMING AND SHIFT")
    hold = clean["since_prev_h"].dropna()
    if len(hold):
        print(f"\n  time between actions: median {hold.median():.1f} h  "
              f"p25 {hold.quantile(.25):.1f}  p75 {hold.quantile(.75):.1f}")
        print(f"  actions within 1 h of the previous: {(hold < 1).mean():.0%}"
              "   (clusters, likely one decision keyed in stages)")

    print(f"\n  by shift (A 06-14, B 14-22, C 22-06):")
    by_shift = clean.groupby("shift").agg(
        n=("delta", "size"), raises=("direction", lambda s: (s == "raise").sum()),
        cuts=("direction", lambda s: (s == "cut").sum()),
        median_abs=("delta", lambda s: s.abs().median()),
    )
    print(by_shift.to_string())
    print("\n  A strong shift skew would suggest handover routine rather than")
    print("  furnace state - worth carrying into the attribution as a control.")

    out = CACHE_DIR / f"operator_action_events_{days}d.parquet"
    frame.to_parquet(out)
    print(f"\n  events written: {out}")


if __name__ == "__main__":
    main()
