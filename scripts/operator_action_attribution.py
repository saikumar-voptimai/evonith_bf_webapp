"""What was the operator looking at? Per-event attribution. Phase 2.

Run:  python scripts/operator_action_attribution.py [days]

WHY THIS EXISTS.

Phase 1 recovered 40 routine coke-setpoint decisions on a normally running
furnace. This asks what each one was a response to: which observations were
deviating in the hours beforehand.

THE METHOD, AND WHY IT IS DELIBERATELY MODEST.

Forty events against roughly twenty candidate observations is not enough to fit
a multivariate model without overfitting it. Anything with per-feature
coefficients would produce confident nonsense at this sample size. So:

  1. A model-free DEVIATION VECTOR per event - robust z-scores of every
     observation against its own trailing baseline, plus the slope over the
     trigger window. Level and trend both matter: an operator reacts to "still
     falling" as much as to "low".

  2. A UNIVARIATE case-control comparison. For each event, matched control
     timestamps are drawn from the same running periods where no action was
     taken. Per-feature effect sizes only - no joint model, no interactions.

Robust statistics throughout (median, IQR) rather than mean and sd, because the
excursions the operator responds to are exactly the outliers that would wreck a
mean-based baseline.

WHAT WOULD INVALIDATE IT.

A PLACEBO observation of pure noise is carried through every step. If it ranks
among the top attributed triggers for a meaningful share of events, the ranking
is fitting noise and none of it should be believed. A shuffle test does the same
job at the population level: permuting event times must collapse the effect
sizes toward zero.

Expect a large "nothing was deviating" bucket. That is a real result - it bounds
how much of operator behaviour these tags can explain at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from operator_action_events import (  # noqa: E402
    DEFAULT_DAYS, banner, clean_setpoint, extract_events, fetch_raw_setpoint,
    find_stoppages, shift_of,
)

GRID = "15min"
# What the operator can see on the screen, grouped by what it tells them.
OBSERVATIONS: dict[str, list[str]] = {
    "thermal": [
        "runner_temp_pci_taphole", "runner_temp_cr_taphole",
        "runner_temp_pci_skimmer", "runner_temp_cr_skimmer", "body_raft",
    ],
    "gas": ["body_etaco", "co_pct", "co2_pct", "h2_pct", "top_temp_avg"],
    "aero": [
        "body_dp_total", "body_dp_top", "body_dp_bottom", "body_perm",
        "top_press_avg", "hot_blast_press", "tuyere_velocity",
    ],
    "burden": ["stock_rod_radar_level", "charges_per_hour", "hm_per_charge"],
    "output": ["production_per_hour"],
}
ALL_OBSERVATIONS = [c for cols in OBSERVATIONS.values() for c in cols]
GROUP_OF = {c: g for g, cols in OBSERVATIONS.items() for c in cols}

# The operator watches the last few hours; burden descent is 6-8 h, so anything
# they are reacting to should be visible inside this window.
TRIGGER_WINDOW = pd.Timedelta("8h")
# The window must STOP SHORT of the decision. Including t itself lets a
# mechanical consequence of the action masquerade as its trigger - hm_per_charge
# moves the moment the charge program changes, so it would be "predicting" an
# action it is actually caused by.
LEAD_GAP = pd.Timedelta("30min")
# Baseline the deviation against - long enough to span several shifts.
BASELINE_WINDOW = pd.Timedelta("7D")
# |z| above this counts as "deviating" rather than normal scatter.
DEVIATION_THRESHOLD = 2.0
# A rolling IQR may not fall below this fraction of the tag's long-run IQR.
# Without it, a tag that happens to sit still for a week reports impossible
# z-scores (top_press_avg produced -133) and wins every ranking on noise.
SCALE_FLOOR_FRACTION = 0.20
Z_CLIP = 10.0
# The baseline needs to be warm before the first event, so the data pull starts
# this much earlier than the analysis window. Otherwise the first days' events
# get no attribution at all and silently drop out of the population.
BASELINE_PREROLL_DAYS = 8
# Controls must sit at least this far from any action, or they are contaminated
# by the very decision they are supposed to contrast with.
CONTROL_EXCLUSION = pd.Timedelta("12h")
CONTROLS_PER_EVENT = 4
# Permutations for the null in section 5. 200 is enough to place a p-value to
# about +/-0.03, which is all the precision this sample size deserves.
PERMUTATIONS = 200
PLACEBO = "PLACEBO_noise"
RNG = np.random.default_rng(42)


@dataclass
class Attribution:
    """What a single event appears to have been a response to."""

    time: pd.Timestamp
    delta: float
    direction: str
    size_class: str
    ranked: list[tuple[str, float, float]]   # (observation, z, slope_per_h)

    @property
    def explained(self) -> bool:
        return bool(self.ranked) and abs(self.ranked[0][1]) >= DEVIATION_THRESHOLD

    def top(self, n: int = 3) -> list[tuple[str, float, float]]:
        return self.ranked[:n]


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Observations on a regular grid, restricted to running periods."""

    blast = pd.to_numeric(raw.get("hot_blast_vol_nm3h"), errors="coerce")
    prod = pd.to_numeric(raw.get("production_per_hour"), errors="coerce")
    running = (blast > 40_000.0) & (prod > 20.0)

    frame = pd.DataFrame(index=raw.index)
    for col in ALL_OBSERVATIONS:
        if col in raw.columns:
            frame[col] = pd.to_numeric(raw[col], errors="coerce")
    frame = frame.where(running)
    panel = frame.resample(GRID).mean()
    # A placebo that cannot possibly explain anything, carried through every
    # step. If it surfaces, the method is fitting noise.
    panel[PLACEBO] = RNG.normal(size=len(panel))
    return panel


def robust_z(panel: pd.DataFrame) -> pd.DataFrame:
    """Deviation from a trailing baseline, in robust sigma.

    Median and IQR rather than mean and sd: the excursions an operator responds
    to are precisely the outliers that would corrupt a mean-based baseline and
    then hide themselves.

    THE SCALE NEEDS A FLOOR. Some tags sit almost perfectly still for days -
    top_press_avg is one - and a rolling IQR that collapses toward zero turns
    ordinary jitter into a z-score of -133. Guarding it against a fraction of
    the tag's OWN long-run spread keeps a genuinely quiet variable from
    dominating the ranking on noise alone.
    """

    steps = max(1, int(BASELINE_WINDOW / pd.Timedelta(GRID)))
    roll = panel.rolling(steps, min_periods=steps // 4)
    median = roll.median()
    iqr = roll.quantile(0.75) - roll.quantile(0.25)

    # The floor needs a global scale that cannot itself collapse. A tag sitting
    # at one value for more than 75% of the record has a global IQR of exactly
    # zero, which made the floor NaN and left the guard doing nothing - the very
    # case it exists for. Standard deviation is non-zero whenever the tag moves
    # at all, so it is the fallback.
    global_iqr = panel.quantile(0.75) - panel.quantile(0.25)
    global_scale = global_iqr.where(global_iqr > 0.0, panel.std())
    floor = (SCALE_FLOOR_FRACTION * global_scale).replace(0.0, np.nan)
    scale = (iqr / 1.349).clip(lower=floor / 1.349, axis=1)

    z = (panel - median) / scale
    # Even with the floor, a real excursion can be enormous. Cap it so ranking
    # reflects "this is far out" rather than the exact size of a tail.
    return z.clip(-Z_CLIP, Z_CLIP)


def rolling_peak_z(zscores: pd.DataFrame) -> pd.DataFrame:
    """Signed peak z over the trailing trigger window, for every timestamp.

    Computed ONCE for the whole panel instead of re-slicing per event. At 244
    events and 200 permutations the per-timestamp version needs a quarter of a
    million frame slices; this is two rolling passes.

    "Signed peak" keeps the direction of the largest excursion: an operator
    responds to how far something went and which way, not to its magnitude
    alone.

    Args:
         - zscores: pd.DataFrame - Robust z-scores on the analysis grid.

    Returns:
         - return pd.DataFrame - Peak z in the window ENDING one LEAD_GAP before
           each timestamp, so nothing at or after the decision leaks in.
    """

    steps = max(2, int(TRIGGER_WINDOW / pd.Timedelta(GRID)))
    lead = max(1, int(LEAD_GAP / pd.Timedelta(GRID)))
    roll = zscores.rolling(steps, min_periods=max(2, steps // 4))
    high, low = roll.max(), roll.min()
    peak = high.where(high.abs() >= low.abs(), low)
    # Shift so the window stops short of the event rather than including it.
    return peak.shift(lead)


def slope_per_hour(panel: pd.DataFrame, at: pd.Timestamp) -> pd.Series:
    """Least-squares trend over the trigger window, per hour."""

    window = panel.loc[at - TRIGGER_WINDOW: at - LEAD_GAP]
    if len(window) < 4:
        return pd.Series(np.nan, index=panel.columns)
    hours = (window.index - window.index[0]).total_seconds() / 3600.0
    out = {}
    for col in window.columns:
        y = window[col].to_numpy(dtype=float)
        ok = np.isfinite(y)
        out[col] = (
            float(np.polyfit(hours[ok], y[ok], 1)[0]) if ok.sum() >= 4 else np.nan
        )
    return pd.Series(out)


def attribute(
    event: Any, zscores: pd.DataFrame, panel: pd.DataFrame
) -> Attribution:
    """Rank observations by how far they had moved before this action."""

    at = event.time
    window = zscores.loc[at - TRIGGER_WINDOW: at - LEAD_GAP]
    if window.empty:
        return Attribution(at, event.delta, event.direction, event.size_class, [])

    # The most extreme value reached in the window, keeping its sign - an
    # operator responds to the peak of an excursion, not its average.
    peak = window.apply(
        lambda s: s.loc[s.abs().idxmax()] if s.notna().any() else np.nan
    )
    slopes = slope_per_hour(panel, at)
    ranked = [
        (col, float(peak[col]), float(slopes.get(col, np.nan)))
        for col in peak.index
        if np.isfinite(peak[col])
    ]
    ranked.sort(key=lambda row: abs(row[1]), reverse=True)
    return Attribution(at, event.delta, event.direction, event.size_class, ranked)


def sample_controls(
    panel: pd.DataFrame, event_times: list[pd.Timestamp], n_per_event: int
) -> list[pd.Timestamp]:
    """Timestamps where the operator could have acted and did not.

    Matched on shift so that time-of-day habits do not masquerade as furnace
    state, and held clear of every action so a control is not just the run-up to
    the next decision.
    """

    # Judge usability on the REAL observations only. The placebo is synthetic
    # noise and is never NaN, so including it here made dropna(how="all") a
    # no-op and drew 33 controls from inside a 38-hour stoppage.
    real = [c for c in panel.columns if c != PLACEBO]
    usable = panel[real].dropna(how="all").index
    events = pd.DatetimeIndex(event_times)
    wanted = {shift_of(t): 0 for t in event_times}
    for t in event_times:
        wanted[shift_of(t)] += n_per_event

    candidates: dict[str, list[pd.Timestamp]] = {"A": [], "B": [], "C": []}
    for ts in usable:
        gap = (events - ts).to_series().abs().min()
        if gap >= CONTROL_EXCLUSION:
            candidates[shift_of(ts)].append(ts)

    chosen: list[pd.Timestamp] = []
    for shift, need in wanted.items():
        pool = candidates.get(shift, [])
        if not pool:
            continue
        take = min(need, len(pool))
        idx = RNG.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[i] for i in idx)
    return chosen


def effect_sizes(
    peak: pd.DataFrame, cases: list[pd.Timestamp], controls: list[pd.Timestamp]
) -> pd.DataFrame:
    """Per-observation separation between action hours and no-action hours.

    Univariate on purpose. Forty events cannot support a joint model, and one
    fitted here would be reporting its own regularisation.
    """

    def peaks(times: list[pd.Timestamp]) -> pd.DataFrame:
        # Events are timestamped to the second; the peak frame lives on the
        # 15-minute grid. ffill takes the most recent grid point at or before
        # each event - an exact index intersection finds nothing at all.
        if not times:
            return pd.DataFrame(columns=peak.columns)
        return peak.reindex(pd.DatetimeIndex(sorted(times)), method="ffill")

    # COMPARE MAGNITUDES, NOT SIGNED PEAKS.
    #
    # The signed peak is bimodal - adjacent windows share 31 of 32 points, so it
    # sits near +2 or -2 and rarely near zero. Its MEDIAN is therefore unstable,
    # flipping between modes on a 50.1/49.9 split. Summarising it that way made
    # the PLACEBO the strongest "discriminator" in the table at a gap of 3.511,
    # purely because cases and controls landed on opposite modes.
    #
    # |peak| is unimodal and answers the question actually being asked: were
    # observations deviating FURTHER before an action than during a quiet spell?
    case_peaks, control_peaks = peaks(cases).abs(), peaks(controls).abs()
    rows = []
    for col in peak.columns:
        a = case_peaks[col].dropna() if col in case_peaks else pd.Series(dtype=float)
        b = control_peaks[col].dropna() if col in control_peaks else pd.Series(dtype=float)
        if len(a) < 5 or len(b) < 5:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        rows.append({
            "observation": col,
            "group": GROUP_OF.get(col, "placebo"),
            "case_med_absz": float(a.median()),
            "ctrl_med_absz": float(b.median()),
            "abs_gap": float(a.median() - b.median()),
            "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled else np.nan,
            "n_case": len(a),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Sort by the SIGNED gap: an observation deviating further before actions is
    # the hypothesis. One deviating LESS is not evidence for it.
    return frame.sort_values("abs_gap", ascending=False)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS

    # Pull extra history so the rolling baseline is already warm at the first
    # event in the analysis window.
    raw = fetch_raw_setpoint(days + BASELINE_PREROLL_DAYS)
    analysis_start = raw.index.max() - pd.Timedelta(days=days)

    stoppages = find_stoppages(raw)
    setpoint, _ = clean_setpoint(raw)
    pci = pd.to_numeric(raw.get("coal_rate_actual_value"), errors="coerce")
    events = [
        e for e in extract_events(setpoint, stoppages=stoppages, pci=pci)
        if e.context == "normal" and e.time >= analysis_start
    ]

    panel = build_panel(raw)
    zscores = robust_z(panel)
    peak = rolling_peak_z(zscores)

    banner("0. SETUP")
    print(f"  {len(events)} routine control events in the {days}-day analysis "
          f"window ({analysis_start.date()} onward)")
    print(f"  baseline pre-rolled from {raw.index.min().date()} so the first "
          "event already has a warm reference")
    have = [c for c in ALL_OBSERVATIONS if c in panel.columns]
    print(f"  {len(have)} observations on a {GRID} grid, plus 1 placebo")
    print(f"  trigger window {TRIGGER_WINDOW}, baseline {BASELINE_WINDOW}, "
          f"|z| >= {DEVIATION_THRESHOLD} counts as deviating")

    attributions = [attribute(e, zscores, panel) for e in events]

    banner("1. THE LEDGER - what was deviating before each action")
    rows = []
    for a in attributions:
        top = a.top(3)
        rows.append({
            "time": a.time.strftime("%m-%d %H:%M"),
            "delta": f"{a.delta:+.0f}",
            "size": a.size_class,
            "explained": "yes" if a.explained else "-",
            "1st": f"{top[0][0]} {top[0][1]:+.1f}" if len(top) > 0 else "",
            "2nd": f"{top[1][0]} {top[1][1]:+.1f}" if len(top) > 1 else "",
            "3rd": f"{top[2][0]} {top[2][1]:+.1f}" if len(top) > 2 else "",
        })
    ledger = pd.DataFrame(rows)
    print(ledger.to_string(index=False))

    banner("2. HOW MANY ARE EXPLAINED - AGAINST AN EMPIRICAL NULL")
    print("  A fixed |z| >= 2 threshold is meaningless here. The peak is taken")
    print(f"  over {len(have)+1} observations x {int(TRIGGER_WINDOW/pd.Timedelta(GRID))}"
          " time points, so pure noise clears 2.0 almost every time.")
    print("  The placebo supplies the honest null: its own peak |z| distribution.")

    placebo_peaks = np.array([
        abs(dict((c, z) for c, z, _ in a.ranked).get(PLACEBO, np.nan))
        for a in attributions
    ])
    placebo_peaks = placebo_peaks[np.isfinite(placebo_peaks)]
    if placebo_peaks.size:
        bar = float(np.quantile(placebo_peaks, 0.95))
        print(f"\n  placebo peak |z|: median {np.median(placebo_peaks):.2f}, "
              f"95th percentile {bar:.2f}")
        naive = sum(a.explained for a in attributions)
        real_explained = sum(
            1 for a in attributions if a.ranked and abs(a.ranked[0][1]) >= bar
        )
        print(f"  at the nominal |z| >= {DEVIATION_THRESHOLD}: "
              f"{naive}/{len(attributions)} 'explained' ({naive/len(attributions):.0%})"
              "  <- meaningless")
        print(f"  against the placebo null (|z| >= {bar:.2f}): "
              f"{real_explained}/{len(attributions)} "
              f"({real_explained/len(attributions):.0%})")
        print(f"  {len(attributions)-real_explained} events had nothing moving "
              "further than noise does.")

    banner("3. WHICH OBSERVATIONS LEAD ACTIONS - by how often they rank first")
    firsts = pd.Series(
        [a.ranked[0][0] for a in attributions if a.ranked]
    ).value_counts()
    globals()["firsts"] = firsts
    for name, count in firsts.items():
        tag = "  <<< PLACEBO" if name == PLACEBO else ""
        print(f"    {name:28s} {count:3d}  ({count/len(attributions):.0%}){tag}")
    if PLACEBO in firsts.index:
        print(f"\n  THE PLACEBO RANKS FIRST {firsts[PLACEBO]} TIMES. Any observation")
        print("  appearing at or below that rate is indistinguishable from noise.")

    banner("4. CASE-CONTROL - do action hours differ from no-action hours?")
    cases = [e.time for e in events]
    controls = sample_controls(panel, cases, CONTROLS_PER_EVENT)
    print(f"  {len(cases)} cases vs {len(controls)} controls "
          f"(matched on shift, >= {CONTROL_EXCLUSION} from any action)")
    if controls:
        idx = pd.DatetimeIndex(controls).sort_values()
        stretches = int((idx.to_series().diff().dt.total_seconds() / 3600 > 6).sum()) + 1
        print(f"  BUT those {len(controls)} controls fall in only {stretches} "
              f"contiguous stretches across {idx.normalize().nunique()} days.")
        print("  Timestamps hours apart in the same quiet spell are not independent")
        print(f"  observations, so the effective control n is nearer {stretches} "
              f"than {len(controls)} - and the power is correspondingly small.")
    table = effect_sizes(peak, cases, controls)
    if table.empty:
        print("  insufficient overlap to compare")
    else:
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        placebo_row = table[table["observation"] == PLACEBO]
        if not placebo_row.empty:
            bar = float(placebo_row["abs_gap"].iloc[0])
            beat = table[table["abs_gap"] > bar]
            print(f"\n  PLACEBO BAR: abs_gap {bar:.3f}. "
                  f"{len(beat)} of {len(table)-1} real observations clear it.")
            print("  Anything below the bar separates cases from controls no better")
            print("  than a random number does.")

    banner("5. PERMUTATION TEST")
    print("  Pool the cases and controls, then re-split them at random into")
    print("  groups of the same size and recompute. Comparing against random")
    print("  TIMES would not do: controls are drawn from quiet spells, so random")
    print("  times differ from them for reasons that have nothing to do with")
    print("  actions. Relabelling the pooled set is the honest null.")

    if not table.empty:
        def top_gap(case_times, control_times) -> float:
            t = effect_sizes(peak, case_times, control_times)
            if t.empty:
                return np.nan
            return float(t[t["observation"] != PLACEBO]["abs_gap"].head(5).mean())

        observed = top_gap(cases, controls)
        pooled = list(cases) + list(controls)
        n_case = len(cases)
        null = []
        for _ in range(PERMUTATIONS):
            order = RNG.permutation(len(pooled))
            null.append(top_gap(
                [pooled[i] for i in order[:n_case]],
                [pooled[i] for i in order[n_case:]],
            ))
        null_arr = np.array([v for v in null if np.isfinite(v)])
        p_value = float((null_arr >= observed).mean()) if null_arr.size else np.nan
        print(f"\n  observed top-5 mean abs_gap: {observed:.3f}")
        print(f"  null distribution ({null_arr.size} permutations): "
              f"median {np.median(null_arr):.3f}, "
              f"95th pct {np.quantile(null_arr, 0.95):.3f}")
        print(f"  p = {p_value:.3f}")
        verdict = ("SIGNAL - action hours differ from no-action hours"
                   if p_value < 0.05 else
                   "NOT DISTINGUISHABLE at this sample size")
        print(f"  -> {verdict}")

    banner("6. WHAT THIS DOES AND DOES NOT SUPPORT")
    if firsts.size:
        chance = 1.0 / max(1, len([c for c in panel.columns]))
        print(f"  Under a null where any of the {len(panel.columns)} observations")
        print(f"  could top the ranking by chance, each would lead ~{chance:.0%} of")
        print(f"  the time. The placebo does exactly that ({firsts.get(PLACEBO,0)}"
              f"/{len(attributions)}).")
        from scipy.stats import binomtest

        leaders = firsts[firsts > firsts.get(PLACEBO, 0)]
        n = len(attributions)
        print(f"\n  Observations leading MORE often than the placebo, with the")
        print(f"  probability of doing so by chance (binomial, p={chance:.3f}):")
        print(f"    {'observation':28s} {'leads':>8s} {'share':>7s} {'p':>9s}")
        for name, count in leaders.items():
            p = binomtest(int(count), n, chance, alternative="greater").pvalue
            mark = "  *" if p < 0.01 else ""
            print(f"    {name:28s} {count:3d}/{n:<4d} {count/n:6.0%} "
                  f"{p:9.2e}{mark}")
        print("\n  That ranking is the firmer of the two results. The case-control")
        print("  comparison is weaker: the controls collapse to a handful of")
        print("  independent quiet spells, so it has little power regardless of")
        print("  what is true.")


if __name__ == "__main__":
    main()
