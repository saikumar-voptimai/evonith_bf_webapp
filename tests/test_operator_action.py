"""Operator action extraction and attribution, against synthetic ground truth.

Every test here pins a bug that was actually made and shipped during this work,
or a property the analysis silently depends on. Real plant data cannot test any
of it - there is no ground truth for "what triggered this decision" - so the
signals are constructed with a known answer and the code must recover it.

The six defects these guard against, in the order they were found:

  1. Filtering the setpoint on VALUE clipped a real 297 -> 560 excursion and
     INVENTED two events that never happened.
  2. A rolling IQR collapsing toward zero produced z = -133 on a quiet tag.
  3. The synthetic placebo column is never NaN, so adding it before
     dropna(how="all") silently disabled the running-period filter.
  4. The trigger window included the decision instant, letting a mechanical
     consequence pose as a trigger.
  5. The signed peak is bimodal, so summarising it with a MEDIAN made pure
     noise the strongest discriminator in the table.
  6. Banking cuts PCI and raises coke together, so a window that only looks
     BACKWARDS never sees it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import operator_action_attribution as A  # noqa: E402
import operator_action_events as E  # noqa: E402

FREQ = "10s"
RUNNING_BLAST = 110_000.0
RUNNING_PROD = 90.0


def _index(hours: float, start: str = "2026-01-01 00:00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=int(hours * 360), freq=FREQ, tz="Asia/Kolkata")


def _raw(levels: list[tuple[float, float]], **overrides) -> pd.DataFrame:
    """Build a raw frame from (hours_held, setpoint_value) segments."""

    total = sum(h for h, _ in levels)
    idx = _index(total)
    setpoint = np.concatenate([
        np.full(int(h * 360), v) for h, v in levels
    ])[: len(idx)]
    frame = pd.DataFrame({
        "coke_rate": setpoint,
        "hot_blast_vol_nm3h": RUNNING_BLAST,
        "production_per_hour": RUNNING_PROD,
        "coal_rate_actual_value": 170.0,
    }, index=idx)
    for key, value in overrides.items():
        frame[key] = value
    return frame


# --- extraction ------------------------------------------------------------------


def test_a_clean_step_is_found_with_its_size_and_time():
    raw = _raw([(6, 300.0), (6, 305.0)])
    setpoint, _ = E.clean_setpoint(raw)

    events = E.extract_events(setpoint)

    assert len(events) == 1
    assert events[0].delta == pytest.approx(5.0)
    assert events[0].direction == "raise"
    assert events[0].level_from == 300.0 and events[0].level_to == 305.0


def test_a_transient_blip_is_not_a_decision():
    """A setpoint that flickers and returns is a keying correction.

    MIN_HOLD is 10 minutes; this blip lasts 5 and must leave no trace - not two
    events, not one.
    """

    raw = _raw([(6, 300.0), (5 / 60, 340.0), (6, 300.0)])
    setpoint, _ = E.clean_setpoint(raw)

    assert E.extract_events(setpoint) == []
    # And prove the debounce is what removed it, by passing the threshold
    # EXPLICITLY. min_hold is a default argument bound at import, so patching
    # the module constant cannot reach it - the mutation has to go through the
    # call site.
    assert len(E.extract_events(setpoint, min_hold=pd.Timedelta("1s"))) == 2


def test_events_reconstruct_the_setpoint_they_came_from():
    """The validity check the whole analysis rests on."""

    raw = _raw([(5, 300.0), (5, 306.0), (5, 302.0), (5, 311.0)])
    setpoint, _ = E.clean_setpoint(raw)

    events = E.extract_events(setpoint)
    check = E.reconstruct(events, setpoint)

    assert len(events) == 3
    assert check["coverage"] > 0.999
    assert check["max_abs_error"] < E.MIN_STEP_KG


def test_a_high_setpoint_on_a_running_furnace_is_kept():
    """DEFECT 1. A value band clipped a real excursion and invented events.

    560 kg/tHM is far outside the BMO guardrail of [280, 420], but that band
    constrains what the optimiser may RECOMMEND - it is not a limit on what an
    operator may enter. With the furnace running, this is one event of +260,
    not two events created by crossing a threshold.
    """

    raw = _raw([(6, 300.0), (6, 560.0)])
    setpoint, notes = E.clean_setpoint(raw)

    events = E.extract_events(setpoint)

    assert notes["kept"] == len(raw), "nothing may be dropped on value alone"
    assert len(events) == 1
    assert events[0].delta == pytest.approx(260.0)


def test_a_setpoint_parked_during_a_stoppage_is_not_a_decision():
    """Blast and production at zero means the tag holds a value, not a choice."""

    raw = _raw([(6, 300.0), (6, 560.0), (6, 300.0)])
    down = (raw.index >= raw.index[0] + pd.Timedelta("6h")) & (
        raw.index < raw.index[0] + pd.Timedelta("12h")
    )
    raw.loc[down, ["hot_blast_vol_nm3h", "production_per_hour"]] = 0.0

    stoppages = E.find_stoppages(raw)
    setpoint, notes = E.clean_setpoint(raw)

    assert len(stoppages) == 1
    assert stoppages.iloc[0]["hours"] == pytest.approx(6.0, abs=0.1)
    assert stoppages.iloc[0]["setpoint_parked"] == 560.0
    assert notes["not_running"] > 0
    # Whatever survives is flagged as restart-related, never routine control.
    events = E.extract_events(setpoint, stoppages=stoppages)
    assert all(e.context == "restart" for e in events)


def test_banking_is_caught_by_looking_forward_as_well_as_back():
    """DEFECT 6. Banking cuts PCI and raises coke as ONE decision.

    An hour before the action PCI is still injecting normally, so a
    backward-only window sees nothing. A single window spanning the event does
    not work either - it averages to 33.5 kg/tHM on real data, above any
    sensible threshold. Before and after must be tested separately.
    """

    raw = _raw([(6, 300.0), (6, 515.0)])
    switch = raw.index[0] + pd.Timedelta("6h")
    raw.loc[raw.index >= switch, "coal_rate_actual_value"] = 0.0

    setpoint, _ = E.clean_setpoint(raw)
    events = E.extract_events(
        setpoint, pci=raw["coal_rate_actual_value"]
    )

    assert len(events) == 1
    assert events[0].context == "pci_off"


def test_a_large_move_is_separated_from_a_trim():
    raw = _raw([(6, 300.0), (6, 305.0), (6, 355.0)])
    setpoint, _ = E.clean_setpoint(raw)

    events = E.extract_events(setpoint)

    assert [e.size_class for e in events] == ["trim", "large"]


# --- z-scores and peaks ------------------------------------------------------------


def _panel(days: float = 12.0, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=int(days * 96), freq=A.GRID,
                        tz="Asia/Kolkata")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "quiet_tag": 100.0 + rng.normal(0, 1.0, len(idx)),
        "normal_tag": 50.0 + rng.normal(0, 3.0, len(idx)),
        A.PLACEBO: rng.normal(size=len(idx)),
    }, index=idx)


def _panel_with_still_stretch(days: float = 40.0, seed: int = 11) -> pd.DataFrame:
    """A tag of ordinary scale that goes unusually still, then resumes.

    This reproduces the ACTUAL top_press_avg failure. During the still stretch
    the rolling IQR collapses; the ordinary jitter that follows is divided by
    almost nothing and explodes.

    Two scenarios that do NOT reproduce it, both tried first: a tag merely small
    in scale (its deviations shrink in proportion, so z stays ~N(0,1) either
    way), and a flat tag that genuinely STEPS (z of 10 is then the CORRECT
    answer - the tag really did move hugely against its own history).

    Kept separate from the shared panel because a tag this awkward distorts the
    ranking tests that use it.
    """

    idx = pd.date_range("2026-01-01", periods=int(days * 96), freq=A.GRID,
                        tz="Asia/Kolkata")
    rng = np.random.default_rng(seed)
    values = 100.0 + rng.normal(0, 2.0, len(idx))
    # The still stretch MUST be longer than BASELINE_WINDOW (7 days), or the
    # rolling window always straddles it and the IQR never actually collapses.
    # At 40 days this span is 10 days; a 4-day version failed to reproduce the
    # bug at all and made the guard look effective when it was not.
    still = slice(int(len(idx) * 0.25), int(len(idx) * 0.50))
    values[still] = 100.0 + rng.normal(0, 1e-4, len(values[still]))
    return pd.DataFrame({"quiet_tag": values}, index=idx)


def test_a_nearly_constant_tag_cannot_produce_an_enormous_z():
    """DEFECT 2. top_press_avg scored z = -133 when its rolling IQR collapsed.

    A tag that sits still for a week has almost no rolling spread, so ordinary
    jitter divided by it explodes. The scale floor keeps a quiet variable from
    winning every ranking on noise.
    """

    z = A.robust_z(_panel_with_still_stretch())

    # Assert on HOW OFTEN the clip is hit, not on the maximum, and not on the
    # clip bound itself. Two earlier versions of this assertion tested nothing:
    # |z| <= Z_CLIP is guaranteed by the clip whether the floor exists or not,
    # and the MAX saturates either way because extreme jitter after a still
    # stretch overwhelms any floor. The floor governs the FREQUENCY. Measured
    # on this fixture: 2.2% of samples saturate with it, 8.5% without.
    saturated = float((z["quiet_tag"].abs() >= A.Z_CLIP - 1e-9).mean())
    assert saturated < 0.05, (
        f"{saturated:.1%} of samples saturated the clip; without the floor a "
        "collapsed rolling IQR does this to ordinary jitter"
    )


def test_the_peak_window_stops_short_of_the_decision():
    """DEFECT 4. Including t itself lets a consequence pose as a trigger.

    Tested from BOTH sides. Asserting only that the spike is excluded would
    pass for a window that excludes everything, so the second half confirms a
    spike just inside the boundary IS still picked up.
    """

    panel = _panel()
    z = A.robust_z(panel).copy()
    at = panel.index[600]

    # Boundaries are LITERAL, not derived from A.LEAD_GAP. Deriving them from
    # the constant under test makes the assertion move with any change to it,
    # which is how the first version of this test passed with LEAD_GAP set to
    # eight hours.
    at_event = z.copy()
    at_event.loc[at, "normal_tag"] = 99.0
    assert abs(A.rolling_peak_z(at_event).loc[at, "normal_tag"]) < 90.0

    # A spike two hours before the decision is well inside an 8 h window and
    # well outside a 30 min lead gap, so it must be visible.
    inside = z.copy()
    inside.loc[at - pd.Timedelta("2h"), "normal_tag"] = 99.0
    assert abs(A.rolling_peak_z(inside).loc[at, "normal_tag"]) > 90.0


def test_the_peak_is_bimodal_so_medians_of_it_are_meaningless():
    """DEFECT 5. This is why the case-control compares |peak|, not peak.

    Adjacent windows share all but one point, so the running extreme sits near
    +2 or -2 and rarely near zero. Summarising THAT with a median flips between
    modes on a near-even split, which made pure noise the strongest
    discriminator in the table.
    """

    z = A.robust_z(_panel(days=30))
    peak = A.rolling_peak_z(z)[A.PLACEBO].dropna()

    near_zero = float((peak.abs() < 0.5).mean())
    assert near_zero < 0.10, "a unimodal peak would sit near zero far more often"
    # The magnitude, by contrast, is well behaved and usable.
    assert peak.abs().median() > 1.5


# --- attribution -------------------------------------------------------------------


def test_an_injected_trigger_is_recovered_and_beats_the_placebo():
    """End to end: plant a known cause, confirm the ranking finds it.

    The single most important test here. If a trigger this obvious cannot be
    recovered, no ranking computed on real data means anything.
    """

    panel = _panel(days=20, seed=3)
    event_at = panel.index[1200]
    window = (panel.index >= event_at - pd.Timedelta("4h")) & (panel.index < event_at)
    panel.loc[window, "normal_tag"] += 40.0     # an unmistakable excursion

    z = A.robust_z(panel)
    event = E.ActionEvent(
        time=event_at, level_from=300.0, level_to=295.0, delta=-5.0,
        direction="cut", held_for=pd.Timedelta("8h"), since_previous=None,
        shift=E.shift_of(event_at),
    )

    result = A.attribute(event, z, panel)

    assert result.ranked, "no observation ranked at all"
    assert result.ranked[0][0] == "normal_tag"
    assert abs(result.ranked[0][1]) > 3.0
    ranks = [name for name, _, _ in result.ranked]
    assert ranks.index("normal_tag") < ranks.index(A.PLACEBO)


def test_with_no_injected_trigger_the_placebo_is_competitive():
    """The complement of the test above, and the reason the placebo exists.

    On data with nothing to find, noise should rank as highly as anything else.
    If real tags reliably beat the placebo on pure noise, the ranking is biased
    and its results on plant data cannot be read.
    """

    wins = 0
    trials = 30
    for seed in range(trials):
        panel = _panel(days=12, seed=seed)
        z = A.robust_z(panel)
        at = panel.index[900]
        event = E.ActionEvent(
            time=at, level_from=300.0, level_to=305.0, delta=5.0,
            direction="raise", held_for=pd.Timedelta("6h"), since_previous=None,
            shift=E.shift_of(at),
        )
        ranked = A.attribute(event, z, panel).ranked
        if ranked and ranked[0][0] == A.PLACEBO:
            wins += 1

    # Three tags, so chance is about a third. Anything near zero would mean the
    # placebo is structurally handicapped and the bar it sets is too low.
    assert 0.10 <= wins / trials <= 0.60, f"placebo led {wins}/{trials}"


def test_controls_are_never_drawn_from_a_gap_in_the_real_data():
    """DEFECT 3. The placebo is synthetic and never NaN, which disabled the
    running-period filter and drew 33 controls from inside a 38-hour stoppage.
    """

    panel = _panel(days=20)
    dead = (panel.index >= panel.index[400]) & (panel.index < panel.index[600])
    panel.loc[dead, ["quiet_tag", "normal_tag"]] = np.nan   # placebo still finite

    controls = A.sample_controls(panel, [panel.index[900]], 20)

    assert controls, "no controls drawn at all"
    assert not any(panel.index[400] <= c < panel.index[600] for c in controls)
