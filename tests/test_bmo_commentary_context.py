"""What the model is told about this blend.

The commentary itself cannot be unit-tested — it is generated prose. What CAN
be tested, and is the thing that actually determines whether it is any good, is
the context: does the model receive the numbers, and does it receive the
caveats that go with them?

A model handed "coke rate 302.4" writes that the furnace will consume 302.4
kg/THM. Handed "coke rate 302.4, which is an energy balance plus a +24.5 kg/THM
fitted offset whose size measures what the balance is still missing", it writes
something an engineer can argue with. These tests pin the second.

They also pin the negative case, which matters more: when a piece of context
could not be gathered, the model must be TOLD it is missing rather than left to
infer from silence. Silence is how a language model ends up inventing a
three-day trend.
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.bmo.coke_calibration import CokeCalibration
from utils.bmo.commentary import (
    SYSTEM_PROMPT,
    build_commentary_context,
    describe_blend,
    summarise_recent_days,
)


class _Blend:
    def __init__(self, **overrides):
        self.shares_pct = {"ore_a": 60.0, "ore_b": 40.0}
        self.quantities_mt = {"ore_a": 600.0, "ore_b": 400.0}
        self.ore_cost_per_thm_rs = 13188.44
        self.objective_rs_per_thm = 26044.81
        self.slag_rate_kg_per_thm = 290.0
        self.slag_basicity = 1.149
        self.slag_t_basicity = 1.365
        self.slag_al2o3_pct = 17.5
        self.slag_mgo_pct = 7.36
        self.violations = []
        self.diagnostics = {
            "flux_cost_per_thm_rs": 18.85,
            "adjusted_fuel_cost_per_thm_rs": 12856.37,
            "adjusted_objective_rs_per_thm": 26044.81,
            "coke_correction_delta_kg_thm": -9.9,
            "fuel_rate_estimate": {
                "coke_rate_kg_thm": 302.4, "nut_coke_rate_kg_thm": 55.2,
                "pci_rate_kg_thm": 170.3, "total_fuel_rate_kg_thm": 527.9,
            },
        }
        self.__dict__.update(overrides)


class _Ore:
    def __init__(self, ore_id, name, stock, price, fe):
        self.ore_id, self.display_name = ore_id, name
        self.stock_mt, self.price_rs_per_mt = stock, price
        self.min_share_pct, self.max_share_pct = 0.0, 100.0
        self.chemistry = type("C", (), {"fe_t_pct": fe})()


def _ores():
    return [_Ore("ore_a", "ORE A", 5000.0, 7200.0, 62.1),
            _Ore("ore_b", "ORE B", 1200.0, 6100.0, 57.4)]


def _frame(rows: int = 72):
    index = pd.date_range("2026-08-30", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "fuel_rate": [520.0 + i * 0.1 for i in range(rows)],
            "body_etaco": [47.0] * rows,
            "production_per_hour": [98.0] * rows,
        },
        index=index,
    )


def _context(**overrides):
    base = dict(
        live_snapshot={"hot_blast_temp": 1150.0, "co_pct": 22.4,
                       "coal_rate_actual_value": 170.3},
        recent_frame=_frame(),
        ores=_ores(),
        lp_blend=_Blend(),
        de_blend=None,
        manual_blend=_Blend(ore_cost_per_thm_rs=13500.0),
        production_target_mt=2350.0,
    )
    base.update(overrides)
    return build_commentary_context(**base)


# --- the numbers get through ---------------------------------------------------


def test_the_blend_and_its_cost_reach_the_model():
    text = _context().text

    assert "ORE A 60.0%" in text
    assert "302.4" in text          # coke rate
    assert "1.365" in text          # T-basicity
    # Total is objective + flux, the same basis the page headlines:
    # 26,044.81 + 18.85 = 26,064. Asserting the SUM rather than the component
    # is what would catch flux being dropped from the model's view of the cost.
    assert "26,064" in text
    assert "flux 19" in text


def test_material_stock_and_price_reach_the_model():
    """Without stock the model cannot tell a good idea from an impossible one."""

    text = _context().text

    assert "ORE B" in text and "1,200 MT" in text and "6,100" in text


def test_the_current_blend_is_included_so_deltas_are_possible():
    """An operator knows today's levels. The change is the useful part."""

    text = _context().text

    assert "CURRENT BLEND" in text
    assert "RECOMMENDED BLEND" in text


def test_the_recent_window_reports_direction_not_just_level():
    text = summarise_recent_days(_frame(), days=3)

    assert "Fuel rate" in text
    assert "trend +" in text, "a rising fuel rate must be visible as rising"


# --- the caveats get through, which is the point --------------------------------


def test_the_coke_rate_is_never_presented_as_a_measurement():
    text = _context().text

    assert "ENERGY BALANCE plus a fitted bias offset, not a measurement" in text
    assert "MAPE 3.37%" in text


def test_the_unresolved_biases_are_stated():
    text = _context().text

    assert "under-read" in text
    assert "shell heat-loss basis is undecided" in text


def test_the_setpoint_versus_charged_distinction_is_stated():
    """Conflating these is the single easiest mistake to make with this data."""

    text = _context().text

    assert "SETPOINT" in text and "CHARGED" in text and "42% of days" in text


def test_the_fuel_model_being_blend_blind_is_stated():
    assert "nearly blend-blind" in _context().text


def test_the_silicon_autocorrelation_caveat_is_stated():
    assert "lagged silicon as an input" in _context().text


def test_a_live_calibration_is_quoted_with_its_age_and_scatter():
    calibration = CokeCalibration(
        offset_kg_per_thm=24.5, sample_days=88, residual_sd_kg_per_thm=10.4,
        window_days=90, fitted_on="2026-08-23",
    )

    text = _context(calibration=calibration).text

    assert "+24.5" in text and "88 days" in text and "+/-10 kg/THM" in text


def test_a_stale_calibration_is_flagged_in_capitals():
    """Stale is not a footnote — the coke level stops being trustworthy."""

    stale = CokeCalibration(
        offset_kg_per_thm=24.5, sample_days=88, residual_sd_kg_per_thm=10.4,
        window_days=90, fitted_on="2026-01-01",
    )

    text = _context(calibration=stale).text

    assert "THE CALIBRATION IS STALE" in text


def test_a_failed_energy_anchor_is_disclosed():
    """A fallback must never pass for the balance's own answer."""

    anchor = type("A", (), {"usable": False, "notes": ["live blast tags unavailable"]})()

    text = _context(energy_anchor=anchor).text

    assert "did NOT solve" in text
    assert "live blast tags unavailable" in text


# --- and the gaps are declared rather than left silent ---------------------------


def test_missing_history_is_declared_not_left_silent():
    """Silence is how a model invents a three-day trend."""

    context = _context(recent_frame=None)

    assert "last 3 days of history" in context.missing
    assert "NOT AVAILABLE FOR THIS RUN" in context.text
    assert "Do not speculate" in context.text


def test_missing_live_tags_are_declared():
    context = _context(live_snapshot=None)

    assert "live furnace tags" in context.missing
    assert "live furnace tags" in context.text


def test_no_solved_blend_is_not_usable():
    context = _context(lp_blend=None, de_blend=None, manual_blend=None)

    assert "a solved blend" in context.missing
    assert not context.is_usable


def test_violations_are_carried_verbatim():
    """A cost saving on an infeasible blend is not a saving."""

    blend = _Blend(violations=["Slag Al2O3 17.50% exceeds max 17.00%"])

    text = describe_blend(blend, "RECOMMENDED BLEND", _ores())

    assert "Slag Al2O3 17.50% exceeds max 17.00%" in text


def test_a_clean_blend_says_none_rather_than_omitting_the_line():
    """An absent line reads as missing data; "none" reads as checked and clear."""

    text = describe_blend(_Blend(), "RECOMMENDED BLEND", _ores())

    assert "Constraint violations: none" in text


# --- the instructions themselves --------------------------------------------------


def test_the_system_prompt_forbids_inventing_numbers():
    assert "Do not invent a number" in SYSTEM_PROMPT


def test_the_system_prompt_allows_saying_do_not_act():
    """A model that must always find a case will manufacture one."""

    assert "not worth acting on" in SYSTEM_PROMPT


def test_the_system_prompt_demands_action_reason_magnitude():
    assert "ACTION, REASON and MAGNITUDE" in SYSTEM_PROMPT


@pytest.mark.parametrize("heading", [
    "Furnace at present", "What the optimizer is proposing",
    "Why this makes sense", "What to watch out for",
])
def test_the_requested_sections_are_the_ones_asked_for(heading):
    assert heading in SYSTEM_PROMPT
