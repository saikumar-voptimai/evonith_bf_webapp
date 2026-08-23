"""Golden benchmark against the plant's own RAFT workbook.

Every expected value here is a cached result from Raft Calculation EML.xlsx, as
tabulated in section 17 of the implementation guide. This is the plant's
engineering ground truth, so these are parity tests: if one fails, our
arithmetic has drifted from the sheet, not the other way round.

The benchmark case is recovered from the guide's intermediates:
    blast 75,000 Nm3/h (total mixed), O2 5,000 Nm3/h, steam 0 t/h,
    humidity 15 g/Nm3, blast 1180 C, coal 14 t/h.
"""

from __future__ import annotations

import pytest

from utils.energy_balance.raft import (
    compute_blast_balance,
    compute_raft,
    oxygen_flow_from_enrichment,
    solve_steam_for_raft,
    steam_sensitivity_nm3_h_per_c,
)

BALANCE = dict(
    blast_volume_nm3_h=75_000.0,
    oxygen_injection_nm3_h=5_000.0,
    steam_injection_t_h=0.0,
    ambient_humidity_g_nm3=15.0,
    coal_injection_t_h=14.0,
    blast_flow_basis="total",
)
CASE = {**BALANCE, "blast_temperature_c": 1180.0}
SETPOINT_C = 2170.0
# The guide quotes intermediates to 6 decimal places, so parity is asserted at
# 1e-5 rather than machine precision - a tighter bound would be testing the
# document's rounding, not our arithmetic.
REL = 1e-5


# --- section 7: blast and moisture balance --------------------------------------


@pytest.mark.parametrize(
    "field, expected",
    [
        ("cold_blast_nm3_h", 70_000.0),
        ("ambient_water_kg_h", 1_050.0),
        ("ambient_steam_nm3_h", 1_306.666667),
        ("dry_blast_nm3_h", 68_693.333333),
        ("o2_from_air_nm3_h", 14_356.906667),
        ("n2_from_air_nm3_h", 54_336.426667),
        ("total_o2_nm3_h", 19_356.906667),
        ("total_n2_nm3_h", 54_336.426667),
        ("total_steam_nm3_h", 1_306.666667),
        ("o2_pct_dry", 26.2668355),
        ("steam_loading_g_per_nm3_dry", 15.285326),
        ("coal_loading_kg_per_nm3_dry", 0.203804),
        ("o2_ratio", 0.0727873),
    ],
)
def test_blast_balance_matches_the_workbook(field, expected):
    balance = compute_blast_balance(**BALANCE)

    assert getattr(balance, field) == pytest.approx(expected, rel=REL)


def test_the_species_sum_back_to_the_entered_blast():
    """O2 + N2 + steam must return the 75,000 that went in.

    On the total basis this is a real check: enrichment oxygen and injected
    steam were subtracted out at the start, so if they do not come back the
    basis handling is wrong.
    """

    b = compute_blast_balance(**BALANCE)

    assert (
        b.total_o2_nm3_h + b.total_n2_nm3_h + b.total_steam_nm3_h
    ) == pytest.approx(75_000.0, rel=1e-9)


# --- section 8: the RAFT equation ------------------------------------------------


def test_raft_matches_the_workbook_to_six_figures():
    assert compute_raft(**CASE).raft_c == pytest.approx(2205.250833, rel=REL)


@pytest.mark.parametrize(
    "component, expected",
    [
        ("base_c", 1559.0),
        ("blast_temp_c", 990.020),
        ("oxygen_c", 361.898292),
        ("steam_c", 92.216372),
        ("coal_c", 613.451087),
    ],
)
def test_each_raft_component_matches(component, expected):
    """Checked individually so a compensating pair of errors cannot hide."""

    assert getattr(compute_raft(**CASE), component) == pytest.approx(expected, rel=REL)


def test_components_reconstruct_the_total():
    r = compute_raft(**CASE)

    assert sum(r.components().values()) == pytest.approx(r.raft_c, rel=1e-9)


# --- section 9: the steam correction, and the defect it fixes --------------------


def test_steam_sensitivity_matches_the_workbook():
    balance = compute_blast_balance(**BALANCE)

    assert steam_sensitivity_nm3_h_per_c(balance.dry_blast_nm3_h) == pytest.approx(
        14.169574, rel=REL
    )


def test_more_steam_is_required_when_raft_is_above_setpoint():
    """THE CRITICAL DEFECT in the source workbook, pinned so it cannot return.

    Sheet1!J27 returns NEGATIVE steam when RAFT is above setpoint - it would
    push RAFT further up. RAFT here is 2205 C against a 2170 C setpoint, so the
    answer must be MORE steam than the zero currently flowing.
    """

    current = compute_raft(**CASE).raft_c
    assert current > SETPOINT_C

    solved = solve_steam_for_raft(target_raft_c=SETPOINT_C, **{
        k: v for k, v in CASE.items() if k != "steam_injection_t_h"
    })

    assert solved["reachable"]
    assert solved["steam_t_h"] > 0.0, "negative steam is the defect being fixed"
    assert solved["raft_c"] == pytest.approx(SETPOINT_C, abs=0.01)


def test_the_full_solve_beats_the_linear_estimate():
    """Guide section 9.5: linear gives +0.4014 t/h, the full solve +0.37935.

    Steam changes the dry blast, which changes all three loadings, so the local
    sensitivity over-states what is needed. The gap is small here but it is the
    reason the guide asks for a root solve rather than a multiplication.
    """

    solved = solve_steam_for_raft(target_raft_c=SETPOINT_C, **{
        k: v for k, v in CASE.items() if k != "steam_injection_t_h"
    })

    assert solved["steam_t_h"] == pytest.approx(0.37935, abs=0.005)
    assert solved["steam_t_h"] < 0.401375


def test_an_unreachable_target_says_so_rather_than_clamping():
    """A clamped number presented as a solution is worse than an honest refusal."""

    result = solve_steam_for_raft(
        target_raft_c=1500.0, max_steam_t_h=2.0,
        **{k: v for k, v in CASE.items() if k != "steam_injection_t_h"},
    )

    assert result["reachable"] is False
    assert "below what" in result["reason"]


# --- directions and guards --------------------------------------------------------


@pytest.mark.parametrize(
    "field, delta, should_rise",
    [
        ("blast_temperature_c", +50.0, True),
        ("oxygen_injection_nm3_h", +1000.0, True),
        ("steam_injection_t_h", +0.5, False),
        ("coal_injection_t_h", +2.0, False),
        ("ambient_humidity_g_nm3", +5.0, False),
    ],
)
def test_each_control_moves_raft_the_right_way(field, delta, should_rise):
    base = compute_raft(**CASE).raft_c
    moved = compute_raft(**{**CASE, field: CASE[field] + delta}).raft_c

    assert (moved > base) is should_rise


def test_a_flow_basis_mistake_is_caught_not_silently_wrong():
    """Subtracting O2 and steam from a cold-blast reading can go negative."""

    with pytest.raises(ValueError, match="blast_flow_basis"):
        compute_blast_balance(**{**BALANCE, "blast_volume_nm3_h": 3_000.0})


def test_enrichment_converts_back_to_the_oxygen_flow_it_came_from():
    """Round trip, because the optimiser works in enrichment and RAFT in flow."""

    balance = compute_blast_balance(**BALANCE)
    o2 = oxygen_flow_from_enrichment(
        enrichment_pct=balance.o2_pct_dry - 20.9,
        blast_volume_nm3_h=CASE["blast_volume_nm3_h"],
        ambient_humidity_g_nm3=CASE["ambient_humidity_g_nm3"],
    )

    assert o2 == pytest.approx(CASE["oxygen_injection_nm3_h"], rel=0.02)
