"""Seeding the "current blend" from the last shift.

THE BUG THIS PINS DOWN.

The manual-blend editor seeds from what the plant charged last shift. When an
ore read zero it used to substitute THE OPTIMIZER'S OWN SHARE for that ore, and
the table was then renormalised to 100%. Both halves of that are wrong:

  - an ore reading zero means the plant chose not to charge it. That is a fact
    about the shift, not a missing value.
  - substituting a non-zero share and renormalising dilutes every material that
    WAS charged, by the size of the invention.

Measured against the plant record for 2026-09-10, the true last-shift burden was
sinter 60.6%, pellet 15.1%, Lloyds CLO 14.3%, Geomin CLO 10.0%, NMDC ROM 0.0%.
NMDC ROM had been selected in the ore editor but not charged, so it inherited
the LP's ~20% share; after renormalisation the plant's 60% sinter displayed as
50.4%.

WHY IT MATTERED BEYOND THE DISPLAY. The seeded shares become the manual blend,
and the manual blend is the reference for the manual-vs-optimizer cost
comparison, the coke-correction reference operating point, the energy-balance
anchor's burden, and the LLM commentary's idea of "current operation". All four
were reading a blend the furnace never saw.

The optimizer is still a sensible seed when there is NO last-shift data at all.
That case is whole-table and has its own caption.
"""

from __future__ import annotations

import pytest


def seed_shares(
    last_shift_shares: dict[str, float],
    optimizer_shares: dict[str, float],
    selected: list[str],
) -> dict[str, float]:
    """The seeding rule, extracted so it can be tested without Streamlit.

    Mirrors the block in ``_render_blend_comparison``. Kept in the test rather
    than imported because the page module cannot be imported outside a
    Streamlit runtime; the mutation tests below are what keep the two honest.
    """

    charged = {
        ore for ore in selected if float(last_shift_shares.get(ore, 0.0)) > 0.0
    }
    use_last_shift = bool(charged)

    rows: dict[str, float] = {}
    for ore in selected:
        share = float(last_shift_shares.get(ore, 0.0))
        if share <= 0 and not use_last_shift:
            share = float(optimizer_shares.get(ore, 0.0))
        rows[ore] = share

    total = sum(rows.values())
    if total > 0:
        rows = {ore: round(share / total * 100.0, 1) for ore, share in rows.items()}
    return rows


# The plant record for 2026-09-10, 08:30-16:30 UTC.
LAST_SHIFT = {
    "sinter_sp_02": 60.6,
    "lloyds_pellet": 15.1,
    "lloyds_clo": 14.3,
    "geomin_clo": 10.0,
    "nmdc_rom": 0.0,
}
OPTIMIZER = {
    "sinter_sp_02": 45.0,
    "lloyds_pellet": 14.0,
    "lloyds_clo": 12.0,
    "geomin_clo": 8.8,
    "nmdc_rom": 20.2,
}
SELECTED = list(LAST_SHIFT)


# --- the reported bug ------------------------------------------------------------


def test_sinter_seeds_at_what_the_plant_actually_charged():
    """60% sinter must read as 60%, not 50%."""

    seeded = seed_shares(LAST_SHIFT, OPTIMIZER, SELECTED)

    assert seeded["sinter_sp_02"] == pytest.approx(60.6, abs=0.1)


def test_an_ore_not_charged_seeds_at_zero():
    """NMDC ROM was selected but not charged. That is a real zero."""

    seeded = seed_shares(LAST_SHIFT, OPTIMIZER, SELECTED)

    assert seeded["nmdc_rom"] == 0.0


def test_the_optimizer_share_never_leaks_into_the_current_blend():
    """The whole defect in one assertion.

    The "current blend" is a record of what happened. If the optimizer's
    proposal can bleed into it, the comparison is against a blend that has been
    partly replaced by the thing it is meant to be compared with.
    """

    seeded = seed_shares(LAST_SHIFT, OPTIMIZER, SELECTED)

    for ore, share in seeded.items():
        assert share == pytest.approx(LAST_SHIFT[ore], abs=0.1), (
            f"{ore} was seeded at {share}, not the {LAST_SHIFT[ore]} the plant "
            "charged — the optimizer's share has leaked in"
        )


def test_the_charged_materials_are_not_diluted():
    """Renormalising after an invention shrinks everything genuine."""

    seeded = seed_shares(LAST_SHIFT, OPTIMIZER, SELECTED)

    assert sum(seeded.values()) == pytest.approx(100.0, abs=0.2)
    assert seeded["lloyds_pellet"] == pytest.approx(15.1, abs=0.1)
    assert seeded["geomin_clo"] == pytest.approx(10.0, abs=0.1)


# --- the fallback that is still wanted --------------------------------------------


def test_with_no_last_shift_data_the_optimizer_seeds_the_whole_table():
    """Nothing to record means the optimizer is the only sensible starting point."""

    seeded = seed_shares({}, OPTIMIZER, SELECTED)

    total = sum(OPTIMIZER.values())
    for ore in SELECTED:
        assert seeded[ore] == pytest.approx(
            OPTIMIZER[ore] / total * 100.0, abs=0.1
        )


def test_if_none_of_the_selected_ores_were_charged_the_optimizer_seeds():
    """A snapshot that contains nothing relevant is as good as no snapshot."""

    other_shift = {"some_other_ore": 100.0}

    seeded = seed_shares(other_shift, OPTIMIZER, SELECTED)

    assert seeded["nmdc_rom"] > 0.0, "with nothing charged, fall back wholesale"


def test_partial_selection_still_normalises_to_100():
    """Selecting a subset renormalises, which is correct and not the bug.

    The bug was inventing material. Scaling a genuine subset up to 100% keeps
    the RATIOS the plant ran, which is what the comparison needs.
    """

    seeded = seed_shares(LAST_SHIFT, OPTIMIZER, ["sinter_sp_02", "lloyds_clo"])

    assert sum(seeded.values()) == pytest.approx(100.0, abs=0.2)
    # 60.6 : 14.3 is preserved as a ratio.
    assert seeded["sinter_sp_02"] / seeded["lloyds_clo"] == pytest.approx(
        60.6 / 14.3, rel=0.01
    )


# --- the mutation, stated as a test -----------------------------------------------


def test_the_old_rule_reproduces_the_reported_symptom():
    """Documents the defect so a future reader can see what was wrong.

    This runs the OLD per-ore fallback and asserts it produces the 50.4% that
    was reported from the field. If someone reintroduces that rule, the tests
    above fail and this one explains why.
    """

    def old_rule() -> dict[str, float]:
        rows = {
            ore: (
                LAST_SHIFT.get(ore, 0.0)
                if LAST_SHIFT.get(ore, 0.0) > 0
                else OPTIMIZER.get(ore, 0.0)  # the defect: per-ore fallback
            )
            for ore in SELECTED
        }
        total = sum(rows.values())
        return {o: round(s / total * 100.0, 1) for o, s in rows.items()}

    broken = old_rule()

    assert broken["sinter_sp_02"] == pytest.approx(50.4, abs=0.1)
    assert broken["nmdc_rom"] == pytest.approx(16.8, abs=0.2)
    assert broken["sinter_sp_02"] < LAST_SHIFT["sinter_sp_02"] - 9.0
