"""Page 9 (Blend Mix Optimiser) output on five recorded plant days is unchanged (SV-3.4, #301).

Each case runs the page source under Streamlit's ``AppTest`` on committed plant data (``tests/bmo_golden.py``) and
compares the LP plan, the result tables, the charged-coke and HM Si panels and the saved snapshot with the golden
record (floats within a relative 1e-6). After an intended change of page-9 output, rebuild the records with
``python -m tests.bmo_golden --write`` and review their diff.
"""

from __future__ import annotations

import os

import pytest

from tests import bmo_golden

CASES = {case.day: case for case in bmo_golden.CASES}


def test_five_recorded_days() -> None:
    assert len(CASES) == 5
    assert all(bmo_golden.golden_path(case).exists() for case in CASES.values())


@pytest.mark.parametrize("day", sorted(CASES))
def test_page9_output_unchanged(day: str) -> None:
    for name in ("COKE_ENERGY_WEATHER_LIVE", "COKE_ENERGY_TODAY"):  # the two the page's output depends on
        expected = bmo_golden.TEST_ENVIRONMENT[name]
        assert os.environ.get(name) == expected, f"{name} differs from the environment the records were written in"
    case = CASES[day]
    record = bmo_golden.normalise(bmo_golden.run_case(case))
    diffs = bmo_golden.compare(bmo_golden.load_golden(case), record)
    assert not diffs, f"{len(diffs)} differences from the golden record of {day}:\n" + "\n".join(diffs[:40])


def test_compare_tolerates_relative_1e6_only() -> None:
    golden = {"a": 1000.0, "b": [1, "x"], "c": None}
    assert bmo_golden.compare(golden, {"a": 1000.0009, "b": [1, "x"], "c": None}) == []
    assert bmo_golden.compare(golden, {"a": 1000.002, "b": [1, "x"], "c": None}) == ["$.a: 1000.0 != 1000.002"]
    assert bmo_golden.compare(golden, {"a": 1000.0, "b": [1, "y"], "c": None}) == ["$.b[1]: 'x' != 'y'"]
    assert bmo_golden.compare(golden, {"a": 1000.0, "b": [1]}) == ["$.c: missing", "$.b: length 2 != 1"]
