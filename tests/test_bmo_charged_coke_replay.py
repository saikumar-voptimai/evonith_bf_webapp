"""End-to-end replay acceptance on the exact bundled dataset.

From source rows through features, model, novelty and policy, the port must
reproduce every computable frozen output and every hourly display state of the
research package's September 15 - October 4 replay.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo.charged_coke.service import ChargedCokeEngine, forecast_hours

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "charged_coke"
FREEZE, END = pd.Timestamp("2026-09-15"), pd.Timestamp("2026-10-04 23:00")
TOLERANCE_KG_THM = 1e-4
FORBIDDEN = ("COKE RATE KG/THM", "ACT. FUEL RATEKG/THM.", "UNITCOST LAKHS/THM")
COMPARED = ["prediction", "spread", "linear", "attention", "novelty", "radius90", "lower", "upper",
            "display_forecast", "last_forecast"]


@pytest.fixture(scope="module")
def engine() -> ChargedCokeEngine:
    return ChargedCokeEngine.load()


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "source_window.csv.gz", parse_dates=["time"]).set_index("time")


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    frame = pd.read_csv(
        FIXTURES / "frozen_replay.csv.gz", parse_dates=["time", "last_issue", "last_window_end"]
    ).set_index("time")
    return frame.loc[FREEZE:END]


def _seed() -> dict:
    seed = json.loads((FIXTURES / "freeze_seed.json").read_text())
    return {
        "initial_streak": seed["initial_streak"],
        "initial_last": {"issue": pd.Timestamp(seed["last_issue"]), "value": seed["last_value"]},
    }


@pytest.fixture(scope="module")
def replay(source, engine) -> pd.DataFrame:
    return forecast_hours(source, engine, start=FREEZE, end=END, **_seed())


def _assert_same(mine: pd.DataFrame, theirs: pd.DataFrame, columns, atol: float) -> None:
    for column in columns:
        np.testing.assert_allclose(
            mine[column].astype(float).to_numpy(), theirs[column].astype(float).to_numpy(),
            atol=atol, rtol=0, equal_nan=True, err_msg=column,
        )


def test_every_computable_frozen_output_matches(replay, reference) -> None:
    mine = replay[replay["prediction"].notna()]
    theirs = reference[reference["prediction"].notna()]

    assert len(mine) == len(theirs) == 463
    assert mine.index.equals(theirs.index)
    _assert_same(mine, theirs, ["prediction", "spread", "linear", "attention"], TOLERANCE_KG_THM)


def test_all_480_display_states_match(replay, reference) -> None:
    assert replay.index.equals(reference.index)
    assert len(replay) == 480
    assert list(replay["state"]) == list(reference["state"])
    assert list(replay["reasons"]) == [json.loads(r) for r in reference["state_reasons_json"]]
    assert list(replay["unusual"]) == list(reference["unusual"].astype(str).str.lower().eq("true"))
    _assert_same(replay, reference, COMPARED, TOLERANCE_KG_THM)
    for column in ("last_issue", "last_window_end"):
        assert replay[column].equals(reference[column]), column
    assert replay["display_forecast"].notna().sum() == 452


def test_target_window_is_rows_two_to_five_after_issue(replay) -> None:
    row = replay.index[100]

    assert replay.loc[row, "target_start_row"] == row + pd.Timedelta(hours=2)
    assert replay.loc[row, "target_end_row"] == row + pd.Timedelta(hours=5)


def test_a_48_hour_replay_with_warmup_equals_full_history(source, engine, replay) -> None:
    start, end = pd.Timestamp("2026-09-28 00:00"), pd.Timestamp("2026-09-29 23:00")
    before = replay.loc[start - pd.Timedelta(hours=1)]
    seed = {"initial_streak": int(before["eligible_streak"]), "initial_last": None}
    if pd.notna(before["last_issue"]):
        seed["initial_last"] = {
            "issue": before["last_issue"],
            "value": float(replay.loc[before["last_issue"], "display_forecast"]),
        }

    short = forecast_hours(source, engine, start=start, end=end, **seed)

    full = replay.loc[start:end]
    assert list(short["state"]) == list(full["state"])
    assert list(short["reasons"]) == list(full["reasons"])
    _assert_same(short, full, COMPARED, 1e-9)
    # Last-row actuals need rows after ``end``; only earlier ones can match.
    _assert_same(short.iloc[:-5], full.iloc[:-5], ["actual"], 1e-9)


def test_48_raw_hours_alone_are_not_enough(source, engine, replay) -> None:
    start, end = pd.Timestamp("2026-09-28 00:00"), pd.Timestamp("2026-09-29 23:00")

    # Only the 48 hours being replayed are read: no warm-up, no context.
    starved = forecast_hours(
        source, engine, start=start, end=end, warmup=pd.Timedelta(0), context=pd.Timedelta(0)
    )

    full = replay.loc[start:end]
    differs = ~np.isclose(
        starved["prediction"].to_numpy(dtype=float), full["prediction"].to_numpy(dtype=float), equal_nan=True
    )
    # Delayed assays (24 h + 12 h fill) and the 24-h references need more.
    assert differs.any()


def test_rows_after_issue_cannot_change_an_issued_row(source, engine, replay) -> None:
    issue = pd.Timestamp("2026-09-28 12:00")
    start = issue - pd.Timedelta(hours=36)
    truncated = source.loc[:issue]
    before = replay.loc[start - pd.Timedelta(hours=1)]
    seed = {"initial_streak": int(before["eligible_streak"]), "initial_last": None}
    if pd.notna(before["last_issue"]):
        seed["initial_last"] = {
            "issue": before["last_issue"],
            "value": float(replay.loc[before["last_issue"], "display_forecast"]),
        }

    cut = forecast_hours(truncated, engine, start=start, end=issue, **seed)

    full = replay.loc[start:issue]
    assert list(cut["state"]) == list(full["state"])
    _assert_same(cut, full, COMPARED, 1e-9)
    # Outcomes are the only thing that needs the future, and only for scoring.
    assert cut["actual"].iloc[-5:].isna().all()


def test_reported_rates_cannot_reach_forecast_or_gate(source, engine) -> None:
    start, end = pd.Timestamp("2026-09-20 00:00"), pd.Timestamp("2026-09-21 23:00")
    altered = source.copy()
    altered["COKE RATE KG/THM"] = np.nan
    altered["ACT. FUEL RATEKG/THM."] = -1e9
    altered["UNITCOST LAKHS/THM"] = 1e9

    base = forecast_hours(source, engine, start=start, end=end)
    changed = forecast_hours(altered, engine, start=start, end=end)

    assert list(base["state"]) == list(changed["state"])
    _assert_same(base, changed, COMPARED, 0.0)
    assert not set(FORBIDDEN) & set(changed.columns)
