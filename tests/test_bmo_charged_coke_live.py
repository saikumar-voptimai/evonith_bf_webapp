"""Live contract: which row is complete, issue once per hour, planned changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo.charged_coke.contract import ForecastWindow, latest_complete_row
from utils.bmo.charged_coke.ledger import ForecastLedger, PlannedChangeRegister
from utils.bmo.charged_coke.service import ChargedCokeEngine, forecast_hours, issue_latest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "charged_coke"
FREEZE, END = pd.Timestamp("2026-09-15"), pd.Timestamp("2026-10-04 23:00")


@pytest.fixture(scope="module")
def engine() -> ChargedCokeEngine:
    return ChargedCokeEngine.load()


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "source_window.csv.gz", parse_dates=["time"]).set_index("time")


@pytest.fixture(scope="module")
def replay(source, engine) -> pd.DataFrame:
    seed = json.loads((FIXTURES / "freeze_seed.json").read_text())
    return forecast_hours(
        source, engine, start=FREEZE, end=END, initial_streak=seed["initial_streak"],
        initial_last={"issue": pd.Timestamp(seed["last_issue"]), "value": seed["last_value"]},
    )


def _published(source: pd.DataFrame, row: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    """The file as built at row + 1 h 06 min: it ends with the hour in progress."""

    return source.loc[: row + pd.Timedelta(hours=1)], row + pd.Timedelta(hours=1, minutes=6)


def test_the_newest_row_of_a_build_is_never_complete() -> None:
    built = pd.Timestamp("2026-10-07 04:06:53")

    assert latest_complete_row(built, pd.Timestamp("2026-10-07 04:00")) == pd.Timestamp("2026-10-07 03:00")
    # Built late in the hour, the newest row's online half has been measured.
    late = pd.Timestamp("2026-10-07 04:45")
    assert latest_complete_row(late, pd.Timestamp("2026-10-07 04:00")) == pd.Timestamp("2026-10-07 04:00")
    # A tz-aware Last-Modified (GMT) is read on the plant clock.
    gmt = pd.Timestamp("2026-10-06 22:36:53", tz="UTC")
    assert latest_complete_row(gmt, pd.Timestamp("2026-10-07 04:00")) == pd.Timestamp("2026-10-07 03:00")


def test_window_keeps_data_issue_and_target_times_apart() -> None:
    window = ForecastWindow.for_row(pd.Timestamp("2026-10-07 03:00"), pd.Timestamp("2026-10-07 04:08"))

    assert window.data_complete_at == pd.Timestamp("2026-10-07 03:30")
    assert window.issued_at == pd.Timestamp("2026-10-07 04:08")
    # Rows 05..08: charges 04:00-08:00, production 04:30-08:30.
    assert (window.coke_start, window.coke_end) == (pd.Timestamp("2026-10-07 04:00"), pd.Timestamp("2026-10-07 08:00"))
    assert (window.production_start, window.production_end) == (
        pd.Timestamp("2026-10-07 04:30"), pd.Timestamp("2026-10-07 08:30"),
    )
    assert window.describe() == "Coke charged 04:00–08:00 (production 04:30–08:30) IST"


def test_hourly_issues_reproduce_the_replay_and_are_written_once(tmp_path, source, engine, replay) -> None:
    ledger, plans = ForecastLedger(tmp_path), PlannedChangeRegister(tmp_path)
    rows = pd.date_range("2026-09-28 10:00", "2026-09-28 21:00", freq="h")  # spans a hold and recovery

    for row in rows:
        published, built = _published(source, row)
        record = issue_latest(published, engine, ledger=ledger, plans=plans, now=built, built_at=built)
        # A refresh five minutes later finds the same record and writes nothing.
        again = issue_latest(published, engine, ledger=ledger, plans=plans, now=built + pd.Timedelta(minutes=5), built_at=built)
        assert again == record

        expected = replay.loc[row]
        assert pd.Timestamp(record["data_row"]) == row
        assert record["state"] == expected["state"], row
        assert record["reasons"][: len(expected["reasons"])] == list(expected["reasons"])
        if record["shown"]:
            assert record["prediction_kg_thm"] == pytest.approx(expected["prediction"], abs=1e-9)
        else:
            assert record["prediction_kg_thm"] is None

    events = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").strip().splitlines()]
    assert sum(e["type"] == "issue" for e in events) == len(rows)


def test_late_revisions_cannot_rewrite_an_issued_forecast(tmp_path, source, engine) -> None:
    ledger, plans = ForecastLedger(tmp_path), PlannedChangeRegister(tmp_path)
    row = pd.Timestamp("2026-09-25 12:00")
    published, built = _published(source, row)
    first = issue_latest(published, engine, ledger=ledger, plans=plans, now=built, built_at=built)

    revised = published.copy()
    revised.loc[row - pd.Timedelta(hours=2): row, "COKE_CALC_MT"] *= 1.2  # republished history
    second = issue_latest(revised, engine, ledger=ledger, plans=plans, now=built + pd.Timedelta(hours=3), built_at=built)

    assert second == first


def test_planned_change_pauses_until_expiry_then_needs_two_eligible_hours(tmp_path, source, engine, replay) -> None:
    ledger, plans = ForecastLedger(tmp_path), PlannedChangeRegister(tmp_path)
    start = pd.Timestamp("2026-09-25 09:00")
    rows = pd.date_range(start, periods=6, freq="h")
    assert replay.loc[rows, "state"].isin(["cruise", "unsettled", "off_cruise"]).all()
    plans.declare(
        reason="Blast cut for tuyere change", valid_from=start + pd.Timedelta(hours=1),
        valid_until=start + pd.Timedelta(hours=3, minutes=6), created_by="shift in-charge",
        now=start + pd.Timedelta(minutes=50),
    )

    states = []
    for row in rows:
        published, built = _published(source, row)
        record = issue_latest(published, engine, ledger=ledger, plans=plans, now=built, built_at=built)
        states.append(record["state"])
        if record["state"] == "planned_change_hold":
            assert record["codes"][0] == "planned_change"
            assert record["planned_change_ids"]

    # Rows 09:00 and 10:00 are due at 10:06 and 11:06, inside the validity:
    # held. Row 11:00 is due at 12:06, the expiry: the first eligible update
    # only recovers, the next shows again. Ordinary checks are back in force.
    assert states[:2] == ["planned_change_hold", "planned_change_hold"]
    assert states[2] == "recovering"
    assert states[3] in {"cruise", "unsettled", "off_cruise"}


def test_planned_change_validity_is_explicit_bounded_and_audited(tmp_path) -> None:
    plans = PlannedChangeRegister(tmp_path)
    now = pd.Timestamp("2026-10-07 09:00")

    with pytest.raises(ValueError, match="at most"):
        plans.declare(reason="reline", valid_from=now, valid_until=now + pd.Timedelta(days=3), created_by="x", now=now)
    with pytest.raises(ValueError, match="reason"):
        plans.declare(reason=" ", valid_from=now, valid_until=now + pd.Timedelta(hours=2), created_by="x", now=now)
    change = plans.declare(reason="PCI off for lance change", valid_from=now,
                           valid_until=now + pd.Timedelta(hours=4), created_by="A. Operator", now=now)
    plans.cancel(change.id, cancelled_by="B. Engineer", now=now + pd.Timedelta(hours=1), note="done early")

    assert plans.active(now + pd.Timedelta(minutes=30))
    assert not plans.active(now + pd.Timedelta(hours=2))
    events = [json.loads(line) for line in plans.path.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["declare", "cancel"]
    assert events[0]["created_by"] == "A. Operator" and events[1]["cancelled_by"] == "B. Engineer"


def test_outcomes_are_appended_once_and_feed_the_scorecard(tmp_path, source, engine) -> None:
    from utils.bmo.charged_coke.monitoring import scorecard

    ledger, plans = ForecastLedger(tmp_path), PlannedChangeRegister(tmp_path)
    rows = pd.date_range("2026-09-25 06:00", periods=10, freq="h")
    for row in rows:
        published, built = _published(source, row)
        issue_latest(published, engine, ledger=ledger, plans=plans, now=built, built_at=built)

    outcomes = ledger.outcomes()
    # A block (rows t+2..t+5) has arrived once row t+5 is complete: by the last
    # issue (row 15:00), rows 06:00..10:00 have matured, later ones have not.
    assert set(outcomes) == set(rows[:5])
    issued = ledger.frame()
    assert issued.loc[rows[0], "state"]  # the issue record is untouched
    card = scorecard(issued, beta=engine.model.beta)
    assert card["hours"] == 10
    assert card["by_state"]["all_shown"]["n"] == 5
    assert card["response"]["signs_as_expected"]
    assert 0 <= card["availability"]["shown_share"] <= 1
