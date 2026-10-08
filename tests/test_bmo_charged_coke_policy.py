from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo.charged_coke import PolicyConfig, apply_policy, audit_hour, decide, novelty

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "charged_coke"
FREEZE = pd.Timestamp("2026-09-15")
_FLAGS = [
    "stable_original", "source_present", "sensor_frozen", "identity_bad",
    "operating_readings_valid", "shadow_possible", "data_ready", "unusual",
]


@pytest.fixture(scope="module")
def config() -> PolicyConfig:
    return PolicyConfig.load()


@pytest.fixture(scope="module")
def replay() -> pd.DataFrame:
    """The research package's own hourly ledger, 13 Sep to 4 Oct."""

    frame = pd.read_csv(FIXTURES / "frozen_replay.csv.gz", parse_dates=["time"])
    for column in _FLAGS:
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    for column in ("audit_reasons_json", "audit_codes_json", "state_reasons_json"):
        frame[column] = frame[column].map(json.loads)
    return frame.set_index("time")


def test_audit_reproduces_the_replay_severity_and_reasons(replay, config) -> None:
    audits = [audit_hour(row, config) for _, row in replay.iterrows()]

    assert [a["severity"] for a in audits] == list(replay["severity"])
    assert [a["data_ready"] for a in audits] == list(replay["data_ready"])
    assert [a["reasons"] for a in audits] == list(replay["audit_reasons_json"])
    assert [a["codes"] for a in audits] == list(replay["audit_codes_json"])


def test_novelty_matches_the_replay_after_the_freeze(replay, config) -> None:
    # Before the freeze the replay scored support hours leave-one-day-out;
    # afterwards every hour is a plain nearest-neighbour distance.
    frozen = replay.loc[FREEZE:]

    distance = novelty(frozen, config)

    np.testing.assert_allclose(distance, frozen["novelty"].to_numpy(), rtol=1e-9, equal_nan=True)
    assert np.isfinite(distance).sum() > 400


def test_hourly_decisions_reproduce_the_operator_replay(replay, config) -> None:
    hours = replay.assign(
        reasons=replay["audit_reasons_json"], codes=replay["audit_codes_json"]
    )

    decided = apply_policy(hours, config)

    frozen = decided.loc[FREEZE:]
    expected = replay.loc[FREEZE:]
    assert list(frozen["state"]) == list(expected["state"])
    np.testing.assert_allclose(
        frozen["radius90"].to_numpy(), expected["radius90"].to_numpy(), equal_nan=True
    )
    assert list(frozen["reasons"]) == list(expected["state_reasons_json"])
    # 452 of 480 hourly updates show a forecast, as the report states.
    assert len(frozen) == 480
    assert frozen["display_forecast"].notna().sum() == 452


def _eligible_hour(**overrides) -> dict:
    row = {
        "source_present": True, "data_ready": True, "prediction": 315.0, "spread": 1.0,
        "severity": "cruise", "novelty": 0.5, "unusual": False, "reasons": [], "codes": [],
    }
    row.update(overrides)
    return row


def test_a_hold_needs_two_eligible_updates_to_resume(config) -> None:
    p = config.parameters

    held = decide(_eligible_hour(severity="large"), p, eligible_streak=5)
    first = decide(_eligible_hour(), p, held["eligible_streak"])
    second = decide(_eligible_hour(), p, first["eligible_streak"])

    assert (held["state"], held["show_prediction"]) == ("large_hold", False)
    assert (first["state"], first["show_prediction"]) == ("recovering", False)
    assert first["reason_codes"][0] == "recovery_debounce"
    assert (second["state"], second["show_prediction"]) == ("cruise", True)
    assert second["radius90"] == pytest.approx(p["bins"]["cruise"]["radius"])


def test_a_planned_change_pauses_even_a_cruising_forecast(config) -> None:
    out = decide(_eligible_hour(), config.parameters, 3, planned_change=True)

    assert out["state"] == "planned_change_hold"
    assert out["reason_codes"][0] == "planned_change"
    assert np.isnan(out["radius90"])


def test_last_forecast_keeps_its_window_and_expires(config) -> None:
    index = pd.date_range("2026-10-06 10:00", periods=8, freq="h")
    hours = pd.DataFrame([_eligible_hour() for _ in index], index=index)
    hours.loc[index[2:], "severity"] = "large"  # paused from 12:00 on

    decided = apply_policy(hours, config)

    shown = decided.loc[index[1]]
    assert shown["state"] == "cruise"
    held = decided.loc[index[4]]
    assert held["state"] == "large_hold"
    assert held["last_issue"] == index[1]
    assert held["last_window_end"] == index[1] + pd.Timedelta(hours=5)
    assert held["last_forecast"] == pytest.approx(315.0)
    # Six hours after its issue the old forecast is gone, never rolled forward.
    assert np.isnan(decided.loc[index[7], "last_forecast"])
