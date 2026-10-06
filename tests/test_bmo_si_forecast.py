from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from utils.bmo.si_prediction import latest_cast_si, si_horizon_hours

ROOT = Path(__file__).resolve().parents[1]


def test_horizon_is_the_shipped_models_newest_previous_si_lag() -> None:
    features = json.loads(
        (ROOT / "src/assets/models/hm_si_feature_columns.json").read_text()
    )

    # CHEM_PCT_SI__lag3h and __lag4h: the cast three hours after the analysis.
    assert si_horizon_hours(features) == 3


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (["CHEM_PCT_SI__lag6h", "CHEM_PCT_SI__lag4h"], 4),
        (["HOT BLAST VOLUMENM3/HR.__lag2h", "CHEM_PCT_SI__lag2"], 2),
        (["HOT BLAST VOLUMENM3/HR.__lag2h", "CHEM_PCT_C"], None),
    ],
)
def test_horizon_follows_whatever_lag_the_model_was_trained_with(
    features, expected
) -> None:
    assert si_horizon_hours(features) == expected


def _history(last_utc: str, value: float) -> pd.DataFrame:
    index = pd.date_range(end=last_utc, periods=4, freq="h", tz="UTC")
    return pd.DataFrame({"CHEM_PCT_SI": [0.4, 0.45, 0.5, value]}, index=index)


def test_the_later_cast_wins() -> None:
    history = _history("2026-10-03 15:00", 0.56)
    report = {"chem_pct_si": 0.91, "sample_timestamp": "2026-10-06T08:59:00+00:00"}

    # The static history trails the plant by days; the HM report is today's.
    assert latest_cast_si(history, report) == (
        0.91,
        pd.Timestamp("2026-10-06 08:59", tz="UTC"),
    )
    older_report = {**report, "sample_timestamp": "2026-10-01T08:59:00+00:00"}
    assert latest_cast_si(history, older_report) == (
        0.56,
        pd.Timestamp("2026-10-03 15:00", tz="UTC"),
    )


def test_missing_or_zero_casts_are_skipped() -> None:
    history = _history("2026-10-03 15:00", 0.56)
    history.iloc[-1, 0] = float("nan")

    value, at = latest_cast_si(history, {"chem_pct_si": 0.0})

    assert value == 0.5
    assert at == pd.Timestamp("2026-10-03 14:00", tz="UTC")
    assert latest_cast_si(None, {"chem_pct_si": 0.7}) == (0.7, None)
    assert latest_cast_si(None, None) == (None, None)


def test_a_naive_history_index_is_read_as_utc() -> None:
    history = _history("2026-10-03 15:00", 0.56)
    history.index = history.index.tz_localize(None)

    assert latest_cast_si(history)[1] == pd.Timestamp("2026-10-03 15:00", tz="UTC")
