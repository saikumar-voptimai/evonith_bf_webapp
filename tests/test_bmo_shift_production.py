from __future__ import annotations

import pandas as pd
import pytest

from utils.bmo.shift_production import (
    last_complete_shift,
    production_change_pct,
    shift_production,
)


@pytest.mark.parametrize(
    ("now_ist", "label", "start", "end"),
    [
        # 02:26 IST: C has only just started, so B (14-22) is the last complete.
        ("2026-09-30 02:26", "B", "2026-09-29 14:00", "2026-09-29 22:00"),
        # Exactly at 06:00 the night shift C (22-06) has ended.
        ("2026-09-30 06:00", "C", "2026-09-29 22:00", "2026-09-30 06:00"),
        ("2026-09-30 13:59", "C", "2026-09-29 22:00", "2026-09-30 06:00"),
        ("2026-09-30 14:00", "A", "2026-09-30 06:00", "2026-09-30 14:00"),
        ("2026-09-30 23:10", "B", "2026-09-30 14:00", "2026-09-30 22:00"),
    ],
)
def test_last_complete_shift_follows_the_plant_schedule(now_ist, label, start, end):
    now = pd.Timestamp(now_ist, tz="Asia/Kolkata")

    got_label, got_start, got_end = last_complete_shift(now)

    assert got_label == label
    assert got_start == pd.Timestamp(start, tz="Asia/Kolkata")
    assert got_end == pd.Timestamp(end, tz="Asia/Kolkata")


def _quarter_hours(start_ist: str, periods: int, charges: list[float]) -> pd.DataFrame:
    index = pd.date_range(start_ist, periods=periods, freq="15min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "charges_per_hour": (charges * periods)[:periods],
            "hm_per_charge": 14.85,
        },
        index=index.tz_convert("UTC"),
    )


def test_shift_production_is_charges_times_hm_per_charge_times_24():
    # 15-minute averages stamped at the END of each quarter hour, as the live
    # tags are, covering shift B and an hour either side.
    samples = _quarter_hours("2026-09-29 13:15", 40, [6.0, 6.5])
    samples.loc[samples.index < pd.Timestamp("2026-09-29 08:30Z"), "charges_per_hour"] = 99.0

    result = shift_production(
        samples, now=pd.Timestamp("2026-09-30 02:26", tz="Asia/Kolkata"), source="live"
    )

    assert result.shift_label == "B"
    assert result.hours == pytest.approx(8.0)
    assert result.charges_per_hour == pytest.approx(6.25)
    assert result.hm_per_charge_mt == pytest.approx(14.85)
    assert result.daily_production_mt == pytest.approx(6.25 * 14.85 * 24)
    assert result.usable
    assert result.describe() == "Shift B, 29 Sep 14:00-22:00 IST"


def test_a_shift_with_too_little_data_is_not_used():
    samples = _quarter_hours("2026-09-29 14:15", 8, [6.2])  # two hours only

    result = shift_production(
        samples, now=pd.Timestamp("2026-09-30 02:26", tz="Asia/Kolkata")
    )

    assert result.hours == pytest.approx(2.0)
    assert not result.usable


def test_hour_start_stamps_select_the_shift_hours():
    index = pd.date_range("2026-09-29 12:00", periods=12, freq="h", tz="Asia/Kolkata")
    samples = pd.DataFrame({"charges_per_hour": 6.0, "hm_per_charge": 15.0}, index=index)

    result = shift_production(
        samples,
        now=pd.Timestamp("2026-09-29 23:00", tz="Asia/Kolkata"),
        timestamps="start",
    )

    # Hours starting 14:00..21:00 are inside shift B; 12:00, 13:00 and 22:00, 23:00 are not.
    assert result.hours == pytest.approx(8.0)
    assert result.daily_production_mt == pytest.approx(6.0 * 15.0 * 24)


def test_production_change_is_relative_to_the_baseline():
    assert production_change_pct(2270.0, 2240.0) == pytest.approx(1.339, abs=1e-3)
    assert production_change_pct(2200.0, 2240.0) < 0
    assert production_change_pct(2270.0, None) is None
    assert production_change_pct(2270.0, 0.0) is None
