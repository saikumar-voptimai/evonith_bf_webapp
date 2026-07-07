from __future__ import annotations

from datetime import date

import pandas as pd

from apps.frontend_streamlit.reports.furnace_report.builder import ShiftBuilder
from apps.frontend_streamlit.reports.furnace_report.data import ShiftRawData
from apps.frontend_streamlit.reports.furnace_report.fetcher import _total_o2_flow_window
from apps.frontend_streamlit.reports.furnace_report.timeframe import get_report_timeframe


def test_shift_timeframe_preserves_configured_shift_window() -> None:
    timeframe = get_report_timeframe("Shift", date(2026, 5, 27), "C")

    assert timeframe.window_id == "2026-05-27_SHIFT_C"
    assert timeframe.shift_label == "C"
    assert timeframe.start_ist.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-27 22:00:00"
    assert timeframe.end_ist.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-28 06:00:00"
    assert timeframe.duration_hours == 8.0


def test_day_timeframe_uses_midnight_to_next_midnight() -> None:
    timeframe = get_report_timeframe("Day", date(2026, 5, 27), "A")

    assert timeframe.window_id == "2026-05-27"
    assert timeframe.shift_label is None
    assert timeframe.start_ist.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-27 00:00:00"
    assert timeframe.end_ist.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-28 00:00:00"
    assert timeframe.duration_hours == 24.0


def test_day_total_o2_flow_window_uses_0015_to_next_day_0014() -> None:
    timeframe = get_report_timeframe("Day", date(2026, 5, 27))
    start_ist, end_ist = _total_o2_flow_window(timeframe)

    assert start_ist.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-27 00:15:00"
    assert end_ist.strftime("%Y-%m-%d %H:%M:%S") == "2026-05-28 00:14:00"


def test_builder_scales_production_by_report_window_duration() -> None:
    timeframe = get_report_timeframe("Day", date(2026, 5, 27))
    report = ShiftBuilder().build(
        ShiftRawData(
            shift_date=timeframe.selected_date,
            shift_label=timeframe.shift_label,
            shift_start_ist=timeframe.start_ist,
            shift_end_ist=timeframe.end_ist,
            online_df=pd.DataFrame({"production_per_hour": [100.0]}),
            hm_slag_df=pd.DataFrame(),
            charge_df=pd.DataFrame(),
            report_type=timeframe.report_type,
        )
    )

    assert report.report_type == "Day"
    assert report.theoretical_production == 2400.0


def test_day_report_ibrm_uses_day_theoretical_production() -> None:
    timeframe = get_report_timeframe("Day", date(2026, 5, 27))
    sample_time = timeframe.start_ist + pd.Timedelta(hours=1)
    report = ShiftBuilder().build(
        ShiftRawData(
            shift_date=timeframe.selected_date,
            shift_label=timeframe.shift_label,
            shift_start_ist=timeframe.start_ist,
            shift_end_ist=timeframe.end_ist,
            online_df=pd.DataFrame({"production_per_hour": [100.0]}),
            hm_slag_df=pd.DataFrame(),
            charge_df=pd.DataFrame(
                {"ore_1_mt": [20.0], "pellet_1_mt": [10.0], "sinter_1_mt": [40.0]}
            ),
            ore_chemistry_df=pd.DataFrame(
                [
                    {"date_time": sample_time, "material_code": "ore_1", "tm": 4.0},
                    {"date_time": sample_time, "material_code": "pellet_1", "tm": 5.0},
                ]
            ),
            report_type=timeframe.report_type,
        )
    )

    assert report.ibrm == 0.03


