"""Unit tests for Furnace Status domain rules and feature orchestration."""

from __future__ import annotations

import html
import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domain import furnace_status as fs
from furnace_data.influx.query import influx_fields
from data import furnace_status_service as fs_data

NOW = datetime(2026, 10, 5, 9, 30, 45, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_feature_caches():
    fs_data.clear_furnace_status_caches()
    yield
    fs_data.clear_furnace_status_caches()


EXPECTED_LABELS = [
    "Production (Theor.)",
    "Production Rate",
    "Hot Blast Volume",
    "Blast Pressure",
    "Top Pressure",
    "CO Utilization",
    "Top Gas H₂",
    "HBT",
    "RAFT",
    "T1",
    "T2",
    "T3",
    "T4",
    "T Avg",
    "Steam Injection",
    "Steam Bypass Flow",
    "O₂ Injection",
    "Oxygen Flow",
    "Fuel Rate",
    "PCI Rate (SP/ACT.)",
    "Slag Rate",
    "Permeability",
    "Tuyere Velocity",
    "Furnace Level",
    "Total Heat Load",
    "Q1 (Staves 1–8)",
    "Q2 (Staves 9–16)",
    "Q3 (Staves 17–24)",
    "Q4 (Staves 25–32)",
    "Pad A",
    "Pad B",
    "Pad C",
    "Pad D",
    "Pad Avg",
]


def make_frame(
    columns: dict[str, list[float]], *, age_minutes: float = 0.0
) -> pd.DataFrame:
    """Build an IST-indexed 1-minute frame whose last row is ``age_minutes`` old."""
    n = max(len(v) for v in columns.values())
    end = pd.Timestamp(NOW - timedelta(minutes=age_minutes)).tz_convert("Asia/Kolkata")
    index = pd.date_range(end=end, periods=n, freq="1min", name="time (IST)")
    return pd.DataFrame(columns, index=index)


# ── Catalogue ────────────────────────────────────────────────────────────────


def test_parameter_keys_are_unique() -> None:
    keys = [p.key for p in fs.PARAMETERS]
    assert len(keys) == len(set(keys))


def test_all_reference_labels_present_in_order() -> None:
    assert [p.label for p in fs.PARAMETERS] == EXPECTED_LABELS


def test_every_parameter_belongs_to_a_known_section() -> None:
    assert {p.section for p in fs.PARAMETERS} <= set(fs.SECTIONS)
    assert sum(len(fs.parameters_in_section(s)) for s in fs.SECTIONS) == 34


def test_source_backed_fields_are_actually_queried() -> None:
    """The query builder only selects fields listed in ``data_mapping``."""
    for spec in fs.PARAMETERS:
        if spec.has_source:
            for field in spec.source_fields:
                assert field in influx_fields(spec.measurement), (spec.key, field)


def test_unsourced_parameters_explain_why() -> None:
    for spec in fs.PARAMETERS:
        if not spec.has_source:
            assert spec.unavailable_reason, spec.key


def test_measurements_are_grouped_not_per_parameter() -> None:
    assert fs.measurements_for() == (
        "process_params",
        "miscellaneous",
        "heatload_delta_t",
        "temperature_profile",
    )


# ── Value selection ──────────────────────────────────────────────────────────


def test_latest_finite_value_is_selected() -> None:
    frame = make_frame({"fuel_rate": [510.0, 520.0, np.nan, np.nan]})
    value, ts = fs.latest_finite_point(frame, "fuel_rate")
    assert value == 520.0
    assert ts == frame.index[1]


def test_infinite_values_are_ignored() -> None:
    frame = make_frame({"fuel_rate": [510.0, np.inf, -np.inf]})
    assert fs.latest_finite_point(frame, "fuel_rate")[0] == 510.0


def test_zero_is_a_real_value() -> None:
    frame = make_frame({"oxygen_flow": [100.0, 0.0]})
    value, _ = fs.latest_finite_point(frame, "oxygen_flow")
    assert value == 0.0

    spec = fs.PARAMETERS_BY_KEY["oxygen_flow"]
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert reading.value == 0.0
    assert fs.format_reading(reading) == "0"
    assert fs.format_reading(reading) != fs.NOT_AVAILABLE


def test_missing_column_and_all_null_series_return_none() -> None:
    frame = make_frame({"fuel_rate": [np.nan, np.nan, np.nan]})
    assert fs.latest_finite_point(frame, "fuel_rate") is None
    assert fs.latest_finite_point(frame, "no_such_field") is None
    assert fs.latest_finite_point(pd.DataFrame(), "fuel_rate") is None
    assert fs.latest_finite_point(None, "fuel_rate") is None


def test_stale_value_is_not_available() -> None:
    frame = make_frame({"fuel_rate": [500.0]}, age_minutes=16)
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert reading.value is None
    assert reading.issue == "stale"


def test_missing_values_are_not_forward_filled_across_columns() -> None:
    frame = make_frame({"fuel_rate": [500.0, np.nan, np.nan]})
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert reading.value == 500.0
    assert reading.timestamp == frame.index[0]  # its own, older timestamp


# ── Derived (summed) parameters ──────────────────────────────────────────────


def _quadrant_frame(values: dict[int, list[float]], **kwargs) -> pd.DataFrame:
    """heat_load_r{6..10}_q{q} columns, each row value = values[q]."""
    return make_frame(
        {
            f"heat_load_r{row}_q{q}": v
            for q, v in values.items()
            for row in range(6, 11)
        },
        **kwargs,
    )


def test_heat_load_specs_sum_rows_r6_to_r10() -> None:
    q1 = fs.PARAMETERS_BY_KEY["heat_load_q1"]
    assert q1.source_fields == tuple(f"heat_load_r{r}_q1" for r in range(6, 11))
    total = fs.PARAMETERS_BY_KEY["heat_load_total"]
    assert len(total.components) == 20
    assert set(total.components) == {
        f
        for q in range(1, 5)
        for f in fs.PARAMETERS_BY_KEY[f"heat_load_q{q}"].components
    }


def test_quadrant_and_total_heat_load_values() -> None:
    frames = {
        "heatload_delta_t": _quadrant_frame(
            {1: [0.5], 2: [0.29], 3: [0.264], 4: [0.306]}
        )
    }
    by_key = {
        k: fs.resolve_reading(fs.PARAMETERS_BY_KEY[k], frames, NOW)
        for k in (
            "heat_load_q1",
            "heat_load_q2",
            "heat_load_q3",
            "heat_load_q4",
            "heat_load_total",
        )
    }
    assert by_key["heat_load_q1"].value == pytest.approx(2.5)
    assert fs.format_reading(by_key["heat_load_q2"]) == "1.45"
    assert by_key["heat_load_total"].value == pytest.approx(2.5 + 1.45 + 1.32 + 1.53)
    assert fs.format_reading(by_key["heat_load_total"]) == "6.80"


def test_heat_load_never_shows_a_partial_sum() -> None:
    frame = _quadrant_frame({1: [0.5]}).drop(columns="heat_load_r8_q1")
    reading = fs.resolve_reading(
        fs.PARAMETERS_BY_KEY["heat_load_q1"], {"heatload_delta_t": frame}, NOW
    )
    assert reading.value is None and reading.issue == "no_data"
    assert fs.format_reading(reading) == "Not available"


def test_heat_load_current_never_mixes_component_timestamps() -> None:
    fresh = _quadrant_frame({1: [0.5]})
    old = make_frame({"heat_load_r6_q1": [0.5]}, age_minutes=20)
    frame = pd.concat([old, fresh.drop(columns="heat_load_r6_q1")])
    reading = fs.resolve_reading(
        fs.PARAMETERS_BY_KEY["heat_load_q1"], {"heatload_delta_t": frame}, NOW
    )
    assert reading.issue == "no_data"


def test_heat_load_trend_sums_per_bin_and_keeps_gaps() -> None:
    frame = _quadrant_frame({1: [0.5, 0.6, 0.7]})
    frame.loc[frame.index[1], "heat_load_r9_q1"] = np.nan
    trend = fs.build_trend_data(fs.PARAMETERS_BY_KEY["heat_load_q1"], frame)

    assert trend.status == "ok"
    assert trend.series.iloc[0] == pytest.approx(2.5)
    assert np.isnan(trend.series.iloc[1])  # one row missing -> gap, not a smaller sum
    assert trend.series.iloc[2] == pytest.approx(3.5)


def test_hearth_pads_use_the_ml_dataset_mapping() -> None:
    from furnace_data.config import load_config

    mapping = load_config("setting_ds_dv.yml")["ml_dataset"]["temperature_params"]
    for pad in "abcd":
        spec = fs.PARAMETERS_BY_KEY[f"hearth_temp_{pad}"]
        assert spec.measurement == "temperature_profile"
        assert spec.title == f"Hearth Temp {pad.upper()}"
        assert mapping[spec.field] == f"hearth_pad_{pad}_c"


def test_hearth_average_is_the_mean_of_all_four_pads() -> None:
    spec = fs.PARAMETERS_BY_KEY["hearth_temp_avg"]
    frame = make_frame(
        {
            "temp_4373_a": [500.0],
            "temp_5411_b": [510.0],
            "temp_5757_c": [520.0],
            "temp_6103_d": [530.0],
        }
    )
    reading = fs.resolve_reading(spec, {"temperature_profile": frame}, NOW)
    assert reading.value == pytest.approx(515.0)
    assert fs.format_reading(reading) == "515"

    partial = fs.resolve_reading(
        spec, {"temperature_profile": frame.drop(columns="temp_6103_d")}, NOW
    )
    assert fs.format_reading(partial) == "Not available"  # no average of 3 pads


def test_hearth_average_trend_is_per_bin_mean_with_gaps() -> None:
    frame = make_frame(
        {
            "temp_4373_a": [500.0, np.nan],
            "temp_5411_b": [510.0, 510.0],
            "temp_5757_c": [520.0, 520.0],
            "temp_6103_d": [530.0, 530.0],
        }
    )
    trend = fs.build_trend_data(fs.PARAMETERS_BY_KEY["hearth_temp_avg"], frame)

    assert trend.series.iloc[0] == pytest.approx(515.0)
    assert np.isnan(trend.series.iloc[1])


def test_uptake_temperatures_read_top_temp_fields() -> None:
    specs = fs.parameters_in_section(fs.SECTION_UPTAKE)
    assert [s.field for s in specs] == [
        "top_temp_1",
        "top_temp_2",
        "top_temp_3",
        "top_temp_4",
        "top_temp_avg",
    ]
    assert all(s.unit == "°C" for s in specs)


# ── Formatting ───────────────────────────────────────────────────────────────


def test_formatting_missing_values_is_exactly_not_available() -> None:
    assert fs.format_value(None, 2) == "Not available"
    assert fs.format_value(float("nan"), 2) == "Not available"
    assert fs.format_value(float("inf"), 2) == "Not available"
    assert fs.NOT_AVAILABLE == "Not available"


def test_format_value_decimals_and_negative_zero() -> None:
    assert fs.format_value(98000.4, 0) == "98,000"
    assert fs.format_value(0.481234, 5) == "0.48123"
    assert fs.format_value(-0.0004, 2) == "0.00"


def test_unavailable_parameter_reading_is_not_available_text() -> None:
    spec = fs.PARAMETERS_BY_KEY["slag_rate"]
    reading = fs.resolve_reading(
        spec, {"process_params": make_frame({"x": [1.0]})}, NOW
    )
    assert reading.issue == "no_source"
    assert fs.format_reading(reading) == "Not available"


def test_steam_injection_is_scaled_from_kg_h_to_tph() -> None:
    spec = fs.PARAMETERS_BY_KEY["steam_injection"]
    frame = make_frame({"steam_injection": [12500.0]})
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert reading.value == pytest.approx(12.5)
    assert fs.format_reading(reading) == "12.50"
    assert spec.unit == "TPH"


def test_co_utilization_keeps_raw_scale() -> None:
    spec = fs.PARAMETERS_BY_KEY["co_utilization"]
    frame = make_frame({"body_etaco": [0.48123456]})
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert fs.format_reading(reading) == "0.48123"
    assert spec.unit == ""


def test_removed_parameters_are_gone() -> None:
    removed = {
        "production_projected",
        "cold_blast_volume",
        "humidity",
        "flare_gas_flow",
        "heat_flow_flux",
    }
    assert not removed & set(fs.PARAMETERS_BY_KEY)


def test_pci_shows_setpoint_and_actual() -> None:
    spec = fs.PARAMETERS_BY_KEY["pci_rate"]
    assert spec.source_fields == ("coal_rate_actual_value", "coal_rate_set_value")
    frame = make_frame(
        {"coal_rate_actual_value": [141.6], "coal_rate_set_value": [145.0]}
    )
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert (reading.setpoint, reading.value) == (145.0, 141.6)
    assert fs.format_reading(reading) == "145 / 142"


def test_pci_with_missing_setpoint_and_available_actual() -> None:
    spec = fs.PARAMETERS_BY_KEY["pci_rate"]
    frame = make_frame(
        {"coal_rate_actual_value": [141.6], "coal_rate_set_value": [np.nan]}
    )
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert fs.format_reading(reading) == "Not available / 142"


def test_pci_with_setpoint_but_no_actual() -> None:
    spec = fs.PARAMETERS_BY_KEY["pci_rate"]
    frame = make_frame({"coal_rate_set_value": [145.0]})
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert reading.issue == "no_data" and reading.setpoint == 145.0
    assert fs.format_reading(reading) == "145 / Not available"
    assert fs.has_display_value(reading)


def test_stale_setpoint_is_not_shown() -> None:
    spec = fs.PARAMETERS_BY_KEY["pci_rate"]
    frame = pd.concat(
        [
            make_frame({"coal_rate_set_value": [145.0]}, age_minutes=30),
            make_frame({"coal_rate_actual_value": [141.6]}),
        ]
    )
    reading = fs.resolve_reading(spec, {"process_params": frame}, NOW)
    assert fs.format_reading(reading) == "Not available / 142"


def test_pci_without_actual_is_plain_not_available() -> None:
    spec = fs.PARAMETERS_BY_KEY["pci_rate"]
    reading = fs.resolve_reading(spec, {"process_params": pd.DataFrame()}, NOW)
    assert fs.format_reading(reading) == "Not available"


# ── Time ranges ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,hours", [("1h", 1), ("4h", 4), ("8h", 8), ("16h", 16), ("24h", 24)]
)
def test_fixed_ranges_resolve_in_utc(label: str, hours: int) -> None:
    start, end = fs.resolve_fixed_range(label, NOW)
    assert end == datetime(2026, 10, 5, 9, 30, tzinfo=timezone.utc)  # minute-floored
    assert end - start == timedelta(hours=hours)
    assert start.utcoffset() == timedelta(0) and end.utcoffset() == timedelta(0)


def test_fixed_range_from_ist_now_is_still_utc() -> None:
    now_ist = NOW.astimezone(fs.IST)
    assert fs.resolve_fixed_range("1h", now_ist) == fs.resolve_fixed_range("1h", NOW)


def test_unknown_fixed_interval_is_rejected() -> None:
    with pytest.raises(fs.RangeError):
        fs.resolve_fixed_range("2h", NOW)


def test_custom_ist_times_convert_to_utc() -> None:
    start, end = fs.resolve_custom_range(
        datetime(2026, 10, 4, 10, 0), datetime(2026, 10, 4, 18, 30), NOW
    )
    assert start == datetime(2026, 10, 4, 4, 30, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 4, 13, 0, tzinfo=timezone.utc)


def test_custom_range_rejects_inverted_and_oversized_ranges() -> None:
    with pytest.raises(fs.RangeError):
        fs.resolve_custom_range(
            datetime(2026, 10, 4, 12), datetime(2026, 10, 4, 10), NOW
        )
    with pytest.raises(fs.RangeError):
        fs.resolve_custom_range(
            datetime(2026, 10, 4, 10), datetime(2026, 10, 4, 10), NOW
        )
    with pytest.raises(fs.RangeError):
        fs.resolve_custom_range(datetime(2026, 5, 1), datetime(2026, 10, 1), NOW)


def test_custom_range_end_is_clamped_to_now() -> None:
    _, end = fs.resolve_custom_range(
        datetime(2026, 10, 5, 12, 0), datetime(2026, 10, 6, 12, 0), NOW
    )
    assert end == datetime(2026, 10, 5, 9, 30, tzinfo=timezone.utc)


def test_default_custom_range_is_previous_eight_hours_in_ist() -> None:
    start, end = fs.default_custom_range(NOW)
    assert end - start == timedelta(hours=8)
    assert end.utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize(
    "duration,expected",
    [
        (timedelta(minutes=10), "1 minute"),
        (timedelta(hours=1), "1 minute"),
        (timedelta(hours=4), "5 minutes"),
        (timedelta(hours=8), "5 minutes"),
        (timedelta(hours=16), "10 minutes"),
        (timedelta(hours=24), "15 minutes"),
        (timedelta(hours=48), "30 minutes"),
        (timedelta(days=7), "1 hour"),
        (timedelta(days=90), "1 hour"),
    ],
)
def test_choose_window(duration: timedelta, expected: str) -> None:
    assert fs.choose_window(duration) == expected


def test_fixed_intervals_use_the_documented_windows() -> None:
    expected = {
        "1h": "1 minute",
        "4h": "5 minutes",
        "8h": "5 minutes",
        "16h": "10 minutes",
        "24h": "15 minutes",
    }
    assert {k: fs.choose_window(v) for k, v in fs.FIXED_INTERVALS.items()} == expected


# ── Fetching through the existing data layer ─────────────────────────────────


class _FakeMeasurementFetcher:
    def __init__(self, owner: "FakeFetcherFactory", measurement: str) -> None:
        self.owner = owner
        self.measurement = measurement

    def fetch_data(
        self,
        time_interval,
        start_time,
        end_time,
        *,
        request_type,
        window_by,
        fields,
    ):
        call = {
            "measurement": self.measurement,
            "time_interval": time_interval,
            "start_time": start_time,
            "end_time": end_time,
            "request_type": request_type,
            "window_by": window_by,
            "fields": fields,
        }
        self.owner.calls.append(call)
        key = (self.measurement, request_type)
        result = self.owner.frames.get(key, self.owner.frames.get(self.measurement))
        if isinstance(result, Exception):
            raise result
        if result is None:
            raise AssertionError(f"No fake frame configured for {key!r}")
        return result.copy()


class FakeFetcherFactory:
    """Stand-in for the generic ``TimeSeriesDataFetcher`` boundary."""

    def __init__(self, frames: dict[object, pd.DataFrame | Exception]) -> None:
        self.frames = frames
        self.calls: list[dict] = []

    def __call__(self, measurement, *, debug, source):
        assert debug is False and source == "historical"
        return _FakeMeasurementFetcher(self, measurement)


def test_transport_adapter_preserves_raw_values_and_canonical_fields() -> None:
    frame = pd.DataFrame(
        {
            "time": ["2026-10-05T09:30:00Z", "2026-10-05T09:29:00Z"],
            "fuel_rate": [200.0, 100.0],
            "table": [1, 1],
            "result": ["_result", "_result"],
        }
    )

    result = fs_data._normalise_frame(frame, ("fuel_rate",))

    assert list(result.columns) == ["fuel_rate"]
    assert list(result["fuel_rate"]) == [100.0, 200.0]
    assert result.index.is_monotonic_increasing
    assert str(result.index.tz) == "Asia/Kolkata"


def test_one_failing_measurement_does_not_discard_the_other(monkeypatch) -> None:
    fake = FakeFetcherFactory(
        {
            "process_params": make_frame(
                {"fuel_rate": [505.0], "production_per_hour": [180.0]}
            ),
            "miscellaneous": RuntimeError("influx exploded: token=SECRET"),
            "heatload_delta_t": _full_frames(0)["heatload_delta_t"],
            "temperature_profile": _full_frames(0)["temperature_profile"],
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    snapshot = fs_data.load_status_snapshot(NOW)

    by_key = {r.spec.key: r for r in snapshot.readings}
    assert by_key["fuel_rate"].value == 505.0
    assert by_key["production_rate"].value == 180.0
    assert by_key["furnace_level"].value is None
    assert by_key["furnace_level"].issue == "fetch_failed"
    assert snapshot.failed_measurements == ("miscellaneous",)
    assert len(snapshot.readings) == len(fs.PARAMETERS)


def test_status_fetch_uses_one_request_per_measurement(monkeypatch) -> None:
    fake = FakeFetcherFactory(
        {
            "process_params": make_frame({"fuel_rate": [505.0]}),
            "miscellaneous": make_frame({"stock_rod_radar_level": [4.2]}),
            "heatload_delta_t": _full_frames(0)["heatload_delta_t"],
            "temperature_profile": _full_frames(0)["temperature_profile"],
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    fs_data.load_status_snapshot(NOW)

    assert [c["measurement"] for c in fake.calls] == [
        "process_params",
        "miscellaneous",
        "heatload_delta_t",
        "temperature_profile",
    ]
    for call in fake.calls:
        assert call["time_interval"] == "last 15 minutes"
        assert call["request_type"] == "ts"
        assert call["window_by"] is None
        assert call["fields"] == fs.source_fields_for_measurement(call["measurement"])


def test_failure_details_never_reach_the_snapshot(monkeypatch) -> None:
    fake = FakeFetcherFactory(
        {
            "process_params": RuntimeError("postgres://user:SECRET@host"),
            "miscellaneous": RuntimeError("token=SECRET"),
            "heatload_delta_t": RuntimeError("SECRET"),
            "temperature_profile": RuntimeError("SECRET"),
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    snapshot = fs_data.load_status_snapshot(NOW)

    assert all(r.value is None for r in snapshot.readings)
    assert snapshot.status.level == "offline"
    assert "SECRET" not in repr(snapshot)


def test_trend_fetch_passes_utc_overrides_and_field_naming(monkeypatch) -> None:
    fake = FakeFetcherFactory(
        {"process_params": make_frame({"hot_blast_vol_nm3h": [98000.0, 99000.0]})}
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)
    spec = fs.PARAMETERS_BY_KEY["hot_blast_volume"]
    start, end = fs.resolve_fixed_range("4h", NOW)

    trend = fs_data.load_trend(spec, start, end, fs.choose_window(end - start))

    (call,) = fake.calls
    assert call["start_time"] == start and call["end_time"] == end
    assert call["start_time"].utcoffset() == timedelta(0)
    assert call["time_interval"] == "over selected range"
    assert call["request_type"] == "windowed-average"
    assert call["window_by"] == "5 minutes"
    assert call["fields"] == spec.source_fields
    assert trend.status == "ok"
    assert list(trend.series) == [98000.0, 99000.0]
    assert str(trend.series.index.tz) == "Asia/Kolkata"


def test_trend_applies_scale_and_pci_charts_setpoint_too(monkeypatch) -> None:
    fake = FakeFetcherFactory(
        {
            "process_params": make_frame(
                {
                    "steam_injection": [10000.0, 12000.0],
                    "coal_rate_actual_value": [140.0, 150.0],
                    "coal_rate_set_value": [145.0, 145.0],
                }
            )
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)
    start, end = fs.resolve_fixed_range("1h", NOW)

    steam = fs_data.load_trend(
        fs.PARAMETERS_BY_KEY["steam_injection"],
        start,
        end,
        "1 minute",
    )
    assert list(steam.series) == [10.0, 12.0]

    pci = fs_data.load_trend(
        fs.PARAMETERS_BY_KEY["pci_rate"],
        start,
        end,
        "1 minute",
    )
    assert list(pci.series) == [140.0, 150.0]
    assert list(pci.setpoint) == [145.0, 145.0]
    assert steam.setpoint is None
    assert fake.calls[-1]["measurement"] == "process_params"  # one request for both
    assert len(fake.calls) == 2


def test_derived_trend_fetches_every_component_in_one_request(monkeypatch) -> None:
    spec = fs.PARAMETERS_BY_KEY["heat_load_total"]
    fake = FakeFetcherFactory(
        {
            "heatload_delta_t": make_frame(
                {field: [1.0, 2.0] for field in spec.source_fields}
            )
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)
    start, end = fs.resolve_fixed_range("1h", NOW)

    trend = fs_data.load_trend(spec, start, end, "1 minute")

    assert trend.status == "ok"
    assert list(trend.series) == [20.0, 40.0]
    assert len(fake.calls) == 1
    assert fake.calls[0]["measurement"] == "heatload_delta_t"
    assert fake.calls[0]["fields"] == spec.source_fields


def test_trend_preserves_gaps(monkeypatch) -> None:
    fake = FakeFetcherFactory(
        {"process_params": make_frame({"fuel_rate": [500.0, np.nan, 520.0]})}
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)
    start, end = fs.resolve_fixed_range("1h", NOW)

    trend = fs_data.load_trend(
        fs.PARAMETERS_BY_KEY["fuel_rate"],
        start,
        end,
        "1 minute",
    )

    assert len(trend.series) == 3 and np.isnan(trend.series.iloc[1])
    stats = fs.compute_trend_stats(trend.series)
    assert (stats.minimum, stats.maximum, stats.mean) == (500.0, 520.0, 510.0)


def test_trend_without_source_never_calls_the_fetcher(monkeypatch) -> None:
    fake = FakeFetcherFactory({})
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)
    start, end = fs.resolve_fixed_range("1h", NOW)

    trend = fs_data.load_trend(
        fs.PARAMETERS_BY_KEY["slag_rate"],
        start,
        end,
        "1 minute",
    )

    assert trend.status == "no_source" and trend.series is None
    assert "daily production report" in trend.message
    assert fake.calls == []


def test_trend_empty_and_failed_requests_are_handled(monkeypatch) -> None:
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    start, end = fs.resolve_fixed_range("1h", NOW)

    monkeypatch.setattr(
        fs_data,
        "TimeSeriesDataFetcher",
        FakeFetcherFactory({"process_params": pd.DataFrame()}),
    )
    assert fs_data.load_trend(spec, start, end, "1 minute").status == "no_data"

    monkeypatch.setattr(
        fs_data,
        "TimeSeriesDataFetcher",
        FakeFetcherFactory({"process_params": RuntimeError("boom")}),
    )
    fs_data.clear_furnace_status_caches()
    failed = fs_data.load_trend(spec, start, end, "1 minute")
    assert failed.status == "error" and "boom" not in failed.message


def test_raw_current_differs_from_final_average_bucket(monkeypatch) -> None:
    """Current/Last data use the final raw point, never the final chart bucket."""
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    start, end = fs.resolve_fixed_range("1h", NOW)
    raw_times = pd.DatetimeIndex(
        [end - timedelta(seconds=40), end - timedelta(seconds=20), end],
        name="time",
    )
    raw = pd.DataFrame(
        {
            "time": raw_times,
            "fuel_rate": [100.0, 120.0, 200.0],
            "table": [0, 0, 0],
        }
    )
    averaged = pd.DataFrame(
        {"fuel_rate": [140.0]},
        index=pd.DatetimeIndex([end], name="time"),
    )
    fake = FakeFetcherFactory(
        {
            ("process_params", "ts"): raw,
            ("process_params", "windowed-average"): averaged,
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    result = fs_data.load_trend_view(
        spec, start, end, "1 minute", use_live_current=True
    )

    assert result.current.value == 200.0
    assert result.current.timestamp == pd.Timestamp(end).tz_convert(fs.IST)
    assert result.trend.series.iloc[-1] == 140.0
    assert result.stats.mean == 140.0
    assert [call["request_type"] for call in fake.calls] == [
        "ts",
        "windowed-average",
    ]


@pytest.mark.parametrize("interval", ["1h", "4h", "8h", "16h", "24h"])
def test_every_fixed_interval_uses_latest_raw_current(monkeypatch, interval) -> None:
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    start, end = fs.resolve_fixed_range(interval, NOW)
    raw = pd.DataFrame(
        {"fuel_rate": [100.0, 200.0]},
        index=pd.DatetimeIndex([end - timedelta(minutes=1), end], name="time"),
    )
    averaged = pd.DataFrame(
        {"fuel_rate": [150.0]}, index=pd.DatetimeIndex([end], name="time")
    )
    fake = FakeFetcherFactory(
        {
            ("process_params", "ts"): raw,
            ("process_params", "windowed-average"): averaged,
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    result = fs_data.load_trend_view(
        spec,
        start,
        end,
        fs.choose_window(end - start),
        use_live_current=True,
    )

    assert result.current.value == 200.0
    assert result.trend.series.iloc[-1] == 150.0


def test_custom_historical_current_is_bounded_to_selected_range(monkeypatch) -> None:
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    end = NOW - timedelta(days=2)
    start = end - timedelta(hours=1)
    raw = pd.DataFrame(
        {"fuel_rate": [190.0, 999.0]},
        index=pd.DatetimeIndex(
            [end - timedelta(minutes=1), end + timedelta(seconds=1)], name="time"
        ),
    )
    averaged = pd.DataFrame(
        {"fuel_rate": [150.0]}, index=pd.DatetimeIndex([end], name="time")
    )
    fake = FakeFetcherFactory(
        {
            ("process_params", "ts"): raw,
            ("process_params", "windowed-average"): averaged,
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    result = fs_data.load_trend_view(
        spec, start, end, "1 minute", use_live_current=False
    )

    assert result.current.value == 190.0
    raw_call = next(call for call in fake.calls if call["request_type"] == "ts")
    assert raw_call["start_time"] == end - fs.STALE_AFTER
    assert raw_call["end_time"] == end
    assert raw_call["window_by"] is None


def test_derived_current_uses_newest_complete_raw_row() -> None:
    spec = fs.PARAMETERS_BY_KEY["heat_load_q1"]
    fields = list(spec.components)
    index = pd.date_range(end=NOW, periods=3, freq="1min", tz="UTC")
    frame = pd.DataFrame({field: [1.0, 2.0, 3.0] for field in fields}, index=index)
    frame.loc[index[-1], fields[-1]] = np.nan

    reading = fs.resolve_reading(spec, {spec.measurement: frame}, NOW)

    assert reading.value == 10.0  # complete middle row: 5 fields × 2
    assert reading.timestamp == index[-2].tz_convert(fs.IST)


def test_current_and_trend_fail_independently(monkeypatch) -> None:
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    start, end = fs.resolve_fixed_range("1h", NOW)
    averaged = pd.DataFrame(
        {"fuel_rate": [500.0, 520.0]},
        index=pd.date_range(end=end, periods=2, freq="1min"),
    )
    fake = FakeFetcherFactory(
        {
            ("process_params", "ts"): RuntimeError("raw token=SECRET"),
            ("process_params", "windowed-average"): averaged,
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    result = fs_data.load_trend_view(
        spec, start, end, "1 minute", use_live_current=True
    )
    assert result.current.issue == "fetch_failed"
    assert result.trend.status == "ok" and result.stats.mean == 510.0

    fs_data.clear_furnace_status_caches()
    raw = pd.DataFrame(
        {"fuel_rate": [530.0]}, index=pd.DatetimeIndex([end], name="time")
    )
    fake = FakeFetcherFactory(
        {
            ("process_params", "ts"): raw,
            ("process_params", "windowed-average"): RuntimeError("trend token=SECRET"),
        }
    )
    monkeypatch.setattr(fs_data, "TimeSeriesDataFetcher", fake)

    result = fs_data.load_trend_view(
        spec, start, end, "1 minute", use_live_current=True
    )
    assert result.current.value == 530.0
    assert result.trend.status == "error" and result.stats is None
    assert "SECRET" not in result.trend.message


def test_compute_trend_stats_empty() -> None:
    assert fs.compute_trend_stats(None) is None
    assert fs.compute_trend_stats(pd.Series([np.nan, np.nan])) is None


def test_clear_caches_is_safe_without_a_runtime() -> None:
    fs_data.clear_furnace_status_caches()


# ── Plant status ─────────────────────────────────────────────────────────────


def _readings(frames: dict[str, pd.DataFrame | None]) -> list[fs.ParameterReading]:
    return list(fs.build_status_snapshot(frames, NOW).readings)


def _full_frames(age_minutes: float) -> dict[str, pd.DataFrame]:
    """One frame per measurement with every source field present."""
    return {
        m: make_frame(
            {
                f: [1.0]
                for p in fs.PARAMETERS
                if p.measurement == m
                for f in p.source_fields
            },
            age_minutes=age_minutes,
        )
        for m in fs.measurements_for()
    }


def test_plant_status_live_partial_offline() -> None:
    live = fs.build_status_snapshot(_full_frames(1), NOW)
    assert live.status.level == "live"

    # Fresh but <70 % of mapped parameters report -> partial.
    sparse = fs.build_status_snapshot(
        {"process_params": make_frame({"fuel_rate": [1.0]}), "miscellaneous": None}, NOW
    )
    assert sparse.status.level == "partial"

    # Complete but 8 minutes old (every source) -> partial.
    old = fs.build_status_snapshot(_full_frames(8), NOW)
    assert old.status.level == "partial"

    gone = fs.build_status_snapshot(
        {"process_params": None, "miscellaneous": None}, NOW
    )
    assert gone.status.level == "offline"
    assert gone.last_updated is None


# ── Query-parameter validation ───────────────────────────────────────────────


def test_view_state_accepts_known_parameter() -> None:
    state = fs.parse_view_state(
        {fs.VIEW_QUERY_KEY: "trend", fs.PARAMETER_QUERY_KEY: "fuel_rate"}
    )
    assert state.view == "trend"
    assert state.spec is fs.PARAMETERS_BY_KEY["fuel_rate"]
    assert not state.needs_reset


def test_view_state_defaults_to_status() -> None:
    state = fs.parse_view_state({})
    assert state.view == "status" and state.spec is None and not state.needs_reset


@pytest.mark.parametrize(
    "params",
    [
        {fs.VIEW_QUERY_KEY: "trend", fs.PARAMETER_QUERY_KEY: "does_not_exist"},
        {
            fs.VIEW_QUERY_KEY: "trend",
            fs.PARAMETER_QUERY_KEY: "<script>alert(1)</script>",
        },
        {fs.VIEW_QUERY_KEY: "trend", fs.PARAMETER_QUERY_KEY: "fuel_rate' OR 1=1 --"},
        {fs.VIEW_QUERY_KEY: "trend", fs.PARAMETER_QUERY_KEY: "../../etc/passwd"},
        {fs.VIEW_QUERY_KEY: "trend", fs.PARAMETER_QUERY_KEY: ""},
        {fs.VIEW_QUERY_KEY: "trend", fs.PARAMETER_QUERY_KEY: 7},
        {fs.VIEW_QUERY_KEY: "trend"},
        {fs.VIEW_QUERY_KEY: "bogus", fs.PARAMETER_QUERY_KEY: "fuel_rate"},
        {fs.VIEW_QUERY_KEY: ["trend"], fs.PARAMETER_QUERY_KEY: []},
    ],
)
def test_unknown_or_malformed_parameters_fall_back_to_status(params) -> None:
    state = fs.parse_view_state(params)
    assert state.view == "status"
    assert state.spec is None
    assert state.needs_reset


def test_view_state_takes_last_value_of_repeated_parameters() -> None:
    state = fs.parse_view_state(
        {
            fs.VIEW_QUERY_KEY: ["status", "trend"],
            fs.PARAMETER_QUERY_KEY: ["x", "raft"],
        }
    )
    assert state.view == "trend" and state.spec.key == "raft"


def test_catalogue_text_is_html_safe_to_escape() -> None:
    """Labels are escaped before rendering; they must survive a round trip."""
    for spec in fs.PARAMETERS:
        assert html.unescape(html.escape(spec.label)) == spec.label


def test_domain_is_pure_and_deleted_fetch_wrappers_are_not_recreated() -> None:
    repo = Path(__file__).resolve().parents[1]
    domain_path = repo / "src" / "domain" / "furnace_status.py"
    source = domain_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {"streamlit", "furnace_data", "influxdb_client_3"} & imported_roots
    for removed in (
        "fetch_status_frame",
        "fetch_trend_frame",
        "cached_status_frame",
        "cached_trend_frame",
    ):
        assert f"def {removed}(" not in source

    service_path = repo / "src" / "data" / "furnace_status_service.py"
    old_orchestration_path = repo / "src" / "ui" / "furnace_status_orchestration.py"
    assert service_path.exists()
    assert not old_orchestration_path.exists()

    feature_source = "\n".join(
        (repo / path).read_text(encoding="utf-8")
        for path in (
            "src/ui/furnace_status_page.py",
            "src/data/furnace_status_service.py",
            "src/custom_pages/3_Data_Visualisation.py",
        )
    )
    assert "InfluxDBClient3" not in feature_source
    assert "fetch_online_df" not in feature_source
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"))
    service_imports = {
        node.module
        for node in ast.walk(service_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "furnace_data.influx.online" not in service_imports
    assert "furnace_data.influx.base" not in service_imports
    assert "furnace_data.influx.query" not in service_imports
    assert not (repo / "src" / "data" / "furnace_status.py").exists()
