"""Pure-logic tests for Key Parameters (no Influx / Streamlit runtime needed)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest
import streamlit as st

if not hasattr(st, "cache_data"):  # another test module stubbed streamlit globally
    pytest.skip("streamlit is stubbed in this session", allow_module_level=True)

from furnace_data.influx.query import influx_fields
from ui.key_parameters import (
    INTERVAL_OPTIONS,
    KEY_PARAMETERS,
    PRESET_HOURS,
    choose_window,
    custom_range_to_utc,
    format_value,
    get_parameter,
    resolve_preset_range,
    window_label,
)


def test_parameter_keys_are_unique() -> None:
    keys = [p.key for p in KEY_PARAMETERS]
    assert len(keys) == len(set(keys))


def test_mapped_parameters_exist_in_influx_config() -> None:
    for p in KEY_PARAMETERS:
        assert (p.measurement is None) == (p.field is None), p.key
        if p.field:
            assert p.field in influx_fields(p.measurement), p.key


def test_unmapped_parameters_are_present_and_blank() -> None:
    unmapped = {p.key for p in KEY_PARAMETERS if p.field is None}
    assert {"prod_proj", "cold_blast_vol", "steam_bypass", "humidity",
            "flare_flow", "slag_rate", "heat_flux"} <= unmapped
    assert format_value(get_parameter("humidity"), None) == "—"


@pytest.mark.parametrize("label,hours", PRESET_HOURS.items())
def test_preset_ranges(label: str, hours: int) -> None:
    now = datetime(2026, 10, 3, 12, 34, 56, tzinfo=timezone.utc)
    start, end = resolve_preset_range(label, now)
    assert end == datetime(2026, 10, 3, 12, 34, tzinfo=timezone.utc)
    assert end - start == timedelta(hours=hours)
    assert set(PRESET_HOURS) | {"Custom"} == set(INTERVAL_OPTIONS)


@pytest.mark.parametrize(
    "span,expected",
    [
        (timedelta(hours=1), "1 minute"),
        (timedelta(hours=4), "5 minutes"),
        (timedelta(hours=8), "5 minutes"),
        (timedelta(hours=16), "10 minutes"),
        (timedelta(hours=24), "15 minutes"),
        (timedelta(days=7), "1 hour"),
        (timedelta(days=500), "1 day"),
    ],
)
def test_choose_window(span: timedelta, expected: str) -> None:
    assert choose_window(span) == expected


def test_choose_window_bounds_point_count() -> None:
    from furnace_data.influx.query import WINDOWING

    minutes = {"1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60,
               "6h": 360, "12h": 720, "1d": 1440}
    for hours in (1, 4, 8, 16, 24, 72, 24 * 14, 24 * 90, 24 * 365):
        span = timedelta(hours=hours)
        points = span.total_seconds() / 60 / minutes[WINDOWING[choose_window(span)]]
        assert points <= 300 or hours >= 24 * 365


def test_custom_range_is_ist_converted_to_utc() -> None:
    start, end = custom_range_to_utc(date(2026, 10, 3), time(6, 0), date(2026, 10, 3), time(14, 0))
    assert start == datetime(2026, 10, 3, 0, 30, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 3, 8, 30, tzinfo=timezone.utc)


def test_custom_range_requires_start_before_end() -> None:
    with pytest.raises(ValueError):
        custom_range_to_utc(date(2026, 10, 3), time(14, 0), date(2026, 10, 3), time(6, 0))


def test_window_label() -> None:
    assert window_label("5 minutes") == "5 minute average"
    assert window_label("1 hour") == "1 hour average"
