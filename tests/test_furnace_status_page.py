"""Tests for Furnace Status UI helpers and the integrated V-Board section.

The helpers are exercised without a Streamlit runtime; the smoke test runs the
actual page script under ``streamlit.testing.v1.AppTest`` with
``fetch_online_df`` monkeypatched (no InfluxDB).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from domain import furnace_status as fs
from ui import furnace_status_page as page
from ui import furnace_status_orchestration as fs_data
from ui import vboard_sections
from ui.vboard_sections import (
    FURNACE_STATUS,
    VBOARD_NAV_KEY,
    VBOARD_SECTIONS,
    VISUALISATIONS,
)

REPO = Path(__file__).resolve().parents[1]
PAGE_FILE = REPO / "src" / "custom_pages" / "3_Data_Visualisation.py"
NOW = datetime(2026, 10, 5, 9, 30, tzinfo=timezone.utc)


def test_vboard_sections_are_ordered_with_visualisations_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_segmented_control(label, options, *, default, key):
        captured.update(label=label, options=options, default=default, key=key)
        return default

    monkeypatch.setattr(vboard_sections.st, "segmented_control", fake_segmented_control)
    assert vboard_sections.select_vboard_section() == VISUALISATIONS
    assert VBOARD_SECTIONS == (VISUALISATIONS, FURNACE_STATUS)
    assert captured["default"] == VISUALISATIONS
    assert captured["key"] == VBOARD_NAV_KEY


# ── Pure helpers ─────────────────────────────────────────────────────────────


def _trend(key: str, values: list[float]) -> fs.TrendData:
    index = pd.date_range(
        end=pd.Timestamp(NOW).tz_convert("Asia/Kolkata"),
        periods=len(values),
        freq="5min",
        name="time (IST)",
    )
    return fs.TrendData(
        fs.PARAMETERS_BY_KEY[key], "ok", pd.Series(values, index=index, name=key)
    )


def test_trend_figure_meets_the_chart_requirements() -> None:
    trend = _trend("hot_blast_volume", [98000.0, np.nan, 99000.0])
    start, end = fs.resolve_fixed_range("1h", NOW)

    fig = page.build_trend_figure(
        trend, (page._naive_ist(start), page._naive_ist(end)), "rev"
    )

    assert fig.layout.hovermode == "x unified"
    assert fig.layout.xaxis.title.text == "Time (IST)"
    assert "Nm³/h" in fig.layout.yaxis.title.text
    assert fig.layout.showlegend is False
    (trace,) = fig.data
    assert trace.connectgaps is False  # gaps are kept, never bridged
    assert np.isnan(trace.y[1])


def test_trend_figure_x_axis_shows_ist_wall_clock() -> None:
    trend = _trend("fuel_rate", [500.0, 510.0])
    start, end = fs.resolve_fixed_range("1h", NOW)
    fig = page.build_trend_figure(
        trend, (page._naive_ist(start), page._naive_ist(end)), "rev"
    )

    assert pd.Timestamp(fig.data[0].x[-1]) == pd.Timestamp(
        "2026-10-05 15:00:00"
    )  # 09:30Z + 5:30
    assert pd.Timestamp(fig.layout.xaxis.range[1]) == pd.Timestamp(
        "2026-10-05 15:00:00"
    )


def test_unitless_parameter_has_plain_y_axis_title() -> None:
    trend = _trend("co_utilization", [0.48, 0.49])
    start, end = fs.resolve_fixed_range("1h", NOW)
    fig = page.build_trend_figure(
        trend, (page._naive_ist(start), page._naive_ist(end)), "rev"
    )
    assert fig.layout.yaxis.title.text == "CO Utilization"


def test_plot_config_is_responsive_and_hides_the_logo() -> None:
    assert page._PLOT_CONFIG["responsive"] is True
    assert page._PLOT_CONFIG["displaylogo"] is False
    assert "lasso2d" in page._PLOT_CONFIG["modeBarButtonsToRemove"]


def test_row_html_escapes_text_and_mutes_not_available() -> None:
    spec = fs.ParameterSpec(
        "x",
        "<img src=x onerror=alert(1)>",
        fs.SECTION_BLAST,
        "process_params",
        "f",
        unit="a&b",
    )
    available = page._row_html(fs.ParameterReading(spec, value=5.0))
    assert "<img" not in available
    assert "&lt;img" in available and "a&amp;b" in available

    missing = page._row_html(fs.ParameterReading(spec, issue="no_data"))
    assert '<span class="fs-na">Not available</span>' in missing
    assert "a&amp;b" not in missing  # no unit next to a missing value
    assert "fs-row--na" in missing


def test_pci_row_mutes_only_the_missing_setpoint() -> None:
    spec = fs.PARAMETERS_BY_KEY["pci_rate"]
    na = '<span class="fs-na">Not available</span>'

    actual_only = page._row_html(fs.ParameterReading(spec, value=141.6))
    assert '<span class="fs-row__main">142</span>' in actual_only
    assert f"SP {na}" in actual_only
    assert "Δ" not in actual_only  # no difference without both values
    assert '<span class="fs-row__unit">kg/THM</span>' in actual_only

    both = page._row_html(fs.ParameterReading(spec, value=141.6, setpoint=145.0))
    assert '<span class="fs-row__main">142</span>' in both and "SP 145" in both
    assert "Δ −3" in both  # actual − setpoint, from the unrounded values

    setpoint_only = page._row_html(
        fs.ParameterReading(spec, issue="no_data", setpoint=145.0)
    )
    assert f'<span class="fs-row__main">{na}</span>' in setpoint_only
    assert "SP 145 kg/THM" in setpoint_only and "Δ" not in setpoint_only
    assert '<span class="fs-row__unit"></span>' in setpoint_only

    neither = page._row_html(fs.ParameterReading(spec, issue="no_data"))
    assert "fs-row--na" in neither and "fs-row__sub" not in neither


def test_trend_figure_draws_setpoint_as_second_line() -> None:
    base = _trend("pci_rate", [140.0, 150.0])
    trend = fs.TrendData(base.spec, "ok", base.series, setpoint=base.series * 0 + 145.0)
    start, end = fs.resolve_fixed_range("1h", NOW)
    fig = page.build_trend_figure(
        trend, (page._naive_ist(start), page._naive_ist(end)), "rev"
    )
    assert [t.name for t in fig.data] == ["Actual", "Setpoint"]
    assert fig.data[1].line.dash == "dash"
    assert fig.layout.showlegend is True


def test_button_label_carries_name_value_and_unit() -> None:
    reading = fs.ParameterReading(
        fs.PARAMETERS_BY_KEY["hot_blast_volume"], value=98250.0
    )
    assert page._button_label(reading) == "Hot Blast Volume: 98,250 Nm³/h. Open trend"
    na = fs.ParameterReading(fs.PARAMETERS_BY_KEY["slag_rate"], issue="no_source")
    assert page._button_label(na) == "Slag Rate: Not available. Open trend"


def test_furnace_svg_is_valid_and_labels_every_zone() -> None:
    svg = page.build_furnace_svg(page._FALLBACK_PROFILE)
    root = ET.fromstring(svg)  # raises if malformed
    assert root.tag.endswith("svg")

    # Labels and callouts are HTML laid over the image (readable at any width).
    readings = {p.key: fs.ParameterReading(p, issue="no_data") for p in fs.PARAMETERS}
    readings["top_pressure"] = fs.ParameterReading(
        fs.PARAMETERS_BY_KEY["top_pressure"], value=0.0
    )
    overlay = page._schematic_html(readings)
    for label in ("Stack", "Belly", "Bosh", "Tuyere", "Hearth", "Hot blast", "Top gas"):
        assert f">{label}<" in overlay
    assert "Hot metal &amp; slag" in overlay
    # Missing readings stay "Not available"; zero is a real reading.
    assert overlay.count('<span class="fs-na">Not available</span>') == (
        len(page._CALLOUTS) - 1
    )
    assert ">0.00<span" in overlay


def test_furnace_profile_comes_from_the_shared_config() -> None:
    page._furnace_profile.cache_clear()
    profile = page._furnace_profile()
    assert profile[0] == (-2.8, 4.374) and profile[-1] == (-2.898, 20.0)


def test_stylesheet_ships_orientation_and_touch_rules() -> None:
    css = (REPO / "src" / "assets" / "css" / "furnace_status.css").read_text(
        encoding="utf-8"
    )
    assert "orientation: portrait" in css and "orientation: landscape" in css
    assert "pointer: coarse" in css
    assert "prefers-reduced-motion" in css
    # Background, header and sidebar follow the app theme, as on the other
    # pages: the stylesheet never restyles them.
    for shared in ('[data-testid="stSidebar"]', '[data-testid="stHeader"]', ".stApp"):
        assert shared not in css
    for zone, _, _ in page._ZONES:  # every zone band links to its readings
        assert f'.fs-zone[data-zone="{zone.lower()}"]:hover' in css


def test_orientation_script_is_defensive_and_user_gesture_driven() -> None:
    script = page._ORIENTATION_HTML
    assert 'lock("landscape")' in script
    assert "requestFullscreen" in script
    assert script.count("try {") >= 4 and "catch" in script
    assert "setInterval" not in script  # no polling loop
    assert "Rotate your device for the best view." in (
        REPO / "src" / "ui" / "furnace_status_page.py"
    ).read_text(encoding="utf-8")


# ── Smoke run of the real page ───────────────────────────────────────────────


def _live_frame(columns: list[str]) -> pd.DataFrame:
    end = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("Asia/Kolkata")
    index = pd.date_range(end=end, periods=15, freq="1min", name="time (IST)")
    return pd.DataFrame(
        {c: np.linspace(10.0, 12.0, len(index)) for c in columns}, index=index
    )


@pytest.fixture
def patched_fetch(monkeypatch):
    calls: list[str] = []
    state = {"fail": set()}

    def fake(selected_measurements, time_range, **kwargs):
        (measurement,) = selected_measurements
        calls.append(measurement)
        if measurement in state["fail"]:
            raise RuntimeError("boom token=SECRET")
        fields = [
            f
            for p in fs.PARAMETERS
            if p.measurement == measurement
            for f in p.source_fields
        ]
        frame = _live_frame(fields)
        start = kwargs.get("start_time_override")
        if start is not None:  # trend request
            end = kwargs["end_time_override"]
            index = pd.date_range(start, end, freq="5min", tz="UTC").tz_convert(
                "Asia/Kolkata"
            )
            frame = pd.DataFrame(
                {c: np.linspace(1.0, 2.0, len(index)) for c in fields}, index=index
            )
        return frame

    monkeypatch.setattr(fs_data, "fetch_online_df", fake)
    fs_data.clear_furnace_status_caches()
    yield calls, state
    fs_data.clear_furnace_status_caches()


def _html_text(at: AppTest) -> str:
    """All ``st.html`` bodies on the page, joined."""
    return " ".join(e.proto.body for e in at.get("html"))


def _app() -> AppTest:
    at = AppTest.from_file(str(PAGE_FILE), default_timeout=60)
    at.session_state[VBOARD_NAV_KEY] = FURNACE_STATUS
    return at


def test_default_visualisations_does_not_fetch_furnace_status(monkeypatch) -> None:
    from ui import vboard_visualisations

    calls: list[object] = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Furnace Status fetch ran while Visualisations was active")

    monkeypatch.setattr(fs_data, "fetch_online_df", fail_if_called)
    monkeypatch.setattr(
        vboard_visualisations,
        "render_visualisations",
        lambda: st.write("visualisations-rendered"),
    )
    fs_data.clear_furnace_status_caches()

    at = AppTest.from_file(str(PAGE_FILE), default_timeout=60).run()

    assert not at.exception
    assert calls == []
    assert any("visualisations-rendered" in item.value for item in at.markdown)
    assert "BF2 blast furnace status" not in _html_text(at)


def test_vboard_furnace_status_renders_every_parameter_with_grouped_fetches(
    patched_fetch,
) -> None:
    calls, _ = patched_fetch
    at = _app().run()

    assert not at.exception
    labels = [b.label for b in at.button if b.label.endswith("Open trend")]
    assert len(labels) == len(fs.PARAMETERS) == 34
    assert calls == list(fs.measurements_for())  # one fetch per measurement
    assert not at.sidebar.selectbox  # Visualisations-only controls stay inactive

    # The mocked fields are identical, so the strips are flat and the quadrants
    # tie: both are stated, with no invented winner.
    html = _html_text(at)
    assert "BF2 blast furnace status" in html
    assert "All shown readings are equal" in html
    assert "All quadrants equal (25.0% each)" in html and "is-peak" not in html
    assert "Spread <b>0</b>" in html


def test_switching_to_visualisations_does_not_render_or_refetch_stale_trend(
    patched_fetch, monkeypatch
) -> None:
    from ui import vboard_visualisations

    calls, _ = patched_fetch
    at = _app().run()
    next(b for b in at.button if b.label.startswith("Fuel Rate")).click()
    at.run()
    calls_before_switch = list(calls)

    monkeypatch.setattr(
        vboard_visualisations,
        "render_visualisations",
        lambda: st.write("visualisations-rendered"),
    )
    at.session_state[VBOARD_NAV_KEY] = VISUALISATIONS
    at.run()

    assert calls == calls_before_switch
    assert "fs-dashboard" not in _html_text(at)
    assert "Fuel Rate" not in _html_text(at)
    assert any("visualisations-rendered" in item.value for item in at.markdown)


def test_clicking_a_row_opens_the_trend_view_and_back_returns(patched_fetch) -> None:
    at = _app()
    at.query_params["ticket"] = "TKT-42"
    at.run()
    next(b for b in at.button if b.label.startswith("Fuel Rate")).click()
    at.run()

    assert not at.exception
    assert at.query_params[fs.VIEW_QUERY_KEY] == ["trend"]
    assert at.query_params[fs.PARAMETER_QUERY_KEY] == ["fuel_rate"]
    assert at.query_params["ticket"] == ["TKT-42"]
    assert at.session_state[VBOARD_NAV_KEY] == FURNACE_STATUS
    html = _html_text(at)
    assert "Fuel Rate" in html and "fs-dashboard" not in html
    assert not [
        b for b in at.button if b.label.endswith("Open trend")
    ]  # dashboard is gone

    next(b for b in at.button if "Back to Furnace Status" in b.label).click()
    at.run()
    assert at.query_params["ticket"] == ["TKT-42"]
    assert len([b for b in at.button if b.label.endswith("Open trend")]) == len(
        fs.PARAMETERS
    )

    next(b for b in at.button if b.label.startswith("T2:")).click()  # a strip tile
    at.run()
    assert at.query_params[fs.PARAMETER_QUERY_KEY] == ["uptake_t2"]
    assert "Uptake Temperature T2" in _html_text(at)


def test_unknown_parameter_in_url_falls_back_to_status(patched_fetch) -> None:
    at = _app()
    at.query_params[fs.VIEW_QUERY_KEY] = "trend"
    at.query_params[fs.PARAMETER_QUERY_KEY] = "<script>alert(1)</script>"
    at.run()

    assert not at.exception
    assert len([b for b in at.button if b.label.endswith("Open trend")]) == len(
        fs.PARAMETERS
    )


def test_unavailable_parameter_trend_explains_itself(patched_fetch) -> None:
    calls, _ = patched_fetch
    at = _app()
    at.query_params[fs.VIEW_QUERY_KEY] = "trend"
    at.query_params[fs.PARAMETER_QUERY_KEY] = "slag_rate"
    at.run()

    assert not at.exception
    html = _html_text(at)
    assert "Not available" in html and "daily production report" in html
    assert calls == []  # nothing to fetch for an unsourced parameter


def test_failing_source_is_contained_and_not_leaked(patched_fetch) -> None:
    _, state = patched_fetch
    state["fail"].update({"miscellaneous", "heatload_delta_t"})
    at = _app().run()

    assert not at.exception
    html = _html_text(at)
    assert "Data source unavailable: miscellaneous" in html
    assert "SECRET" not in html
    level = next(b for b in at.button if b.label.startswith("Furnace Level"))
    assert "Not available" in level.label
    assert any(b.label.startswith("Fuel Rate: 12") for b in at.button)
    # No quadrant data: the ring stays neutral instead of inventing shares.
    assert "Shares need all four quadrant readings." in html
    assert "Highest share" not in html
    total = next(b for b in at.button if b.label.startswith("Total Heat Load"))
    assert "Not available" in total.label


def test_refresh_clears_only_this_pages_caches(patched_fetch) -> None:
    calls, _ = patched_fetch
    unrelated_runs: list[int] = []

    @st.cache_data
    def unrelated_cache(x: int) -> int:
        unrelated_runs.append(x)
        return x

    unrelated_cache.clear()
    unrelated_cache(1)

    at = _app().run()
    assert calls == list(fs.measurements_for())

    at.run()  # second render within the TTL: served from cache
    assert calls == list(fs.measurements_for())

    next(b for b in at.button if b.label == "Refresh").click()
    at.run()
    assert calls == list(fs.measurements_for()) * 2  # status caches cleared

    unrelated_cache(1)
    assert unrelated_runs == [1]  # an unrelated cache was left alone
    unrelated_cache.clear()


def test_status_snapshot_age_logic_is_wall_clock_independent() -> None:
    """Guard: freshness is judged against the injected ``now``, not the system clock."""
    frame = pd.DataFrame(
        {"fuel_rate": [500.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp(NOW).tz_convert("Asia/Kolkata")], name="time (IST)"
        ),
    )
    spec = fs.PARAMETERS_BY_KEY["fuel_rate"]
    assert fs.resolve_reading(spec, {"process_params": frame}, NOW).value == 500.0
    assert (
        fs.resolve_reading(
            spec, {"process_params": frame}, NOW + timedelta(minutes=20)
        ).issue
        == "stale"
    )
