"""Furnace status service (SV-2.1, #293): the snapshot equals the Streamlit V-Board for the same timestamp, limit
states, live-status tags, hot-metal tags, trend parity and decimation, one snapshot query per 10-s slot.

The InfluxDB boundary (``TimeSeriesDataFetcher`` in ``data.furnace_status.repository``) is replaced by a counting fake
that serves a recorded-like 15-min frame per measurement; no InfluxDB, Neon or Redis is reached.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data import furnace_status as fs
from data.furnace_status import repository as fs_repository
from services import status
from services.events import LiveStatusData, TagValue, WorkerStatus

NOW = datetime(2026, 10, 27, 9, 30, 15, tzinfo=UTC)  # 15:00:15 IST
IST = "Asia/Kolkata"
KPI_KEYS = (
    "production_rate",
    "hot_blast_volume",
    "o2_injection",
    "pci_rate",
    "co_utilization",
    "raft",
    "charges_per_hr",
)


# ---------------------------------------------------------------- fake InfluxDB


def plant_frames(now: datetime = NOW) -> dict[str, pd.DataFrame | Exception]:
    """A 15-min snapshot of the four measurements at 1-min cadence ending at ``now`` (floored to the minute): typical
    BF-02 values, blast pressure above and RAFT below their limits, an incomplete last fuel-rate row, miscellaneous
    failing, one hearth pad stale (17 min) and one without data."""
    end = pd.Timestamp(now).tz_convert(IST).floor("min")
    idx = pd.date_range(end=end, periods=20, freq="1min", name="time (IST)")
    ramp = np.linspace(0.0, 1.0, len(idx))
    process = pd.DataFrame(
        {
            "theoretical_production_per_day": 2310.0 + 15.0 * ramp,
            "production_per_hour": 96.4 + 0.8 * ramp,
            "charges_per_hour": np.full(len(idx), 6.12),
            "hot_blast_vol_nm3h": 101_500.0 + 900.0 * ramp,
            "hot_blast_press": 2.80 + 0.12 * ramp,  # 2.92 bar at the end: above Pb_max 2.85
            "top_press_avg": np.full(len(idx), 1.413),
            "body_dp_total": 1.33 + 0.01 * ramp,
            "body_dp_bottom": np.full(len(idx), 0.981),
            "body_etaco": 47.3 + 0.2 * ramp,
            "h2_pct": np.full(len(idx), 3.07),
            "hot_blast_temp": 1185.0 + 2.0 * ramp,
            "body_raft": 2080.0 + 5.0 * ramp,  # below RAFT_min 2,100
            "top_temp_1": 128.0 + ramp,
            "top_temp_2": 141.0 - ramp,
            "top_temp_3": np.full(len(idx), 135.2),
            "top_temp_4": 150.0 + 2.0 * ramp,
            "top_temp_avg": np.full(len(idx), 138.6),
            "steam_injection": np.full(len(idx), 2450.0),  # kg/h, shown in TPH
            "oxygen_enrichment_pct": np.full(len(idx), 4.21),
            "oxygen_flow": 5320.0 + 10.0 * ramp,
            "fuel_rate": np.full(len(idx), 545.0),  # the fuel-rate setpoint
            "coal_rate_actual_value": 152.0 + ramp,
            "coke_rate": 338.0 - ramp,
            "nut_coke_rate": np.r_[np.full(len(idx) - 1, 48.0), np.nan],  # last row incomplete
            "coal_rate_set_value": np.full(len(idx), 150.0),
            "body_perm": np.full(len(idx), 2.64),
            "tuyere_velocity": np.full(len(idx), 236.4),
        },
        index=idx,
    )
    heat = pd.DataFrame(
        {
            f"heat_load_r{row}_q{q}": 0.30 + 0.02 * row + 0.05 * q + 0.01 * ramp
            for row in range(6, 11)
            for q in range(1, 5)
        },
        index=idx,
    )
    hearth = pd.DataFrame(
        {
            "temp_4373_b": np.r_[np.full(3, 412.0), np.full(len(idx) - 3, np.nan)],  # last value 17 min old: stale
            "temp_5411_c": np.full(len(idx), np.nan),  # no data in the window
            "temp_5757_c": np.full(len(idx), 388.5),
            "temp_6103_b": 371.0 + ramp,
        },
        index=idx,
    )
    return {
        "process_params": process,
        "miscellaneous": RuntimeError("influx timeout token=SECRET"),
        "heatload_delta_t": heat,
        "temperature_profile": hearth,
    }


class CountingFetcher:
    """Stand-in for ``TimeSeriesDataFetcher``: serves ``frames`` (snapshot requests) or a 1-min ramp over the range
    (trend requests) and counts every request per measurement."""

    def __init__(self, frames: dict[str, pd.DataFrame | Exception], *, delay_s: float = 0.0) -> None:
        self.frames = frames
        self.delay_s = delay_s
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def __call__(self, measurement: str, *, debug: bool, source: str) -> CountingFetcher._One:
        assert debug is False and source == "historical"
        return CountingFetcher._One(self, measurement)

    @property
    def snapshot_queries(self) -> int:
        """Snapshot queries (one request per measurement each)."""
        live = [c for c in self.calls if c["start_time"] is None]
        return len(live) // len(fs.measurements_for())

    class _One:
        def __init__(self, owner: CountingFetcher, measurement: str) -> None:
            self.owner, self.measurement = owner, measurement

        def fetch_data(self, time_interval, start_time, end_time, *, request_type, window_by, fields):
            with self.owner._lock:
                self.owner.calls.append(
                    {"measurement": self.measurement, "start_time": start_time, "end_time": end_time, "fields": fields}
                )
            if self.owner.delay_s:
                time.sleep(self.owner.delay_s)
            if start_time is not None:  # trend: a ramp over the range at 1-min cadence
                index = pd.date_range(start_time, end_time, freq="1min").tz_convert(IST)
                return pd.DataFrame({f: np.linspace(1.0, 2.0, len(index)) for f in fields}, index=index)
            frame = self.owner.frames[self.measurement]
            if isinstance(frame, Exception):
                raise frame
            return frame.loc[:, [f for f in fields if f in frame.columns]].copy()


@pytest.fixture(autouse=True)
def _fresh_caches():
    status.clear_cache()
    yield
    status.clear_cache()


@pytest.fixture()
def fetcher(monkeypatch) -> CountingFetcher:
    fake = CountingFetcher(plant_frames())
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", fake)
    return fake


def rel_close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def utc(ts: pd.Timestamp | None) -> datetime | None:
    return None if ts is None else pd.Timestamp(ts).tz_convert("UTC").to_pydatetime()


# ---------------------------------------------------------------- parity with the Streamlit V-Board


def assert_snapshot_equals_page(view: status.StatusSnapshotView, page: fs.StatusSnapshot) -> None:
    """The view shows what the V-Board shows for the same snapshot: values (relative difference <= 1e-6), times,
    display text, telemetry header, operating state, failed sources, zones."""
    from ui.furnace_status import components
    from ui.furnace_status.layout import ZONE_READINGS, ZONES

    assert [p.key for p in view.parameters] == [r.spec.key for r in page.readings] == [s.key for s in fs.PARAMETERS]
    for p, r in zip(view.parameters, page.readings, strict=True):
        assert rel_close(p.value, r.value), p.key
        assert rel_close(p.setpoint, r.setpoint), p.key
        assert p.at == utc(r.timestamp) and p.setpoint_at == utc(r.setpoint_timestamp), p.key
        assert p.display == fs.format_reading(r), p.key
        assert (p.label, p.unit, p.decimals, p.section) == (r.spec.label, r.spec.unit, r.spec.decimals, r.spec.section)
        assert p.state == (r.issue or p.state) and (r.issue is None) == (p.state in ("ok", "low", "high")), p.key
        # The page's accessible row label carries the same text the API returns.
        unit = f" {p.unit}" if (p.value is not None or p.setpoint is not None) and p.unit else ""
        assert components.button_label(r).startswith(f"{p.label}: {p.display}{unit}."), p.key
    f = view.freshness
    assert (f.telemetry, f.telemetry_label, f.detail) == (page.status.level, page.status.label, page.status.detail)
    assert rel_close(f.coverage, page.status.coverage)
    assert f.data_at == utc(page.last_updated)
    assert f.stale == (page.status.level == "offline")
    assert f.failed_sources == list(page.failed_measurements)
    o = view.operating_state
    assert (o.level, o.label, o.detail) == (
        page.operating_state.level,
        page.operating_state.label,
        page.operating_state.detail,
    )
    assert view.sections == list(fs.SECTIONS)
    assert [(z.name, z.lower_m, z.upper_m) for z in view.zones] == list(ZONES)
    assert {z.name: tuple(z.readings) for z in view.zones} == ZONE_READINGS


def test_snapshot_equals_the_streamlit_page_for_the_same_timestamp(fetcher) -> None:
    page = fs.load_status_snapshot(NOW)  # what the V-Board renders (ui.furnace_status.dashboard)
    view = status.build_snapshot_view(NOW)
    assert_snapshot_equals_page(view, page)

    by_key = {p.key: p for p in view.parameters}
    states = {k: by_key[k].state for k in ("blast_pressure", "raft", "hot_blast_volume", "furnace_level", "slag_rate")}
    assert states == {
        "blast_pressure": "high",
        "raft": "low",
        "hot_blast_volume": "ok",
        "furnace_level": "fetch_failed",
        "slag_rate": "no_source",
    }
    assert by_key["hearth_temp_a"].state == "stale" and by_key["hearth_temp_b"].state == "no_data"
    assert by_key["steam_injection"].value == pytest.approx(2.45) and by_key["steam_injection"].unit == "TPH"
    # Fuel rate: setpoint and the newest complete component row (the last row lacks nut coke).
    assert by_key["fuel_rate"].setpoint == 545.0
    assert by_key["fuel_rate"].value == pytest.approx(152.0 + 18 / 19 + 338.0 - 18 / 19 + 48.0)
    assert by_key["fuel_rate"].display == "545 / 538"
    assert view.freshness.computed_at == NOW and view.freshness.source == "request"
    assert view.freshness.telemetry == "live" and view.freshness.failed_sources == ["miscellaneous"]
    assert view.operating_state.level == "normal" and view.operating_state.state == "normal"
    assert view.operating_state.value == pytest.approx(102_400.0)
    assert fetcher.snapshot_queries == 1  # the second path read the 5-s request cache


@pytest.mark.parametrize(
    ("volume", "level", "state"), [(26_000.0, "normal", "normal"), (24_999.0, "abnormal", "reduced")]
)
def test_operating_state_wording_of_the_live_event(monkeypatch, volume, level, state) -> None:
    frames = plant_frames()
    frames["process_params"]["hot_blast_vol_nm3h"] = volume
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", CountingFetcher(frames))
    view = status.build_snapshot_view(NOW)
    assert (view.operating_state.level, view.operating_state.state) == (level, state)
    assert status.live_status_fields(view)["operating_state"] == state


def test_operating_state_is_null_without_a_fresh_blast_volume(monkeypatch) -> None:
    frames = plant_frames()
    frames["process_params"]["hot_blast_vol_nm3h"] = np.nan
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", CountingFetcher(frames))
    view = status.build_snapshot_view(NOW)
    assert view.operating_state.level == "unknown" and view.operating_state.state is None


def test_a_snapshot_without_any_source_is_stale_and_offline(monkeypatch) -> None:
    down = {m: RuntimeError("down") for m in fs.measurements_for()}
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", CountingFetcher(down))
    view = status.build_snapshot_view(NOW)
    assert view.freshness.stale and view.freshness.telemetry == "offline" and view.freshness.data_at is None
    assert all(p.value is None for p in view.parameters)
    fields = status.live_status_fields(view)
    assert fields["stale"] is True and fields["data_at"] is None and fields["operating_state"] is None


# ---------------------------------------------------------------- limits


def test_every_limit_reference_resolves_in_the_fact_register() -> None:
    from utils.burden_engine.facts import load_register

    register = load_register()
    referenced = [spec for spec in fs.PARAMETERS if spec.limits is not None]
    assert {s.key for s in referenced} == {
        "hot_blast_volume",
        "blast_pressure",
        "top_pressure",
        "body_dp_total",
        "hbt",
        "raft",
        "o2_injection",
        "oxygen_flow",
        "fuel_rate",
    }
    for spec in referenced:
        limits = status.parameter_limits(spec.key)
        assert limits is not None, spec.key
        params = register.active(spec.limits.fact).params
        assert limits.low == (params[spec.limits.low] if spec.limits.low else None), spec.key
        assert limits.high == (params[spec.limits.high] if spec.limits.high else None), spec.key
        if limits.low is not None and limits.high is not None:
            assert limits.low < limits.high, spec.key
    assert status.parameter_limits("raft").model_dump() == {
        "low": 2100.0,
        "high": 2350.0,
        "fact": "ASM-11",
        "basis": "Process limits (observed Jun-Oct envelope, not equipment ratings)",
    }
    assert (status.parameter_limits("fuel_rate").low, status.parameter_limits("fuel_rate").high) == (520.0, 600.0)
    assert status.parameter_limits("production_rate") is None


@pytest.mark.parametrize(
    ("value", "issue", "expected"),
    [
        (2099.999, None, "low"),
        (2100.0, None, "ok"),
        (2350.0, None, "ok"),
        (2350.001, None, "high"),
        (None, "stale", "stale"),
        (None, "fetch_failed", "fetch_failed"),
    ],
)
def test_limit_state_compares_the_unrounded_reading(value, issue, expected) -> None:
    spec = fs.PARAMETERS_BY_KEY["raft"]
    reading = fs.ParameterReading(spec, value=value, issue=issue)
    assert status.reading_state(reading, status.parameter_limits("raft")) == expected


def test_a_catalogue_limit_needs_a_fact_and_a_bound(tmp_path) -> None:
    import yaml

    from data.furnace_status.catalogue import CONFIG_PATH, CatalogueError, load_catalogue

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raft = next(p for p in raw["parameters"] if p["key"] == "raft")
    for bad, message in (({"fact": "ASM-11"}, "must name low, high or both"), ({"low": "RAFT_min"}, "limits.fact")):
        raft["limits"] = bad
        path = tmp_path / "furnace_status.yml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(CatalogueError, match=message):
            load_catalogue(path)


# ---------------------------------------------------------------- live status tags


def test_live_status_fields_carry_every_sourced_parameter_and_the_hot_metal(fetcher) -> None:
    view = status.build_snapshot_view(NOW)
    hot_metal = {
        "hmt": TagValue(value=1492.0, unit="°C", at=NOW - timedelta(minutes=40)),
        "hm_si": TagValue(value=0.46, unit="%", at=NOW - timedelta(minutes=40)),
    }
    fields = status.live_status_fields(view, hot_metal)
    sourced = {spec.key for spec in fs.PARAMETERS if spec.has_source}
    assert set(fields["tags"]) == sourced | {"hmt", "hm_si"} and "slag_rate" not in fields["tags"]
    assert set(KPI_KEYS) <= set(fields["tags"])
    by_key = {p.key: p for p in view.parameters}
    for key in sourced:
        tag = fields["tags"][key]
        assert (tag.value, tag.unit, tag.at) == (by_key[key].value, by_key[key].unit, by_key[key].at), key
    assert fields["tags"]["furnace_level"].value is None  # its measurement failed
    assert fields["operating_state"] == "normal" and fields["stale"] is False and fields["telemetry"] == "live"
    assert fields["data_at"] == view.freshness.data_at
    assert (fields["stale_after_minutes"], fields["live_max_age_minutes"]) == (15.0, 5.0)

    worker = WorkerStatus(name="host:1", started_at=NOW, running=0, queued=0, pool_size=8)
    event = LiveStatusData(worker=worker, **fields)
    again = LiveStatusData.model_validate_json(event.model_dump_json())
    assert again.tags["raft"].value == by_key["raft"].value and again.tags["hmt"].unit == "°C"
    assert again.worker == worker
    bare = LiveStatusData(worker=worker)  # a beat without clients: the worker part only
    assert bare.tags == {} and bare.stale is None and bare.operating_state is None


def _hm_frame() -> pd.DataFrame:
    t = pd.to_datetime(
        ["2026-10-27 11:05", "2026-10-27 12:40", "2026-10-27 13:55", "2026-10-27 14:30", "2026-10-27 14:50"]
    )
    return pd.DataFrame(
        {
            "t": t,
            "hmt_gt_1480c": [1488.0, 1495.0, 1502.0, 1210.0, np.nan],  # 14:30 is below the 1,350 °C tap filter
            "chem_pct_si": [0.41, 0.52, np.nan, 0.47, 0.0],  # 14:50 is a zero (no analysis)
        }
    )


def test_latest_hot_metal_takes_the_latest_valid_samples() -> None:
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls.append((start, end))
        return _hm_frame()

    tags = status.latest_hot_metal(NOW, fetch=fetch)
    assert tags["hmt"] == TagValue(value=1502.0, unit="°C", at=datetime(2026, 10, 27, 8, 25, tzinfo=UTC))
    assert tags["hm_si"] == TagValue(value=0.47, unit="%", at=datetime(2026, 10, 27, 9, 0, tzinfo=UTC))
    ((start, end),) = calls
    assert end - start == pd.Timedelta(hours=24, minutes=1) and start == pd.Timestamp("2026-10-26 15:00:15")


def test_latest_hot_metal_is_empty_when_neon_is_not_reachable() -> None:
    def down(start, end):
        raise ConnectionError("could not connect to server")

    assert status.latest_hot_metal(NOW, fetch=down) == {}
    assert status.latest_hot_metal(NOW, fetch=lambda s, e: pd.DataFrame(columns=["t"])) == {}


# ---------------------------------------------------------------- trends


def test_trend_equals_the_streamlit_trend_view(fetcher) -> None:
    spec = fs.PARAMETERS_BY_KEY["pci_rate"]
    start, end = fs.resolve_fixed_range("1h", NOW)
    page = fs.load_trend_view(spec, start, end, use_live_current=True)  # the V-Board trend view
    trend = status.build_trend(["pci_rate"], "1h", now=NOW)
    (series,) = trend.series
    assert (trend.start, trend.end) == (start, end) and series.status == "ok"
    assert series.raw_points == len(page.trend.series) == 61
    assert series.times == [utc(t) for t in page.trend.series.index]
    assert all(rel_close(a, float(b)) for a, b in zip(series.values, page.trend.series, strict=True))
    assert series.setpoint_times == [utc(t) for t in page.trend.setpoint.index]
    assert rel_close(series.stats.mean, page.stats.mean) and rel_close(series.stats.maximum, page.stats.maximum)
    assert series.current.display == fs.format_reading(page.current) == "150 / 153"
    assert series.current.state == "ok"


def test_trend_of_a_parameter_without_source_explains_itself(fetcher) -> None:
    (series,) = status.build_trend(["slag_rate"], "8h", now=NOW).series
    assert series.status == "no_source" and "daily production report" in series.message
    assert series.times == [] and series.current.state == "no_source"


def test_custom_trend_range_is_ist_and_at_most_seven_days(fetcher) -> None:
    trend = status.build_trend(
        ["raft"], "custom", datetime(2026, 10, 26, 6, 0), datetime(2026, 10, 26, 14, 0), max_points=200, now=NOW
    )
    assert trend.start == datetime(2026, 10, 26, 0, 30, tzinfo=UTC)
    assert trend.end == datetime(2026, 10, 26, 8, 30, tzinfo=UTC)
    (series,) = trend.series
    assert series.raw_points == 481 and len(series.times) <= 200
    with pytest.raises(status.RangeError, match="at most 7 days"):
        status.build_trend(["raft"], "custom", datetime(2026, 10, 18), datetime(2026, 10, 26), now=NOW)
    with pytest.raises(status.RangeError, match="needs start and end"):
        status.build_trend(["raft"], "custom", now=NOW)
    with pytest.raises(status.RangeError, match="Unknown interval"):
        status.build_trend(["raft"], "2h", now=NOW)
    with pytest.raises(KeyError):
        status.build_trend(["no_such_key"], "1h", now=NOW)


def _seven_days() -> pd.Series:
    idx = pd.date_range("2026-10-20 06:00", periods=7 * 8640, freq="10s", tz=IST, name="time (IST)")
    values = 2200.0 + 30.0 * np.sin(np.arange(len(idx)) / 600.0)
    series = pd.Series(values, index=idx, name="RAFT")
    series.iloc[20_000] = 2450.0  # a one-sample spike
    series.iloc[40_000] = 1900.0  # a one-sample dip
    hole = (idx >= pd.Timestamp("2026-10-23 10:00", tz=IST)) & (idx < pd.Timestamp("2026-10-23 12:00", tz=IST))
    return series[~hole]


def test_decimation_keeps_extremes_gaps_and_the_point_budget() -> None:
    raw = _seven_days()
    out = status.decimate(raw, 2000)
    assert len(raw) == 59_760 and len(out) <= 2000
    assert out.max() == 2450.0 and out.min() == 1900.0  # one-sample extremes survive
    gaps = out[out.isna()]
    assert len(gaps) == 1
    assert pd.Timestamp("2026-10-23 10:00", tz=IST) < gaps.index[0] < pd.Timestamp("2026-10-23 12:00", tz=IST)
    assert out.index.is_monotonic_increasing and str(out.index.tz) == IST
    short = raw.iloc[:500]
    assert status.decimate(short, 2000) is short  # a series that fits is untouched


def test_decimation_bound_holds_for_sparse_data_with_many_holes() -> None:
    # 30 bursts of 10 samples, 20 h apart: every burst sits in its own bucket, every gap is a hole.
    starts = pd.date_range("2026-10-01", periods=30, freq="20h", tz=IST)
    idx = pd.DatetimeIndex([s + pd.Timedelta(seconds=10 * i) for s in starts for i in range(10)])
    raw = pd.Series(np.arange(300.0), index=idx)
    out = status.decimate(raw, 150)
    assert len(out) <= 150 and int(out.isna().sum()) == 29
    # 3-min cadence (each step a raw gap): no hole at the scale of the buckets, the budget holds.
    steady = pd.Series(np.arange(3000.0), index=pd.date_range("2026-10-20", periods=3000, freq="3min", tz=IST))
    for points in (100, 101, 1000):
        out = status.decimate(steady, points)
        assert len(out) <= points and not out.isna().any()


# ---------------------------------------------------------------- one query per slot


def test_current_snapshot_is_taken_once_per_ten_second_slot(monkeypatch) -> None:
    fake = CountingFetcher(plant_frames(), delay_s=0.2)
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", fake)
    clock = {"now": NOW}
    monkeypatch.setattr(status, "clock", lambda: clock["now"])

    views: list[status.StatusSnapshotView] = []
    threads = [threading.Thread(target=lambda: views.append(status.current_snapshot_view())) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(views) == 12 and fake.snapshot_queries == 1  # 12 concurrent callers, one query
    assert len({v.freshness.computed_at for v in views}) == 1

    fs_repository.clear_cache()  # past the 5-s request cache as well
    clock["now"] = NOW + timedelta(seconds=4)  # same slot [09:30:10, 09:30:20)
    status.current_snapshot_view()
    assert fake.snapshot_queries == 1
    clock["now"] = NOW + timedelta(seconds=6)  # next slot
    status.current_snapshot_view()
    assert fake.snapshot_queries == 2
