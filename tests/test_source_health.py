"""Source health (SV-6.5, #319): the registry of the five sources, stale badges within a minute of an InfluxDB
outage, jobs that skip with a reason while a source is down, and the route ``GET /api/twin/now``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from services import coke_energy as ce
from services import source_health, twin_views
from services import twin as twin_service
from services.jobs import TwinEvaluateParams
from services.source_health import SourceDown, SourceHealth, observe_status_view
from tests.test_api_twin import client, get, twin  # noqa: F401 - fixtures
from tests.test_scheduled_jobs import context, record, store, utc_of  # noqa: F401 - fixtures
from worker.kinds import twin as twin_kind
from worker.runtime import SkipJob

T0 = datetime(2026, 10, 27, 5, 0, tzinfo=UTC)
M = timedelta(minutes=1)


def registry(start: datetime = T0, retry_after_s: float = 300.0) -> tuple[SourceHealth, dict[str, datetime]]:
    now = {"t": start}
    return SourceHealth(retry_after_s=retry_after_s, clock=lambda: now["t"]), now


# ------------------------------------------------------------------------------------------------- the registry
def test_a_failure_marks_the_source_failed_then_down_for_the_hold_and_an_ok_clears_it() -> None:
    health, now = registry()
    health.record_ok("influxdb", data_end=pd.Timestamp("2026-10-27 10:20"), value={"co": 23.1})  # IST naive
    state = health.state("influxdb")
    assert state.data_end == datetime(2026, 10, 27, 4, 50, tzinfo=UTC) and not state.failed
    now["t"] = T0 + 2 * M
    health.record_failure("influxdb", ConnectionRefusedError("influx.local:8086"))
    assert health.failed("influxdb") and health.down("influxdb")
    assert health.last_value("influxdb") == {"co": 23.1}  # the last value is kept
    assert health.reason("influxdb") == (
        "InfluxDB unreachable since 27 Oct 10:32 IST (ConnectionRefusedError: influx.local:8086); "
        "next try after 27 Oct 10:37 IST"
    )
    with pytest.raises(SourceDown, match="InfluxDB unreachable") as caught:
        health.require("hourly", "influxdb")
    assert caught.value.key == "influxdb"
    now["t"] = T0 + 7 * M + timedelta(seconds=1)  # the hold ended: the next job probes again
    assert health.failed("influxdb") and not health.down("influxdb")
    health.require("influxdb")
    assert (
        health.reason("influxdb")
        == "InfluxDB unreachable since 27 Oct 10:32 IST (ConnectionRefusedError: influx.local:8086)"
    )
    health.record_failure("influxdb", "timeout")
    assert health.down("influxdb") and health.state("influxdb").failures == 2
    health.record_ok("influxdb", data_end=T0 + 8 * M)
    row = next(r for r in health.view() if r["key"] == "influxdb")
    assert not row["failed"] and not row["down"] and row["failures"] == 0 and row["data_end"] == T0 + 8 * M
    assert row["has_value"] and row["last_error"] == "timeout"  # the last error stays readable
    with pytest.raises(KeyError):
        health.record_ok("qdrant")


def test_observe_records_the_outcome_and_raises_source_down() -> None:
    health, now = registry()
    frame = pd.DataFrame({"ts": pd.to_datetime(["2026-10-27 10:00", "2026-10-27 10:10"])})
    out = health.observe("influxdb", lambda: frame, data_end=lambda f: f["ts"].max(), keep=True)
    assert out is frame and health.state("influxdb").data_end == datetime(2026, 10, 27, 4, 40, tzinfo=UTC)
    assert health.last_value("influxdb") is frame

    def boom() -> None:
        raise OSError("dataset host unreachable")

    with pytest.raises(SourceDown, match="Hourly dataset unreachable since") as caught:
        health.observe("hourly", boom)
    assert isinstance(caught.value.__cause__, OSError) and health.down("hourly")
    assert health.state("hourly").last_error == "OSError: dataset host unreachable"


def test_the_status_snapshot_view_records_influxdb() -> None:
    health, now = registry()

    def view(failed: list[str], data_at: datetime | None, at: datetime) -> Any:
        return SimpleNamespace(freshness=SimpleNamespace(failed_sources=failed, data_at=data_at, computed_at=at))

    assert observe_status_view(view([], T0 - M, T0), health=health)
    assert health.state("influxdb").data_end == T0 - M and health.state("influxdb").last_ok_at == T0
    assert observe_status_view(view(["heatload_delta_t"], T0 - M, T0 + 10 * M), health=health)  # one measurement
    assert not observe_status_view(view(["process_params", "temperature_profile"], None, T0 + 20 * M), health=health)
    state = health.state("influxdb")
    assert state.failed and state.last_failure_at == T0 + 20 * M
    assert state.last_error == "no reading; measurements not read: process_params, temperature_profile"
    assert observe_status_view(SimpleNamespace(), health=health) is False  # no freshness: unchanged
    # the stale copy an API process reads from Redis reports the producer's outage, not a reading
    stale = view(["process_params"], T0 - M, T0 + 30 * M)
    stale.freshness.telemetry = "offline"
    health.record_ok("influxdb", at=T0 + 25 * M)
    assert not observe_status_view(stale, health=health) and health.failed("influxdb")


def test_the_status_snapshot_keeps_the_last_values_while_influxdb_is_down(monkeypatch) -> None:
    from domain.furnace_status.status import build_status_snapshot
    from services import status as status_service

    good = status_service.StatusSnapshotView(
        parameters=[],
        sections=["Blast"],
        zones=[],
        operating_state=status_service.StatusOperatingState(
            level="normal", state="normal", label="Normal", detail="", parameter="hot_blast_volume", normal_min=25_000
        ),
        freshness=status_service.StatusFreshness(
            computed_at=T0,
            data_at=T0 - M,
            age_s=60.0,
            stale=False,
            telemetry="live",
            telemetry_label="Live",
            detail="40 of 41 mapped parameters reporting",
            coverage=0.98,
            stale_after_minutes=15.0,
            live_max_age_minutes=5.0,
            live_min_coverage=0.8,
            source="producer",
        ),
    )
    assert observe_status_view(good)  # the producer's good view, kept as the last value
    monkeypatch.setattr(status_service, "load_status_snapshot", lambda at: build_status_snapshot({}, at))
    view = status_service.build_snapshot_view(T0 + 10 * M, source="request")
    assert view.freshness.telemetry == "offline" and view.freshness.stale and view.freshness.coverage == 0.0
    assert view.freshness.data_at == T0 - M and view.freshness.age_s == 660.0 and view.freshness.source == "request"
    assert view.freshness.failed_sources and view.freshness.detail.startswith("InfluxDB not reachable; last values")
    assert view.sections == ["Blast"] and view.operating_state.level == "normal"  # the last values
    assert source_health.HEALTH.failed("influxdb")  # down for RETRY_AFTER_S on the live clock; T0 is in the past
    assert observe_status_view(view) is False  # the copy itself never counts as a reading


# ------------------------------------------------------------------------------------------- stale within 1 min
def latest_evaluation(at: datetime) -> Any:
    ends = {
        "influxdb": (at - 10 * M).isoformat(),
        "hourly": (at - 90 * M).isoformat(),
        "hm_slag": (at - 3 * timedelta(hours=1)).isoformat(),
        "dpr": None,
        "open_meteo": (at - timedelta(hours=20)).isoformat(),
    }
    return SimpleNamespace(flags={"sources": ends, "provisional_dpr": True}, evaluated_at=at)


def test_an_influxdb_outage_shows_the_stale_badge_within_a_minute() -> None:
    health, now = registry()
    latest = latest_evaluation(T0)
    limits = twin_views.STALE_AFTER_H
    rows = {r["key"]: r for r in twin_views.source_freshness(latest, T0 + M, limits, health.view(T0 + M))}
    assert not rows["influxdb"]["stale"] and not rows["influxdb"]["down"] and rows["influxdb"]["error"] is None
    assert rows["dpr"]["stale"] and rows["dpr"]["provisional"] and rows["dpr"]["data_end"] is None

    # 10:30 beat of the worker: InfluxDB answers; 10:40: it does not. The badge is stale from that beat on,
    # though the newest reading (05:30 UTC) is only 10 min old against the 30-min limit.
    health.record_ok("influxdb", data_end=T0 + 30 * M, at=T0 + 30 * M)
    rows = {r["key"]: r for r in twin_views.source_freshness(latest, T0 + 31 * M, limits, health.view(T0 + 31 * M))}
    assert rows["influxdb"]["data_end"] == T0 + 30 * M and rows["influxdb"]["recorded_at"] == T0 + 30 * M  # newer
    assert not rows["influxdb"]["stale"] and rows["influxdb"]["age_h"] == pytest.approx(1 / 60)
    health.record_failure("influxdb", "no reading; measurements not read: process_params", at=T0 + 40 * M)
    at = T0 + 40 * M + timedelta(seconds=10)
    rows = {r["key"]: r for r in twin_views.source_freshness(latest, at, limits, health.view(at))}
    influx = rows["influxdb"]
    assert influx["stale"] and influx["down"] and influx["failed_at"] == T0 + 40 * M
    assert influx["error"] == "no reading; measurements not read: process_params"
    assert influx["data_end"] == T0 + 30 * M  # the last values are kept
    assert not rows["hourly"]["stale"] and not rows["hourly"]["down"]  # the others are untouched
    # the next beat with a reading clears it
    health.record_ok("influxdb", data_end=T0 + 50 * M, at=T0 + 50 * M)
    rows = {r["key"]: r for r in twin_views.source_freshness(latest, T0 + 51 * M, limits, health.view(T0 + 51 * M))}
    assert not rows["influxdb"]["stale"] and not rows["influxdb"]["down"]
    # without the registry the row is as before SV-6.5
    before = twin_views.source_freshness(latest, T0 + M, limits)
    assert [r["down"] for r in before] == [False] * 5 and not before[0]["stale"]


def test_the_route_carries_the_live_health(client, twin) -> None:  # noqa: F811 - fixtures
    source_health.HEALTH.record_failure("hourly", OSError("dataset host unreachable"))
    body = get(client, "op", "/api/twin/now").json()
    rows = {s["key"]: s for s in body["sources"]}
    assert rows["hourly"]["down"] and rows["hourly"]["stale"] and rows["hourly"]["failed_at"] is not None
    assert rows["hourly"]["error"] == "OSError: dataset host unreachable"
    assert not rows["influxdb"]["down"] and rows["influxdb"]["error"] is None


# ------------------------------------------------------------------------------------- load_frames and the job
KEYS = ce.DataKeys("synthetic.csv", 1, "2026-05-15T10:00:00+05:30", 0, "2026-04-01", "{}", "", 30.0)


def loaders(fail: str | None = None) -> ce.Loaders:
    fd = pd.DataFrame({"time": pd.to_datetime(["2026-05-15 08:00", "2026-05-15 09:00"])})
    m10 = pd.DataFrame({"ts": pd.to_datetime(["2026-05-15 09:50", "2026-05-15 10:00"])})
    hm = pd.DataFrame({"t": pd.to_datetime(["2026-05-15 07:30"])})
    dpr = pd.DataFrame({"hm_t": [1.0, 2.0]}, index=pd.to_datetime(["2026-05-13", "2026-05-14"]))

    def failing(*args: Any) -> Any:
        raise ConnectionError(f"{fail} down")

    return ce.Loaders(
        hourly=failing if fail == "hourly" else (lambda path, mtime: fd),
        online=failing if fail == "influxdb" else (lambda *a: m10),
        neon=failing if fail == "neon" else (lambda *a: (hm, dpr, [pd.Timestamp("2026-05-14").date()])),
        lab=lambda *a: (pd.DataFrame(), pd.DataFrame()),
        names=dict,
    )


def test_load_frames_records_every_source() -> None:
    frames = twin_service.load_frames(KEYS, loaders())
    assert len(frames.fd) == 2 and frames.provisional == (pd.Timestamp("2026-05-14").date(),)
    ends = {r["key"]: r["data_end"] for r in source_health.HEALTH.view()}
    assert ends["hourly"] == utc_of("2026-05-15 09:00") and ends["influxdb"] == utc_of("2026-05-15 10:00")
    assert ends["hm_slag"] == utc_of("2026-05-15 07:30")
    assert ends["dpr"] == utc_of("2026-05-13 00:00")  # the provisional date is not a DPR
    for fail, keys in (("hourly", ["hourly"]), ("influxdb", ["influxdb"]), ("neon", ["hm_slag", "dpr"])):
        source_health.HEALTH.reset()
        with pytest.raises(SourceDown, match="unreachable since"):
            twin_service.load_frames(KEYS, loaders(fail))
        assert [k for k in source_health.SOURCES if source_health.HEALTH.down(k)] == keys


def test_the_hourly_job_skips_with_the_reason_while_a_source_is_down(store, monkeypatch) -> None:  # noqa: F811
    calls: list[Any] = []
    monkeypatch.setattr(
        twin_service, "ensure_dataset", lambda end: calls.append(end) or (True, twin_service.DatasetState(end, end))
    )
    source_health.HEALTH.record_failure("hm_slag", "Neon: connection timed out")
    job, bus = context("twin.evaluate", record("twin.evaluate", slot=utc_of("2026-05-15 10:10")))
    with pytest.raises(SkipJob, match=r"HM/slag unreachable since .* \(Neon: connection timed out\); next try after"):
        asyncio.run(twin_kind.evaluate(job, TwinEvaluateParams(backfill_hours=0, strata=False)))
    assert calls == [] and asyncio.run(bus.recent(10)) == []  # nothing probed, nothing published

    def no_dataset(end: Any) -> Any:
        raise OSError("dataset host unreachable")

    source_health.HEALTH.reset()
    monkeypatch.setattr(twin_service, "ensure_dataset", no_dataset)
    with pytest.raises(SkipJob, match="Hourly dataset unreachable since"):
        asyncio.run(twin_kind.evaluate(job, TwinEvaluateParams(backfill_hours=0, strata=False)))
    assert source_health.HEALTH.down("hourly")

    def evaluate(t: Any, strata: bool = True) -> Any:
        raise SourceDown("influxdb", "InfluxDB unreachable since 15 May 10:10 IST (timeout)")

    source_health.HEALTH.reset()
    monkeypatch.setattr(twin_service, "ensure_dataset", lambda end: (True, twin_service.DatasetState(end, end)))
    monkeypatch.setattr(twin_service, "evaluate", evaluate)
    with pytest.raises(SkipJob, match="InfluxDB unreachable since 15 May 10:10 IST"):
        asyncio.run(twin_kind.evaluate(job, TwinEvaluateParams(backfill_hours=0, strata=False)))
