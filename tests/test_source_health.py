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


def test_a_failed_refresh_of_the_hourly_dataset_skips_the_job_without_retries(store, monkeypatch) -> None:  # noqa: F811
    """``refresh_static_dataset`` returns its error instead of raising: the twin job used to retry to HH:25 and fail."""
    end = pd.Timestamp("2026-05-15 10:00")
    real = twin_service.ensure_dataset

    def ensure(window_end: Any) -> Any:
        return real(
            window_end,
            cfg={},
            now=end,
            refresh_fn=lambda cfg: {"refreshed": False, "error": "HTTPError: 503 Service Unavailable"},
            state_fn=lambda cfg: twin_service.DatasetState(end - 2 * twin_service.HOUR, end - twin_service.HOUR),
        )

    monkeypatch.setattr(twin_service, "ensure_dataset", ensure)
    job, bus = context("twin.evaluate", record("twin.evaluate", slot=utc_of("2026-05-15 10:10")))
    with pytest.raises(SkipJob, match=r"Hourly dataset unreachable since .* \(refresh failed: HTTPError: 503") as e:
        asyncio.run(twin_kind.evaluate(job, TwinEvaluateParams(backfill_hours=0, strata=False)))
    assert e.value.source == "hourly" and source_health.HEALTH.down("hourly")
    # a refresh that worked marks the source up with the local copy's last row
    source_health.record_dataset_refresh({"refreshed": True}, pd.Timestamp("2026-05-15 09:00"))
    assert not source_health.HEALTH.failed("hourly")
    assert source_health.HEALTH.state("hourly").data_end == utc_of("2026-05-15 09:00")
    source_health.record_dataset_refresh({"refreshed": False})  # a late build without an error records nothing
    assert not source_health.HEALTH.failed("hourly")


# ------------------------------------------------------------------------------------- between the processes
def test_the_worker_registry_reaches_the_api_through_redis() -> None:
    from tests.test_services_live_status import FakeRedis

    worker, _ = registry()
    api, _ = registry()
    redis = FakeRedis()
    worker.record_ok("hourly", data_end=pd.Timestamp("2026-10-27 09:00"), at=T0)
    worker.record_failure("hourly", OSError("dataset host unreachable"), at=T0 + 10 * M)
    worker.record_ok("dpr", data_end=pd.Timestamp("2026-10-26"), at=T0)
    api.record_ok("influxdb", data_end=T0 + 9 * M, at=T0 + 10 * M)  # the API's own status snapshot
    asyncio.run(source_health.publish(redis, health=worker))
    assert asyncio.run(source_health.follow(redis, health=api))
    hourly = api.state("hourly")
    assert hourly.failed and hourly.failures == 1 and hourly.last_error == "OSError: dataset host unreachable"
    assert hourly.data_end == utc_of("2026-10-27 09:00") and api.down("hourly", T0 + 11 * M)
    assert api.state("dpr").data_end == utc_of("2026-10-26 00:00") and not api.failed("dpr")
    influx = api.state("influxdb")  # the worker knew nothing of InfluxDB: the API's own state stands
    assert not influx.failed and influx.data_end == T0 + 9 * M and influx.last_ok_at == T0 + 10 * M

    api.record_ok("hourly", at=T0 + 20 * M)  # a newer success of this process is not undone by the older failure
    asyncio.run(source_health.follow(redis, health=api))
    assert not api.failed("hourly")
    worker.record_failure("hourly", "timeout", at=T0 + 30 * M)  # the worker's newer failure is adopted
    asyncio.run(source_health.publish(redis, health=worker))
    asyncio.run(source_health.follow(redis, health=api))
    assert api.failed("hourly") and api.state("hourly").last_error == "timeout" and api.state("hourly").failures == 2

    assert not asyncio.run(source_health.follow(FakeRedis(), health=api))  # no worker
    broken = FakeRedis()
    asyncio.run(broken.set(source_health.REDIS_KEY, "not json"))
    assert not asyncio.run(source_health.follow(broken, health=api))


def test_the_route_shows_a_source_the_worker_found_down(client, twin) -> None:  # noqa: F811 - fixtures
    from tests.test_services_live_status import FakeRedis

    worker = SourceHealth()
    worker.record_failure("dpr", "Neon: connection timed out")
    live = client.app.state.live
    redis, live.redis = live.redis, FakeRedis()
    try:
        asyncio.run(source_health.publish(live.redis, health=worker))
        rows = {s["key"]: s for s in get(client, "op", "/api/twin/now").json()["sources"]}
    finally:
        live.redis = redis
    assert rows["dpr"]["down"] and rows["dpr"]["stale"] and rows["dpr"]["error"] == "Neon: connection timed out"
    assert not rows["hourly"]["down"]


# ------------------------------------------------------------------------------- the skip in the worker runtime
class _Jobs:
    """The parts of ``JobStore`` ``run_job`` uses."""

    def __init__(self, rec: Any) -> None:
        self.rec, self.finished = rec, []

    async def get(self, job_id: str) -> Any:
        return self.rec

    async def update(self, job_id: str, **changes: Any) -> None:
        return None

    async def finish(self, job_id: str, status: str, **kw: Any) -> None:
        self.finished.append((status, kw.get("error")))


def test_a_skip_ends_cancelled_and_raises_the_source_stale_alarm(store, monkeypatch) -> None:  # noqa: F811
    from services.events import MemoryEventBus
    from worker import runtime

    async def handler(job: Any, params: Any) -> Any:
        raise SkipJob("InfluxDB unreachable since 15 May 10:10 IST (timeout)", source="influxdb")

    monkeypatch.setitem(runtime.HANDLERS, "forecast.si", handler)
    rec = record("forecast.si", slot=utc_of("2026-05-15 10:15")).model_copy(update={"status": "queued"})
    jobs, bus = _Jobs(rec), MemoryEventBus()
    ctx = {"store": jobs, "bus": bus, "running": set(), "job_try": 1}
    asyncio.run(runtime.run_job(ctx, rec.id))
    assert jobs.finished == [("cancelled", "Skipped: InfluxDB unreachable since 15 May 10:10 IST (timeout)")]
    events = asyncio.run(bus.recent(10))
    assert [e.type for e in events] == ["job.progress", "job.done", "alarm.raised"]
    assert events[1].data.status == "cancelled"
    (alarm,) = store.alarms.open_alarms()  # no job_failed: a skip leaves that alarm as it is
    assert (alarm.kind, alarm.subject) == ("source_stale", "influxdb") and "unreachable" in alarm.message
    asyncio.run(runtime.run_job(ctx, rec.id))  # the next skip updates the open alarm
    raised = [e.type for e in asyncio.run(bus.recent(10))].count("alarm.raised")
    assert len(store.alarms.open_alarms()) == 1 and raised == 1


# ------------------------------------------------------------------------------------------ the forecast jobs
def test_the_si_reads_record_influxdb_and_hm_slag(monkeypatch) -> None:
    from data.bmo import si_forecast_context
    from services import forecasts

    def down(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("influx.local:8086 refused")

    monkeypatch.setattr(si_forecast_context, "fetch_online_df", down)
    with pytest.raises(SourceDown, match="InfluxDB unreachable since"):
        forecasts._observed_online(selected_measurements=["process_params"])
    assert source_health.HEALTH.down("influxdb")
    reads: list[str] = []
    monkeypatch.setattr(
        si_forecast_context, "fetch_offline_data", lambda **kw: reads.append(kw["table_name"]) or pd.DataFrame()
    )
    forecasts._observed_offline(table_name=forecasts.HM_SLAG_TABLE, query_type="raw")
    forecasts._observed_offline(table_name="offline_feed.charge_data", query_type="raw")
    assert reads == [forecasts.HM_SLAG_TABLE, "offline_feed.charge_data"]
    assert source_health.HEALTH.state("hm_slag").last_ok_at is not None  # the charge table is no tracked source
    source = forecasts.live_si_source(bundle_dir=forecasts.si_paths().default_bundle, static_dataset_path="x.csv")
    assert source.online_fetch is forecasts._observed_online and source.offline_fetch is forecasts._observed_offline


def test_the_forecast_jobs_skip_with_an_attempt_while_their_source_is_down(tmp_path, monkeypatch) -> None:
    from services import forecasts
    from services.jobs import CATALOG, NoParams
    from tests.test_forecast_jobs import SI_BUNDLE, _cfg
    from worker.kinds import forecasts as forecast_kinds
    from worker.runtime import JobContext

    def job(kind: str, attempt: int = 1) -> JobContext:
        return JobContext(job_id="j1", kind=CATALOG[kind], attempt=attempt, store=None, bus=None, pool=None)

    skips: list[tuple[str, str]] = []
    monkeypatch.setattr(forecasts, "record_skip", lambda forecast, reason: skips.append((forecast, reason)))
    monkeypatch.setattr(forecasts, "issue_si", lambda: pytest.fail("a down source is not probed"))
    source_health.HEALTH.record_failure("influxdb", "no reading; measurements not read: process_params")
    with pytest.raises(SkipJob, match="InfluxDB unreachable since") as e:
        asyncio.run(forecast_kinds.si(job("forecast.si"), NoParams()))
    assert e.value.source == "influxdb" and skips[0][0] == "si" and skips[0][1].startswith("InfluxDB unreachable")

    def failed_read() -> Any:  # the read itself fails (recorded by the observed fetch, attempt logged as failed)
        raise SourceDown("hm_slag", "HM/slag unreachable since 15 May 10:10 IST (timeout)")

    source_health.HEALTH.reset()
    monkeypatch.setattr(forecasts, "issue_si", failed_read)
    with pytest.raises(SkipJob, match="HM/slag unreachable") as e:
        asyncio.run(forecast_kinds.si(job("forecast.si"), NoParams()))
    assert e.value.source == "hm_slag" and len(skips) == 1

    # charged coke: the refresh of the hourly dataset failed while the build is late: no retries
    attempt = {"forecast": "charged_coke", "status": "cruise", "written": False, "behind": True}

    def issue_charged() -> dict[str, Any]:
        source_health.record_dataset_refresh({"refreshed": False, "error": "HTTPError: 503"})
        return {"attempt": attempt, "record": {"data_row": None}}

    monkeypatch.setattr(forecasts, "issue_charged", issue_charged)
    with pytest.raises(SkipJob, match=r"Hourly dataset unreachable since .*refresh failed: HTTPError: 503") as e:
        asyncio.run(forecast_kinds.charged_coke(job("forecast.charged_coke"), NoParams()))
    assert e.value.source == "hourly"
    with pytest.raises(SkipJob, match="next try after"):  # the next run within the hold does not probe
        asyncio.run(forecast_kinds.charged_coke(job("forecast.charged_coke", attempt=2), NoParams()))
    assert skips[-1][0] == "charged_coke"

    # the skipped attempt is in the attempt log: the gap check names it and page 9 shows the reason
    monkeypatch.undo()
    cfg = _cfg(tmp_path)
    origin = pd.Timestamp("2026-10-20T04:30:00Z")
    out = forecasts.record_skip("si", "InfluxDB unreachable since 20 Oct 09:55 IST (timeout)", now=origin, cfg=cfg)
    assert out["status"] == "skipped" and out["key"] == origin and not out["written"]
    view = forecasts.si_for_page(tmp_path / "si", SI_BUNDLE, now=origin + pd.Timedelta(minutes=1))
    assert view.record["status"] == "insufficient_data" and "InfluxDB unreachable" in view.record["reasons"][0]
    window = dict(start=origin - pd.Timedelta(hours=1), end=origin + pd.Timedelta(hours=1))
    gaps = forecasts.gaps("si", tmp_path / "si", limit=pd.Timedelta(minutes=10), **window)
    assert gaps and gaps[0].statuses == {"skipped": 1} and "InfluxDB unreachable" in gaps[0].errors[0]
    forecasts.record_skip("charged_coke", "Hourly dataset unreachable", now=origin, cfg=cfg)
    assert forecasts.AttemptLog(tmp_path / "cc").latest()["status"] == "skipped"
    legacy = _cfg(tmp_path, si_forecast={**cfg["si_forecast"], "mode": "legacy"})
    assert forecasts.record_skip("si", "x", now=origin, cfg=legacy) is None
