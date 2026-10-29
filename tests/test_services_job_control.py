"""Jobs and system health (SV-4.8, #311), service layer: audited cancel and retry (``services.job_audit``), the job
history (``services.job_history``), the sources and services (``services.system_health``) and the host reading
(``services.host_health``: tegrastats, psutil). SQLite twin store, in-memory Redis."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore
from services import host_health, job_audit, job_history, schedule, system_health
from services.jobs import JobStore
from tests.test_api_bmo import IndexedRedis

T0 = datetime(2026, 10, 29, 4, 0, tzinfo=UTC)
ADMIN = Actor("adm", "admin", "127.0.0.1")


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    yield store
    store.dispose()


def clock(start: datetime = T0):
    times = iter(start + timedelta(minutes=i) for i in range(10_000))
    return lambda: next(times)


def entries(twin: TwinStore) -> list[Any]:
    rows, _ = twin.audit.page(limit=50, offset=0)
    return list(reversed(rows))  # oldest first


# --------------------------------------------------------------------------------------------- audited control
def test_a_cancel_writes_its_entry_and_ends_a_queued_job(twin) -> None:
    jobs = JobStore(IndexedRedis(), clock=clock())
    record, _ = asyncio.run(jobs.submit("system.selftest", {"steps": 2}, "op"))

    out = asyncio.run(job_audit.cancel(jobs, twin, ADMIN, record.id, origin="admin jobs"))

    assert out.ended and out.record.status == "cancelled" and out.record.cancel_requested
    assert out.record.error == job_audit.CANCELLED_BEFORE_START
    (entry,) = entries(twin)
    assert entry.id == out.audit_id and (entry.action, entry.object_type, entry.object_id) == (
        "job.cancel",
        "job",
        record.id,
    )
    assert entry.details == {"kind": "system.selftest", "title": entry.details["title"], "status": "queued",
                             "origin": "admin jobs"}  # fmt: skip
    assert entry.before["status"] == "queued" and entry.before["cancel_requested"] is False
    assert entry.after["status"] == "cancelled" and entry.after["cancel_requested"] is True
    with pytest.raises(job_audit.JobConflict):  # ended
        asyncio.run(job_audit.cancel(jobs, twin, ADMIN, record.id, origin="admin jobs"))
    with pytest.raises(job_audit.JobNotFound):
        asyncio.run(job_audit.cancel(jobs, twin, ADMIN, "0" * 32, origin="admin jobs"))
    assert len(entries(twin)) == 1


def test_a_running_job_is_only_marked_and_a_failed_entry_leaves_it_untouched(twin) -> None:
    jobs = JobStore(IndexedRedis(), clock=clock())
    record, _ = asyncio.run(jobs.submit("system.selftest", {}, "op"))
    asyncio.run(jobs.update(record.id, status="running", started_at=T0))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("database gone")

    original = twin.audit.record
    twin.audit.record = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        asyncio.run(job_audit.cancel(jobs, twin, ADMIN, record.id, origin="jobs"))
    assert asyncio.run(jobs.get(record.id)).cancel_requested is False  # no change without its entry
    twin.audit.record = original  # type: ignore[method-assign]

    out = asyncio.run(job_audit.cancel(jobs, twin, ADMIN, record.id, origin="jobs"))
    assert not out.ended and out.record.status == "running" and out.record.cancel_requested
    (entry,) = entries(twin)
    assert entry.after["status"] == "running" and entry.after["cancel_requested"] is True


def test_a_retry_queues_the_same_parameters_and_records_what_it_retried(twin) -> None:
    jobs = JobStore(IndexedRedis(), clock=clock())
    record, _ = asyncio.run(jobs.submit("system.selftest", {"steps": 3}, "op"))
    with pytest.raises(job_audit.JobConflict):  # queued: nothing to retry
        asyncio.run(job_audit.retry(jobs, twin, ADMIN, record.id, origin="admin jobs"))
    asyncio.run(jobs.finish(record.id, "failed", error="boom"))

    out = asyncio.run(job_audit.retry(jobs, twin, ADMIN, record.id, origin="admin jobs"))

    assert not out.duplicate and out.record.id != record.id and out.record.params == record.params
    assert out.record.submitted_by == "adm" and out.record.status == "queued"
    (entry,) = entries(twin)
    assert (entry.action, entry.object_id) == ("job.retry", out.record.id)
    assert entry.details["retry_of"] == record.id and entry.details["retry_of_status"] == "failed"
    assert entry.before is None and entry.after["id"] == out.record.id
    # a supervisor may not start an admin kind by retrying it
    with pytest.raises(job_audit.RoleRefused):
        asyncio.run(job_audit.retry(jobs, twin, Actor("sup", "supervisor"), record.id, origin="admin jobs"))


def test_a_retry_reads_an_archived_job_when_redis_no_longer_holds_it(twin) -> None:
    jobs = JobStore(IndexedRedis(), clock=clock())
    twin.jobs.upsert(
        dict(
            id="a" * 32,
            kind="system.selftest",
            status="cancelled",
            params={"steps": 4, "step_s": 1.0, "cpu": True},
            input_hash="h",
            submitted_by="scheduler (skipped)",
            submitted_at=T0 - timedelta(days=20),
            scheduled_for=T0 - timedelta(days=20),
        )  # fmt: skip
    )
    out = asyncio.run(job_audit.retry(jobs, twin, ADMIN, "a" * 32, origin="admin jobs"))
    assert out.record.params["steps"] == 4 and out.record.scheduled_for is None
    (entry,) = entries(twin)
    assert entry.details["retry_of_scheduled_for"] == (T0 - timedelta(days=20)).isoformat()


# ---------------------------------------------------------------------------------------------------- history
def test_the_history_merges_the_archive_with_redis_and_gives_the_durations(twin) -> None:
    jobs = JobStore(IndexedRedis(), clock=clock())
    done, _ = asyncio.run(jobs.submit("system.selftest", {"steps": 1}, "op"))
    asyncio.run(jobs.update(done.id, status="running", started_at=T0 + timedelta(seconds=30)))
    asyncio.run(jobs.finish(done.id, "done", run_s=12.5))
    running, _ = asyncio.run(jobs.submit("system.selftest", {"steps": 2}, "op"))
    asyncio.run(jobs.update(running.id, status="running", started_at=T0 + timedelta(minutes=10), progress=0.5))
    # the archive holds an old failed job and the running job in its queued state
    old = dict(id="b" * 32, kind="twin.evaluate", status="failed", params={}, input_hash="h",
               submitted_by=schedule.SCHEDULER, submitted_at=T0 - timedelta(days=3),
               started_at=T0 - timedelta(days=3) + timedelta(minutes=1), finished_at=T0 - timedelta(days=3, minutes=-3),
               scheduled_for=T0 - timedelta(days=3), error="SourceDown: HM/slag", tries=2)  # fmt: skip
    twin.jobs.upsert_many([old, running.model_dump(include=set(_ARCHIVED))])
    now = T0 + timedelta(minutes=20)

    rows, sources = asyncio.run(
        job_history.read(jobs, twin, since=T0 - timedelta(days=7), until=now, limit=50, now=now)
    )

    assert sources == ["twin.jobs", "redis"]
    assert [r["id"] for r in rows] == [running.id, done.id, "b" * 32]  # newest first
    live = rows[0]
    assert live["live"] and live["status"] == "running" and live["progress"] == 0.5 and live["can_cancel"]
    assert live["elapsed_s"] == pytest.approx(600.0) and live["run_s"] is None
    assert rows[1]["run_s"] == 12.5 and rows[1]["status"] == "done" and not rows[1]["can_retry"]
    failed = rows[2]
    assert not failed["live"] and failed["scheduled"] and failed["can_retry"] and not failed["can_cancel"]
    assert failed["wait_s"] == pytest.approx(60.0) and failed["error"].startswith("SourceDown") and failed["tries"] == 2

    only_failed, _ = asyncio.run(
        job_history.read(jobs, twin, since=T0 - timedelta(days=7), until=now, status="failed", now=now)
    )
    assert [r["id"] for r in only_failed] == ["b" * 32]
    kind_rows, _ = asyncio.run(
        job_history.read(jobs, None, since=T0 - timedelta(days=7), until=now, kind="twin.evaluate", now=now)
    )
    assert kind_rows == []  # Redis alone does not hold the archived job
    with pytest.raises(job_history.Unavailable):
        asyncio.run(job_history.read(None, None, since=T0, until=now))


_ARCHIVED = ("id", "kind", "status", "params", "input_hash", "submitted_by", "submitted_at")


def test_a_queued_job_waits_from_its_submission() -> None:
    record = SimpleNamespace(id="c", kind="system.selftest", status="queued", submitted_by="op", submitted_at=T0,
                             started_at=None, finished_at=None, scheduled_for=None, params={})  # fmt: skip
    row = job_history.history_row(record, now=T0 + timedelta(seconds=90), live=True)
    assert row["wait_s"] == 90.0 and row["can_cancel"] and row["elapsed_s"] is None and row["run_s"] is None
    skipped = SimpleNamespace(**{**vars(record), "status": "cancelled", "submitted_by": schedule.SKIPPED_BY})
    assert job_history.history_row(skipped, now=T0, live=False)["skipped"]


# ---------------------------------------------------------------------------------------------------- sources
NOW = datetime(2026, 10, 29, 6, 30, tzinfo=UTC)


def latest_evaluation(**sources: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        window_end=datetime(2026, 10, 29, 6, 0, tzinfo=UTC),
        evaluated_at=datetime(2026, 10, 29, 6, 7, tzinfo=UTC),
        flags={"sources": {k: v.isoformat() for k, v in sources.items()}, "provisional_dpr": True},
    )


def test_every_source_carries_its_last_timestamp_and_age() -> None:
    """Acceptance: every source shows last timestamp and age."""
    ends = {
        "influxdb": NOW - timedelta(minutes=5),
        "hourly": NOW - timedelta(hours=1, minutes=30),
        "hm_slag": NOW - timedelta(hours=9),
        "dpr": NOW - timedelta(days=1),
        "open_meteo": NOW - timedelta(hours=6),
    }
    health = [{"key": "influxdb", "failed": True, "last_error": "timeout", "last_failure_at": NOW,
               "data_end": None, "last_ok_at": None}]  # fmt: skip
    rows = system_health.source_rows(latest_evaluation(**ends), now=NOW, cfg={}, health=health)

    assert [r["key"] for r in rows] == ["influxdb", "hourly", "hm_slag", "dpr", "open_meteo", "evaluation"]
    for row in rows:
        assert row["last_at"] is not None and row["age_s"] == (NOW - row["last_at"]).total_seconds(), row["key"]
    by = {r["key"]: r for r in rows}
    assert by["influxdb"]["state"] == "down" and by["influxdb"]["error"] == "timeout"
    assert by["hourly"]["state"] == "ok" and by["hourly"]["stale_after_s"] == 2.5 * 3600
    assert by["hm_slag"]["state"] == "stale"  # 9 h > 6 h
    assert by["dpr"]["note"].startswith("DPR date carried forward")
    assert by["evaluation"]["last_at"] == datetime(2026, 10, 29, 6, 0, tzinfo=UTC)
    assert by["evaluation"]["age_s"] == 1800.0 and by["evaluation"]["group"] == "twin"


def test_a_source_without_a_time_is_missing_and_the_evaluation_off_without_a_store() -> None:
    rows = system_health.source_rows(None, now=NOW, cfg={}, health=[], store_configured=False)
    assert all(r["last_at"] is None and r["age_s"] is None for r in rows)
    assert [r["state"] for r in rows] == ["missing"] * 5 + ["off"]


class HeartbeatRedis(IndexedRedis):
    async def ping(self) -> bool:
        return True

    async def zcard(self, key: str) -> int:
        return 3


def test_the_worker_row_reads_the_heartbeat_and_the_redis_row_the_queue() -> None:
    from worker.runtime import ALIVE_KEY

    assert system_health.ALIVE_KEY == ALIVE_KEY
    redis = HeartbeatRedis()
    asyncio.run(redis.set(ALIVE_KEY, (NOW - timedelta(seconds=12)).isoformat()))
    row = asyncio.run(system_health.worker_row(redis, now=NOW))
    assert row["state"] == "ok" and row["age_s"] == 12.0 and row["last_at"] == NOW - timedelta(seconds=12)
    late = asyncio.run(system_health.worker_row(redis, now=NOW + timedelta(minutes=5)))
    assert late["state"] == "down"
    assert asyncio.run(system_health.worker_row(IndexedRedis(), now=NOW))["state"] == "missing"
    queue = asyncio.run(system_health.redis_row(redis, configured=True))
    assert queue["state"] == "ok" and queue["detail"].startswith("3 jobs") and queue["latency_ms"] >= 0
    assert asyncio.run(system_health.redis_row(None, configured=False))["state"] == "off"
    assert asyncio.run(system_health.redis_row(None, configured=True))["state"] == "down"


def test_the_store_row_names_the_schema_revision(twin) -> None:
    assert system_health.store_row(None)["state"] == "off"
    assert system_health.store_row(twin)["state"] == "down"  # created without alembic
    from sqlalchemy import text

    with twin.sessions.begin() as s:
        s.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        s.execute(text("INSERT INTO alembic_version VALUES ('twin_0099')"))
    row = system_health.store_row(twin)
    assert row["state"] == "ok" and "twin_0099" in row["detail"]


# ------------------------------------------------------------------------------------------------------- host
ORIN = (
    "10-29-2026 12:00:01 RAM 5182/62841MB (lfb 2x4MB) SWAP 12/31420MB (cached 0MB) "
    "CPU [2%@729,10%@729,off,off,0%@2201,0%@2201] EMC_FREQ 0%@3199 GR3D_FREQ 0%@[305,305] "
    "cpu@46.875C soc2@43.218C gpu@42.656C tj@96.5C PMIC@100C VDD_GPU_SOC 2795mW/2795mW"
)


def test_tegrastats_gives_ram_cores_and_temperatures() -> None:
    parsed = host_health.parse_tegrastats(ORIN)
    assert (parsed["ram_used_mb"], parsed["ram_total_mb"]) == (5182.0, 62841.0)
    assert (parsed["swap_used_mb"], parsed["swap_total_mb"]) == (12.0, 31420.0)
    assert parsed["cores_pct"] == [2.0, 10.0, None, None, 0.0, 0.0]
    assert parsed["cpu_pct"] == pytest.approx(3.0)  # mean of the online cores
    assert [t["sensor"] for t in parsed["temperatures"]] == ["cpu", "soc2", "gpu", "tj"]  # PMIC placeholder dropped


def test_the_host_reading_uses_tegrastats_on_the_jetson_and_psutil_elsewhere() -> None:
    disk = [{"path": "/", "used_gb": 90.0, "total_gb": 100.0, "pct": 90.0, "state": "warning"}]
    jetson = host_health.collect(tegrastats=lambda: ORIN, disk_rows=lambda: disk, now=NOW)
    assert jetson["source"] == "tegrastats" and jetson["ram_pct"] == pytest.approx(100 * 5182 / 62841)
    tj = next(t for t in jetson["temperatures"] if t["sensor"] == "tj")
    assert tj["state"] == "critical" and jetson["state"] == "critical"

    figures = {"cpu_pct": 12.0, "ram_used_mb": 1000.0, "ram_total_mb": 4000.0, "swap_used_mb": 0.0,
               "swap_total_mb": 0.0, "temperatures": []}  # fmt: skip
    other = host_health.collect(tegrastats=lambda: None, figures=lambda: figures, disk_rows=lambda: [], now=NOW)
    assert other["source"] == "psutil" and other["ram_pct"] == 25.0 and other["state"] == "ok"
    assert other["temperatures"] == [] and "No temperature sensor" in other["notes"][0]

    real = host_health.collect(tegrastats=lambda: None)  # this machine, through psutil
    assert real["source"] == "psutil" and real["ram_total_mb"] > 0 and real["disks"]
    assert 0 <= real["cpu_pct"] <= 100 and real["disks"][0]["total_gb"] > 0


def test_limit_states() -> None:
    assert [host_health.state_of(v, "temp_c") for v in (None, 79.9, 80.0, 95.0)] == [None, "ok", "warning", "critical"]
    assert host_health.worst(["ok", None, "warning"]) == "warning" and host_health.worst([]) == "ok"
