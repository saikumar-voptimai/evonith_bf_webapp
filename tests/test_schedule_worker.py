"""The schedule in the worker (SV-1.7, #287): the cron entry records the slot it serves and a slot whose previous job
is still active as skipped; a worker started after a slot runs the latest missed one within 30 min and records the
others since its last heartbeat as skipped; the heartbeat marks the time; execution time is summed over the tries.
Needs redis-server."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from services import schedule as sch
from services.jobs import CATALOG, JobKind, NoParams
from tests.redis_server import needs_redis, redis_server
from tests.test_jobs import connect, wait_for, worker_for
from worker import runtime
from worker.runtime import ALIVE_KEY, TransientJobError, scheduled

pytestmark = needs_redis


@pytest.fixture()
def redis_url(tmp_path):
    with redis_server(tmp_path) as url:
        yield url


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def test_the_cron_entry_records_its_slot_and_a_busy_slot_as_skipped(redis_url) -> None:
    kind = CATALOG["twin.evaluate"]
    slot = datetime(2026, 10, 27, 5, 40, tzinfo=UTC)  # 11:10 IST

    async def run():
        pool, store = await connect(redis_url)
        try:
            entry = scheduled(kind.name)
            await entry({"store": store, "score": _ms(slot + timedelta(seconds=0.123456))})
            (first,) = await store.list(10)
            assert first.submitted_by == "scheduler" and first.scheduled_for == slot and first.status == "queued"
            # The next slot comes while the first job is still queued (no worker here): recorded as skipped.
            await entry({"store": store, "score": _ms(slot + timedelta(hours=1, seconds=0.12))})
            records = await store.list(10)
            skipped = [r for r in records if r.submitted_by == sch.SKIPPED_BY]
            assert len(records) == 2 and len(skipped) == 1
            assert skipped[0].status == "cancelled" and skipped[0].scheduled_for == slot + timedelta(hours=1)
            assert first.id in skipped[0].error and "12:10 IST" in skipped[0].error
            audit = sch.audit(records, [kind], slot, slot + timedelta(hours=2))[kind.name]
            assert audit.counts["running"] == 1 and audit.counts["skipped"] == 1 and not audit.clean
            # The same slot again while its job is active (catch-up and cron entry both): one job, nothing skipped.
            await entry({"store": store, "score": _ms(slot + timedelta(seconds=0.2))})
            assert len(await store.list(10)) == 2
        finally:
            await pool.aclose()

    asyncio.run(run())


def _tick_kind(now: datetime) -> tuple[JobKind, datetime, datetime]:
    """A kind with two slots after the last heartbeat: 5 and 10 minutes before ``now`` (to the minute)."""
    latest = (now - timedelta(minutes=5)).replace(second=0, microsecond=0)
    older = (now - timedelta(minutes=10)).replace(second=0, microsecond=0)
    minutes = {latest.astimezone(sch.IST).minute, older.astimezone(sch.IST).minute}
    return JobKind("test.tick", "Tick", schedule=(None, minutes), timeout_s=10, params=NoParams), latest, older


def test_a_worker_started_after_a_slot_catches_it_up(redis_url, monkeypatch) -> None:
    now = datetime.now(UTC)
    kind, latest, older = _tick_kind(now)
    ran = []

    async def tick(job, params):
        ran.append(job.job_id)
        return {"ok": True}

    monkeypatch.setitem(CATALOG, kind.name, kind)
    monkeypatch.setitem(runtime.HANDLERS, kind.name, tick)
    monkeypatch.setenv("BF_WORKER_CATCH_UP", "1")
    monkeypatch.setenv(
        "BF_WORKER_SKIP_SCHEDULES", "forecast.,twin.,optimiser.,calibration.,learn.,report.,shift.,system."
    )
    monkeypatch.setattr(runtime, "CATCH_UP_DELAY_S", 0.2)

    async def run():
        pool, store = await connect(redis_url)
        try:
            await pool.set(ALIVE_KEY, (now - timedelta(minutes=12)).isoformat())  # the stopped worker's last beat
            worker = worker_for(redis_url, cron_jobs=[])
            task = asyncio.create_task(worker.main())
            caught = await wait_for(lambda: _by(store, sch.CATCH_UP_BY, "done"), timeout=20)
            skipped = await wait_for(lambda: _by(store, sch.SKIPPED_BY, "cancelled"), timeout=5)
            assert caught.kind == kind.name and caught.scheduled_for == latest and caught.result == {"ok": True}
            assert skipped.scheduled_for == older and "was not running" in skipped.error
            assert ran == [caught.id]
            audit = sch.audit(await store.list(100), [kind], older, now)[kind.name]
            assert audit.counts["done"] == 1 and audit.counts["skipped"] == 1
            await worker.close()
            task.cancel()
            alive = await runtime.read_alive(pool)  # written again at shutdown
            assert alive is not None and alive > now
        finally:
            await pool.aclose()

    asyncio.run(run())


async def _by(store, submitted_by: str, status: str):
    for record in await store.list(50):
        if record.submitted_by == submitted_by and record.status == status:
            return record
    return None


def test_the_catch_up_can_be_turned_off(redis_url, monkeypatch) -> None:
    now = datetime.now(UTC)
    kind, _, _ = _tick_kind(now)
    monkeypatch.setitem(CATALOG, kind.name, kind)
    monkeypatch.setenv("BF_WORKER_CATCH_UP", "0")
    monkeypatch.setenv(
        "BF_WORKER_SKIP_SCHEDULES", "forecast.,twin.,optimiser.,calibration.,learn.,report.,shift.,system."
    )

    async def run():
        pool, store = await connect(redis_url)
        try:
            await pool.set(ALIVE_KEY, (now - timedelta(minutes=12)).isoformat())
            worker = worker_for(redis_url, cron_jobs=[], burst=True)
            await worker.main()
            await worker.close()
            assert await store.list(10) == []
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_the_execution_time_is_summed_over_the_tries(redis_url, monkeypatch) -> None:
    async def flaky(job, params):
        await asyncio.sleep(0.3)
        if job.attempt == 1:
            raise TransientJobError("InfluxDB timed out")
        return {}

    monkeypatch.setitem(CATALOG, "test.flaky", JobKind("test.flaky", "Flaky", max_tries=2, retry_delay_s=1.0))
    monkeypatch.setitem(runtime.HANDLERS, "test.flaky", flaky)

    async def run():
        pool, store = await connect(redis_url)
        try:
            record, _ = await store.submit("test.flaky", {}, "adm")
            worker = worker_for(redis_url, burst=True)
            await worker.main()
            await worker.close()
            done = await store.get(record.id)
            elapsed = (done.finished_at - done.started_at).total_seconds()
            assert done.status == "done" and done.tries == 2
            assert 0.55 <= done.run_s <= 1.0 < elapsed  # two tries of 0.3 s; the 1 s wait between them is not counted
        finally:
            await pool.aclose()

    asyncio.run(run())
