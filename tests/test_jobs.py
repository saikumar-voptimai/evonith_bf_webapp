"""Job queue and scheduler (#277): records, de-duplication, execution, retries, time limits, cancellation, cron start
time, and survival of a worker kill. Needs redis-server; the worker runs in-process or as `python -m worker`."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from arq import cron
from arq.connections import RedisSettings, create_pool
from arq.jobs import Job
from arq.worker import create_worker
from pydantic import ValidationError

from services.events import RedisEventBus
from services.jobs import ARQ_FUNCTION, CATALOG, QUEUE_NAME, JobKind, JobStore, NoParams
from tests.redis_server import needs_redis, redis_server
from worker import runtime
from worker.runtime import TransientJobError, scheduled
from worker.settings import WorkerSettings

pytestmark = needs_redis
REPO = Path(__file__).resolve().parents[1]


async def connect(url: str):
    pool = await create_pool(RedisSettings.from_dsn(url), default_queue_name=QUEUE_NAME)

    async def enqueue(job_id: str) -> None:
        await pool.enqueue_job(ARQ_FUNCTION, job_id, _job_id=job_id, _queue_name=QUEUE_NAME)

    return pool, JobStore(pool, enqueue=enqueue)


def worker_for(url: str, **overrides):
    settings = {"redis_settings": RedisSettings.from_dsn(url), "poll_delay": 0.05, "handle_signals": False, **overrides}
    return create_worker(WorkerSettings, **settings)


async def wait_for(predicate, timeout: float, interval: float = 0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {timeout} s")


@pytest.fixture()
def redis_url(tmp_path):
    with redis_server(tmp_path) as url:
        yield url


# ---------------------------------------------------------------- records and de-duplication


def test_duplicate_request_returns_the_running_job(redis_url) -> None:
    async def run() -> None:
        pool, store = await connect(redis_url)
        try:
            a, dup_a = await store.submit("system.selftest", {"steps": 3}, "adm")
            b, dup_b = await store.submit("system.selftest", {"steps": 3}, "sup")
            c, dup_c = await store.submit("system.selftest", {"steps": 4}, "adm")
            assert (dup_a, dup_b, dup_c) == (False, True, False)
            assert b.id == a.id and c.id != a.id
            assert a.params == {"steps": 3, "step_s": 1.0, "cpu": True}  # defaults filled before hashing

            await store.finish(a.id, "done", result={})
            d, dup_d = await store.submit("system.selftest", {"steps": 3}, "adm")
            assert not dup_d and d.id != a.id  # a finished job does not block a new run

            listed = [r.id for r in await store.list(10)]
            assert listed[:3] == [d.id, c.id, a.id]
            with pytest.raises(ValidationError):
                await store.submit("system.selftest", {"steps": 0}, "adm")
            with pytest.raises(ValidationError):
                await store.submit("system.selftest", {"unknown": 1}, "adm")
        finally:
            await pool.aclose()

    asyncio.run(run())


# ---------------------------------------------------------------- execution


def test_selftest_runs_in_the_worker_and_its_pool(redis_url) -> None:
    async def run() -> None:
        pool, store = await connect(redis_url)
        bus = RedisEventBus(pool)
        try:
            record, _ = await store.submit("system.selftest", {"steps": 3, "step_s": 0}, "adm")
            worker = worker_for(redis_url, burst=True)
            await worker.main()
            await worker.close()
            done = await store.get(record.id)
            assert done.status == "done" and done.progress == 1.0 and done.tries == 1
            assert done.result["pool"]["pid"] != done.result["pool"]["worker_pid"]  # ran in a pool process
            events = [e for e in await bus.recent(50) if getattr(e.data, "job_id", None) == record.id]
            assert [e.type for e in events] == ["job.progress"] * 4 + ["job.done"]
            assert [e.data.progress for e in events[:4]] == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]
            assert events[-1].data.status == "done"
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_transient_failure_is_retried_and_time_limit_ends_a_job(redis_url, monkeypatch) -> None:
    calls = {"flaky": 0}

    async def flaky(job, params):
        calls["flaky"] += 1
        if job.attempt == 1:
            raise TransientJobError("InfluxDB timed out")
        return {"attempt": job.attempt}

    async def slow(job, params):
        await asyncio.sleep(5)
        return {}

    monkeypatch.setitem(CATALOG, "test.flaky", JobKind("test.flaky", "Flaky", max_tries=2, retry_delay_s=0.2))
    monkeypatch.setitem(CATALOG, "test.slow", JobKind("test.slow", "Slow", timeout_s=0.5, params=NoParams))
    monkeypatch.setitem(runtime.HANDLERS, "test.flaky", flaky)
    monkeypatch.setitem(runtime.HANDLERS, "test.slow", slow)

    async def run() -> None:
        pool, store = await connect(redis_url)
        try:
            flaky_job, _ = await store.submit("test.flaky", {}, "adm")
            slow_job, _ = await store.submit("test.slow", {}, "adm")
            worker = worker_for(redis_url, burst=True)
            await worker.main()
            await worker.close()
            f = await store.get(flaky_job.id)
            s = await store.get(slow_job.id)
            assert f.status == "done" and f.tries == 2 and f.result == {"attempt": 2} and calls["flaky"] == 2
            assert s.status == "failed" and s.error == "Stopped at the time limit of 0.5 s."
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_cancelling_a_running_job(redis_url) -> None:
    async def run() -> None:
        pool, store = await connect(redis_url)
        try:
            record, _ = await store.submit("system.selftest", {"steps": 20, "step_s": 0.5, "cpu": False}, "adm")
            worker = worker_for(redis_url)
            task = asyncio.create_task(worker.main())
            await wait_for(lambda: _status(store, record.id, "running"), timeout=10)
            await store.update(record.id, cancel_requested=True)
            try:
                await Job(record.id, pool, _queue_name=QUEUE_NAME).abort(timeout=5)
            except TimeoutError:
                pass
            done = await wait_for(lambda: _status(store, record.id, "cancelled"), timeout=10)
            assert done.error == "Cancelled while running."
            await worker.close()
            task.cancel()
        finally:
            await pool.aclose()

    asyncio.run(run())


async def _status(store: JobStore, job_id: str, status: str):
    record = await store.get(job_id)
    return record if record and record.status == status else None


# ---------------------------------------------------------------- schedule


def test_scheduled_job_starts_within_30_s_of_its_time(redis_url) -> None:
    async def run() -> None:
        pool, store = await connect(redis_url)
        try:
            due = datetime.now(UTC).replace(microsecond=0).timestamp() + 3
            second = int(due) % 60
            job = cron(scheduled("system.selftest"), second={second}, microsecond=0, unique=True)
            worker = worker_for(redis_url, cron_jobs=[job])
            task = asyncio.create_task(worker.main())
            started = await wait_for(lambda: _scheduled_started(store), timeout=40)
            delay = started.started_at.timestamp() - due
            assert started.submitted_by == "scheduler"
            assert 0 <= delay < 30, f"started {delay:.1f} s after its time"
            await worker.close()
            task.cancel()
        finally:
            await pool.aclose()

    asyncio.run(run())


async def _scheduled_started(store: JobStore):
    for record in await store.list(10):
        if record.submitted_by == "scheduler" and record.started_at is not None:
            return record
    return None


def test_cron_entries_follow_the_catalogue_in_ist() -> None:
    names = {c.name: c for c in WorkerSettings.cron_jobs}
    housekeeping = names["cron:schedule_system_housekeeping"]
    assert housekeeping.hour == {4} and housekeeping.minute == {0} and housekeeping.second == 0
    assert WorkerSettings.timezone.utcoffset(None).total_seconds() == 5.5 * 3600


# ---------------------------------------------------------------- worker kill


def test_queued_jobs_survive_a_worker_kill(redis_url, tmp_path) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "furnace_data")]),
        "BF_REDIS_URL": redis_url,
        "BF_WORKER_POOL": "0",
        "BF_WORKER_MAX_JOBS": "1",
        "BF_WORKER_JOB_TIMEOUT": "6",
    }

    log = tmp_path / "worker.log"

    def start():
        with open(log, "a", encoding="utf-8") as out:
            return subprocess.Popen([sys.executable, "-m", "worker"], cwd=REPO, env=env, stdout=out, stderr=out)

    async def run() -> None:
        pool, store = await connect(redis_url)
        first = start()
        second = None
        try:
            running, _ = await store.submit("system.selftest", {"steps": 4, "step_s": 1, "cpu": False}, "adm")
            queued, _ = await store.submit("system.selftest", {"steps": 1, "step_s": 0, "cpu": False}, "adm")
            await wait_for(lambda: _status(store, running.id, "running"), timeout=20)
            first.send_signal(signal.SIGKILL)
            first.wait(timeout=5)
            assert (await store.get(queued.id)).status == "queued"

            second = start()
            await wait_for(lambda: _status(store, queued.id, "done"), timeout=20)
            # The interrupted job runs again once its in-progress mark expires (job timeout 6 s + 10 s).
            again = await wait_for(lambda: _status(store, running.id, "done"), timeout=40, interval=0.5)
            assert again.tries == 2
        finally:
            for proc in (first, second):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
            await pool.aclose()

    asyncio.run(run())
