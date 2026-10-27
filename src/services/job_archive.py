"""Archive of job records in ``twin.jobs`` (SV-0.9, #278).

The worker's ``JobStore`` hands every change of state of a job (queued, running, done, failed, cancelled; start and
finish times; tries) to ``TwinJobArchive.record``, which only queues it. One task writes the queue in order, in a
thread (the store is synchronous), coalescing changes of the same job; so a slow or unreachable database never delays
a job. Writes that fail are counted and logged at most once a minute; the nightly housekeeping job copies the records
of the last 7 days again (``sync``), which repairs any gap shorter than the Redis retention.

Redis stays the live store of job records; ``twin.jobs`` is their history (admin space, SV-4.8).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from data.twin.repositories import TwinStore
from services.jobs import JobRecord

log = logging.getLogger("worker.archive")

BATCH = 100
WARN_INTERVAL_S = 60.0


class TwinJobArchive:
    """Ordered, non-blocking writer of job records into ``twin.jobs``."""

    def __init__(self, store: TwinStore, *, max_queue: int = 2000) -> None:
        self.store = store
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task[None] | None = None
        self._warned_at = 0.0
        self.written = 0
        self.failed = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="twin-job-archive")

    def record(self, record: JobRecord) -> None:
        try:
            self._queue.put_nowait(record.model_dump())
        except asyncio.QueueFull:
            self.failed += 1
            self._warn("job archive queue full; record %s not archived until the next housekeeping", record.id)

    async def sync(self, records: list[JobRecord]) -> int:
        return await asyncio.to_thread(self.store.jobs.upsert_many, [r.model_dump() for r in records])

    async def flush(self, timeout: float = 10.0) -> None:
        """Wait until every queued record has been written or has failed."""
        await asyncio.wait_for(self._queue.join(), timeout)

    async def close(self, timeout: float = 10.0) -> None:
        """Write what is queued (at most ``timeout`` s), stop the writer and release the connections."""
        if self._task is not None:
            with contextlib.suppress(TimeoutError):
                await self.flush(timeout)
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await asyncio.to_thread(self.store.dispose)

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            while len(batch) < BATCH and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            latest: dict[str, dict[str, Any]] = {}
            for item in batch:  # the last state of each job in the batch
                latest[item["id"]] = item
            try:
                await asyncio.to_thread(self.store.jobs.upsert_many, list(latest.values()))
                self.written += len(latest)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failed += len(latest)
                self._warn("job archive write failed (%d records): %s", len(latest), exc)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _warn(self, message: str, *args: object) -> None:
        now = time.monotonic()
        if now - self._warned_at >= WARN_INTERVAL_S:
            self._warned_at = now
            log.warning(message, *args)


async def open_job_archive(application_name: str = "twin-worker") -> TwinJobArchive | None:
    """The archive on ``BF_TWIN_DB_URL`` / ``DATABASE_URL``, started; None when no URL is set. The schema state is
    logged once; a database that is down or not yet migrated only makes the writes fail until it is fixed."""
    store = TwinStore.from_url(application_name=application_name)
    if store is None:
        log.info("job archive off: neither BF_TWIN_DB_URL nor DATABASE_URL is set")
        return None
    try:
        revision = await asyncio.to_thread(store.revision)
    except Exception as exc:
        log.warning("job archive: database not reachable now (%s); records are written once it is", exc)
    else:
        if revision is None:
            log.warning("job archive: the twin schema is not migrated; run `alembic -n twin upgrade head`")
        else:
            log.info("job archive on: twin.jobs, schema revision %s", revision)
    archive = TwinJobArchive(store)
    archive.start()
    return archive
