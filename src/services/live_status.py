"""Live furnace status in Redis (SV-2.1, #293): one InfluxDB snapshot query per 10 s, whatever the number of clients.

* Every API process records how many clients hold its event stream (``twin:live:subscribers:<process>``, TTL
  ``SUBSCRIBERS_TTL_S``; written every ``SUBSCRIBERS_REPORT_S`` and on every change, deleted at shutdown).
* The worker's heartbeat (``worker.runtime.heartbeat_loop``, every 10 s) is the single producer: while the total over
  the processes is above zero, ``LiveStatusProducer.tick`` takes one snapshot (``services.status.build_snapshot_view``),
  writes it to ``twin:live:status`` and returns the plant part of the ``live.status`` event. A lock key
  (``twin:live:producer``, held by the worker name) keeps a second worker from querying as well.
* The API answers ``GET /api/status/snapshot`` from ``twin:live:status`` while it is at most ``LATEST_FRESH_S`` old;
  otherwise (no worker, no client, Redis down) it takes the snapshot itself, once per 10-s slot and process
  (``services.status.current_snapshot_view``).

A beat waits ``SNAPSHOT_WAIT_S`` for its snapshot; one that finishes later is published by the next beat, which then
starts the next query (one query at a time, at most one per beat). A query running longer than ``SNAPSHOT_ABANDON_S``
is given up and a new one started while fewer than ``MAX_ABANDONED`` given-up queries still run. The hot-metal tags
(``hmt``, ``hm_si``) come from Neon at most every ``HOT_METAL_REFRESH_S`` in a task of their own, so a slow Neon does
not hold up the beat. The blocking calls run in daemon threads of their own: a hung query takes no thread of the
worker's default executor (the job kinds use it) and does not hold up the worker's exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from services import source_health
from services.events import TagValue
from services.status import StatusSnapshotView, build_snapshot_view, latest_hot_metal, live_status_fields

log = logging.getLogger(__name__)

SUBSCRIBERS_PREFIX = "twin:live:subscribers:"
LATEST_KEY = "twin:live:status"
PRODUCER_KEY = "twin:live:producer"

PERIOD_S = 10.0  # the worker's heartbeat
SUBSCRIBERS_REPORT_S = 5.0
SUBSCRIBERS_TTL_S = 20
LATEST_TTL_S = 120
LATEST_FRESH_S = 25.0  # two beats and a margin
PRODUCER_LOCK_S = 25
SNAPSHOT_WAIT_S = 8.0  # longest wait of a beat for its snapshot; a slower one is published by the next beat
SNAPSHOT_ABANDON_S = 60.0  # a snapshot query running longer is given up (its thread ends on its own)
MAX_ABANDONED = 2  # given-up queries still running; at the bound the running query is kept (threads stay bounded)
HOT_METAL_REFRESH_S = 120.0
HOT_METAL_WAIT_S = 2.0


def _text(value: Any) -> str | None:
    return value.decode() if isinstance(value, bytes) else value


# ---------------------------------------------------------------- subscriber counts (API processes)


async def record_subscribers(redis: Any, process_id: str, count: int) -> None:
    """The number of event-stream clients of one API process, kept ``SUBSCRIBERS_TTL_S``."""
    await redis.set(SUBSCRIBERS_PREFIX + process_id, str(max(0, int(count))), ex=SUBSCRIBERS_TTL_S)


async def clear_subscribers(redis: Any, process_id: str) -> None:
    await redis.delete(SUBSCRIBERS_PREFIX + process_id)


async def subscriber_total(redis: Any) -> int:
    """Event-stream clients over every API process whose count has not expired."""
    keys = [key async for key in redis.scan_iter(match=SUBSCRIBERS_PREFIX + "*", count=100)]
    if not keys:
        return 0
    total = 0
    for value in await redis.mget(keys):
        try:
            total += max(0, int(_text(value) or 0))
        except ValueError:
            continue
    return total


# ---------------------------------------------------------------- producer lock and latest snapshot


async def claim_producer(redis: Any, name: str) -> bool:
    """Whether ``name`` is (or now becomes) the single producer; the lock lapses ``PRODUCER_LOCK_S`` after its
    holder's last beat."""
    if await redis.set(PRODUCER_KEY, name, nx=True, ex=PRODUCER_LOCK_S):
        return True
    if _text(await redis.get(PRODUCER_KEY)) == name:
        await redis.expire(PRODUCER_KEY, PRODUCER_LOCK_S)
        return True
    return False


async def write_latest(redis: Any, view: StatusSnapshotView) -> None:
    await redis.set(LATEST_KEY, view.model_dump_json(), ex=LATEST_TTL_S)


async def read_latest(
    redis: Any, *, now: datetime | None = None, max_age_s: float = LATEST_FRESH_S
) -> StatusSnapshotView | None:
    """The producer's latest snapshot when it was taken at most ``max_age_s`` ago, else None."""
    raw = await redis.get(LATEST_KEY)
    if not raw:
        return None
    try:
        view = StatusSnapshotView.model_validate_json(raw)
    except ValidationError:
        log.warning("live status: %s does not hold a snapshot view; ignored", LATEST_KEY)
        return None
    age = ((now or datetime.now(UTC)) - view.freshness.computed_at).total_seconds()
    if age > max_age_s:
        return None
    source_health.observe_status_view(view)  # this process's registry follows the producer's finding (SV-6.5)
    return view


# ---------------------------------------------------------------- producer (worker)


def _in_thread[T](fn: Callable[..., T], *args: Any) -> asyncio.Future[T]:
    """``fn(*args)`` in a daemon thread of its own (with the caller's context, as ``asyncio.to_thread``), as a future of
    the running loop. A call that hangs takes no thread of the loop's default executor and does not hold up the exit
    of the process."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()
    context = contextvars.copy_context()

    def settle(result: Any, error: Exception | None) -> None:
        if future.done():  # cancelled by close()
            return
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)

    def run() -> None:
        try:
            result, error = context.run(fn, *args), None
        except Exception as exc:  # noqa: BLE001 - handed to the beat that awaits the future
            result, error = None, exc
        with contextlib.suppress(RuntimeError):  # the loop closed meanwhile (worker stopped)
            loop.call_soon_threadsafe(settle, result, error)

    threading.Thread(target=run, name=f"live-status-{getattr(fn, '__name__', 'call')}", daemon=True).start()
    return future


class LiveStatusProducer:
    """The worker's live-status producer: at most one snapshot query per beat, only while clients are connected.

    ``snapshot(now)`` and ``hot_metal(now)`` are blocking (InfluxDB, Neon) and run in daemon threads (``_in_thread``).
    One snapshot query runs at a time. A beat waits ``snapshot_wait_s`` for it; a snapshot that finishes after that
    wait is published by the next beat when it is at most ``LATEST_FRESH_S`` old, and that beat starts the next query
    (a query slower than the wait still yields a snapshot every beat, one beat late). A query running longer than
    ``snapshot_abandon_s`` is given up and a new one started, as long as fewer than ``max_abandoned`` given-up queries
    are still running (a hung source holds at most ``max_abandoned`` + 1 threads); a given-up query's result is not
    used.
    """

    def __init__(
        self,
        name: str,
        *,
        snapshot: Callable[[datetime], StatusSnapshotView] | None = None,
        hot_metal: Callable[[datetime], dict[str, TagValue]] | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        snapshot_wait_s: float = SNAPSHOT_WAIT_S,
        snapshot_abandon_s: float = SNAPSHOT_ABANDON_S,
        max_abandoned: int = MAX_ABANDONED,
        hot_metal_refresh_s: float = HOT_METAL_REFRESH_S,
        hot_metal_wait_s: float = HOT_METAL_WAIT_S,
    ) -> None:
        self.name = name
        self._snapshot = snapshot or (lambda now: build_snapshot_view(now, source="producer"))
        self._hot_metal = hot_metal or latest_hot_metal
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic
        self.snapshot_wait_s = snapshot_wait_s
        self.snapshot_abandon_s = snapshot_abandon_s
        self.max_abandoned = max_abandoned
        self.hot_metal_refresh_s = hot_metal_refresh_s
        self.hot_metal_wait_s = hot_metal_wait_s
        self._task: asyncio.Future[StatusSnapshotView] | None = None  # the query whose snapshot is not used yet
        self._task_started = 0.0  # monotonic start of self._task, s
        self._abandoned: set[asyncio.Future[StatusSnapshotView]] = set()  # given up, thread still running
        self._hm_task: asyncio.Future[dict[str, TagValue]] | None = None
        self._hm_started: float | None = None
        self._hm: dict[str, TagValue] = {}
        self.queries = 0  # snapshot queries started (tests, diagnostics)
        self.abandoned = 0  # snapshot queries given up (tests, diagnostics)
        self._warned = 0.0

    def _warn(self, message: str, *args: Any) -> None:
        if time.monotonic() - self._warned > 60:  # once a minute at most
            log.warning(message, *args)
            self._warned = time.monotonic()

    async def tick(self, redis: Any) -> dict[str, Any]:
        """Plant fields of this beat's ``live.status`` (``LiveStatusData`` keywords); empty when no client is
        connected, another worker produces, or no snapshot is ready."""
        try:
            if await subscriber_total(redis) <= 0 or not await claim_producer(redis, self.name):
                return {}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Redis briefly unavailable: no snapshot this beat
            self._warn("live status: subscriber count not read (%s)", type(exc).__name__)
            return {}
        now = self._clock()
        deadline = asyncio.get_running_loop().time() + self.snapshot_wait_s
        late = self._late_snapshot(now)  # finished after the last beat stopped waiting for it
        task = self._query(now)
        hot_metal = await self._hot_metal_tags(now, deadline)
        view = await self._await_snapshot(task, deadline)
        if view is None:
            view = late
        if view is None:
            return {}
        try:
            await write_latest(redis, view)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._warn("live status: latest snapshot not written (%s)", type(exc).__name__)
        return live_status_fields(view, hot_metal)

    def _late_snapshot(self, now: datetime) -> StatusSnapshotView | None:
        """The snapshot that finished after the last beat stopped waiting for it, when it is at most
        ``LATEST_FRESH_S`` old. Taken off in any case, so this beat starts the next query."""
        task = self._task
        if task is None or not task.done():
            return None
        self._task = None
        if task.cancelled():
            return None
        error = task.exception()
        if error is not None:
            log.error("live status: snapshot failed", exc_info=error)
            return None
        view = task.result()
        age_s = (now - view.freshness.computed_at).total_seconds()
        return view if age_s <= LATEST_FRESH_S else None

    def _query(self, now: datetime) -> asyncio.Future[StatusSnapshotView]:
        """The running snapshot query, or a new one when none runs or the running one is given up."""
        task = self._task
        if task is not None and self._monotonic() - self._task_started >= self.snapshot_abandon_s:
            if len(self._abandoned) < self.max_abandoned:
                self._abandon(task)
                task = None
            else:
                self._warn(
                    "live status: snapshot query running for %.0f s; %d given-up queries still run, none started",
                    self._monotonic() - self._task_started,
                    len(self._abandoned),
                )
        if task is None:
            task = self._task = _in_thread(self._snapshot, now)
            self._task_started = self._monotonic()
            self.queries += 1
        return task

    def _abandon(self, task: asyncio.Future[StatusSnapshotView]) -> None:
        log.warning(
            "live status: snapshot query running for %.0f s given up; a new one starts",
            self._monotonic() - self._task_started,
        )
        self._task = None
        self.abandoned += 1
        self._abandoned.add(task)
        task.add_done_callback(self._forget)

    def _forget(self, task: asyncio.Future[StatusSnapshotView]) -> None:
        """A given-up query returned: its thread has ended; its result or error is dropped."""
        self._abandoned.discard(task)
        if not task.cancelled():
            task.exception()  # retrieved, so a late failure is not reported as unhandled

    async def _await_snapshot(
        self, task: asyncio.Future[StatusSnapshotView], deadline: float
    ) -> StatusSnapshotView | None:
        """The query's snapshot when it finishes before ``deadline`` (then used: the next beat starts a new query);
        None while it runs or when it failed."""
        loop = asyncio.get_running_loop()
        try:
            view = await asyncio.wait_for(asyncio.shield(task), max(0.0, deadline - loop.time()))
        except TimeoutError:
            self._warn("live status: snapshot slower than %g s; published by the next beat", self.snapshot_wait_s)
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("live status: snapshot failed")
            view = None
        if self._task is task:
            self._task = None
        return view

    async def _hot_metal_tags(self, now: datetime, deadline: float) -> dict[str, TagValue]:
        """The latest hot-metal tags: a refresh every ``hot_metal_refresh_s`` in its own task, awaited at most
        ``hot_metal_wait_s`` (a slower one lands in a later beat)."""
        started = self._monotonic()
        if self._hm_task is None and (
            self._hm_started is None or started - self._hm_started >= self.hot_metal_refresh_s
        ):
            self._hm_started = started
            self._hm_task = _in_thread(self._hot_metal, now)
        task = self._hm_task
        if task is None:
            return self._hm
        if not task.done():
            wait = min(self.hot_metal_wait_s, max(0.0, deadline - asyncio.get_running_loop().time()))
            await asyncio.wait({task}, timeout=wait)
        if task.done():
            self._hm_task = None
            try:
                self._hm = task.result()
            except asyncio.CancelledError:
                self._hm = {}
            except Exception as exc:  # latest_hot_metal catches source errors; this is a defect
                log.warning("live status: hot-metal tags failed (%s)", type(exc).__name__)
                self._hm = {}
        return self._hm

    async def close(self) -> None:
        """Cancel the queries of a stopping worker (their daemon threads end on their own)."""
        for task in (self._task, self._hm_task, *self._abandoned):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
