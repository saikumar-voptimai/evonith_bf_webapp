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

The hot-metal tags (``hmt``, ``hm_si``) come from Neon at most every ``HOT_METAL_REFRESH_S`` in a task of their own,
so a slow Neon does not hold up the beat.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

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
SNAPSHOT_WAIT_S = 8.0  # longest wait of a beat for its snapshot; a slower query is used by no beat
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
    return view if age <= max_age_s else None


# ---------------------------------------------------------------- producer (worker)


class LiveStatusProducer:
    """The worker's live-status producer: at most one snapshot query per beat, only while clients are connected.

    ``snapshot(now)`` and ``hot_metal(now)`` are blocking (InfluxDB, Neon) and run in threads. A snapshot still running
    when the beat stops waiting (``snapshot_wait_s``) is not started again: the next beat starts a new one only after
    it has finished, so slow queries never pile up.
    """

    def __init__(
        self,
        name: str,
        *,
        snapshot: Callable[[datetime], StatusSnapshotView] | None = None,
        hot_metal: Callable[[datetime], dict[str, TagValue]] | None = None,
        clock: Callable[[], datetime] | None = None,
        snapshot_wait_s: float = SNAPSHOT_WAIT_S,
        hot_metal_refresh_s: float = HOT_METAL_REFRESH_S,
        hot_metal_wait_s: float = HOT_METAL_WAIT_S,
    ) -> None:
        self.name = name
        self._snapshot = snapshot or (lambda now: build_snapshot_view(now, source="producer"))
        self._hot_metal = hot_metal or latest_hot_metal
        self._clock = clock or (lambda: datetime.now(UTC))
        self.snapshot_wait_s = snapshot_wait_s
        self.hot_metal_refresh_s = hot_metal_refresh_s
        self.hot_metal_wait_s = hot_metal_wait_s
        self._task: asyncio.Task[StatusSnapshotView] | None = None
        self._hm_task: asyncio.Task[dict[str, TagValue]] | None = None
        self._hm_started: float | None = None
        self._hm: dict[str, TagValue] = {}
        self.queries = 0  # snapshot queries started (tests, diagnostics)
        self._warned = 0.0

    def _warn(self, message: str, *args: Any) -> None:
        if time.monotonic() - self._warned > 60:  # once a minute at most
            log.warning(message, *args)
            self._warned = time.monotonic()

    async def tick(self, redis: Any) -> dict[str, Any]:
        """Plant fields of this beat's ``live.status`` (``LiveStatusData`` keywords); empty when no client is
        connected, another worker produces, or the snapshot is not ready."""
        try:
            if await subscriber_total(redis) <= 0 or not await claim_producer(redis, self.name):
                return {}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Redis briefly unavailable: no snapshot this beat
            self._warn("live status: subscriber count not read (%s)", type(exc).__name__)
            return {}
        now = self._clock()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.snapshot_wait_s
        task = self._task
        if task is None or task.done():
            task = self._task = asyncio.create_task(asyncio.to_thread(self._snapshot, now))
            self.queries += 1
        hot_metal = await self._hot_metal_tags(now, deadline)
        try:
            view = await asyncio.wait_for(asyncio.shield(task), max(0.0, deadline - loop.time()))
        except TimeoutError:
            self._warn("live status: snapshot slower than %g s; beat published without it", self.snapshot_wait_s)
            return {}
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("live status: snapshot failed")
            return {}
        try:
            await write_latest(redis, view)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._warn("live status: latest snapshot not written (%s)", type(exc).__name__)
        return live_status_fields(view, hot_metal)

    async def _hot_metal_tags(self, now: datetime, deadline: float) -> dict[str, TagValue]:
        """The latest hot-metal tags: a refresh every ``hot_metal_refresh_s`` in its own task, awaited at most
        ``hot_metal_wait_s`` (a slower one lands in a later beat)."""
        started = time.monotonic()
        if self._hm_task is None and (
            self._hm_started is None or started - self._hm_started >= self.hot_metal_refresh_s
        ):
            self._hm_started = started
            self._hm_task = asyncio.create_task(asyncio.to_thread(self._hot_metal, now))
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
        """Cancel the tasks of a stopping worker (their threads finish on their own)."""
        for task in (self._task, self._hm_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
