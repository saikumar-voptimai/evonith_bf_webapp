"""Live state of an API process: the Redis connection, the event bus, the job store and the SSE fan-out.

Started in the application lifespan. With ``BF_REDIS_URL`` set, events come from the Redis stream written by the
worker and jobs can be submitted; while Redis is not reachable the process keeps answering, the stream carries
heartbeats only, job routes answer 503, and the connection is retried every 15 s. Without ``BF_REDIS_URL`` the bus is
in-process (development without Redis).

The process records its number of event-stream clients in Redis (``services.live_status``: every 5 s and on every
change, TTL 20 s); the worker takes the live furnace-status snapshot only while the total over the processes is above
zero (SV-2.1).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from api.errors import ApiError
from api.settings import ApiSettings
from services import live_status
from services.events import EventBus, MemoryEventBus, RedisEventBus, StreamEvent
from services.jobs import ARQ_FUNCTION, QUEUE_NAME, JobStore

log = logging.getLogger("api.live")

RECONNECT_S = 15.0
SUBSCRIBER_QUEUE = 1000


@dataclass(eq=False)
class Subscriber:
    queue: asyncio.Queue[StreamEvent | None] = field(default_factory=lambda: asyncio.Queue(SUBSCRIBER_QUEUE))
    closed: bool = False


class EventHub:
    """Reads the bus once per process and hands every event to each connected client."""

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._subscribers: set[Subscriber] = set()
        self._on_change = on_change

    @property
    def size(self) -> int:
        return len(self._subscribers)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def subscribe(self) -> Subscriber:
        sub = Subscriber()
        self._subscribers.add(sub)
        self._changed()
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        if sub in self._subscribers:
            self._subscribers.discard(sub)
            self._changed()

    def fan_out(self, event: StreamEvent) -> None:
        dropped = False
        for sub in list(self._subscribers):
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                # A client that cannot keep up is disconnected; it reconnects with Last-Event-ID and gets the replay.
                sub.closed = True
                self._subscribers.discard(sub)
                dropped = True
        if dropped:
            self._changed()

    def close_all(self) -> None:
        for sub in list(self._subscribers):
            sub.closed = True
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(None)
        self._subscribers.clear()
        self._changed()


class LiveState:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.redis: Any = None
        self.bus: EventBus = MemoryEventBus()
        self.store: JobStore | None = None
        # Key of this process's client count in Redis (several API processes may share one host).
        self.process_id = f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(3)}"
        self._subscribers_changed = asyncio.Event()
        self.hub = EventHub(on_change=self._subscribers_changed.set)
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def redis_connected(self) -> bool:
        return self.redis is not None

    async def start(self) -> None:
        if self.settings.redis_url and not await self._connect():
            self._tasks.append(asyncio.create_task(self._reconnect_loop()))
        self._tasks.append(asyncio.create_task(self._pump()))
        self._tasks.append(asyncio.create_task(self._report_subscribers()))

    async def stop(self) -> None:
        self.hub.close_all()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.redis is not None:
            with contextlib.suppress(Exception):
                await live_status.clear_subscribers(self.redis, self.process_id)
            await self.redis.aclose()

    async def _report_subscribers(self) -> None:
        """Write this process's event-stream client count to Redis on every change and every 5 s (the key expires
        after 20 s, so a process that stops without ``stop`` drops out of the total)."""
        warned = 0.0
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._subscribers_changed.wait(), live_status.SUBSCRIBERS_REPORT_S)
            self._subscribers_changed.clear()
            redis = self.redis
            if redis is None:
                continue
            try:
                await live_status.record_subscribers(redis, self.process_id, self.hub.size)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Redis briefly unavailable: the next report retries
                if time.monotonic() - warned > 60:
                    log.warning("client count not written to Redis (%s)", type(exc).__name__)
                    warned = time.monotonic()

    async def _connect(self) -> bool:
        from arq.connections import RedisSettings, create_pool

        redis_settings = RedisSettings.from_dsn(self.settings.redis_url or "")
        redis_settings.conn_retries = 0
        redis_settings.conn_timeout = 2
        try:
            redis = await create_pool(redis_settings, default_queue_name=QUEUE_NAME)
        except Exception as exc:
            log.warning("Redis not reachable (%s); retrying every %.0f s", type(exc).__name__, RECONNECT_S)
            return False

        async def enqueue(job_id: str) -> None:
            await redis.enqueue_job(ARQ_FUNCTION, job_id, _job_id=job_id, _queue_name=QUEUE_NAME)

        self.redis = redis
        self.bus = RedisEventBus(redis)
        self.store = JobStore(redis, enqueue=enqueue)
        log.info("connected to Redis at %s:%s", redis_settings.host, redis_settings.port)
        return True

    async def _reconnect_loop(self) -> None:
        while self.redis is None:
            await asyncio.sleep(RECONNECT_S)
            await self._connect()

    async def _pump(self) -> None:
        bus: EventBus | None = None
        last = "0-0"
        while True:
            try:
                if bus is not self.bus:
                    bus = self.bus
                    last = await bus.last_id()
                for event in await bus.read(last, block_ms=5000):
                    last = event.id
                    self.hub.fan_out(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("event stream read failed (%s); retrying", type(exc).__name__)
                await asyncio.sleep(2)


def live_state(request: Request) -> LiveState:
    live = getattr(request.app.state, "live", None)
    if live is None:
        raise ApiError(503, "The live service is starting; try again in a few seconds.")
    return live


def job_store(request: Request) -> JobStore:
    store = live_state(request).store
    if store is None:
        raise ApiError(503, "The job queue is not available (Redis is not connected).")
    return store
