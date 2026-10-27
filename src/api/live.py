"""Live state of an API process: the Redis connection, the event bus, the job store and the SSE fan-out.

Started in the application lifespan. With ``BF_REDIS_URL`` set, events come from the Redis stream written by the
worker and jobs can be submitted; while Redis is not reachable the process keeps answering, the stream carries
heartbeats only, job routes answer 503, and the connection is retried every 15 s. Without ``BF_REDIS_URL`` the bus is
in-process (development without Redis).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from api.errors import ApiError
from api.settings import ApiSettings
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

    def __init__(self) -> None:
        self._subscribers: set[Subscriber] = set()

    @property
    def size(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> Subscriber:
        sub = Subscriber()
        self._subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        self._subscribers.discard(sub)

    def fan_out(self, event: StreamEvent) -> None:
        for sub in list(self._subscribers):
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                # A client that cannot keep up is disconnected; it reconnects with Last-Event-ID and gets the replay.
                sub.closed = True
                self._subscribers.discard(sub)

    def close_all(self) -> None:
        for sub in list(self._subscribers):
            sub.closed = True
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(None)
        self._subscribers.clear()


class LiveState:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.redis: Any = None
        self.bus: EventBus = MemoryEventBus()
        self.store: JobStore | None = None
        self.hub = EventHub()
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def redis_connected(self) -> bool:
        return self.redis is not None

    async def start(self) -> None:
        if self.settings.redis_url and not await self._connect():
            self._tasks.append(asyncio.create_task(self._reconnect_loop()))
        self._tasks.append(asyncio.create_task(self._pump()))

    async def stop(self) -> None:
        self.hub.close_all()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.redis is not None:
            await self.redis.aclose()

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
