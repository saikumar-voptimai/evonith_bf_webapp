"""Live furnace status (SV-2.1, #293): API processes record their event-stream clients, the worker's heartbeat takes one
snapshot per beat while the total is above zero, a second worker stays idle, slow sources do not pile up, and the API
reads the latest snapshot while it is fresh.

Redis is an in-memory fake of the commands ``services.live_status`` and ``api.live`` use (redis-server is not on every
development machine; the same commands run against a real server in CI through the job and event tests).
"""

from __future__ import annotations

import asyncio
import fnmatch
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from api.live import LiveState
from api.settings import ApiSettings
from data.furnace_status import repository as fs_repository
from services import live_status, status
from services.events import MemoryEventBus, TagValue
from tests.test_services_status import NOW, CountingFetcher, plant_frames
from worker import runtime


class FakeRedis:
    """Async in-memory Redis: get, set (ex, nx, xx), mget, delete, expire, scan_iter, aclose; keys may be bytes."""

    def __init__(self, clock=time.monotonic) -> None:
        self.clock = clock
        self._data: dict[str, tuple[bytes, float | None]] = {}
        self._lock = threading.Lock()
        self.closed = False

    @staticmethod
    def _key(key: str | bytes) -> str:
        return key.decode() if isinstance(key, bytes) else key

    def _live(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and self.clock() >= expires:
            del self._data[key]
            return None
        return value

    async def get(self, key):
        with self._lock:
            return self._live(self._key(key))

    async def set(self, key, value, ex=None, nx=False, xx=False):
        key = self._key(key)
        with self._lock:
            exists = self._live(key) is not None
            if (nx and exists) or (xx and not exists):
                return None
            data = value.encode() if isinstance(value, str) else value
            self._data[key] = (data, self.clock() + ex if ex else None)
            return True

    async def mget(self, keys):
        with self._lock:
            return [self._live(self._key(k)) for k in keys]

    async def delete(self, *keys):
        with self._lock:
            return sum(self._data.pop(self._key(k), None) is not None for k in keys)

    async def expire(self, key, seconds):
        key = self._key(key)
        with self._lock:
            value = self._live(key)
            if value is None:
                return False
            self._data[key] = (value, self.clock() + seconds)
            return True

    async def scan_iter(self, match=None, count=None):
        with self._lock:
            keys = [k for k in list(self._data) if self._live(k) is not None]
        for key in keys:
            if match is None or fnmatch.fnmatchcase(key, match):
                yield key.encode()

    async def aclose(self) -> None:
        self.closed = True


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _fresh_caches():
    status.clear_cache()
    yield
    status.clear_cache()


@pytest.fixture()
def fetcher(monkeypatch) -> CountingFetcher:
    fake = CountingFetcher(plant_frames(datetime.now(UTC)))
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", fake)
    return fake


def no_hot_metal(now: datetime) -> dict[str, TagValue]:
    return {}


# ---------------------------------------------------------------- subscriber counts


def test_client_counts_sum_over_processes_and_expire() -> None:
    clock = Clock()
    redis = FakeRedis(clock)

    async def run() -> None:
        assert await live_status.subscriber_total(redis) == 0
        await live_status.record_subscribers(redis, "api-a", 3)
        await live_status.record_subscribers(redis, "api-b", 4)
        await redis.set("twin:other", "9")
        assert await live_status.subscriber_total(redis) == 7
        await live_status.record_subscribers(redis, "api-a", 0)
        assert await live_status.subscriber_total(redis) == 4
        clock.t += live_status.SUBSCRIBERS_TTL_S  # api-b stopped reporting (crashed): its count lapses
        assert await live_status.subscriber_total(redis) == 0
        await live_status.record_subscribers(redis, "api-b", 2)
        await live_status.clear_subscribers(redis, "api-b")
        assert await live_status.subscriber_total(redis) == 0

    asyncio.run(run())


def test_an_api_process_reports_its_stream_clients() -> None:
    async def run() -> None:
        live = LiveState(ApiSettings(static_dir=None, redis_url=None))
        await live.start()
        redis = FakeRedis()
        live.redis = redis  # as after a successful connect
        try:
            subs = [live.hub.subscribe() for _ in range(3)]
            await _until(lambda: _count(redis, live.process_id), 3)
            live.hub.unsubscribe(subs[0])
            live.hub.unsubscribe(subs[0])  # twice: counted once
            await _until(lambda: _count(redis, live.process_id), 2)
            assert await live_status.subscriber_total(redis) == 2
        finally:
            await live.stop()
        assert await redis.get(live_status.SUBSCRIBERS_PREFIX + live.process_id) is None  # removed at shutdown

    asyncio.run(run())


async def _count(redis: FakeRedis, process_id: str) -> int | None:
    raw = await redis.get(live_status.SUBSCRIBERS_PREFIX + process_id)
    return None if raw is None else int(raw)


async def _until(read, expected: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await read() == expected:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"value is {await read()!r}, expected {expected!r}")


# ---------------------------------------------------------------- producer


def test_one_snapshot_query_per_beat_whatever_the_number_of_clients(fetcher) -> None:
    """Acceptance criterion: at most one InfluxDB snapshot query per 10-s beat for 7 clients on two API processes,
    and none from the API while the latest snapshot is fresh."""
    redis = FakeRedis()

    async def run() -> None:
        await live_status.record_subscribers(redis, "api-a", 3)
        await live_status.record_subscribers(redis, "api-b", 4)
        producer = live_status.LiveStatusProducer("worker-1", hot_metal=no_hot_metal)
        for beat in range(1, 7):
            fields = await producer.tick(redis)
            assert set(fields["tags"]) >= {"production_rate", "hot_blast_volume", "raft"}
            assert fetcher.snapshot_queries == beat  # one query per beat, not per client
            fs_repository.clear_cache()  # the next beat is 10 s later: past the 5-s request cache
            for _ in range(20):  # clients reading the snapshot route meanwhile
                view = await live_status.read_latest(redis)
                assert view is not None and view.freshness.source == "producer"
        assert producer.queries == 6 and fetcher.snapshot_queries == 6
        assert len(fetcher.calls) == 6 * 4  # four measurements per snapshot query

    asyncio.run(run())


def test_no_query_without_clients(fetcher) -> None:
    redis = FakeRedis()

    async def run() -> None:
        producer = live_status.LiveStatusProducer("worker-1", hot_metal=no_hot_metal)
        assert await producer.tick(redis) == {}
        await live_status.record_subscribers(redis, "api-a", 0)
        assert await producer.tick(redis) == {}
        assert await redis.get(live_status.LATEST_KEY) is None

    asyncio.run(run())
    assert fetcher.calls == []


def test_a_second_worker_does_not_query(fetcher) -> None:
    clock = Clock()
    redis = FakeRedis(clock)

    async def run() -> None:
        await live_status.record_subscribers(redis, "api-a", 1)
        first = live_status.LiveStatusProducer("worker-1", hot_metal=no_hot_metal)
        second = live_status.LiveStatusProducer("worker-2", hot_metal=no_hot_metal)
        assert (await first.tick(redis))["tags"]
        assert await second.tick(redis) == {}
        assert (first.queries, second.queries) == (1, 0)
        clock.t += live_status.PRODUCER_LOCK_S  # the first worker stopped beating: the lock lapses
        await live_status.record_subscribers(redis, "api-a", 1)
        fs_repository.clear_cache()
        assert (await second.tick(redis))["tags"]
        assert (first.queries, second.queries) == (1, 1)

    asyncio.run(run())


def test_a_slow_snapshot_is_not_started_again_while_it_runs() -> None:
    release = threading.Event()
    started: list[datetime] = []

    def slow(now: datetime) -> status.StatusSnapshotView:
        started.append(now)
        release.wait(5)
        return status.snapshot_view(_snapshot_at(now), now, source="producer")

    redis = FakeRedis()

    async def run() -> None:
        await live_status.record_subscribers(redis, "api-a", 2)
        producer = live_status.LiveStatusProducer(
            "worker-1", snapshot=slow, hot_metal=no_hot_metal, snapshot_wait_s=0.05
        )
        assert await producer.tick(redis) == {}  # not ready within the beat
        assert await producer.tick(redis) == {}  # still running: no second query
        assert producer.queries == 1 and len(started) == 1
        producer.snapshot_wait_s = 5.0
        release.set()
        fields = await producer.tick(redis)  # still the first query (finishing): awaited, not started again
        assert fields["tags"] and producer.queries == 1 and len(started) == 1
        fields = await producer.tick(redis)  # the next beat: a new query
        assert fields["tags"] and producer.queries == 2 and len(started) == 2

    asyncio.run(run())


def _snapshot_at(now: datetime):
    from data.furnace_status import build_status_snapshot

    frames = {m: f for m, f in plant_frames(now).items() if not isinstance(f, Exception)}
    return build_status_snapshot(frames, now)


def test_hot_metal_is_read_every_two_minutes_beside_the_beat(fetcher) -> None:
    calls: list[datetime] = []

    def hot_metal(now: datetime) -> dict[str, TagValue]:
        calls.append(now)
        return {"hmt": TagValue(value=1490.0, unit="°C", at=now - timedelta(minutes=30))}

    redis = FakeRedis()

    async def run() -> None:
        await live_status.record_subscribers(redis, "api-a", 1)
        producer = live_status.LiveStatusProducer("worker-1", hot_metal=hot_metal)
        for _ in range(3):
            fields = await producer.tick(redis)
            assert fields["tags"]["hmt"].value == 1490.0
        assert len(calls) == 1  # refreshed every 120 s, not every beat
        producer.hot_metal_refresh_s = 0.0
        await producer.tick(redis)
        assert len(calls) == 2

    asyncio.run(run())


def test_hot_metal_that_hangs_does_not_hold_up_the_beat(fetcher) -> None:
    release = threading.Event()

    def hanging(now: datetime) -> dict[str, TagValue]:
        release.wait(5)
        return {"hm_si": TagValue(value=0.5, unit="%", at=now)}

    redis = FakeRedis()

    async def run() -> None:
        await live_status.record_subscribers(redis, "api-a", 1)
        producer = live_status.LiveStatusProducer("worker-1", hot_metal=hanging, hot_metal_wait_s=0.05)
        t0 = time.monotonic()
        fields = await producer.tick(redis)
        assert time.monotonic() - t0 < 2.0 and fields["tags"] and "hm_si" not in fields["tags"]
        release.set()
        await asyncio.sleep(0.1)
        fs_repository.clear_cache()
        fields = await producer.tick(redis)
        assert fields["tags"]["hm_si"].value == 0.5  # lands in a later beat
        await producer.close()

    asyncio.run(run())


def test_the_latest_snapshot_is_read_while_fresh(fetcher) -> None:
    redis = FakeRedis()

    async def run() -> None:
        view = status.build_snapshot_view(NOW, source="producer")
        await live_status.write_latest(redis, view)
        fresh = await live_status.read_latest(redis, now=NOW + timedelta(seconds=live_status.LATEST_FRESH_S))
        assert fresh == view
        assert await live_status.read_latest(redis, now=NOW + timedelta(seconds=26)) is None
        await redis.set(live_status.LATEST_KEY, "not json")
        assert await live_status.read_latest(redis, now=NOW) is None

    asyncio.run(run())


# ---------------------------------------------------------------- worker heartbeat


class _Store:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis

    async def queued_count(self) -> int:
        return 0


def test_the_heartbeat_publishes_the_snapshot_while_clients_are_connected(fetcher, monkeypatch) -> None:
    monkeypatch.setattr(runtime, "HEARTBEAT_S", 0.25)
    redis = FakeRedis()
    bus = MemoryEventBus()

    async def run() -> None:
        ctx = {
            "store": _Store(redis),
            "bus": bus,
            "name": "worker-1",
            "started_at": datetime.now(UTC),
            "running": set(),
            "pool_size": 0,
            "live_status": live_status.LiveStatusProducer("worker-1", hot_metal=no_hot_metal),
        }
        await live_status.record_subscribers(redis, "api-a", 5)
        task = asyncio.create_task(runtime.heartbeat_loop(ctx))
        await asyncio.sleep(1.4)
        await live_status.record_subscribers(redis, "api-a", 0)  # the last client left
        await asyncio.sleep(0.8)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        events = [e for e in await bus.recent(100) if e.type == "live.status"]
        with_plant = [e for e in events if e.data.tags]
        assert len(with_plant) >= 4 and len(events) - len(with_plant) >= 2
        assert ctx["live_status"].queries == len(with_plant)  # one query per beat that carried a snapshot
        first = with_plant[0].data
        assert first.worker.name == "worker-1" and first.operating_state == "normal" and first.stale is False
        assert first.tags["hot_blast_volume"].unit == "Nm³/h" and first.telemetry == "live"
        assert all(e.data.tags == {} and e.data.stale is None for e in events[len(with_plant) :])
        assert await runtime.read_alive(redis) is not None

    asyncio.run(run())
