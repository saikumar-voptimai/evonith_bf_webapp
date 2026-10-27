"""Event bus (#276): schemas, ordering, replay after Last-Event-ID, trimming; in memory and on Redis."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from services.events import (
    REPLAY_LIMIT,
    MemoryEventBus,
    RedisEventBus,
    id_key,
    is_event_id,
    make_event,
    sse_frame,
)
from tests.redis_server import needs_redis, redis_server

PROGRESS = {"job_id": "j1", "kind": "system.selftest", "status": "running", "progress": 0.5, "message": "Step 1"}


def test_events_are_validated_against_their_schema() -> None:
    event = make_event("1-0", "job.progress", datetime.now(UTC), PROGRESS)
    assert event.type == "job.progress" and event.data.progress == 0.5
    with pytest.raises(ValidationError):
        make_event("1-0", "job.progress", datetime.now(UTC), {**PROGRESS, "progress": 1.5})
    with pytest.raises(ValidationError):
        make_event("1-0", "no.such.type", datetime.now(UTC), {})


def test_sse_frame_and_ids() -> None:
    event = make_event(
        "1730000000000-3",
        "alarm.raised",
        datetime(2026, 10, 27, tzinfo=UTC),
        {"alarm_id": "a1", "code": "OMEGA_DRIFT", "severity": "warning", "message": "CUSUM above 5"},
    )
    frame = sse_frame(event)
    assert frame.startswith("id: 1730000000000-3\nevent: alarm.raised\ndata: {")
    assert frame.endswith("\n\n")
    assert "id:" not in sse_frame(event, with_id=False)
    assert id_key("10-2") > id_key("9-9") and id_key("10-10") > id_key("10-9")
    assert is_event_id("1730000000000-0") and not is_event_id("x") and not is_event_id(None)


async def _exercise(bus) -> None:
    first = await bus.publish("job.progress", PROGRESS)
    ids = [first.id] + [
        (await bus.publish("job.progress", {**PROGRESS, "message": f"m{i}"})).id for i in range(REPLAY_LIMIT + 20)
    ]
    assert ids == sorted(ids, key=id_key)
    assert await bus.last_id() == ids[-1]

    replay = await bus.since(ids[-6])
    assert [e.id for e in replay] == ids[-5:]
    capped = await bus.since(first.id)
    assert len(capped) == REPLAY_LIMIT and capped[-1].id == ids[-1]  # at most the last 100
    assert [e.id for e in await bus.recent(3)] == ids[-3:]

    async def later() -> None:
        await asyncio.sleep(0.2)
        await bus.publish("alarm.cleared", {"alarm_id": "a1", "code": "X", "severity": "warning", "message": "ok"})

    task = asyncio.create_task(later())
    got = await bus.read(ids[-1], block_ms=3000)
    await task
    assert [e.type for e in got] == ["alarm.cleared"]
    assert await bus.read(got[-1].id, block_ms=50) == []


def test_memory_bus() -> None:
    asyncio.run(_exercise(MemoryEventBus()))


@needs_redis
def test_redis_bus(tmp_path) -> None:
    import redis.asyncio as aioredis

    async def run(url: str) -> None:
        client = aioredis.from_url(url)
        try:
            await _exercise(RedisEventBus(client, key="test:events", maxlen=1000))
            entries = await client.xlen("test:events")
            assert entries == REPLAY_LIMIT + 22
        finally:
            await client.aclose()

    with redis_server(tmp_path) as url:
        asyncio.run(run(url))
