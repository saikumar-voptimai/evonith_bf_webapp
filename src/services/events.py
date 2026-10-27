"""Live events from the worker to the browser (SV-0.7, #276).

Producers (worker jobs, schedulers) append events to the Redis stream ``twin:events``; every API process reads the
stream once and fans the events out to its server-sent-event clients (``api.routers.stream``). The stream id of an
entry is the SSE event id, so a client that reconnects with ``Last-Event-ID`` receives the events it missed (at most
the last ``REPLAY_LIMIT``). Without Redis, ``MemoryEventBus`` keeps the same interface inside one process.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field, TypeAdapter

STREAM_KEY = "twin:events"
STREAM_MAXLEN = 1000
REPLAY_LIMIT = 100

EventType = Literal[
    "live.status",
    "twin.snapshot",
    "forecast.si",
    "forecast.charged_coke",
    "job.progress",
    "job.done",
    "alarm.raised",
    "alarm.cleared",
    "heartbeat",
]
JobState = Literal["queued", "running", "done", "failed", "cancelled"]


# ---------------------------------------------------------------- payloads


class WorkerStatus(BaseModel):
    name: str = Field(..., description="Worker process name (host:pid)")
    started_at: datetime
    running: int = Field(..., description="Jobs running in this worker")
    queued: int = Field(..., description="Jobs waiting in the queue")
    pool_size: int = Field(..., description="Processes of the model-solve pool")


class TagValue(BaseModel):
    value: float | None
    unit: str | None = None
    at: datetime | None = Field(None, description="Time of the reading (UTC)")


class LiveStatusData(BaseModel):
    worker: WorkerStatus
    tags: dict[str, TagValue] = Field(
        default_factory=dict, description="Latest plant readings (not yet filled: KPI strip of the Live Twin, SV-1.9)"
    )


class JobProgressData(BaseModel):
    job_id: str
    kind: str
    status: JobState
    progress: float | None = Field(None, ge=0, le=1, description="Fraction done, when the job reports it")
    message: str | None = None


class JobDoneData(BaseModel):
    job_id: str
    kind: str
    status: Literal["done", "failed", "cancelled"]
    message: str | None = None
    duration_s: float | None = None


class TwinSnapshotData(BaseModel):
    """One hourly twin evaluation (``services.twin.snapshot_values``, table ``twin.snapshot``, SV-1.1): the window
    [window_start, window_end) in UTC, coke rates in kg/tHM dry, omega dimensionless, other quantities with their units
    in the keys, data flags with the notes of the evaluation, model versions."""

    window_start: datetime
    window_end: datetime
    evaluated_at: datetime
    job_id: str | None = None
    status: Literal["complete", "partial", "failed"]
    k_trace: float | None = Field(None, description="Burnt coke, oxygen balance + 14-day offset, kg/tHM dry")
    k_forge: float | None = Field(None, description="Two-zone balance at the 14-day medians, kg/tHM dry")
    k_strata: float | None = Field(None, description="Two-zone balance at omega_STRATA, kg/tHM dry")
    k_strata_sd: float | None = Field(None, description="1 sigma of k_strata, kg/tHM dry")
    k_charged: float | None = Field(None, description="Coke charged, kg/tHM dry")
    omega_imp: float | None = Field(None, description="Implied shaft efficiency of the window, dimensionless")
    omega_strata: float | None = Field(None, description="STRATA 1.5 shaft efficiency of the window, dimensionless")
    omega_strata_sd: float | None = Field(None, description="1 sigma of omega_strata, dimensionless")
    omega_cal: float | None = Field(
        None, description="14-day median omega (the anchor of FORGE and STRATA), dimensionless"
    )
    quantities: dict[str, float] = Field(default_factory=dict)
    flags: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, Any] = Field(default_factory=dict)


class SiForecastPoint(BaseModel):
    horizon_minutes: int = Field(..., description="Target start after the origin, min")
    label: str = Field(..., description="Now, +1 h, +2 h, +3 h")
    target_start: datetime = Field(..., description="Start of the target band (UTC)")
    target_end: datetime = Field(..., description="End of the target band (UTC)")
    si_pct: float | None = Field(None, description="Forecast hot-metal Si, mass %")
    lower_pct: float | None = Field(None, description="Lower end of the 90 % range, mass %")
    upper_pct: float | None = Field(None, description="Upper end of the 90 % range, mass %")
    status: str | None = None


class SiForecastData(BaseModel):
    """One attempt of the hot-metal Si forecast at a new origin (``services.forecasts.issue_si``, SV-1.5). Usable
    paths (status ok, warning, out_of_domain) are in the ledger; others carry the reason the inputs were not usable."""

    origin_at: datetime = Field(..., description="Origin of the path (UTC), floored in IST to the model cadence")
    issued_at: datetime = Field(..., description="When the path was issued (UTC)")
    status: str = Field(..., description="ok, warning, out_of_domain, stale_inputs, insufficient_data, ...")
    recorded: bool = Field(..., description="The path is in the ledger")
    mode: str | None = Field(None, description="shadow or forecast")
    model_version: str | None = None
    si_pct: float | None = Field(None, description="Headline (Now) hot-metal Si, mass %")
    last_sample_si_pct: float | None = Field(None, description="Latest raw cast Si at the origin, mass %")
    last_sample_at: datetime | None = Field(None, description="Time of that sample (UTC)")
    horizons: list[SiForecastPoint] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ChargedCokePoint(BaseModel):
    horizon: int = Field(..., description="Rows after the data row (one per hour), h")
    at: datetime = Field(..., description="End of the 4-h charge window of the rate")
    prediction_kg_thm: float | None = Field(None, description="Charged coke, 4-h rate, kg/tHM")
    lower_kg_thm: float | None = Field(None, description="Lower end of the 90 % range, kg/tHM")
    upper_kg_thm: float | None = Field(None, description="Upper end of the 90 % range, kg/tHM")


class ChargedCokeForecastData(BaseModel):
    """A charged-coke forecast written to the ledger (``services.forecasts.issue_charged``, SV-1.5); times in IST
    with their offset."""

    data_row: datetime = Field(..., description="Label of the hourly row the forecast was made from (IST)")
    issued_at: datetime = Field(..., description="When the forecast was issued (IST)")
    state: str = Field(..., description="cruise, unsettled, off_cruise (shown) or a pause state")
    shown: bool = Field(..., description="The forecast may be shown and used as the Data-Driven level")
    prediction_kg_thm: float | None = Field(None, description="Charged coke of the 4-h block, kg/tHM")
    lower_kg_thm: float | None = Field(None, description="Lower end of the 90 % range, kg/tHM")
    upper_kg_thm: float | None = Field(None, description="Upper end of the 90 % range, kg/tHM")
    coke_start: datetime | None = Field(None, description="Start of the charges the block sums (IST)")
    coke_end: datetime | None = Field(None, description="End of the charges the block sums (IST)")
    model_version: str | None = None
    path: list[ChargedCokePoint] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AlarmData(BaseModel):
    alarm_id: str
    code: str = Field(..., description="Alarm code, e.g. OMEGA_DRIFT")
    severity: Literal["warning", "critical"]
    message: str
    value: float | None = None


class HeartbeatData(BaseModel):
    at: datetime


# ---------------------------------------------------------------- events


class _Event(BaseModel):
    id: str = Field(..., description="Stream id; the SSE event id used for Last-Event-ID replay")
    at: datetime = Field(..., description="Publication time (UTC)")


class LiveStatusEvent(_Event):
    type: Literal["live.status"]
    data: LiveStatusData


class TwinSnapshotEvent(_Event):
    type: Literal["twin.snapshot"]
    data: TwinSnapshotData


class ForecastSiEvent(_Event):
    type: Literal["forecast.si"]
    data: SiForecastData


class ForecastChargedCokeEvent(_Event):
    type: Literal["forecast.charged_coke"]
    data: ChargedCokeForecastData


class JobProgressEvent(_Event):
    type: Literal["job.progress"]
    data: JobProgressData


class JobDoneEvent(_Event):
    type: Literal["job.done"]
    data: JobDoneData


class AlarmRaisedEvent(_Event):
    type: Literal["alarm.raised"]
    data: AlarmData


class AlarmClearedEvent(_Event):
    type: Literal["alarm.cleared"]
    data: AlarmData


class HeartbeatEvent(_Event):
    type: Literal["heartbeat"]
    data: HeartbeatData


StreamEvent = Annotated[
    LiveStatusEvent
    | TwinSnapshotEvent
    | ForecastSiEvent
    | ForecastChargedCokeEvent
    | JobProgressEvent
    | JobDoneEvent
    | AlarmRaisedEvent
    | AlarmClearedEvent
    | HeartbeatEvent,
    Field(discriminator="type"),
]
EVENT_ADAPTER: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


def make_event(event_id: str, type_: str, at: datetime, data: Any) -> StreamEvent:
    """Validate an event against its schema (raises ``pydantic.ValidationError``)."""
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    return EVENT_ADAPTER.validate_python({"id": event_id, "type": type_, "at": at, "data": data})


_ID = re.compile(r"^\d{1,15}-\d{1,10}$")


def is_event_id(value: str | None) -> bool:
    """A well-formed stream id (``<ms>-<seq>``), e.g. a client's ``Last-Event-ID``."""
    return bool(value) and bool(_ID.match(value or ""))


def id_key(event_id: str) -> tuple[int, int]:
    """Order of stream ids (``<ms>-<seq>``)."""
    ms, _, seq = event_id.partition("-")
    return int(ms), int(seq or 0)


def sse_frame(event: StreamEvent, *, with_id: bool = True) -> str:
    """One server-sent event: ``id``, ``event`` (the type) and the event as JSON ``data``."""
    head = f"id: {event.id}\n" if with_id else ""
    return f"{head}event: {event.type}\ndata: {event.model_dump_json()}\n\n"


# ---------------------------------------------------------------- buses


class EventBus(Protocol):
    async def publish(self, type_: str, data: Any) -> StreamEvent: ...

    async def read(self, after: str, block_ms: int) -> list[StreamEvent]:
        """Events after ``after`` (``"0-0"`` for all), waiting up to ``block_ms`` for the first one."""
        ...

    async def since(self, last_id: str, limit: int = REPLAY_LIMIT) -> list[StreamEvent]:
        """The most recent ``limit`` events after ``last_id``, oldest first."""
        ...

    async def recent(self, limit: int = REPLAY_LIMIT) -> list[StreamEvent]: ...

    async def last_id(self) -> str: ...


class MemoryEventBus:
    """In-process bus with the interface of ``RedisEventBus`` (development without Redis, tests)."""

    def __init__(self, maxlen: int = STREAM_MAXLEN) -> None:
        self._events: deque[StreamEvent] = deque(maxlen=maxlen)
        self._changed = asyncio.Condition()
        self._last = (0, 0)

    def _next_id(self) -> str:
        ms = int(time.time() * 1000)
        seq = self._last[1] + 1 if ms <= self._last[0] else 0
        self._last = (max(ms, self._last[0]), seq)
        return f"{self._last[0]}-{self._last[1]}"

    async def publish(self, type_: str, data: Any) -> StreamEvent:
        event = make_event(self._next_id(), type_, datetime.now(UTC), data)
        async with self._changed:
            self._events.append(event)
            self._changed.notify_all()
        return event

    def _after(self, after: str) -> list[StreamEvent]:
        key = id_key(after)
        return [e for e in self._events if id_key(e.id) > key]

    async def read(self, after: str, block_ms: int) -> list[StreamEvent]:
        async with self._changed:
            found = self._after(after)
            if found or block_ms <= 0:
                return found
            try:
                await asyncio.wait_for(self._changed.wait_for(lambda: bool(self._after(after))), block_ms / 1000)
            except TimeoutError:
                return []
            return self._after(after)

    async def since(self, last_id: str, limit: int = REPLAY_LIMIT) -> list[StreamEvent]:
        return self._after(last_id)[-limit:]

    async def recent(self, limit: int = REPLAY_LIMIT) -> list[StreamEvent]:
        return list(self._events)[-limit:]

    async def last_id(self) -> str:
        return self._events[-1].id if self._events else "0-0"


class RedisEventBus:
    """Events in the Redis stream ``twin:events``, trimmed to about ``maxlen`` entries."""

    def __init__(self, redis: Any, key: str = STREAM_KEY, maxlen: int = STREAM_MAXLEN) -> None:
        self._redis = redis
        self._key = key
        self._maxlen = maxlen

    @staticmethod
    def _decode(entry_id: Any, fields: dict[Any, Any]) -> StreamEvent:
        def text(v: Any) -> str:
            return v.decode() if isinstance(v, bytes) else str(v)

        f = {text(k): text(v) for k, v in fields.items()}
        return make_event(text(entry_id), f["type"], datetime.fromisoformat(f["at"]), json.loads(f["data"]))

    async def publish(self, type_: str, data: Any) -> StreamEvent:
        at = datetime.now(UTC)
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
        make_event("0-1", type_, at, payload)  # validate before writing
        entry_id = await self._redis.xadd(
            self._key,
            {"type": type_, "at": at.isoformat(), "data": json.dumps(payload, separators=(",", ":"))},
            maxlen=self._maxlen,
            approximate=True,
        )
        return make_event(entry_id.decode() if isinstance(entry_id, bytes) else entry_id, type_, at, payload)

    async def read(self, after: str, block_ms: int) -> list[StreamEvent]:
        reply = await self._redis.xread({self._key: after}, count=500, block=block_ms or None)
        events: list[StreamEvent] = []
        for _stream, entries in reply or []:
            events.extend(self._decode(entry_id, fields) for entry_id, fields in entries)
        return events

    async def since(self, last_id: str, limit: int = REPLAY_LIMIT) -> list[StreamEvent]:
        # Inclusive range (Redis 6.0 on JetPack 6 has no exclusive "(" bound); the entry last_id itself is dropped.
        entries = await self._redis.xrevrange(self._key, max="+", min=last_id, count=limit + 1)
        events = [self._decode(i, f) for i, f in reversed(entries)]
        return [e for e in events if e.id != last_id][-limit:]

    async def recent(self, limit: int = REPLAY_LIMIT) -> list[StreamEvent]:
        entries = await self._redis.xrevrange(self._key, max="+", min="-", count=limit)
        return [self._decode(i, f) for i, f in reversed(entries)]

    async def last_id(self) -> str:
        entries = await self._redis.xrevrange(self._key, max="+", min="-", count=1)
        if not entries:
            return "0-0"
        entry_id = entries[0][0]
        return entry_id.decode() if isinstance(entry_id, bytes) else entry_id
