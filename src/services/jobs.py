"""Background jobs: the catalogue of job kinds, job records and submission (SV-0.8, #277).

A job is a model run executed by the worker (``python -m worker``, arq on Redis). The API and the scheduler submit it
through ``submit``: the parameters are validated against the kind's model, a record is written to Redis
(``twin:job:<id>``, listed in ``twin:jobs``) and the arq queue receives the job id. A request whose kind and
parameters equal those of a queued or running job returns that job instead of starting another (input hash in
``twin:job-active:<kind>:<hash>``). Records are kept in Redis for 7 days; a ``JobArchive`` (the worker's
``services.job_archive.TwinJobArchive``, table ``twin.jobs`` in Neon, #278) receives every change of state and keeps
them longer. Worker implementations of the kinds are in ``worker.kinds``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from services.auth import Role
from services.events import JobState

QUEUE_NAME = "twin:queue"
RECORD_TTL_S = 7 * 86400
RECORD_KEY = "twin:job:{}"
INDEX_KEY = "twin:jobs"
ACTIVE_KEY = "twin:job-active:{}:{}"
ARQ_FUNCTION = "run_job"


class NoParams(BaseModel):
    model_config = {"extra": "forbid"}


class SelfTestParams(BaseModel):
    model_config = {"extra": "forbid"}

    steps: int = Field(5, ge=1, le=60, description="Progress steps")
    step_s: float = Field(1.0, ge=0, le=30, description="Seconds per step")
    cpu: bool = Field(True, description="Also run one task in the model-solve process pool")


@dataclass(frozen=True)
class JobKind:
    name: str
    title: str
    role: Role = "admin"
    timeout_s: float = 300.0
    max_tries: int = 1
    retry_delay_s: float = 30.0
    params: type[BaseModel] = NoParams
    #: Cron times in IST, ``(hours, minutes)`` with ``None`` for every hour; recorded jobs only.
    schedule: tuple[set[int] | None, set[int]] | None = None
    description: str = ""


CATALOG: dict[str, JobKind] = {}


def register(kind: JobKind) -> JobKind:
    if kind.name in CATALOG:
        raise ValueError(f"job kind {kind.name} is registered twice")
    CATALOG[kind.name] = kind
    return kind


register(
    JobKind(
        name="system.selftest",
        title="Worker self-test",
        role="admin",
        timeout_s=120,
        params=SelfTestParams,
        description="Reports progress in steps and runs one task in the model-solve pool.",
    )
)
register(
    JobKind(
        name="system.housekeeping",
        title="Housekeeping",
        role="admin",
        timeout_s=300,
        max_tries=2,
        schedule=({4}, {0}),
        description="Copies the job records of the last 7 days to the archive and removes older index entries.",
    )
)


class JobRecord(BaseModel):
    id: str
    kind: str
    status: JobState
    params: dict[str, Any] = Field(default_factory=dict)
    input_hash: str
    progress: float | None = Field(None, ge=0, le=1)
    message: str | None = None
    submitted_by: str
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tries: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False


class UnknownJobKind(KeyError):
    pass


class JobArchive(Protocol):
    """Long-term store of job records. ``record`` must not block: it is called on every change of state."""

    def record(self, record: JobRecord) -> None: ...

    async def sync(self, records: list[JobRecord]) -> int:
        """Write the records now (reconciliation); returns how many were written."""
        ...


#: Fields whose change is a change of state worth archiving (progress and messages are not).
ARCHIVED_CHANGES = frozenset({"status", "started_at", "finished_at", "tries"})


def input_hash(kind: str, params: dict[str, Any]) -> str:
    canonical = json.dumps({"kind": kind, "params": params}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class JobStore:
    """Job records in Redis. ``enqueue`` puts a job id on the arq queue (``ArqRedis.enqueue_job``)."""

    redis: Any
    enqueue: Callable[[str], Any] | None = None
    clock: Callable[[], datetime] = field(default=_now)
    archive: JobArchive | None = None

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self.redis.get(RECORD_KEY.format(job_id))
        return JobRecord.model_validate_json(raw) if raw else None

    async def save(self, record: JobRecord) -> None:
        await self.redis.set(RECORD_KEY.format(record.id), record.model_dump_json(), ex=RECORD_TTL_S)

    async def update(self, job_id: str, **changes: Any) -> JobRecord | None:
        record = await self.get(job_id)
        if record is None:
            return None
        record = record.model_copy(update=changes)
        await self.save(record)
        if self.archive is not None and ARCHIVED_CHANGES.intersection(changes):
            self.archive.record(record)
        return record

    async def submit(self, kind_name: str, params: dict[str, Any], submitted_by: str) -> tuple[JobRecord, bool]:
        """Create and queue a job; ``(record, True)`` when an identical job is already queued or running."""
        kind = CATALOG.get(kind_name)
        if kind is None:
            raise UnknownJobKind(kind_name)
        clean = kind.params.model_validate(params).model_dump(mode="json")
        digest = input_hash(kind.name, clean)
        active_key = ACTIVE_KEY.format(kind.name, digest)
        job_id = uuid.uuid4().hex
        hold_s = int(kind.timeout_s * max(kind.max_tries, 1) + kind.retry_delay_s * kind.max_tries + 600)
        if not await self.redis.set(active_key, job_id, nx=True, ex=hold_s):
            existing_id = await self.redis.get(active_key)
            existing = await self.get(existing_id.decode() if isinstance(existing_id, bytes) else existing_id)
            if existing is not None and existing.status in ("queued", "running"):
                return existing, True
            await self.redis.set(active_key, job_id, ex=hold_s)  # stale hold of a finished or expired job
        record = JobRecord(
            id=job_id,
            kind=kind.name,
            status="queued",
            params=clean,
            input_hash=digest,
            submitted_by=submitted_by,
            submitted_at=self.clock(),
        )
        await self.save(record)
        await self.redis.zadd(INDEX_KEY, {job_id: record.submitted_at.timestamp()})
        if self.archive is not None:
            self.archive.record(record)
        if self.enqueue is not None:
            await self.enqueue(job_id)
        return record, False

    async def finish(
        self, job_id: str, status: JobState, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> JobRecord | None:
        record = await self.update(
            job_id,
            status=status,
            finished_at=self.clock(),
            result=result,
            error=error,
            progress=1.0 if status == "done" else None,
        )
        if record is not None:
            active_key = ACTIVE_KEY.format(record.kind, record.input_hash)
            held = await self.redis.get(active_key)
            if held is not None and (held.decode() if isinstance(held, bytes) else held) == job_id:
                await self.redis.delete(active_key)
        return record

    async def list(self, limit: int = 50) -> list[JobRecord]:
        ids = await self.redis.zrevrange(INDEX_KEY, 0, max(limit, 1) - 1)
        if not ids:
            return []
        raws = await self.redis.mget([RECORD_KEY.format(i.decode() if isinstance(i, bytes) else i) for i in ids])
        return [JobRecord.model_validate_json(r) for r in raws if r]

    async def queued_count(self) -> int:
        return int(await self.redis.zcard(QUEUE_NAME))

    async def trim(self, older_than_s: float = RECORD_TTL_S) -> int:
        cutoff = self.clock().timestamp() - older_than_s
        return int(await self.redis.zremrangebyscore(INDEX_KEY, "-inf", cutoff))
