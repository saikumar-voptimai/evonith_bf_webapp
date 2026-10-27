"""Background jobs: the catalogue of job kinds, job records and submission (SV-0.8, #277).

A job is a model run executed by the worker (``python -m worker``, arq on Redis). The API and the scheduler submit it
through ``submit``: the parameters are validated against the kind's model, a record is written to Redis
(``twin:job:<id>``, listed in ``twin:jobs``) and the arq queue receives the job id. A request whose kind and
parameters equal those of a queued or running job returns that job instead of starting another (input hash in
``twin:job-active:<kind>:<hash>``). Records are kept in Redis for 7 days; a ``JobArchive`` (the worker's
``services.job_archive.TwinJobArchive``, table ``twin.jobs`` in Neon, #278) receives every change of state and keeps
them longer. Worker implementations of the kinds are in ``worker.kinds``.

Scheduled kinds (``JobKind.schedule``, IST) are submitted by the worker's cron entries with ``submitted_by="scheduler"``
and ``scheduled_for`` = the slot; the catch-up at worker start and skipped slots are recorded the same way
(``services.schedule``, SV-1.7).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

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


class OptimiserRunParams(BaseModel):
    """One Burden & Fuel Optimiser run with page 11's defaults (``services.optimiser.default_case``)."""

    model_config = {"extra": "forbid"}

    mode: Literal["step1", "step2", "max_production"] = Field(
        ..., description="step1: cheapest production level; step2: cheapest blend at a fixed production; max_production"
    )
    omega_mode: Literal["strata", "fixed"] = Field(
        "strata", description="Shaft efficiency: STRATA-ω (blend-dependent, ASM-18) or Fixed-ω (calibration, ASM-01)"
    )
    production_t_d: float | None = Field(
        None, ge=1500, le=3000, description="Step 2 hot-metal target, t/d (default: today's production)"
    )
    certainty: Literal["low", "any", "medium", "high"] = Field(
        "low", description="STRATA-ω certainty filter: low excludes Investigate plans (page 11's default)"
    )
    risk_k: float = Field(
        0.0, ge=0, le=3, description="STRATA-ω risk aversion: objective = cost + k × σ(cost), dimensionless"
    )
    auto: bool = Field(False, description="Also run the three decision settings of auto mode (STRATA-ω)")
    calibration: Literal["day", "live"] = Field(
        "day",
        description=(
            "Two-zone calibration: day = the one made at 02:30 on the 14 days before today (calibration.nightly; "
            "the live one while it is missing), live = page 11's, on the stable shifts balanced so far"
        ),
    )


class TwinEvaluateParams(BaseModel):
    """One twin evaluation (``twin.evaluate``): the window ending at a full hour, and the windows before it that are
    missing."""

    model_config = {"extra": "forbid"}

    window_end: datetime | None = Field(
        None,
        description=(
            "End of the 8-h window, a full hour (aware, or IST when naive); default: the hour of the schedule slot, "
            "else of now"
        ),
    )
    backfill_hours: int = Field(
        24,
        ge=0,
        le=168,
        description=(
            "Windows ending in these hours before window_end without a snapshot, or evaluated before their last "
            "hourly row was published, are evaluated too, h"
        ),
    )
    strata: bool = Field(True, description="Evaluate K_STRATA (3 network solves per window and 1 per 14-day anchor)")


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
register(
    JobKind(
        name="optimiser.run",
        title="Optimiser run",
        role="supervisor",
        # Step 1 with STRATA-ω: 18.4 min on the development host (offline case, both oversize options, 670 network
        # solves); the AGX Orin may need twice that until the solves run in parallel (SV-1.8)
        timeout_s=3600,
        max_tries=1,
        params=OptimiserRunParams,
        description=(
            "Runs the Burden & Fuel Optimiser with page 11's defaults (live yard and calibration, today's controls) in "
            "the model-solve pool and stores the plan in twin.optimiser_run (SV-1.6)."
        ),
    )
)
register(
    JobKind(
        name="forecast.si",
        title="HM Si forecast",
        role="supervisor",
        timeout_s=150,
        max_tries=1,  # the next run is 5 min later
        schedule=(None, set(range(0, 60, 5))),
        description=(
            "Issues the hot-metal Si path (Now, +1, +2, +3 h) at the current origin into the Si ledger and matches "
            "matured outcomes to the raw samples (SV-1.5)."
        ),
    )
)
register(
    JobKind(
        name="forecast.charged_coke",
        title="Charged-coke forecast",
        role="supervisor",
        timeout_s=300,
        max_tries=5,  # HH:07 to HH:27 while the build of the hour is not published
        retry_delay_s=300,
        schedule=(None, {7}),
        description=(
            "Fetches the hourly dataset when the local build predates this hour and issues the 4-h charged-coke "
            "block for the latest complete row into the charged-coke ledger (SV-1.5)."
        ),
    )
)
register(
    JobKind(
        name="twin.evaluate",
        title="Twin evaluation",
        role="supervisor",
        timeout_s=900,
        max_tries=4,  # HH:10 to HH:25 while the hourly dataset lacks the window's last row
        retry_delay_s=300,
        params=TwinEvaluateParams,
        schedule=(None, {10}),
        description=(
            "Evaluates the 8-h window ending at the full hour (K_TRACE, K_FORGE, K_STRATA, implied ω, transit) into "
            "twin.snapshot once the hourly dataset holds the window's last row, with the windows of the last 24 h "
            "that are missing; runs the implied-ω drift alarm and publishes twin.snapshot (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="optimiser.step2",
        title="Scheduled Step 2",
        role="supervisor",
        timeout_s=900,
        max_tries=3,  # 06:15 to 06:25 while the yard or the calibration falls back to the reference
        retry_delay_s=300,
        schedule=({6, 14, 22}, {15}),
        description=(
            "Step 2 at the current production (latest twin window) with STRATA-ω and Fixed-ω in parallel: live yard "
            "and prices, the day's calibration; two runs in twin.optimiser_run (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="optimiser.step1",
        title="Scheduled Step 1",
        role="supervisor",
        timeout_s=3600,  # as optimiser.run
        max_tries=2,
        retry_delay_s=300,
        schedule=({6, 14, 22}, {30}),
        description=(
            "Step 1 (cheapest production level) with STRATA-ω: live yard and prices, the day's calibration; one run "
            "in twin.optimiser_run (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="calibration.nightly",
        title="Nightly calibration",
        role="admin",
        timeout_s=900,
        max_tries=3,
        retry_delay_s=600,
        schedule=({2}, {30}),
        description=(
            "Two-zone calibration on the stable shifts of the 14 days before today, saved as the day's calibration of "
            "the optimiser runs, and the STRATA-Flow network Jacobian at it (17 solves, disk cache) (SV-1.7)."
        ),
    )
)
register(
    JobKind(
        name="learn.shadow",
        title="STRATA-Learn shadow run",
        role="admin",
        timeout_s=600,
        max_tries=1,
        schedule=({3}, {0}),
        description=(
            "STRATA-Learn in shadow mode on the stored evidence (Kalman challenger, champion/challenger gate, CUSUM "
            "drift, register proposal) into src/storage/burden_engine/learn; changes nothing in the asset, the "
            "register or the pages (SV-1.7)."
        ),
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
    scheduled_for: datetime | None = Field(
        None, description="Schedule slot a scheduler job serves (its IST cron time); null for jobs a user started"
    )
    run_s: float | None = Field(None, description="Execution time summed over the tries, s")


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

    async def submit(
        self,
        kind_name: str,
        params: dict[str, Any],
        submitted_by: str,
        *,
        scheduled_for: datetime | None = None,
    ) -> tuple[JobRecord, bool]:
        """Create and queue a job; ``(record, True)`` when an identical job is already queued or running.
        ``scheduled_for``: the schedule slot the job serves (scheduler, catch-up)."""
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
            scheduled_for=scheduled_for,
        )
        await self.save(record)
        await self.redis.zadd(INDEX_KEY, {job_id: record.submitted_at.timestamp()})
        if self.archive is not None:
            self.archive.record(record)
        if self.enqueue is not None:
            await self.enqueue(job_id)
        return record, False

    async def record_skipped(
        self, kind_name: str, slot: datetime, reason: str, *, submitted_by: str = "scheduler (skipped)"
    ) -> JobRecord:
        """Record a schedule slot that did not run: a cancelled job record of the kind with ``scheduled_for`` =
        ``slot`` and the reason (worker not running at the slot, or the previous job of the kind still active)."""
        kind = CATALOG.get(kind_name)
        if kind is None:
            raise UnknownJobKind(kind_name)
        now = self.clock()
        record = JobRecord(
            id=uuid.uuid4().hex,
            kind=kind.name,
            status="cancelled",
            params={},
            input_hash=input_hash(kind.name, {}),
            message=reason,
            submitted_by=submitted_by,
            submitted_at=now,
            finished_at=now,
            error=reason,
            scheduled_for=slot,
        )
        await self.save(record)
        await self.redis.zadd(INDEX_KEY, {record.id: now.timestamp()})
        if self.archive is not None:
            self.archive.record(record)
        return record

    async def finish(
        self,
        job_id: str,
        status: JobState,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        run_s: float | None = None,
    ) -> JobRecord | None:
        changes: dict[str, Any] = {} if run_s is None else {"run_s": run_s}
        record = await self.update(
            job_id,
            status=status,
            finished_at=self.clock(),
            result=result,
            error=error,
            progress=1.0 if status == "done" else None,
            **changes,
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
