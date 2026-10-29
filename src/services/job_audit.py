"""Audited job starts (SV-4.6, #309): a user's start of a job kind marked ``audited`` in ``services.jobs.CATALOG``
(``calibration.nightly``) writes an entry to the audit log, action ``job.start``, object ``job`` / the job id.

The job is queued first (Redis), then the entry is recorded (``twin.audit_log``, its own transaction: a job start
writes nothing else in the twin store). When the entry cannot be written, the new job is cancelled before it starts
(the worker skips a job with ``cancel_requested``) and the error is raised: a job of an audited kind does not run
without its entry. A request that returns a job already queued or running (``duplicate``) is recorded too, with the
job's fields before and after. No Streamlit (tests/test_services_architecture.py).

Job control (SV-4.8, #311): every cancel (``job.cancel``; ``POST /api/jobs/{id}/cancel`` and the admin view) and
every retry (``job.retry``: the kind and parameters of a failed or cancelled job queued again by a user) is recorded
the same way. A cancel writes its entry first (before = the job, after = the job as the cancel leaves it) and changes
the job only once the entry is written; a retry is a start with ``details.retry_of``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from data.twin.repositories import Actor, TwinStore
from services import schedule
from services.audit import json_safe
from services.auth import has_role, normalise_role
from services.jobs import CATALOG, JobRecord, JobStore, UnknownJobKind

logger = logging.getLogger(__name__)

ACTION = "job.start"
CANCEL_ACTION = "job.cancel"
RETRY_ACTION = "job.retry"
OBJECT_TYPE = "job"
#: Fields of the job the entry records.
FIELDS = ("id", "kind", "status", "params", "submitted_by", "submitted_at", "scheduled_for")
#: Fields a cancel records: the start fields and the state the cancel changes.
CONTROL_FIELDS = (*FIELDS, "started_at", "tries", "cancel_requested", "error")
NOT_STARTED = "Not started: the audit log could not be written."
CANCELLED_BEFORE_START = "Cancelled before start."
ACTIVE = ("queued", "running")
RETRYABLE = ("failed", "cancelled")


class JobNotFound(LookupError):
    """No record of the job (Redis keeps 7 days; a retry also reads ``twin.jobs``)."""


class JobConflict(ValueError):
    """The job's state does not allow the action (a cancel of an ended job, a retry of one that did not fail)."""

    def __init__(self, message: str, status: str) -> None:
        super().__init__(message)
        self.status = status


class RoleRefused(PermissionError):
    """The actor's role is below the kind's."""


@dataclass(frozen=True)
class AuditedStart:
    record: JobRecord
    duplicate: bool
    audit_id: int


@dataclass(frozen=True)
class AuditedCancel:
    record: JobRecord
    #: The job was queued and ended here (``cancelled`` at once); a running job ends when the worker stops it.
    ended: bool
    audit_id: int


def job_fields(record: Any, fields: tuple[str, ...] = FIELDS) -> dict[str, Any]:
    return json_safe({name: getattr(record, name) for name in fields})


async def submit(
    jobs: JobStore,
    store: TwinStore,
    actor: Actor,
    kind: str,
    params: dict[str, Any] | None = None,
    *,
    origin: str,
    action: str = ACTION,
    details: dict[str, Any] | None = None,
) -> AuditedStart:
    """Queue ``kind`` for ``actor`` and record the start; ``origin`` names the view or route that started it;
    ``action`` and ``details`` (merged into the entry's) mark a retry."""
    job_kind = CATALOG.get(kind)
    if job_kind is None:
        raise UnknownJobKind(kind)
    record, duplicate = await jobs.submit(kind, dict(params or {}), actor.name)
    fields = job_fields(record)
    context = {"kind": kind, "title": job_kind.title, "duplicate": duplicate, "origin": origin, **(details or {})}
    try:
        entry = await asyncio.to_thread(
            store.audit.record,
            actor,
            action,
            OBJECT_TYPE,
            record.id,
            context,
            before=fields if duplicate else None,
            after=fields,
        )
    except Exception:
        if not duplicate:
            try:
                await jobs.update(record.id, cancel_requested=True)
                await jobs.finish(record.id, "cancelled", error=NOT_STARTED)
            except Exception:  # noqa: BLE001 - the audit error is the one to report
                logger.exception(
                    "job %s (%s) is queued without its audit entry and could not be cancelled", record.id, kind
                )
        raise
    return AuditedStart(record=record, duplicate=duplicate, audit_id=int(entry.id))


async def cancel(jobs: JobStore, store: TwinStore, actor: Actor, job_id: str, *, origin: str) -> AuditedCancel:
    """Cancel a queued or running job for ``actor``: the entry (``job.cancel``) is written first, then the job is
    marked (``cancel_requested``: the worker stops a running job and skips a queued one) and a queued job is ended
    (``cancelled``). An entry that cannot be written leaves the job as it was. Raises ``JobNotFound`` and
    ``JobConflict`` (the job has ended)."""
    record = await jobs.get(job_id)
    if record is None:
        raise JobNotFound(job_id)
    if record.status not in ACTIVE:
        raise JobConflict(f"The job has already ended ({record.status}).", record.status)
    queued = record.status == "queued"
    ending = {"status": "cancelled", "error": CANCELLED_BEFORE_START} if queued else {}
    planned = record.model_copy(update={"cancel_requested": True, **ending})
    kind = CATALOG.get(record.kind)
    details = {
        "kind": record.kind,
        "title": kind.title if kind is not None else record.kind,
        "status": record.status,
        "origin": origin,
    }
    entry = await asyncio.to_thread(
        store.audit.record,
        actor,
        CANCEL_ACTION,
        OBJECT_TYPE,
        record.id,
        details,
        before=job_fields(record, CONTROL_FIELDS),
        after=job_fields(planned, CONTROL_FIELDS),
    )
    await jobs.update(job_id, cancel_requested=True)
    ended = await jobs.finish(job_id, "cancelled", error=CANCELLED_BEFORE_START) if queued else None
    current = await jobs.get(job_id)
    return AuditedCancel(record=current or ended or planned, ended=ended is not None, audit_id=int(entry.id))


async def retry(jobs: JobStore, store: TwinStore, actor: Actor, job_id: str, *, origin: str) -> AuditedStart:
    """Queue the kind and parameters of a failed or cancelled job again for ``actor`` (``job.retry``, with
    ``retry_of``). The record is read from Redis, else from ``twin.jobs``. A scheduler job's retry is a user's start:
    its parameters (often none) take their defaults at the time of the retry, not at the slot. Raises ``JobNotFound``,
    ``JobConflict`` (neither failed nor cancelled), ``UnknownJobKind`` (the kind left the catalogue),
    ``RoleRefused`` and ``pydantic.ValidationError`` (parameters the kind no longer accepts)."""
    record: Any = await jobs.get(job_id)
    if record is None:
        record = await asyncio.to_thread(store.jobs.get, job_id)
    if record is None:
        raise JobNotFound(job_id)
    if record.status not in RETRYABLE:
        raise JobConflict(f"Only a failed or cancelled job is retried; this one is {record.status}.", record.status)
    kind = CATALOG.get(record.kind)
    if kind is None:
        raise UnknownJobKind(record.kind)
    role = normalise_role(actor.role)
    if role is None or not has_role(role, kind.role):
        raise RoleRefused(f"Starting {kind.title} needs the {kind.role} role; signed in as {actor.role}.")
    details = {
        "retry_of": record.id,
        "retry_of_status": record.status,
        "retry_of_scheduled_for": json_safe(schedule.utc(record.scheduled_for)) if record.scheduled_for else None,
    }
    return await submit(
        jobs, store, actor, kind.name, dict(record.params or {}), origin=origin, action=RETRY_ACTION, details=details
    )


__all__ = [
    "ACTION",
    "ACTIVE",
    "CANCELLED_BEFORE_START",
    "CANCEL_ACTION",
    "CONTROL_FIELDS",
    "FIELDS",
    "NOT_STARTED",
    "OBJECT_TYPE",
    "RETRYABLE",
    "RETRY_ACTION",
    "AuditedCancel",
    "AuditedStart",
    "JobConflict",
    "JobNotFound",
    "RoleRefused",
    "cancel",
    "job_fields",
    "retry",
    "submit",
]
