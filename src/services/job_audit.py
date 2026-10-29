"""Audited job starts (SV-4.6, #309): a user's start of a job kind marked ``audited`` in ``services.jobs.CATALOG``
(``calibration.nightly``) writes an entry to the audit log, action ``job.start``, object ``job`` / the job id.

The job is queued first (Redis), then the entry is recorded (``twin.audit_log``, its own transaction: a job start
writes nothing else in the twin store). When the entry cannot be written, the new job is cancelled before it starts
(the worker skips a job with ``cancel_requested``) and the error is raised: a job of an audited kind does not run
without its entry. A request that returns a job already queued or running (``duplicate``) is recorded too, with the
job's fields before and after. No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from data.twin.repositories import Actor, TwinStore
from services.audit import json_safe
from services.jobs import CATALOG, JobRecord, JobStore, UnknownJobKind

logger = logging.getLogger(__name__)

ACTION = "job.start"
OBJECT_TYPE = "job"
#: Fields of the job the entry records.
FIELDS = ("id", "kind", "status", "params", "submitted_by", "submitted_at", "scheduled_for")
NOT_STARTED = "Not started: the audit log could not be written."


@dataclass(frozen=True)
class AuditedStart:
    record: JobRecord
    duplicate: bool
    audit_id: int


def job_fields(record: JobRecord) -> dict[str, Any]:
    return json_safe({name: getattr(record, name) for name in FIELDS})


async def submit(
    jobs: JobStore,
    store: TwinStore,
    actor: Actor,
    kind: str,
    params: dict[str, Any] | None = None,
    *,
    origin: str,
) -> AuditedStart:
    """Queue ``kind`` for ``actor`` and record the start; ``origin`` names the view or route that started it."""
    job_kind = CATALOG.get(kind)
    if job_kind is None:
        raise UnknownJobKind(kind)
    record, duplicate = await jobs.submit(kind, dict(params or {}), actor.name)
    fields = job_fields(record)
    details = {"kind": kind, "title": job_kind.title, "duplicate": duplicate, "origin": origin}
    try:
        entry = await asyncio.to_thread(
            store.audit.record,
            actor,
            ACTION,
            OBJECT_TYPE,
            record.id,
            details,
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


__all__ = ["ACTION", "FIELDS", "NOT_STARTED", "OBJECT_TYPE", "AuditedStart", "job_fields", "submit"]
