"""Shift handover record (SV-2.5, #297): ``shift.close`` at 06:10, 14:10, 22:10 IST.

The shift that ended before the slot (C, A, B), or the ``shift_key`` of the request, is frozen into
``twin.shift_record`` (``services.shift_records``): its hourly twin snapshots, the shift window's coke rates and
implied ω, the latest plan decision taken in the shift and the link of its report. While ``twin.evaluate`` has not
stored the shift's window the job tries again 5 min later (to HH:30); the last try freezes what there is. A shift that
has a record is not written again (idempotent per shift); the ``backfill_shifts`` shifts before it without a record are
frozen first. Neon unreachable: tried again like a missing window.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from services import outbox, shift_records
from services.jobs import ShiftCloseParams
from services.store import twin_store
from worker.runtime import JobContext, SkipJob, TransientJobError, handles


def _close(
    target: shift_records.Shift, backfill: int, *, wait: bool, job_id: str | None, submitted_by: str | None
) -> dict[str, Any]:
    store = twin_store("twin-worker")
    if store is None:
        raise SkipJob("no twin store: set BF_TWIN_DB_URL or DATABASE_URL")
    due = shift_records.due_shifts(store, target, backfill)
    if wait and target in due and not shift_records.shift_window_ready(store, target):
        _, end = shift_records.bounds(target)
        raise TransientJobError(f"twin.evaluate has not stored the window of shift {target.key} ending {end:%H:%M} UTC")
    frozen = [shift_records.freeze(store, s, job_id=job_id, submitted_by=submitted_by) for s in due]  # oldest first
    if target not in due:  # recorded before: this run returns it
        frozen.append((store.shifts.get(target.key), False))
    return shift_records.job_result(target, frozen)


@handles("shift.close")
async def close(job: JobContext, params: ShiftCloseParams) -> dict[str, Any]:
    record = await job.store.get(job.job_id)
    now = datetime.now(UTC)
    if params.shift_key:
        target = shift_records.Shift.parse(params.shift_key)
        if shift_records.bounds(target)[1] > now:
            raise ValueError(f"shift {target.key} has not ended yet")
    else:
        target = shift_records.closed_shift(
            record.scheduled_for if record is not None and record.scheduled_for else now
        )
    await job.progress(0.05, f"Handover record of shift {target.key}", force=True)
    try:
        return await asyncio.to_thread(
            _close,
            target,
            params.backfill_shifts,
            wait=job.attempt < job.kind.max_tries,
            job_id=job.job_id if record is not None else None,
            submitted_by=record.submitted_by if record is not None else None,
        )
    except Exception as exc:
        if outbox.is_outage(exc) and job.attempt < job.kind.max_tries:
            raise TransientJobError(f"twin store unreachable: {type(exc).__name__}") from exc
        raise
