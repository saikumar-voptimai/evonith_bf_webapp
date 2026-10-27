"""Shift reports (SV-2.3, #295): ``report.shift`` at 06:20, 14:20, 22:20 IST and ``report.live`` on demand.

report.shift   the shift that ended before the slot (C, A, B), or the ``shift_key`` of the request (a re-run)
report.live    the running shift (``POST /api/reports/live``), data up to now

The report is generated in a thread (``services.reports.generate``: the Reports tab's data, then the LLM narrative with
one retry) and stored in ``twin.shift_report``; without a twin store it is returned in the job result only. A report
whose data cannot be fetched (InfluxDB or Neon unavailable) is tried again after the kind's retry delay; an LLM that
fails twice does not fail the job: the report is stored without narrative (``narrative_status`` failed).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from services import reports
from services.jobs import JobRecord, LiveReportParams, ShiftReportParams
from services.store import twin_store
from worker.runtime import JobContext, TransientJobError, handles


def _store(values: dict[str, Any], job_record: dict[str, Any] | None) -> str | None:
    store = twin_store("twin-worker")
    if store is None:
        return None
    if job_record is not None:  # the report refers to its job; the archive may not have written it yet
        store.jobs.upsert(job_record)
    return str(store.reports.add(values).id)


async def _report(
    job: JobContext, record: JobRecord | None, shift: reports.Shift, *, kind: reports.ReportKind, narrative: bool
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def progress(fraction: float, message: str) -> None:  # called from the generating thread
        asyncio.run_coroutine_threadsafe(job.progress(fraction, message, force=True), loop)

    await job.progress(0.02, f"{'Live report' if kind == 'live' else 'Report'} of shift {shift.key}", force=True)
    try:
        report = await asyncio.to_thread(reports.generate, shift, kind=kind, narrative=narrative, progress=progress)
    except reports.ReportDataError as exc:
        if job.attempt < job.kind.max_tries:
            raise TransientJobError(str(exc)) from exc
        raise
    by_scheduler = record is not None and record.submitted_by.startswith("scheduler")
    values = reports.store_values(
        report,
        job_id=job.job_id if record is not None else None,
        trigger="schedule" if by_scheduler else "user",
        requested_by=record.submitted_by if record is not None else "unknown",
    )
    await job.progress(0.95, "Storing the report", force=True)
    report_id = await asyncio.to_thread(_store, values, record.model_dump() if record is not None else None)
    return reports.job_result(report, report_id)


@handles("report.shift")
async def shift(job: JobContext, params: ShiftReportParams) -> dict[str, Any]:
    record = await job.store.get(job.job_id)
    now = datetime.now(UTC)
    if params.shift_key:
        target = reports.Shift.parse(params.shift_key)
        start, end = (t.astimezone(UTC) for t in (target.timeframe().start_ist, target.timeframe().end_ist))
        if start > now:
            raise ValueError(f"shift {target.key} has not started yet")
        kind: reports.ReportKind = "closed" if end <= now else "live"
    else:
        target = reports.closed_shift(record.scheduled_for if record is not None and record.scheduled_for else now)
        kind = "closed"
    return await _report(job, record, target, kind=kind, narrative=params.narrative)


@handles("report.live")
async def live(job: JobContext, params: LiveReportParams) -> dict[str, Any]:
    record = await job.store.get(job.job_id)
    return await _report(job, record, reports.running_shift(), kind="live", narrative=params.narrative)
