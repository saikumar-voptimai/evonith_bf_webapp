"""Hourly twin evaluation (SV-1.7, #287): ``twin.evaluate`` at HH:10.

1. The window W = [HH-8, HH) of the slot (or of the request) needs the hourly row HH-1, which the published dataset
   first holds in its build of about HH:06; without it the local copy is fetched again (``services.twin.
   ensure_dataset``). Still missing: try again in 5 min (to HH:25); the last try evaluates on the rows there are and
   flags every window whose last row is missing, which a later run evaluates again.
2. The windows of the last ``backfill_hours`` that have no snapshot, or were flagged so, are evaluated first (oldest
   first), then W; each is upserted into ``twin.snapshot`` with this job's id.
3. The implied-omega drift alarm runs on the stored series (``services.drift.run``: ``alarm.raised`` /
   ``alarm.cleared``), then the Live Twin alarms (``services.twin_alarms.run``: STRATA - TRACE gap, stale sources,
   implied lag; SV-1.9), and W is published as ``twin.snapshot``.

The evaluation runs in a thread (the Coke Energy sources and the pipeline are memoised in the worker process). Without
a twin store the window is evaluated and returned in the job result only.

Outages (SV-6.5): a source that failed within ``source_health.RETRY_AFTER_S`` skips the job with the reason
(``SkipJob``); a loader that fails marks its source and skips the job as well. The snapshots go through the buffered
store (``services.outbox``): while Neon is unreachable they are kept in the local outbox and replayed by the heartbeat,
the backfill lookup, the drift and the alarms are skipped with a note, and the job still ends ``done``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from services import coke_energy as ce
from services import drift, outbox, source_health, twin, twin_alarms
from services.jobs import TwinEvaluateParams
from utils.burden_engine import strata_network
from worker.runtime import JobContext, SkipJob, TransientJobError, handles

BACKFILL_SOLVERS_FROM = 3  # windows from which their network solves run in solver processes
SOURCES = ("hourly", "influxdb", "hm_slag", "dpr")  # the evaluation's sources; a down one skips the job


def _outage_note(step: str, exc: BaseException) -> str:
    return f"{step} skipped: twin store unreachable ({type(exc).__name__}: {str(exc).splitlines()[0][:200]})"


@handles("twin.evaluate")
async def evaluate(job: JobContext, params: TwinEvaluateParams) -> dict[str, Any]:
    record = await job.store.get(job.job_id)
    end = twin.target_window_end(params.window_end, record.scheduled_for if record is not None else None)
    try:
        source_health.HEALTH.require(*SOURCES)
    except source_health.SourceDown as exc:
        raise SkipJob(exc.reason) from exc
    await job.progress(0.02, f"Checking the hourly dataset for the row of {end - twin.HOUR:%H:%M}", force=True)
    try:
        ready, data = await asyncio.to_thread(twin.ensure_dataset, end)
    except Exception as exc:
        source_health.HEALTH.record_failure("hourly", exc)
        raise SkipJob(source_health.HEALTH.reason("hourly") or f"hourly dataset: {exc}") from exc
    if not ready and job.attempt < job.kind.max_tries:
        raise TransientJobError(
            f"the hourly dataset ends at {data.data_end} (build {data.built_at}); the window ending {end:%d %b %H:%M} "
            f"needs the row of {end - twin.HOUR:%H:%M}"
        )
    notes: list[str] = []
    store = await asyncio.to_thread(outbox.buffered_twin_store, "twin-worker")
    windows = [end]
    if store is not None:
        if store.pending and not store.holding:  # Neon may be back: the queued writes first (SV-6.5)
            await asyncio.to_thread(store.replay)
        try:
            windows = await asyncio.to_thread(twin.due_windows, store, end, params.backfill_hours)
        except Exception as exc:
            if not outbox.is_outage(exc):
                raise
            notes.append(_outage_note("backfill lookup", exc))  # the missing windows are found by a later run
    if store is not None and record is not None:  # the snapshots refer to their job; the archive may lag
        await asyncio.to_thread(store.jobs.upsert, record.model_dump())
    job_id = job.job_id if record is not None else None
    results: list[twin.StoredWindow] = []
    # A backfill of several windows solves each window's three STRATA states in solver processes (SV-1.8); for one
    # window the start of the processes would cost about what it saves. Starting and stopping them blocks (a thread);
    # the executor holds for this job's context, which the evaluation threads copy, not for other jobs.
    solvers = min(3, strata_network.solver_count()) if params.strata and len(windows) >= BACKFILL_SOLVERS_FROM else 0
    pool = await asyncio.to_thread(strata_network.start_solvers, solvers)
    try:
        with strata_network.parallel_solves(pool):
            for i, w in enumerate(windows):
                await job.progress(
                    0.05 + 0.85 * i / len(windows),
                    f"Window {i + 1} of {len(windows)}: {w - 8 * twin.HOUR:%d %b %H:%M} to {w:%H:%M}",
                )
                try:
                    results.append(
                        await asyncio.to_thread(
                            twin.evaluate_and_store,
                            store,
                            w,
                            job_id=job_id,
                            last_row_pending=not data.holds(w - twin.HOUR),  # a backfilled window may lack its row
                            strata=params.strata,
                        )
                    )
                except source_health.SourceDown as exc:
                    raise SkipJob(exc.reason) from exc
    finally:
        await asyncio.to_thread(strata_network.stop_solvers, pool)
    latest = results[-1].values
    alarms: list[str] = []
    buffered = store.pending if store is not None else 0
    if store is not None and buffered:
        notes.append(f"{buffered} write{'' if buffered == 1 else 's'} held in the local outbox until Neon returns")
    if store is not None and not buffered:
        try:
            await job.progress(0.95, "Implied-ω drift alarm", force=True)
            report = await drift.run(store, job.bus, drift.DriftConfig.from_config(ce.config()))
            alarms = [f"{d} open" for d in report.open_episodes()]
            await job.progress(0.97, "Live Twin alarms", force=True)
            changes = await twin_alarms.run(store, job.bus, ce.config())
            alarms += [f"{k}{'/' + sub if sub else ''} {event.removeprefix('alarm.')}" for event, k, sub in changes]
        except Exception as exc:
            if not outbox.is_outage(exc):
                raise
            notes.append(_outage_note("drift and alarms", exc))
    elif store is not None:
        notes.append("drift and alarms skipped: twin store unreachable (writes buffered)")
    await job.bus.publish("twin.snapshot", twin.snapshot_event(latest))
    return {
        "window_end": end.isoformat(),
        "windows": [twin.headline(r.values) for r in results],
        "backfilled": len(results) - 1,
        "last_row_pending": not ready,
        "dataset": data.as_dict(),
        "drift": alarms,
        "stored": store is not None,
        "buffered": buffered,
        "notes": [*results[-1].notes, *notes],
    }
