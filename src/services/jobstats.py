"""Runtime statistics per job kind (SV-1.7, #287).

From job records (Redis ``JobRecord`` or ``twin.jobs`` rows) of a period, per kind:

    runs              records of the period (submitted in it), skipped slots excluded
    done, failed, cancelled, skipped
    runtime p50/p95/max, s   execution time of the jobs that ended ``done``, summed over their tries (``run_s``;
                             records without it: finish minus start)
    start delay p95, s       start minus the schedule slot (scheduler jobs) or the submission (other jobs), first try
    last                      finish time and status of the latest job that ended

Percentiles are linear interpolations between order statistics (numpy's default); with fewer than 20 runs the p95 is
close to the maximum. ``GET /api/jobs/stats`` computes them on request from ``twin.jobs`` (Redis without an archive);
the nightly housekeeping job stores the statistics of the last 14 days in its result (``twin.jobs``, one document per
day).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

from services.schedule import SKIPPED_BY, describe, utc


def _get(row: Any, key: str) -> Any:
    return row.get(key) if isinstance(row, Mapping) else getattr(row, key, None)


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


@dataclass
class KindStats:
    kind: str
    title: str
    schedule: str | None
    timeout_s: float | None
    runs: int = 0
    done: int = 0
    failed: int = 0
    cancelled: int = 0
    skipped: int = 0
    running: int = 0
    runtime_p50_s: float | None = None
    runtime_p95_s: float | None = None
    runtime_max_s: float | None = None
    start_delay_p95_s: float | None = None
    last_finished_at: datetime | None = None
    last_status: str | None = None


def runtime_s(row: Any) -> float | None:
    """Execution time of a finished job, s: ``run_s`` (over all tries), else finish minus start."""
    value = _get(row, "run_s")
    if value is not None:
        return float(value)
    started, finished = _get(row, "started_at"), _get(row, "finished_at")
    if started is None or finished is None:
        return None
    return max((utc(finished) - utc(started)).total_seconds(), 0.0)


def start_delay_s(row: Any) -> float | None:
    started = _get(row, "started_at")
    if started is None:
        return None
    ref = _get(row, "scheduled_for") or _get(row, "submitted_at")
    return max((utc(started) - utc(ref)).total_seconds(), 0.0)


def stats(records: Iterable[Any], kinds: Iterable[Any], *, start: datetime, end: datetime) -> dict[str, KindStats]:
    """Statistics of the records submitted in [start, end), per kind of ``kinds`` (catalogue order)."""
    start, end = utc(start), utc(end)
    by_kind: dict[str, list[Any]] = {}
    for row in records:
        submitted = _get(row, "submitted_at")
        if submitted is None or not start <= utc(submitted) < end:
            continue
        by_kind.setdefault(str(_get(row, "kind")), []).append(row)
    out: dict[str, KindStats] = {}
    for kind in kinds:
        rows = by_kind.get(kind.name, [])
        st = KindStats(kind.name, kind.title, describe(kind.schedule), kind.timeout_s)
        runtimes, delays = [], []
        last: Any | None = None
        for row in rows:
            status = _get(row, "status")
            if status == "cancelled" and _get(row, "submitted_by") == SKIPPED_BY:
                st.skipped += 1
                continue
            st.runs += 1
            if status == "done":
                st.done += 1
                if (r := runtime_s(row)) is not None:
                    runtimes.append(r)
            elif status == "failed":
                st.failed += 1
            elif status == "cancelled":
                st.cancelled += 1
            else:
                st.running += 1
            if (d := start_delay_s(row)) is not None:
                delays.append(d)
            finished = _get(row, "finished_at")
            if finished is not None and (last is None or utc(finished) > utc(_get(last, "finished_at"))):
                last = row
        st.runtime_p50_s, st.runtime_p95_s = _pct(runtimes, 50), _pct(runtimes, 95)
        st.runtime_max_s = round(max(runtimes), 3) if runtimes else None
        st.start_delay_p95_s = _pct(delays, 95)
        if last is not None:
            st.last_finished_at, st.last_status = utc(_get(last, "finished_at")), str(_get(last, "status"))
        out[kind.name] = st
    return out


def document(table: Mapping[str, KindStats], *, start: datetime, end: datetime, source: str) -> dict[str, Any]:
    """The statistics as one JSON document (the housekeeping result)."""

    def plain(st: KindStats) -> dict[str, Any]:
        d = asdict(st)
        if d["last_finished_at"] is not None:
            d["last_finished_at"] = d["last_finished_at"].isoformat()
        return d

    return dict(
        start=utc(start).isoformat(),
        end=utc(end).isoformat(),
        source=source,
        kinds=[plain(st) for st in table.values()],
    )


__all__ = ["KindStats", "document", "runtime_s", "start_delay_s", "stats"]
