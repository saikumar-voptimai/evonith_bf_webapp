"""Scheduled jobs: slots, the catch-up at worker start, and the audit of the schedule (SV-1.7, #287).

A kind's schedule is ``(hours, minutes)`` in IST (``services.jobs.JobKind.schedule``; hours None = every hour). The
worker's cron entry submits a job of the kind at every slot with ``submitted_by="scheduler"`` and ``scheduled_for`` =
the slot (``slot_of``: from arq's run time of the entry). A slot is *served* when a scheduler record carries it:

    scheduler              the cron entry at the slot
    scheduler (catch-up)   submitted at worker start for the latest slot, when the worker started within ``CATCH_UP``
                           (30 min) of it
    scheduler (skipped)    a cancelled record of a slot that did not run: the worker was not running at the slot (or
                           started more than 30 min after the latest one), or the previous job of the kind was still
                           queued or running (``JobStore.submit`` returns it instead of a second job)

``catch_up_plan(kinds, records, now, since=...)``: the slots of each scheduled kind in (``since``, now] that are not
served, ``since`` being the worker's last heartbeat before it stopped (Redis ``twin:worker:alive``). The latest of
them is submitted when it is at most ``CATCH_UP`` old, every other one is recorded as skipped (at most
``MAX_SKIPPED`` per kind, newest first). Without a heartbeat (first start) only the latest slot is considered, and it
is not recorded as skipped when too old.

``audit(records, kinds, start, end)``: for every slot in [start, end) whether a scheduler job served it and how that job
ended (done, failed, cancelled, still running, skipped) or that no record exists (missed). Records written before
``scheduled_for`` existed (migration twin_0002) are matched to the first slot at or before their submission, within
the kind's slot interval and ``CATCH_UP``.

A slot the worker could not serve because it was not running is *off*, not missed or skipped: a skipped record whose
reason starts with ``OFF_PREFIX`` (written by the catch-up), or a slot without a record inside an outage of the worker
(``outages``: from its last activity before a start, which the catch-up records mark, to that start; with
``down_since``, the worker's last heartbeat while it is down now, also from then to now). Missed and skipped then
count only what went wrong while the worker ran (``SlotAudit.clean``). Acceptance of #287: 14 consecutive days without
a missed, skipped or off slot (``scripts/schedule_audit.py``, ``GET /api/jobs/schedule``).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
CATCH_UP = timedelta(minutes=30)
MAX_SKIPPED = 300
HEARTBEAT_MARGIN = timedelta(seconds=15)  # the worker's heartbeat interval (10 s) and some
SETTLE = timedelta(minutes=1)  # the audit leaves out the slots of the last minute: their records may not exist yet
SCHEDULER = "scheduler"
CATCH_UP_BY = "scheduler (catch-up)"
SKIPPED_BY = "scheduler (skipped)"
#: Reason of a slot the catch-up records as skipped because the worker was not running (``catch_up_plan``).
OFF_PREFIX = "Skipped: the worker was not running"

Schedule = tuple[set[int] | None, set[int]]


def utc(value: Any) -> datetime:
    """An aware UTC datetime from a datetime (naive = UTC, as SQLite returns it) or an ISO string."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _get(row: Any, key: str) -> Any:
    return row.get(key) if isinstance(row, Mapping) else getattr(row, key, None)


def is_scheduler(row: Any) -> bool:
    return str(_get(row, "submitted_by") or "").startswith(SCHEDULER)


# ------------------------------------------------------------------------------------------------------------ slots
def slots(schedule: Schedule, start: datetime, end: datetime) -> list[datetime]:
    """The slots in [start, end), aware UTC, in order."""
    hours, minutes = schedule
    start, end = utc(start), utc(end)
    if not start < end:
        return []
    out = []
    day = start.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    last_day = end.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    hour_list = sorted(hours) if hours is not None else range(24)
    minute_list = sorted(minutes)
    while day <= last_day:
        for h in hour_list:
            for m in minute_list:
                t = day.replace(hour=h, minute=m).astimezone(UTC)
                if start <= t < end:
                    out.append(t)
        day += timedelta(days=1)  # IST has no daylight saving: every day has 24 h
    return out


def last_slot(schedule: Schedule, at: datetime) -> datetime | None:
    """The latest slot at or before ``at`` (within two days)."""
    at = utc(at)
    found = slots(schedule, at - timedelta(days=2), at + timedelta(microseconds=1))
    return found[-1] if found else None


def next_slot(schedule: Schedule, at: datetime) -> datetime | None:
    """The first slot after ``at``."""
    at = utc(at)
    found = slots(schedule, at + timedelta(microseconds=1), at + timedelta(days=2))
    return found[0] if found else None


def interval(schedule: Schedule) -> timedelta:
    """The shortest time between two consecutive slots."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    found = slots(schedule, start, start + timedelta(days=2))
    gaps = [b - a for a, b in zip(found, found[1:], strict=False)]
    return min(gaps) if gaps else timedelta(days=1)


def describe(schedule: Schedule | None) -> str | None:
    """'04:00', '06:15, 14:15, 22:15'; for every hour 'every hour at :07' or 'every 5 min'."""
    if schedule is None:
        return None
    hours, minutes = schedule
    mm = sorted(minutes)
    if hours is None:
        step = mm[1] - mm[0] if len(mm) > 1 else 60
        if len(mm) > 1 and 60 % step == 0 and mm == list(range(mm[0], 60, step)):
            return f"every {step} min" + (f" from :{mm[0]:02d}" if mm[0] else "")
        return "every hour at " + ", ".join(f":{m:02d}" for m in mm)
    return ", ".join(f"{h:02d}:{m:02d}" for h in sorted(hours) for m in mm)


def slot_of(schedule: Schedule, run_at: datetime) -> datetime | None:
    """The slot a cron run at ``run_at`` serves: the latest slot at or before it (arq runs the entry at second 0.12 of
    the slot, later when the worker is busy)."""
    return last_slot(schedule, utc(run_at) + timedelta(seconds=2))


# ------------------------------------------------------------------------------------------------------- matching
def served(records: Iterable[Any], kind: str, schedule: Schedule) -> dict[datetime, Any]:
    """Slot -> the scheduler record serving it, for one kind: a job that ran (or runs) before a skipped record of the
    same slot, else the latest by submission."""
    tolerance = min(CATCH_UP, interval(schedule))
    out: dict[datetime, Any] = {}
    rows = sorted(
        (r for r in records if _get(r, "kind") == kind and is_scheduler(r)), key=lambda r: utc(_get(r, "submitted_at"))
    )
    for row in rows:
        slot = _get(row, "scheduled_for")
        if slot is not None:
            slot = utc(slot)
        else:  # before twin_0002: the first slot at or before the submission
            submitted = utc(_get(row, "submitted_at"))
            slot = last_slot(schedule, submitted)
            if slot is None or submitted - slot >= tolerance or slot in out:
                continue
        held = out.get(slot)
        if held is None or _get(held, "submitted_by") == SKIPPED_BY or _get(row, "submitted_by") != SKIPPED_BY:
            out[slot] = row
    return out


# ------------------------------------------------------------------------------------------------------- catch-up
@dataclass(frozen=True)
class Decision:
    kind: str
    slot: datetime
    action: str  # submit | skip
    reason: str


def _ist(t: datetime) -> str:
    return f"{t.astimezone(IST):%d %b %H:%M} IST"


def catch_up_plan(
    kinds: Iterable[Any],
    records: Iterable[Any],
    now: datetime,
    *,
    since: datetime | None,
    window: timedelta = CATCH_UP,
    max_skipped: int = MAX_SKIPPED,
) -> list[Decision]:
    """What the worker does at start for the slots it did not serve (see the module notes)."""
    now = utc(now)
    records = list(records)
    out: list[Decision] = []
    for kind in kinds:
        if kind.schedule is None:
            continue
        done = served(records, kind.name, kind.schedule)
        newest = last_slot(kind.schedule, now)
        if since is None:
            todo = [newest] if newest is not None and newest not in done and now - newest <= window else []
        else:  # from the last heartbeat (less one heartbeat interval: the worker may have stopped just after it)
            start = max(utc(since) - HEARTBEAT_MARGIN, now - timedelta(days=7))
            todo = [t for t in slots(kind.schedule, start, now + timedelta(microseconds=1)) if t not in done]
        if not todo:
            continue
        *older, latest = todo
        if latest == newest and now - latest <= window:  # only the latest slot of a kind runs late
            out.append(
                Decision(kind.name, latest, "submit", f"missed by {(now - latest).total_seconds() / 60:.0f} min")
            )
        else:
            older.append(latest)
        for slot in sorted(older, reverse=True)[:max_skipped]:
            late = (now - slot).total_seconds() / 60
            rule = (
                f"only the latest slot of a kind, {_ist(newest)}, runs at start"
                if newest is not None and slot < newest and now - slot <= window
                else f"a missed slot runs at start only within {window.total_seconds() / 60:.0f} min"
            )
            out.append(
                Decision(
                    kind.name,
                    slot,
                    "skip",
                    f"{OFF_PREFIX} at {_ist(slot)}; it started {late:.0f} min later ({rule}).",
                )
            )
    return out


# -------------------------------------------------------------------------------------------------------- outages
EPOCH = datetime(2000, 1, 1, tzinfo=UTC)
Interval = tuple[datetime, datetime]


def _reason(row: Any) -> str:
    return str(_get(row, "error") or _get(row, "message") or "")


def is_off_skip(row: Any) -> bool:
    """A record of a slot the catch-up skipped because the worker was not running."""
    return _get(row, "submitted_by") == SKIPPED_BY and _reason(row).startswith(OFF_PREFIX)


def activity_times(records: Iterable[Any]) -> list[datetime]:
    """Times the worker is known to have run, in order: a job started, or the scheduler submitted a job (its cron entry,
    the catch-up's late run, or a slot skipped because the previous job was still active). The catch-up's records of
    slots skipped while the worker was off are written at its next start for earlier slots and do not count."""
    out = []
    for row in records:
        started = _get(row, "started_at")
        if started is not None:
            out.append(utc(started))
        if is_scheduler(row) and not is_off_skip(row):
            out.append(utc(_get(row, "submitted_at")))
    return sorted(out)


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def outages(
    records: Iterable[Any], *, now: datetime | None = None, down_since: datetime | None = None
) -> list[Interval]:
    """Periods the worker was not running, from its job records, merged and in order.

    Every start after a stop leaves records submitted at the start (the catch-up's late run of a slot, or slots it
    records as skipped with ``OFF_PREFIX``); the outage before it began at the worker's last activity
    (``activity_times``) before the start, or at ``EPOCH`` without one. ``down_since`` (the last heartbeat of a worker
    that is down now) adds the period from then, or from the last activity when that is later, to ``now``."""
    records = list(records)
    acts = activity_times(records)
    starts = sorted(
        {utc(_get(r, "submitted_at")) for r in records if _get(r, "submitted_by") == CATCH_UP_BY or is_off_skip(r)}
    )
    found = []
    for start in starts:
        i = bisect_left(acts, start - SETTLE)  # the catch-up's own records are at the start
        found.append((acts[i - 1] if i else EPOCH, start))
    if down_since is not None and now is not None:
        last = max([utc(down_since), *acts[-1:]])
        if last < utc(now):
            found.append((last, utc(now)))
    return _merge(found)


def within(t: datetime, intervals: Sequence[Interval]) -> bool:
    """``t`` inside one of the (merged, ordered) intervals, ends excluded."""
    i = bisect_right(intervals, (t, datetime.max.replace(tzinfo=UTC))) - 1
    return i >= 0 and intervals[i][0] < t < intervals[i][1]


# ---------------------------------------------------------------------------------------------------------- audit
SLOT_STATES = ("done", "failed", "cancelled", "running", "skipped", "missed", "off")


@dataclass
class SlotAudit:
    kind: str
    title: str = ""
    schedule: str = ""
    expected: int = 0
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SLOT_STATES, 0))
    problems: list[dict[str, Any]] = field(default_factory=list)  # slots not done: slot, state, job, reason

    @property
    def clean(self) -> bool:
        """No slot missed or skipped while the worker ran (failed runs are runs, and slots while the worker was off
        are off; both are counted separately)."""
        return self.counts["missed"] == 0 and self.counts["skipped"] == 0

    def as_dict(self, limit: int = 50) -> dict[str, Any]:
        return dict(
            kind=self.kind,
            title=self.title,
            schedule=self.schedule,
            expected=self.expected,
            **self.counts,
            clean=self.clean,
            problems=self.problems[-limit:],
        )


def slot_state(row: Any | None) -> str:
    if row is None:
        return "missed"
    status = _get(row, "status")
    if status == "cancelled" and _get(row, "submitted_by") == SKIPPED_BY:
        return "off" if is_off_skip(row) else "skipped"
    if status in ("queued", "running"):
        return "running"
    return str(status)


def audit(
    records: Iterable[Any],
    kinds: Iterable[Any],
    start: datetime,
    end: datetime,
    *,
    now: datetime | None = None,
    down_since: datetime | None = None,
) -> dict[str, SlotAudit]:
    """Per scheduled kind: the slots in [start, end) and what became of each (see the module notes). Slots less than
    ``SETTLE`` before ``now`` are not counted (their job may not be recorded yet). A slot without a record inside an
    outage of the worker (``outages``; ``down_since`` with ``now``: the worker is down now) is off, not missed."""
    records = list(records)
    off = outages(records, now=now, down_since=down_since)
    end = min(utc(end), utc(now) - SETTLE) if now is not None else utc(end)
    out: dict[str, SlotAudit] = {}
    for kind in kinds:
        if kind.schedule is None:
            continue
        done = served(records, kind.name, kind.schedule)
        res = SlotAudit(kind.name, getattr(kind, "title", ""), describe(kind.schedule) or "")
        for slot in slots(kind.schedule, start, end):
            res.expected += 1
            row = done.get(slot)
            state = slot_state(row)
            if state == "missed" and within(slot, off):
                state = "off"
            res.counts[state] += 1
            if state != "done":
                res.problems.append(
                    dict(
                        slot=slot,
                        state=state,
                        job_id=_get(row, "id") if row is not None else None,
                        reason=(_get(row, "error") or _get(row, "message")) if row is not None else None,
                    )
                )
        out[kind.name] = res
    return out


__all__ = [
    "CATCH_UP",
    "CATCH_UP_BY",
    "IST",
    "MAX_SKIPPED",
    "OFF_PREFIX",
    "SCHEDULER",
    "SETTLE",
    "SKIPPED_BY",
    "SLOT_STATES",
    "Decision",
    "SlotAudit",
    "activity_times",
    "audit",
    "catch_up_plan",
    "describe",
    "interval",
    "is_off_skip",
    "is_scheduler",
    "last_slot",
    "next_slot",
    "outages",
    "served",
    "slot_of",
    "slot_state",
    "slots",
    "utc",
    "within",
]
