"""Repositories of the ``twin`` schema (SV-0.9, #278).

``TwinStore`` holds one repository per table. Each call runs in its own transaction and returns detached rows
(readable after the session closes). User actions (plan decision, fact-register proposal and review, alarm
acknowledgement, a new version of the operator inputs) write their audit row in the same transaction as the change
(``audit_row``); admin writes do so through ``services.audit.admin_write``, which also records the object's values
before and after (SV-4.1). Times passed in must be timezone-aware; they are stored in UTC.

The store is synchronous (SQLAlchemy 2.0 with psycopg2); async callers run it in a thread (``asyncio.to_thread``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from data.twin.engine import build_engine, resolve_twin_url, session_factory
from data.twin.models import (
    DECISIONS,
    OPERATOR_INPUT_ORIGINS,
    SCHEMA,
    SEVERITIES,
    AlarmEvent,
    AuditLog,
    FactsVersion,
    FmConversation,
    FmFeedback,
    FmMessage,
    Job,
    OperatorInputs,
    OptimiserCommentary,
    OptimiserRun,
    PlanDecision,
    ShiftRecord,
    ShiftReport,
    Snapshot,
    TwinBase,
    WhatIfCase,
)

M = TypeVar("M", bound=TwinBase)

SHIFT_KEY = re.compile(r"^\d{4}-\d{2}-\d{2}_([ABC])$")
JOB_FIELDS = (
    "id",
    "kind",
    "status",
    "params",
    "input_hash",
    "submitted_by",
    "submitted_at",
    "started_at",
    "finished_at",
    "tries",
    "message",
    "result",
    "error",
    "scheduled_for",
    "run_s",
)


class TwinStoreError(RuntimeError):
    """Base of the store's refusals."""


class NotFound(TwinStoreError):
    pass


class AlreadyRecorded(TwinStoreError):
    """The row exists and is written once (shift record)."""


class ReviewRefused(TwinStoreError):
    """A fact-register version that is not pending, or a status change reviewed by its own author."""


class VersionConflict(TwinStoreError):
    """An edit of the operator inputs that started from a version which is no longer the current one."""

    def __init__(self, base_version: int, current_version: int) -> None:
        super().__init__(
            f"the edit started from version {base_version}, but version {current_version} is current: "
            "reload the operator inputs and apply the change again"
        )
        self.base_version = base_version
        self.current_version = current_version


@dataclass(frozen=True)
class Actor:
    """The user behind an action, as the audit log records it."""

    name: str
    role: str | None = None
    client_ip: str | None = None


def utcnow() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | None, name: str = "time") -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got {value!r}")
    return value.astimezone(UTC)


def _stored_utc(value: datetime) -> datetime:
    """A time read back from the store: aware on PostgreSQL, naive UTC on SQLite."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _columns(model: type[TwinBase]) -> set[str]:
    return {c.key for c in model.__table__.columns}


def _checked(model: type[TwinBase], values: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(values) - _columns(model)
    if unknown:
        raise ValueError(f"{model.__tablename__} has no column {', '.join(sorted(unknown))}")
    out = dict(values)
    for column in model.__table__.columns:
        if column.key in out and isinstance(out[column.key], datetime):
            out[column.key] = _utc(out[column.key], column.key)
    return out


def _upsert(session: Session, model: type[TwinBase], values: dict[str, Any], keys: Sequence[str]) -> None:
    """INSERT ... ON CONFLICT (keys) DO UPDATE on PostgreSQL and SQLite."""
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg = pg_insert(model).values(**values)
        changes = {c: pg.excluded[c] for c in values if c not in keys}
        session.execute(pg.on_conflict_do_update(index_elements=list(keys), set_=changes))
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        lite = sqlite_insert(model).values(**values)
        changes = {c: lite.excluded[c] for c in values if c not in keys}
        session.execute(lite.on_conflict_do_update(index_elements=list(keys), set_=changes))
    else:  # pragma: no cover - the store runs on PostgreSQL (Neon) and SQLite (outage buffer)
        raise NotImplementedError(f"upsert on {dialect}")


def audit_row(
    session: Session,
    actor: Actor,
    action: str,
    object_type: str,
    object_id: str | None,
    details: Mapping[str, Any] | None,
    at: datetime,
    *,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> AuditLog:
    """Add the audit row of an action to ``session``, the transaction of the change it records. ``before`` and
    ``after`` are the object's fields around an admin write (JSON-safe values, ``services.audit.json_safe``)."""
    row = AuditLog(
        at=at,
        actor=actor.name,
        role=actor.role,
        action=action,
        object_type=object_type,
        object_id=object_id,
        details=dict(details or {}),
        client_ip=actor.client_ip,
        before=None if before is None else dict(before),
        after=None if after is None else dict(after),
    )
    session.add(row)
    return row


def overlay_hash(overlay: Mapping[str, Any]) -> str:
    """SHA-256 of the overlay in canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(overlay, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _leaves(doc: Any, prefix: str = "") -> dict[str, Any]:
    """``{"a.b.c": value}`` of a JSON document; lists and scalars are leaves."""
    if isinstance(doc, Mapping) and doc:
        out: dict[str, Any] = {}
        for key, value in doc.items():
            out.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: doc} if prefix else {}


def document_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The leaves that differ between two JSON documents, by path: ``{"path", "before", "after"}`` (None = absent)."""
    old, new = _leaves(before), _leaves(after)
    return [
        {"path": path, "before": old.get(path), "after": new.get(path)}
        for path in sorted(old.keys() | new.keys())
        if old.get(path) != new.get(path) or (path in old) != (path in new)
    ]


class _Repository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions


class JobRepository(_Repository):
    """Job records mirrored from Redis (``services.jobs.JobRecord`` fields; progress and cancel flag excluded)."""

    def upsert(self, record: Mapping[str, Any], *, at: datetime | None = None) -> None:
        self.upsert_many([record], at=at)

    def upsert_many(self, records: Iterable[Mapping[str, Any]], *, at: datetime | None = None) -> int:
        stamp = _utc(at) or utcnow()
        count = 0
        with self._sessions.begin() as session:
            for record in records:
                values = _checked(Job, {k: record[k] for k in JOB_FIELDS if k in record})
                values.setdefault("params", {})
                values.setdefault("tries", 0)
                values["archived_at"] = stamp
                _upsert(session, Job, values, ("id",))
                count += 1
        return count

    def get(self, job_id: str) -> Job | None:
        with self._sessions() as session:
            return session.get(Job, job_id)

    def window(self, since: datetime, until: datetime, *, kinds: Sequence[str] | None = None) -> list[Job]:
        """Jobs submitted in [since, until) or serving a schedule slot in it, oldest first by submission (the schedule
        audit and the runtime statistics read a period with this)."""
        s, u = _utc(since, "since"), _utc(until, "until")
        query = (
            select(Job)
            .where(
                or_(
                    and_(Job.submitted_at >= s, Job.submitted_at < u),
                    and_(Job.scheduled_for >= s, Job.scheduled_for < u),
                )
            )
            .order_by(Job.submitted_at)
        )
        if kinds:
            query = query.where(Job.kind.in_(list(kinds)))
        with self._sessions() as session:
            return list(session.scalars(query))

    def recent(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """Newest first by submission time."""
        query = select(Job).order_by(Job.submitted_at.desc()).limit(limit)
        if kind:
            query = query.where(Job.kind == kind)
        if status:
            query = query.where(Job.status == status)
        if since:
            query = query.where(Job.submitted_at >= _utc(since, "since"))
        if until:
            query = query.where(Job.submitted_at < _utc(until, "until"))
        with self._sessions() as session:
            return list(session.scalars(query))


class SnapshotRepository(_Repository):
    """Hourly twin evaluations, one per window end. A re-evaluation of a window replaces its row entirely: a value the
    new evaluation does not give is cleared, never left over from the earlier one."""

    def upsert(self, values: Mapping[str, Any]) -> Snapshot:
        data = _checked(Snapshot, values)
        data.pop("id", None)
        for key in ("window_start", "window_end", "evaluated_at", "status"):
            if data.get(key) is None:
                raise ValueError(f"snapshot needs {key}")
        row: dict[str, Any] = {c.key: None for c in Snapshot.__table__.columns if c.key != "id"}
        row.update({"quantities": {}, "flags": {}, "versions": {}})
        row.update(data)
        with self._sessions.begin() as session:
            _upsert(session, Snapshot, row, ("window_end",))
            return session.scalars(select(Snapshot).where(Snapshot.window_end == data["window_end"])).one()

    def get(self, window_end: datetime) -> Snapshot | None:
        with self._sessions() as session:
            query = select(Snapshot).where(Snapshot.window_end == _utc(window_end, "window_end"))
            return session.scalars(query).one_or_none()

    def latest(self) -> Snapshot | None:
        with self._sessions() as session:
            return session.scalars(select(Snapshot).order_by(Snapshot.window_end.desc()).limit(1)).one_or_none()

    def series(self, start: datetime, end: datetime) -> list[Snapshot]:
        """Snapshots with ``start <= window_end <= end``, oldest first."""
        query = (
            select(Snapshot)
            .where(Snapshot.window_end >= _utc(start, "start"), Snapshot.window_end <= _utc(end, "end"))
            .order_by(Snapshot.window_end)
        )
        with self._sessions() as session:
            return list(session.scalars(query))


class OptimiserRunRepository(_Repository):
    """Optimiser runs and the supervisors' decisions on them."""

    def add(self, values: Mapping[str, Any]) -> OptimiserRun:
        data = _checked(OptimiserRun, values)
        data["id"] = uuid.UUID(str(data["id"])) if data.get("id") else uuid.uuid4()
        with self._sessions.begin() as session:
            run = OptimiserRun(**data)
            session.add(run)
        return run

    def get(self, run_id: uuid.UUID | str) -> OptimiserRun | None:
        with self._sessions() as session:
            return session.get(OptimiserRun, uuid.UUID(str(run_id)))

    def latest(self, *, mode: str | None = None, status: str | None = "ok") -> OptimiserRun | None:
        query = select(OptimiserRun).order_by(OptimiserRun.started_at.desc()).limit(1)
        if mode:
            query = query.where(OptimiserRun.mode == mode)
        if status:
            query = query.where(OptimiserRun.status == status)
        with self._sessions() as session:
            return session.scalars(query).one_or_none()

    def recent(
        self,
        *,
        mode: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
    ) -> list[OptimiserRun]:
        """Newest first by start time."""
        query = select(OptimiserRun).order_by(OptimiserRun.started_at.desc()).limit(limit)
        if mode:
            query = query.where(OptimiserRun.mode == mode)
        if since:
            query = query.where(OptimiserRun.started_at >= _utc(since, "since"))
        if until:
            query = query.where(OptimiserRun.started_at < _utc(until, "until"))
        with self._sessions() as session:
            return list(session.scalars(query))

    def decide(
        self,
        run_id: uuid.UUID | str,
        decision: str,
        actor: Actor,
        *,
        at: datetime | None = None,
        shift_key: str | None = None,
        comment: str | None = None,
    ) -> PlanDecision:
        """Record ``accepted`` or ``rejected`` with its audit row; a later decision on the same run supersedes it."""
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {', '.join(DECISIONS)}")
        when = _utc(at) or utcnow()
        key = uuid.UUID(str(run_id))
        with self._sessions.begin() as session:
            if session.get(OptimiserRun, key) is None:
                raise NotFound(f"optimiser run {key}")
            row = PlanDecision(
                run_id=key,
                decision=decision,
                decided_by=actor.name,
                decided_at=when,
                shift_key=shift_key,
                comment=comment,
            )
            session.add(row)
            audit_row(
                session,
                actor,
                "plan.decide",
                "optimiser_run",
                str(key),
                {"decision": decision, "shift_key": shift_key, "comment": comment},
                when,
            )
        return row

    def decision(self, run_id: uuid.UUID | str) -> PlanDecision | None:
        """The decision that applies: the latest."""
        query = (
            select(PlanDecision)
            .where(PlanDecision.run_id == uuid.UUID(str(run_id)))
            .order_by(PlanDecision.decided_at.desc(), PlanDecision.id.desc())
            .limit(1)
        )
        with self._sessions() as session:
            return session.scalars(query).one_or_none()

    def decisions(self, run_id: uuid.UUID | str) -> list[PlanDecision]:
        """Every decision on the run, oldest first."""
        query = (
            select(PlanDecision)
            .where(PlanDecision.run_id == uuid.UUID(str(run_id)))
            .order_by(PlanDecision.decided_at, PlanDecision.id)
        )
        with self._sessions() as session:
            return list(session.scalars(query))

    def add_commentary(self, values: Mapping[str, Any]) -> OptimiserCommentary:
        """Store a commentary on a run (``NotFound`` when the run does not exist)."""
        data = _checked(OptimiserCommentary, values)
        data["run_id"] = uuid.UUID(str(data["run_id"]))
        with self._sessions.begin() as session:
            if session.get(OptimiserRun, data["run_id"]) is None:
                raise NotFound(f"optimiser run {data['run_id']}")
            row = OptimiserCommentary(**data)
            session.add(row)
        return row

    def commentary(self, run_id: uuid.UUID | str, *, status: str | None = None) -> OptimiserCommentary | None:
        """The latest commentary on the run (with ``status``: the latest of that status)."""
        query = (
            select(OptimiserCommentary)
            .where(OptimiserCommentary.run_id == uuid.UUID(str(run_id)))
            .order_by(OptimiserCommentary.generated_at.desc())
            .limit(1)
        )
        if status:
            query = query.where(OptimiserCommentary.status == status)
        with self._sessions() as session:
            return session.scalars(query).one_or_none()


class ShiftRecordRepository(_Repository):
    """Shift handover records: written once at the close of the shift, never changed."""

    def freeze(self, values: Mapping[str, Any]) -> ShiftRecord:
        data = _checked(ShiftRecord, values)
        if data.get("plan_run_id"):
            data["plan_run_id"] = uuid.UUID(str(data["plan_run_id"]))
        key = str(data.get("shift_key", ""))
        match = SHIFT_KEY.match(key)
        if not match:
            raise ValueError(f"shift_key must read YYYY-MM-DD_A|B|C, got {key!r}")
        data.setdefault("letter", match.group(1))
        if data["letter"] != match.group(1):
            raise ValueError(f"letter {data['letter']} does not match shift_key {key}")
        data.setdefault("closed_at", utcnow())
        try:
            with self._sessions.begin() as session:
                row = ShiftRecord(**data)
                session.add(row)
        except IntegrityError as exc:
            with self._sessions() as session:
                exists = session.get(ShiftRecord, key) is not None
            if exists:
                raise AlreadyRecorded(f"shift record {key} exists and cannot be changed") from exc
            raise
        return row

    def get(self, shift_key: str) -> ShiftRecord | None:
        with self._sessions() as session:
            return session.get(ShiftRecord, shift_key)

    def between(self, start: datetime, end: datetime) -> list[ShiftRecord]:
        """Records of shifts starting in [start, end), oldest first."""
        query = (
            select(ShiftRecord)
            .where(ShiftRecord.shift_start >= _utc(start, "start"), ShiftRecord.shift_start < _utc(end, "end"))
            .order_by(ShiftRecord.shift_start)
        )
        with self._sessions() as session:
            return list(session.scalars(query))


class ShiftReportRepository(_Repository):
    """Shift reports (SV-2.3): one row per generated report; a re-run of a shift adds a row, the latest applies."""

    def add(self, values: Mapping[str, Any]) -> ShiftReport:
        data = _checked(ShiftReport, values)
        data["id"] = uuid.UUID(str(data["id"])) if data.get("id") else uuid.uuid4()
        key = str(data.get("shift_key", ""))
        match = SHIFT_KEY.match(key)
        if not match:
            raise ValueError(f"shift_key must read YYYY-MM-DD_A|B|C, got {key!r}")
        data.setdefault("letter", match.group(1))
        if data["letter"] != match.group(1):
            raise ValueError(f"letter {data['letter']} does not match shift_key {key}")
        with self._sessions.begin() as session:
            row = ShiftReport(**data)
            session.add(row)
        return row

    def get(self, report_id: uuid.UUID | str) -> ShiftReport | None:
        with self._sessions() as session:
            return session.get(ShiftReport, uuid.UUID(str(report_id)))

    def latest(self, shift_key: str, *, kind: str | None = None) -> ShiftReport | None:
        """The latest report of the shift (of ``kind`` when given)."""
        query = select(ShiftReport).where(ShiftReport.shift_key == shift_key)
        if kind:
            query = query.where(ShiftReport.kind == kind)
        with self._sessions() as session:
            return session.scalars(query.order_by(ShiftReport.generated_at.desc()).limit(1)).one_or_none()

    def recent(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        letter: str | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[ShiftReport]:
        """Reports of shifts starting in [start, end), newest shift first and, within a shift, newest report first."""
        query = (
            select(ShiftReport).order_by(ShiftReport.shift_start.desc(), ShiftReport.generated_at.desc()).limit(limit)
        )
        if start:
            query = query.where(ShiftReport.shift_start >= _utc(start, "start"))
        if end:
            query = query.where(ShiftReport.shift_start < _utc(end, "end"))
        if letter:
            query = query.where(ShiftReport.letter == letter)
        if kind:
            query = query.where(ShiftReport.kind == kind)
        with self._sessions() as session:
            return list(session.scalars(query))


class AlarmRepository(_Repository):
    """Alarms from raise to clear; at most one open alarm per kind and subject."""

    def _open(self, session: Session, kind: str, subject: str) -> AlarmEvent | None:
        query = select(AlarmEvent).where(
            AlarmEvent.kind == kind, AlarmEvent.subject == subject, AlarmEvent.cleared_at.is_(None)
        )
        return session.scalars(query).one_or_none()

    def raise_alarm(
        self,
        kind: str,
        *,
        severity: str,
        message: str,
        subject: str = "",
        details: Mapping[str, Any] | None = None,
        at: datetime | None = None,
    ) -> tuple[AlarmEvent, bool]:
        """Open an alarm, or update the open one of the same kind and subject (severity, message, details).
        Returns the alarm and whether it was newly raised."""
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {', '.join(SEVERITIES)}")
        when = _utc(at) or utcnow()
        for _ in range(2):  # a concurrent raise of the same alarm wins the unique index; then update that one
            try:
                with self._sessions.begin() as session:
                    current = self._open(session, kind, subject)
                    if current is not None:
                        current.severity = severity
                        current.message = message
                        if details is not None:
                            current.details = dict(details)
                        return current, False
                    row = AlarmEvent(
                        kind=kind,
                        subject=subject,
                        severity=severity,
                        message=message,
                        details=dict(details or {}),
                        raised_at=when,
                    )
                    session.add(row)
                return row, True
            except IntegrityError:
                continue
        raise TwinStoreError(f"alarm {kind}/{subject} could not be raised")

    def clear_alarm(self, kind: str, *, subject: str = "", at: datetime | None = None) -> AlarmEvent | None:
        """Close the open alarm of this kind and subject; None when none is open."""
        with self._sessions.begin() as session:
            current = self._open(session, kind, subject)
            if current is None:
                return None
            # Never before the raise (the table's check): a clear stamped by a clock slightly behind the raiser's.
            current.cleared_at = max(_utc(at) or utcnow(), _stored_utc(current.raised_at))
            return current

    def acknowledge(self, alarm_id: int, actor: Actor, *, at: datetime | None = None) -> AlarmEvent:
        when = _utc(at) or utcnow()
        with self._sessions.begin() as session:
            row = session.get(AlarmEvent, alarm_id)
            if row is None:
                raise NotFound(f"alarm {alarm_id}")
            row.acknowledged_by = actor.name
            row.acknowledged_at = when
            audit_row(session, actor, "alarm.acknowledge", "alarm_event", str(alarm_id), {"kind": row.kind}, when)
            return row

    def open_alarms(self) -> list[AlarmEvent]:
        """Open alarms, newest first."""
        query = select(AlarmEvent).where(AlarmEvent.cleared_at.is_(None)).order_by(AlarmEvent.raised_at.desc())
        with self._sessions() as session:
            return list(session.scalars(query))

    def history(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[AlarmEvent]:
        """Alarms raised in [since, until), newest first."""
        query = select(AlarmEvent).order_by(AlarmEvent.raised_at.desc()).limit(limit)
        if kind:
            query = query.where(AlarmEvent.kind == kind)
        if since:
            query = query.where(AlarmEvent.raised_at >= _utc(since, "since"))
        if until:
            query = query.where(AlarmEvent.raised_at < _utc(until, "until"))
        with self._sessions() as session:
            return list(session.scalars(query))


class FactsVersionRepository(_Repository):
    """Versioned overlay of the fact register: proposals and their review."""

    def propose(
        self,
        *,
        base_version: str,
        overlay: Mapping[str, Any],
        actor: Actor,
        second_review: bool = False,
        parent_version: int | None = None,
        note: str | None = None,
        at: datetime | None = None,
    ) -> FactsVersion:
        """A pending version numbered after the highest one; ``second_review`` when it changes a fact's status."""
        when = _utc(at) or utcnow()
        for _ in range(3):  # two proposals at once: the second takes the next number
            try:
                with self._sessions.begin() as session:
                    number = (session.scalar(select(func.max(FactsVersion.version))) or 0) + 1
                    row = FactsVersion(
                        version=number,
                        base_version=base_version,
                        parent_version=parent_version,
                        status="pending",
                        overlay=dict(overlay),
                        content_hash=overlay_hash(overlay),
                        second_review=second_review,
                        note=note,
                        created_by=actor.name,
                        created_at=when,
                    )
                    session.add(row)
                    audit_row(
                        session,
                        actor,
                        "facts.propose",
                        "facts_version",
                        str(number),
                        {"base_version": base_version, "facts": sorted(overlay), "second_review": second_review},
                        when,
                    )
                return row
            except IntegrityError:
                continue
        raise TwinStoreError("no version number could be assigned")

    def review(
        self,
        version: int,
        *,
        approve: bool,
        actor: Actor,
        note: str | None = None,
        at: datetime | None = None,
    ) -> FactsVersion:
        """Approve or reject a pending version. A version that changes a fact's status needs a second administrator."""
        when = _utc(at) or utcnow()
        with self._sessions.begin() as session:
            row = session.scalars(select(FactsVersion).where(FactsVersion.version == version)).one_or_none()
            if row is None:
                raise NotFound(f"fact-register version {version}")
            if row.status != "pending":
                raise ReviewRefused(f"version {version} is {row.status}, not pending")
            if row.second_review and actor.name == row.created_by:
                raise ReviewRefused(f"version {version} changes a fact status: a second administrator must review it")
            row.status = "approved" if approve else "rejected"
            row.reviewed_by = actor.name
            row.reviewed_at = when
            audit_row(
                session,
                actor,
                "facts.review",
                "facts_version",
                str(version),
                {"status": row.status, "note": note},
                when,
            )
            return row

    def latest_approved(self, base_version: str | None = None) -> FactsVersion | None:
        """The approved version with the highest number; with ``base_version`` only among those made against it."""
        query = (
            select(FactsVersion).where(FactsVersion.status == "approved").order_by(FactsVersion.version.desc()).limit(1)
        )
        if base_version is not None:
            query = query.where(FactsVersion.base_version == base_version)
        with self._sessions() as session:
            return session.scalars(query).one_or_none()

    def get(self, version: int) -> FactsVersion | None:
        with self._sessions() as session:
            return session.scalars(select(FactsVersion).where(FactsVersion.version == version)).one_or_none()

    def recent(self, *, limit: int = 50) -> list[FactsVersion]:
        """Newest version first."""
        with self._sessions() as session:
            return list(session.scalars(select(FactsVersion).order_by(FactsVersion.version.desc()).limit(limit)))


@dataclass(frozen=True)
class Seed:
    """The operator inputs in force before the first version (the YAML), recorded as version 1 on an empty table."""

    inputs: Mapping[str, Any]
    valid_from: datetime
    note: str | None = None


#: Changes listed in the details of one ``operator_inputs.update`` audit row (the total is always given).
AUDIT_CHANGES_MAX = 200
#: Transactions one save may start: each retry follows a concurrent save that got in first.
WRITE_ATTEMPTS = 5


class OperatorInputsRepository(_Repository):
    """Versions of the operator inputs (``operator_inputs``, SCD-2): the current one has no ``valid_upto``.

    ``write`` applies a change to the current document and stores the result as the next version, closing the current
    one at the same instant, with its audit row (``operator_inputs.update``, the changed values in the details) in the
    same transaction; a change that leaves the document as it is writes nothing. On an empty table the ``seed`` is
    recorded first as version 1 (origin ``seed``, audit ``operator_inputs.seed``), so the history starts from what was
    in force before. Two saves at once: the second waits for the first (row lock on PostgreSQL), then reads the first
    one's result in a new transaction and applies its change to it, or, with a ``base_version``, fails with
    ``VersionConflict`` (tests/test_twin_store.py, two threads on PostgreSQL).
    """

    def current(self) -> OperatorInputs | None:
        with self._sessions() as session:
            return session.scalars(select(OperatorInputs).where(OperatorInputs.valid_upto.is_(None))).one_or_none()

    def current_version(self) -> int | None:
        """Number of the current version (one indexed row): the cache token of the readers."""
        with self._sessions() as session:
            query = select(OperatorInputs.version).where(OperatorInputs.valid_upto.is_(None))
            return session.scalars(query).one_or_none()

    def at(self, when: datetime) -> OperatorInputs | None:
        """The version valid at ``when`` (valid_from <= when < valid_upto); None before the first version."""
        t = _utc(when, "when")
        query = (
            select(OperatorInputs)
            .where(
                OperatorInputs.valid_from <= t,
                or_(OperatorInputs.valid_upto.is_(None), OperatorInputs.valid_upto > t),
            )
            .order_by(OperatorInputs.valid_from.desc())
            .limit(1)
        )
        with self._sessions() as session:
            return session.scalars(query).one_or_none()

    def get(self, version: int) -> OperatorInputs | None:
        with self._sessions() as session:
            return session.scalars(select(OperatorInputs).where(OperatorInputs.version == version)).one_or_none()

    def history(self, *, limit: int = 50) -> list[OperatorInputs]:
        """Newest version first."""
        query = select(OperatorInputs).order_by(OperatorInputs.version.desc()).limit(limit)
        with self._sessions() as session:
            return list(session.scalars(query))

    def write(
        self,
        change: Callable[[dict[str, Any]], Mapping[str, Any]],
        actor: Actor,
        *,
        origin: str,
        note: str | None = None,
        base_version: int | None = None,
        seed: Callable[[], Seed | None] | None = None,
        at: datetime | None = None,
    ) -> tuple[OperatorInputs, bool]:
        """Store ``change(current inputs)`` as the next version. Returns the version in force afterwards and whether
        a version was written (also true when only the seed was).

        ``base_version``: the version the edit started from (0: no version stored yet, the seed); when another one is
        current, ``VersionConflict``.
        ``seed``: called only on an empty table. ``at``: the save time (default now); a version never starts before
        the one it closes."""
        if origin not in OPERATOR_INPUT_ORIGINS or origin == "seed":
            raise ValueError("origin must be streamlit or api")
        when = _utc(at) or utcnow()
        for _ in range(WRITE_ATTEMPTS):  # a concurrent save got in first: start again from its result
            try:
                with self._sessions.begin() as session:
                    query = select(OperatorInputs).order_by(OperatorInputs.version.desc()).limit(1)
                    latest = session.scalars(query.with_for_update()).one_or_none()
                    if latest is not None and latest.valid_upto is not None and self._newer(session, latest.version):
                        # This save waited for the row lock of a concurrent one, which closed the version: READ
                        # COMMITTED hands back that row, not the new one. The next statement sees its result.
                        continue
                    seeded = False
                    if latest is None and seed is not None:
                        recorded = seed()
                        if recorded is not None:
                            latest, seeded = self._seed(session, recorded, actor, when), True
                    current = latest.version if latest is not None else 0
                    # base_version 0: the edit started from the seed (the YAML), no version was stored yet.
                    if base_version is not None and base_version != (0 if seeded else current):
                        raise VersionConflict(base_version, current)
                    before = dict(latest.inputs) if latest is not None else {}
                    after = dict(change(copy.deepcopy(before)))
                    digest = overlay_hash(after)
                    if latest is not None and digest == latest.content_hash:
                        return latest, seeded
                    start = when
                    if latest is not None:
                        start = max(when, _stored_utc(latest.valid_from) + timedelta(microseconds=1))
                        if latest.valid_upto is None:  # closed by hand (no current version): nothing to close
                            latest.valid_upto = start
                            session.flush()  # closed before the next version opens: one current version
                    row = OperatorInputs(
                        version=current + 1,
                        valid_from=start,
                        author=actor.name,
                        origin=origin,
                        note=note,
                        inputs=after,
                        content_hash=digest,
                    )
                    session.add(row)
                    changes = document_changes(before, after)
                    details = {
                        "version": row.version,
                        "previous_version": current or None,
                        "origin": origin,
                        "note": note,
                        "changes": changes[:AUDIT_CHANGES_MAX],
                        "changes_total": len(changes),
                    }
                    audit_row(
                        session, actor, "operator_inputs.update", "operator_inputs", str(row.version), details, start
                    )
                return row, True
            except IntegrityError:
                continue
        raise TwinStoreError("no version number could be assigned to the operator inputs")

    @staticmethod
    def _newer(session: Session, version: int) -> bool:
        """A version above ``version`` is committed (a new statement: a fresh snapshot under READ COMMITTED)."""
        newest = session.scalar(select(func.max(OperatorInputs.version)))
        return newest is not None and newest > version

    @staticmethod
    def _seed(session: Session, seed: Seed, actor: Actor, when: datetime) -> OperatorInputs:
        start = min(_utc(seed.valid_from, "valid_from") or when, when - timedelta(milliseconds=1))
        row = OperatorInputs(
            version=1,
            valid_from=start,
            author="seed",
            origin="seed",
            note=seed.note,
            inputs=dict(seed.inputs),
            content_hash=overlay_hash(seed.inputs),
        )
        session.add(row)
        session.flush()
        details = {"version": 1, "origin": "seed", "valid_from": start.isoformat(), "note": seed.note}
        audit_row(session, actor, "operator_inputs.seed", "operator_inputs", "1", details, when)
        return row


class AuditRepository(_Repository):
    """Append-only log of admin writes and plan decisions."""

    def record(
        self,
        actor: Actor,
        action: str,
        object_type: str,
        object_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        *,
        at: datetime | None = None,
        before: Mapping[str, Any] | None = None,
        after: Mapping[str, Any] | None = None,
    ) -> AuditLog:
        """An entry in its own transaction (an action without a write in the twin store); writes record theirs
        through ``services.audit.admin_write``."""
        with self._sessions.begin() as session:
            return audit_row(
                session,
                actor,
                action,
                object_type,
                object_id,
                details,
                _utc(at) or utcnow(),
                before=before,
                after=after,
            )

    @staticmethod
    def _filtered(
        query: Any,
        *,
        actor: str | None = None,
        action: str | None = None,
        object_type: str | None = None,
        object_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Any:
        if actor:
            query = query.where(AuditLog.actor == actor)
        if action:
            query = query.where(AuditLog.action == action)
        if object_type:
            query = query.where(AuditLog.object_type == object_type)
        if object_id:
            query = query.where(AuditLog.object_id == object_id)
        if since:
            query = query.where(AuditLog.at >= _utc(since, "since"))
        if until:
            query = query.where(AuditLog.at < _utc(until, "until"))
        return query

    def query(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        object_type: str | None = None,
        object_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 200,
    ) -> list[AuditLog]:
        """Entries in [since, until), newest first."""
        filters = dict(
            actor=actor, action=action, object_type=object_type, object_id=object_id, since=since, until=until
        )
        query = self._filtered(select(AuditLog), **filters).order_by(AuditLog.at.desc(), AuditLog.id.desc())
        with self._sessions() as session:
            return list(session.scalars(query.limit(limit)))

    def page(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        object_type: str | None = None,
        object_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditLog], int]:
        """One page of the entries in [since, until), newest first, and the number of entries that match."""
        if limit < 1 or offset < 0:
            raise ValueError(f"limit must be >= 1 and offset >= 0, got {limit} and {offset}")
        filters = dict(
            actor=actor, action=action, object_type=object_type, object_id=object_id, since=since, until=until
        )
        rows = self._filtered(select(AuditLog), **filters).order_by(AuditLog.at.desc(), AuditLog.id.desc())
        count = self._filtered(select(func.count()).select_from(AuditLog), **filters)
        with self._sessions() as session:
            total = int(session.scalar(count) or 0)
            return list(session.scalars(rows.limit(limit).offset(offset))), total

    def facets(self) -> dict[str, list[str]]:
        """The actors, actions and object types present in the log, sorted (the viewer's filter menus)."""
        columns = {"actors": AuditLog.actor, "actions": AuditLog.action, "object_types": AuditLog.object_type}
        with self._sessions() as session:
            return {key: sorted(session.scalars(select(column).distinct())) for key, column in columns.items()}


class FurnaceMindRepository(_Repository):
    """Conversations and messages of the FurnaceMind agent service (SV-5.1, #312), and the feedback on its answers
    (SV-5.2, #313).

    ``append`` numbers the messages of a conversation 1, 2, ... in the same transaction that advances the
    conversation's ``message_count`` (row lock on PostgreSQL), so two writers cannot take the same number.
    ``rate`` writes the feedback on an answer with its audit row in one transaction.
    """

    TITLE_CHARS = 200

    def create(
        self,
        *,
        owner: str,
        reasoning_level: str,
        owner_id: str | None = None,
        title: str = "",
        meta: Mapping[str, Any] | None = None,
        at: datetime | None = None,
    ) -> FmConversation:
        now = _utc(at) or utcnow()
        row = FmConversation(
            id=uuid.uuid4(),
            owner=owner,
            owner_id=owner_id,
            title=title[: self.TITLE_CHARS],
            reasoning_level=reasoning_level,
            created_at=now,
            updated_at=now,
            message_count=0,
            meta=dict(meta or {}),
        )
        with self._sessions.begin() as session:
            session.add(row)
        return row

    def get(self, conversation_id: uuid.UUID | str) -> FmConversation | None:
        with self._sessions() as session:
            return session.get(FmConversation, uuid.UUID(str(conversation_id)))

    def recent(self, owner: str, *, limit: int = 30) -> list[FmConversation]:
        """The owner's conversations, most recently updated first."""
        query = (
            select(FmConversation)
            .where(FmConversation.owner == owner)
            .order_by(FmConversation.updated_at.desc(), FmConversation.created_at.desc())
            .limit(limit)
        )
        with self._sessions() as session:
            return list(session.scalars(query))

    def messages(self, conversation_id: uuid.UUID | str, *, last: int | None = None) -> list[FmMessage]:
        """Messages in order (``seq`` ascending); with ``last``, only the last ``last`` of them."""
        key = uuid.UUID(str(conversation_id))
        query = select(FmMessage).where(FmMessage.conversation_id == key)
        with self._sessions() as session:
            if last is None:
                return list(session.scalars(query.order_by(FmMessage.seq)))
            rows = list(session.scalars(query.order_by(FmMessage.seq.desc()).limit(last)))
            return rows[::-1]

    def append(
        self,
        conversation_id: uuid.UUID | str,
        *,
        role: str,
        content: str,
        parts: Sequence[Mapping[str, Any]] = (),
        status: str = "complete",
        model: str | None = None,
        turn_id: str | None = None,
        meta: Mapping[str, Any] | None = None,
        at: datetime | None = None,
    ) -> FmMessage:
        """Add the next message; the first user message also titles an untitled conversation."""
        key = uuid.UUID(str(conversation_id))
        now = _utc(at) or utcnow()
        with self._sessions.begin() as session:
            conversation = session.scalars(
                select(FmConversation).where(FmConversation.id == key).with_for_update()
            ).one_or_none()
            if conversation is None:
                raise NotFound(f"no FurnaceMind conversation {key}")
            seq = conversation.message_count + 1
            row = FmMessage(
                conversation_id=key,
                seq=seq,
                role=role,
                content=content,
                parts=[dict(p) for p in parts],
                status=status,
                model=model,
                turn_id=turn_id,
                created_at=now,
                meta=dict(meta or {}),
            )
            session.add(row)
            conversation.message_count = seq
            conversation.updated_at = now
            if role == "user" and not conversation.title:
                conversation.title = " ".join(content.split())[: self.TITLE_CHARS]
        return row

    def feedback(self, conversation_id: uuid.UUID | str) -> list[FmFeedback]:
        """The feedback on the conversation's answers, by ``seq``."""
        key = uuid.UUID(str(conversation_id))
        query = select(FmFeedback).where(FmFeedback.conversation_id == key).order_by(FmFeedback.seq)
        with self._sessions() as session:
            return list(session.scalars(query))

    def rate(
        self,
        conversation_id: uuid.UUID | str,
        seq: int,
        actor: Actor,
        *,
        rating: str,
        comment: str = "",
        author_id: str | None = None,
        at: datetime | None = None,
    ) -> FmFeedback:
        """Save the feedback on answer ``seq`` (created, or changed in place) with its audit row ``fm.feedback``
        (the rating and comment before and after) in the same transaction. ``NotFound`` when the conversation has no
        assistant message ``seq``."""
        key = uuid.UUID(str(conversation_id))
        now = _utc(at) or utcnow()
        with self._sessions.begin() as session:
            # Two saves at once: the conversation's row lock (PostgreSQL) makes the second wait, then change the row.
            session.scalars(select(FmConversation.id).where(FmConversation.id == key).with_for_update()).one_or_none()
            role = session.scalars(
                select(FmMessage.role).where(FmMessage.conversation_id == key, FmMessage.seq == seq)
            ).one_or_none()
            if role != "assistant":
                raise NotFound(f"no answer {seq} in FurnaceMind conversation {key}")
            row = session.scalars(
                select(FmFeedback).where(FmFeedback.conversation_id == key, FmFeedback.seq == seq)
            ).one_or_none()
            before = None if row is None else {"rating": row.rating, "comment": row.comment}
            if row is None:
                row = FmFeedback(conversation_id=key, seq=seq, author=actor.name, created_at=now)
                session.add(row)
            row.author = actor.name
            row.author_id = author_id
            row.rating = rating
            row.comment = comment
            row.updated_at = now
            audit_row(
                session,
                actor,
                "fm.feedback",
                "fm_message",
                f"{key}/{seq}",
                {"conversation_id": str(key), "seq": seq},
                now,
                before=before,
                after={"rating": rating, "comment": comment},
            )
        return row


class WhatIfRepository(_Repository):
    """Saved what-if cases of the admin space (SV-4.3, #306): reads by owner and the worker's result.

    Saves and deletes are admin writes with their audit row (``services.whatif.save_case``, ``delete_case`` through
    ``services.audit.admin_write``); ``store_result`` is the worker's and writes no audit row."""

    def list(self, owner: str, kind: str | None = None, *, limit: int = 100) -> list[WhatIfCase]:
        """The owner's cases, most recently changed first."""
        query = select(WhatIfCase).where(WhatIfCase.owner == owner)
        if kind is not None:
            query = query.where(WhatIfCase.kind == kind)
        query = query.order_by(WhatIfCase.updated_at.desc(), WhatIfCase.name).limit(max(1, limit))
        with self._sessions() as session:
            return list(session.scalars(query))

    def get(self, case_id: uuid.UUID | str) -> WhatIfCase | None:
        try:
            key = uuid.UUID(str(case_id))
        except ValueError:
            return None
        with self._sessions() as session:
            return session.get(WhatIfCase, key)

    def store_result(
        self,
        case_id: uuid.UUID | str,
        *,
        owner: str,
        inputs: Mapping[str, Any],
        result: Mapping[str, Any],
        job_id: str | None,
        facts_version: str | None,
        at: datetime | None = None,
    ) -> bool:
        """Store a run's result with the case when the case is ``owner``'s and still holds ``inputs`` (the case the
        run solved); False otherwise (deleted, another owner's, or changed since the run started)."""
        try:
            key = uuid.UUID(str(case_id))
        except ValueError:
            return False
        with self._sessions.begin() as session:
            row = session.get(WhatIfCase, key, with_for_update=True)
            if row is None or row.owner != owner or dict(row.inputs or {}) != dict(inputs):
                return False
            row.result = dict(result)
            row.result_job_id = job_id
            row.result_at = _utc(at) or utcnow()
            row.facts_version = facts_version
            return True


class TwinStore:
    """The repositories of the ``twin`` schema over one engine."""

    def __init__(self, sessions: sessionmaker[Session], engine: Engine | None = None) -> None:
        self.sessions = sessions
        self.engine = engine
        self.jobs = JobRepository(sessions)
        self.snapshots = SnapshotRepository(sessions)
        self.optimiser = OptimiserRunRepository(sessions)
        self.shifts = ShiftRecordRepository(sessions)
        self.reports = ShiftReportRepository(sessions)
        self.alarms = AlarmRepository(sessions)
        self.facts = FactsVersionRepository(sessions)
        self.operator_inputs = OperatorInputsRepository(sessions)
        self.audit = AuditRepository(sessions)
        self.fm = FurnaceMindRepository(sessions)
        self.whatif = WhatIfRepository(sessions)

    @classmethod
    def from_url(cls, url: str | None = None, *, application_name: str = "twin") -> TwinStore | None:
        """Store on ``url`` (default ``BF_TWIN_DB_URL``, then ``DATABASE_URL``); None when no URL is configured.
        No connection is opened here."""
        resolved = resolve_twin_url(url)
        if resolved is None:
            return None
        engine = build_engine(resolved, application_name=application_name)
        return cls(session_factory(engine), engine)

    def revision(self) -> str | None:
        """Alembic revision of the schema (``twin.alembic_version``), None when the schema is not migrated.
        A database that cannot be reached raises (``OperationalError``)."""
        sqlite = self.sessions.kw["bind"].dialect.name == "sqlite"
        table = "alembic_version" if sqlite else f"{SCHEMA}.alembic_version"
        try:
            with self.sessions() as session:
                return session.execute(text(f"SELECT version_num FROM {table}")).scalar_one_or_none()
        except ProgrammingError:  # PostgreSQL: undefined table
            return None
        except OperationalError:  # SQLite: no such table
            if sqlite:
                return None
            raise

    def dispose(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
