"""Repositories of the ``twin`` schema (SV-0.9, #278).

``TwinStore`` holds one repository per table. Each call runs in its own transaction and returns detached rows
(readable after the session closes). User actions (plan decision, fact-register proposal and review, alarm
acknowledgement) write their audit row in the same transaction as the change. Times passed in must be timezone-aware;
they are stored in UTC.

The store is synchronous (SQLAlchemy 2.0 with psycopg2); async callers run it in a thread (``asyncio.to_thread``).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from data.twin.engine import build_engine, resolve_twin_url, session_factory
from data.twin.models import (
    DECISIONS,
    SCHEMA,
    SEVERITIES,
    AlarmEvent,
    AuditLog,
    FactsVersion,
    Job,
    OptimiserRun,
    PlanDecision,
    ShiftRecord,
    Snapshot,
    TwinBase,
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
)


class TwinStoreError(RuntimeError):
    """Base of the store's refusals."""


class NotFound(TwinStoreError):
    pass


class AlreadyRecorded(TwinStoreError):
    """The row exists and is written once (shift record)."""


class ReviewRefused(TwinStoreError):
    """A fact-register version that is not pending, or a status change reviewed by its own author."""


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


def _audit(
    session: Session,
    actor: Actor,
    action: str,
    object_type: str,
    object_id: str | None,
    details: Mapping[str, Any] | None,
    at: datetime,
) -> AuditLog:
    row = AuditLog(
        at=at,
        actor=actor.name,
        role=actor.role,
        action=action,
        object_type=object_type,
        object_id=object_id,
        details=dict(details or {}),
        client_ip=actor.client_ip,
    )
    session.add(row)
    return row


def overlay_hash(overlay: Mapping[str, Any]) -> str:
    """SHA-256 of the overlay in canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(overlay, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


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
            _audit(
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
            _audit(session, actor, "alarm.acknowledge", "alarm_event", str(alarm_id), {"kind": row.kind}, when)
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
                    _audit(
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
            _audit(
                session,
                actor,
                "facts.review",
                "facts_version",
                str(version),
                {"status": row.status, "note": note},
                when,
            )
            return row

    def latest_approved(self) -> FactsVersion | None:
        query = (
            select(FactsVersion).where(FactsVersion.status == "approved").order_by(FactsVersion.version.desc()).limit(1)
        )
        with self._sessions() as session:
            return session.scalars(query).one_or_none()

    def get(self, version: int) -> FactsVersion | None:
        with self._sessions() as session:
            return session.scalars(select(FactsVersion).where(FactsVersion.version == version)).one_or_none()

    def recent(self, *, limit: int = 50) -> list[FactsVersion]:
        """Newest version first."""
        with self._sessions() as session:
            return list(session.scalars(select(FactsVersion).order_by(FactsVersion.version.desc()).limit(limit)))


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
    ) -> AuditLog:
        with self._sessions.begin() as session:
            return _audit(session, actor, action, object_type, object_id, details, _utc(at) or utcnow())

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
        query = select(AuditLog).order_by(AuditLog.at.desc(), AuditLog.id.desc()).limit(limit)
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
        with self._sessions() as session:
            return list(session.scalars(query))


class TwinStore:
    """The repositories of the ``twin`` schema over one engine."""

    def __init__(self, sessions: sessionmaker[Session], engine: Engine | None = None) -> None:
        self.sessions = sessions
        self.engine = engine
        self.jobs = JobRepository(sessions)
        self.snapshots = SnapshotRepository(sessions)
        self.optimiser = OptimiserRunRepository(sessions)
        self.shifts = ShiftRecordRepository(sessions)
        self.alarms = AlarmRepository(sessions)
        self.facts = FactsVersionRepository(sessions)
        self.audit = AuditRepository(sessions)

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
