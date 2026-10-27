"""ORM models of the ``twin`` schema in Neon (SV-0.9, #278): the persistent store of the twin-api and the twin-worker.

| Table | Content | Written by |
|---|---|---|
| ``jobs`` | Job records beyond the 7 days kept in Redis | worker (job archive), housekeeping |
| ``snapshot`` | Hourly twin evaluation of the window [t − 8 h, t) | ``twin.evaluate`` (SV-1.1) |
| ``optimiser_run`` | Optimiser runs (Step 1, Step 2, maximum production), inputs and result | optimiser jobs (SV-1.7) |
| ``plan_decision`` | Accept / reject of a run by a supervisor, append-only | API (SV-1.6) |
| ``shift_record`` | Shift handover record frozen at 06:10, 14:10, 22:10, immutable | ``shift.close`` (SV-2.5) |
| ``alarm_event`` | Alarms raised and cleared (ω drift, model gap, stale source, job failure, lag) | twin and worker |
| ``facts_version`` | Versioned overlay of the fact register with second-admin review | admin space (SV-4.2) |
| ``audit_log`` | Admin writes and plan decisions, append-only | API |

Times are ``timestamptz`` in UTC. Columns queried by time range are indexed (``ix_<table>_<column>``). Values that are
plotted or filtered have typed columns with the unit in the name (``k_trace`` kg/tHM); everything else of a result is
kept in a JSON column (``JSONB`` on PostgreSQL, ``JSON`` on SQLite, which the outage buffer of SV-6.5 uses).

The migration in ``alembic/twin/versions`` creates exactly these tables; ``tests/test_twin_store.py`` compares the two.
Rows of ``shift_record``, ``plan_decision`` and ``audit_log`` cannot be updated or deleted: the migration installs a
trigger that refuses it on PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "twin"

NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

#: JSON document: ``JSONB`` on PostgreSQL, ``JSON`` elsewhere.
JSON_DOC = JSON().with_variant(JSONB(), "postgresql")
#: Surrogate key: ``bigint`` identity on PostgreSQL, ``integer`` row id on SQLite.
BIG_ID = BigInteger().with_variant(Integer(), "sqlite")
#: Server default of the JSON columns that are never null.
EMPTY_JSON = text("'{}'")

JOB_STATES = ("queued", "running", "done", "failed", "cancelled")
SNAPSHOT_STATES = ("complete", "partial", "failed")
RUN_STATES = ("ok", "infeasible", "failed")
RUN_TRIGGERS = ("schedule", "user")
OMEGA_MODES = ("strata", "fixed")
DECISIONS = ("accepted", "rejected")
SEVERITIES = ("warning", "critical")
FACTS_STATES = ("pending", "approved", "rejected")
SHIFT_LETTERS = ("A", "B", "C")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class TwinBase(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA, naming_convention=NAMING)
    type_annotation_map = {datetime: DateTime(timezone=True), dict[str, Any]: JSON_DOC}


class Job(TwinBase):
    """A job record of ``services.jobs`` (``JobRecord``), mirrored at every change of state."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    input_hash: Mapped[str] = mapped_column(String(64))
    submitted_by: Mapped[str] = mapped_column(String(128))
    submitted_at: Mapped[datetime]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    tries: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    message: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None]
    error: Mapped[str | None] = mapped_column(Text)
    archived_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint(_in("status", JOB_STATES), name="status"),
        Index("ix_jobs_submitted_at", "submitted_at"),
        Index("ix_jobs_kind_submitted_at", "kind", "submitted_at"),
        Index("ix_jobs_finished_at", "finished_at"),
    )


class Snapshot(TwinBase):
    """Twin evaluation of the window [``window_start``, ``window_end``) (8 h), one row per window end."""

    __tablename__ = "snapshot"

    id: Mapped[int] = mapped_column(BIG_ID, Identity(), primary_key=True)
    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    evaluated_at: Mapped[datetime]
    job_id: Mapped[str | None] = mapped_column(ForeignKey(f"{SCHEMA}.jobs.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(16))
    #: Coke rates, kg/tHM: burnt (TRACE), two-zone balance (FORGE), STRATA with its 1σ, charged.
    k_trace: Mapped[float | None] = mapped_column(Float)
    k_forge: Mapped[float | None] = mapped_column(Float)
    k_strata: Mapped[float | None] = mapped_column(Float)
    k_strata_sd: Mapped[float | None] = mapped_column(Float)
    k_charged: Mapped[float | None] = mapped_column(Float)
    #: Shaft efficiency, dimensionless: implied, STRATA for the window's blend with its 1σ, 14-day calibration.
    omega_imp: Mapped[float | None] = mapped_column(Float)
    omega_strata: Mapped[float | None] = mapped_column(Float)
    omega_strata_sd: Mapped[float | None] = mapped_column(Float)
    omega_cal: Mapped[float | None] = mapped_column(Float)
    #: Every other quantity of the evaluation (production, HMT, Si, moisture, transit), with units in the keys.
    quantities: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    #: Data flags of the window (stale or missing sources, provisional DPR).
    flags: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    #: Model, calibration and fact-register versions used.
    versions: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)

    __table_args__ = (
        UniqueConstraint("window_end"),
        CheckConstraint(_in("status", SNAPSHOT_STATES), name="status"),
        CheckConstraint("window_start < window_end", name="window"),
        Index("ix_snapshot_evaluated_at", "evaluated_at"),
    )


class OptimiserRun(TwinBase):
    """One optimiser run: the case it solved (``inputs``), the plan (``result``) and its headline numbers."""

    __tablename__ = "optimiser_run"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[str | None] = mapped_column(ForeignKey(f"{SCHEMA}.jobs.id", ondelete="SET NULL"))
    trigger: Mapped[str] = mapped_column(String(16))
    requested_by: Mapped[str] = mapped_column(String(128))
    #: ``step1`` (cheapest production level), ``step2`` (fixed production), ``max_production``.
    mode: Mapped[str] = mapped_column(String(32))
    omega_mode: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime]
    finished_at: Mapped[datetime]
    status: Mapped[str] = mapped_column(String(16))
    production_t_d: Mapped[float | None] = mapped_column(Float)
    #: Ore + flux + fuel − slag credit, Rs/tHM; saving against the charged blend of the last 24 h, Rs/tHM.
    cost_rs_thm: Mapped[float | None] = mapped_column(Float)
    saving_rs_thm: Mapped[float | None] = mapped_column(Float)
    coke_kg_thm: Mapped[float | None] = mapped_column(Float)
    coke_sd_kg_thm: Mapped[float | None] = mapped_column(Float)
    fuel_rate_kg_thm: Mapped[float | None] = mapped_column(Float)
    #: Certainty tier of the plan: High, Medium, Low, Investigate.
    certainty: Mapped[str | None] = mapped_column(String(16))
    inputs: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    result: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    versions: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)

    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATES), name="status"),
        CheckConstraint(_in("trigger", RUN_TRIGGERS), name="trigger"),
        CheckConstraint(_in("omega_mode", OMEGA_MODES), name="omega_mode"),
        CheckConstraint("started_at <= finished_at", name="times"),
        Index("ix_optimiser_run_started_at", "started_at"),
        Index("ix_optimiser_run_mode_started_at", "mode", "started_at"),
    )


class PlanDecision(TwinBase):
    """A supervisor's decision on a run; a changed decision is a new row, the latest one applies."""

    __tablename__ = "plan_decision"

    id: Mapped[int] = mapped_column(BIG_ID, Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.optimiser_run.id", ondelete="CASCADE"))
    decision: Mapped[str] = mapped_column(String(16))
    decided_by: Mapped[str] = mapped_column(String(128))
    decided_at: Mapped[datetime]
    #: Shift in which the decision was taken, e.g. ``2026-10-27_A``.
    shift_key: Mapped[str | None] = mapped_column(String(16))
    comment: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(_in("decision", DECISIONS), name="decision"),
        Index("ix_plan_decision_decided_at", "decided_at"),
        Index("ix_plan_decision_run_id_decided_at", "run_id", "decided_at"),
    )


class ShiftRecord(TwinBase):
    """Handover record of a shift: twin values, the plan and its decision, the report. Written once."""

    __tablename__ = "shift_record"

    #: ``YYYY-MM-DD_A`` (date of the shift start in IST).
    shift_key: Mapped[str] = mapped_column(String(16), primary_key=True)
    letter: Mapped[str] = mapped_column(String(1))
    shift_start: Mapped[datetime]
    shift_end: Mapped[datetime]
    closed_at: Mapped[datetime]
    #: Shift means: coke rates kg/tHM, implied shaft efficiency.
    k_trace: Mapped[float | None] = mapped_column(Float)
    k_forge: Mapped[float | None] = mapped_column(Float)
    k_strata: Mapped[float | None] = mapped_column(Float)
    k_charged: Mapped[float | None] = mapped_column(Float)
    omega_imp: Mapped[float | None] = mapped_column(Float)
    #: Frozen twin values of the shift (hourly snapshots, aggregates, flags).
    twin: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    plan_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.optimiser_run.id"))
    plan_decision: Mapped[str | None] = mapped_column(String(16))
    #: Identifier or link of the shift report.
    report_ref: Mapped[str | None] = mapped_column(String(256))

    __table_args__ = (
        CheckConstraint(_in("letter", SHIFT_LETTERS), name="letter"),
        CheckConstraint(f"plan_decision IS NULL OR {_in('plan_decision', DECISIONS)}", name="plan_decision"),
        CheckConstraint("shift_start < shift_end", name="times"),
        Index("ix_shift_record_shift_start", "shift_start"),
    )


class AlarmEvent(TwinBase):
    """An alarm from raise to clear. At most one open alarm per kind and subject."""

    __tablename__ = "alarm_event"

    id: Mapped[int] = mapped_column(BIG_ID, Identity(), primary_key=True)
    #: ``omega_drift``, ``strata_trace_gap``, ``source_stale``, ``job_failed``, ``implied_lag``, ...
    kind: Mapped[str] = mapped_column(String(64))
    #: What the alarm is about (a data source, a job kind); empty when the kind needs no subject.
    subject: Mapped[str] = mapped_column(String(128), default="", server_default="")
    severity: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    raised_at: Mapped[datetime]
    cleared_at: Mapped[datetime | None]
    acknowledged_by: Mapped[str | None] = mapped_column(String(128))
    acknowledged_at: Mapped[datetime | None]

    __table_args__ = (
        CheckConstraint(_in("severity", SEVERITIES), name="severity"),
        CheckConstraint("cleared_at IS NULL OR cleared_at >= raised_at", name="cleared_after_raised"),
        Index("ix_alarm_event_raised_at", "raised_at"),
        Index(
            "uq_alarm_event_open",
            "kind",
            "subject",
            unique=True,
            postgresql_where=text("cleared_at IS NULL"),
            sqlite_where=text("cleared_at IS NULL"),
        ),
    )


class FactsVersion(TwinBase):
    """A version of the fact-register overlay: the facts it changes against the register file ``base_version``."""

    __tablename__ = "facts_version"

    id: Mapped[int] = mapped_column(BIG_ID, Identity(), primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    #: Version of ``config/facts_register.json`` the overlay applies to, e.g. ``1.8``.
    base_version: Mapped[str] = mapped_column(String(16))
    parent_version: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    #: Changed facts by id: ``{"ASM-08": {"value": 185, "band": [170, 200]}}``.
    overlay: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    content_hash: Mapped[str] = mapped_column(String(64))
    #: The overlay changes the status of a fact: a second administrator must review it.
    second_review: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    note: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime]
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None]

    __table_args__ = (
        UniqueConstraint("version"),
        CheckConstraint(_in("status", FACTS_STATES), name="status"),
        CheckConstraint(
            "NOT second_review OR reviewed_by IS NULL OR reviewed_by <> created_by", name="second_reviewer"
        ),
        CheckConstraint("(status = 'pending') = (reviewed_at IS NULL)", name="reviewed_when_decided"),
        Index("ix_facts_version_created_at", "created_at"),
    )


class AuditLog(TwinBase):
    """One admin write or plan decision: who, what, on which object, when, with the details of the change."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BIG_ID, Identity(), primary_key=True)
    at: Mapped[datetime] = mapped_column(server_default=func.now())
    actor: Mapped[str] = mapped_column(String(128))
    role: Mapped[str | None] = mapped_column(String(16))
    #: ``plan.decide``, ``facts.propose``, ``facts.review``, ``alarm.acknowledge``, ``job.cancel``, ...
    action: Mapped[str] = mapped_column(String(64))
    object_type: Mapped[str] = mapped_column(String(64))
    object_id: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=EMPTY_JSON)
    client_ip: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_audit_log_at", "at"),
        Index("ix_audit_log_actor_at", "actor", "at"),
        Index("ix_audit_log_object_type_object_id_at", "object_type", "object_id", "at"),
    )


#: Tables whose rows are written once; the migration's trigger refuses UPDATE and DELETE on PostgreSQL.
APPEND_ONLY = ("shift_record", "plan_decision", "audit_log")

#: Time columns queried by range, each the leading column of an index.
TIME_INDEXED = {
    "jobs": ("submitted_at", "finished_at"),
    "snapshot": ("window_end", "evaluated_at"),
    "optimiser_run": ("started_at",),
    "plan_decision": ("decided_at",),
    "shift_record": ("shift_start",),
    "alarm_event": ("raised_at",),
    "facts_version": ("created_at",),
    "audit_log": ("at",),
}
