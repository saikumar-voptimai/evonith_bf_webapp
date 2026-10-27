"""Twin schema: jobs, snapshot, optimiser_run, plan_decision, shift_record, alarm_event, facts_version, audit_log.

Revision ID: twin_0001
Revises:
Create Date: 2026-10-27

SV-0.9 (#278). Tables of ``data.twin.models``; every time column queried by range leads an index. Rows of
``shift_record``, ``plan_decision`` and ``audit_log`` are written once: the trigger ``reject_change`` refuses UPDATE
and DELETE on them (SQLSTATE 23001, restrict_violation). The schema itself is created by ``alembic/twin/env.py``
before the version table; the downgrade drops everything in it except that table.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "twin_0001"
down_revision = None
branch_labels = None
depends_on = None

S = "twin"
TZ = sa.DateTime(timezone=True)
JSON = postgresql.JSONB(astext_type=sa.Text())
EMPTY = sa.text("'{}'::jsonb")
APPEND_ONLY = ("shift_record", "plan_decision", "audit_log")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _identity() -> sa.Column:
    return sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False)


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("params", JSON, server_default=EMPTY, nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("submitted_by", sa.String(128), nullable=False),
        sa.Column("submitted_at", TZ, nullable=False),
        sa.Column("started_at", TZ, nullable=True),
        sa.Column("finished_at", TZ, nullable=True),
        sa.Column("tries", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("result", JSON, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("archived_at", TZ, server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.CheckConstraint(
            _in("status", ("queued", "running", "done", "failed", "cancelled")), name=op.f("ck_jobs_status")
        ),
        schema=S,
    )
    op.create_index(op.f("ix_jobs_submitted_at"), "jobs", ["submitted_at"], schema=S)
    op.create_index(op.f("ix_jobs_kind_submitted_at"), "jobs", ["kind", "submitted_at"], schema=S)
    op.create_index(op.f("ix_jobs_finished_at"), "jobs", ["finished_at"], schema=S)

    op.create_table(
        "snapshot",
        _identity(),
        sa.Column("window_start", TZ, nullable=False),
        sa.Column("window_end", TZ, nullable=False),
        sa.Column("evaluated_at", TZ, nullable=False),
        sa.Column("job_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        *(
            sa.Column(name, sa.Float(), nullable=True)
            for name in (
                "k_trace",
                "k_forge",
                "k_strata",
                "k_strata_sd",
                "k_charged",
                "omega_imp",
                "omega_strata",
                "omega_strata_sd",
                "omega_cal",
            )
        ),
        sa.Column("quantities", JSON, server_default=EMPTY, nullable=False),
        sa.Column("flags", JSON, server_default=EMPTY, nullable=False),
        sa.Column("versions", JSON, server_default=EMPTY, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_snapshot")),
        sa.UniqueConstraint("window_end", name=op.f("uq_snapshot_window_end")),
        sa.ForeignKeyConstraint(
            ["job_id"], [f"{S}.jobs.id"], name=op.f("fk_snapshot_job_id_jobs"), ondelete="SET NULL"
        ),
        sa.CheckConstraint(_in("status", ("complete", "partial", "failed")), name=op.f("ck_snapshot_status")),
        sa.CheckConstraint("window_start < window_end", name=op.f("ck_snapshot_window")),
        schema=S,
    )
    op.create_index(op.f("ix_snapshot_evaluated_at"), "snapshot", ["evaluated_at"], schema=S)

    op.create_table(
        "optimiser_run",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.String(64), nullable=True),
        sa.Column("trigger", sa.String(16), nullable=False),
        sa.Column("requested_by", sa.String(128), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("omega_mode", sa.String(16), nullable=False),
        sa.Column("started_at", TZ, nullable=False),
        sa.Column("finished_at", TZ, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        *(
            sa.Column(name, sa.Float(), nullable=True)
            for name in (
                "production_t_d",
                "cost_rs_thm",
                "saving_rs_thm",
                "coke_kg_thm",
                "coke_sd_kg_thm",
                "fuel_rate_kg_thm",
            )
        ),
        sa.Column("certainty", sa.String(16), nullable=True),
        sa.Column("inputs", JSON, server_default=EMPTY, nullable=False),
        sa.Column("result", JSON, server_default=EMPTY, nullable=False),
        sa.Column("versions", JSON, server_default=EMPTY, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_optimiser_run")),
        sa.ForeignKeyConstraint(
            ["job_id"], [f"{S}.jobs.id"], name=op.f("fk_optimiser_run_job_id_jobs"), ondelete="SET NULL"
        ),
        sa.CheckConstraint(_in("status", ("ok", "infeasible", "failed")), name=op.f("ck_optimiser_run_status")),
        sa.CheckConstraint(_in("trigger", ("schedule", "user")), name=op.f("ck_optimiser_run_trigger")),
        sa.CheckConstraint(_in("omega_mode", ("strata", "fixed")), name=op.f("ck_optimiser_run_omega_mode")),
        sa.CheckConstraint("started_at <= finished_at", name=op.f("ck_optimiser_run_times")),
        schema=S,
    )
    op.create_index(op.f("ix_optimiser_run_started_at"), "optimiser_run", ["started_at"], schema=S)
    op.create_index(op.f("ix_optimiser_run_mode_started_at"), "optimiser_run", ["mode", "started_at"], schema=S)

    op.create_table(
        "plan_decision",
        _identity(),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("decided_by", sa.String(128), nullable=False),
        sa.Column("decided_at", TZ, nullable=False),
        sa.Column("shift_key", sa.String(16), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_plan_decision")),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{S}.optimiser_run.id"],
            name=op.f("fk_plan_decision_run_id_optimiser_run"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(_in("decision", ("accepted", "rejected")), name=op.f("ck_plan_decision_decision")),
        schema=S,
    )
    op.create_index(op.f("ix_plan_decision_decided_at"), "plan_decision", ["decided_at"], schema=S)
    op.create_index(op.f("ix_plan_decision_run_id_decided_at"), "plan_decision", ["run_id", "decided_at"], schema=S)

    op.create_table(
        "shift_record",
        sa.Column("shift_key", sa.String(16), nullable=False),
        sa.Column("letter", sa.String(1), nullable=False),
        sa.Column("shift_start", TZ, nullable=False),
        sa.Column("shift_end", TZ, nullable=False),
        sa.Column("closed_at", TZ, nullable=False),
        *(
            sa.Column(name, sa.Float(), nullable=True)
            for name in ("k_trace", "k_forge", "k_strata", "k_charged", "omega_imp")
        ),
        sa.Column("twin", JSON, server_default=EMPTY, nullable=False),
        sa.Column("plan_run_id", sa.Uuid(), nullable=True),
        sa.Column("plan_decision", sa.String(16), nullable=True),
        sa.Column("report_ref", sa.String(256), nullable=True),
        sa.PrimaryKeyConstraint("shift_key", name=op.f("pk_shift_record")),
        sa.ForeignKeyConstraint(
            ["plan_run_id"], [f"{S}.optimiser_run.id"], name=op.f("fk_shift_record_plan_run_id_optimiser_run")
        ),
        sa.CheckConstraint(_in("letter", ("A", "B", "C")), name=op.f("ck_shift_record_letter")),
        sa.CheckConstraint(
            f"plan_decision IS NULL OR {_in('plan_decision', ('accepted', 'rejected'))}",
            name=op.f("ck_shift_record_plan_decision"),
        ),
        sa.CheckConstraint("shift_start < shift_end", name=op.f("ck_shift_record_times")),
        schema=S,
    )
    op.create_index(op.f("ix_shift_record_shift_start"), "shift_record", ["shift_start"], schema=S)

    op.create_table(
        "alarm_event",
        _identity(),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("subject", sa.String(128), server_default="", nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", JSON, server_default=EMPTY, nullable=False),
        sa.Column("raised_at", TZ, nullable=False),
        sa.Column("cleared_at", TZ, nullable=True),
        sa.Column("acknowledged_by", sa.String(128), nullable=True),
        sa.Column("acknowledged_at", TZ, nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alarm_event")),
        sa.CheckConstraint(_in("severity", ("warning", "critical")), name=op.f("ck_alarm_event_severity")),
        sa.CheckConstraint(
            "cleared_at IS NULL OR cleared_at >= raised_at", name=op.f("ck_alarm_event_cleared_after_raised")
        ),
        schema=S,
    )
    op.create_index(op.f("ix_alarm_event_raised_at"), "alarm_event", ["raised_at"], schema=S)
    op.create_index(
        op.f("uq_alarm_event_open"),
        "alarm_event",
        ["kind", "subject"],
        unique=True,
        schema=S,
        postgresql_where=sa.text("cleared_at IS NULL"),
    )

    op.create_table(
        "facts_version",
        _identity(),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("base_version", sa.String(16), nullable=False),
        sa.Column("parent_version", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("overlay", JSON, server_default=EMPTY, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("second_review", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(128), nullable=False),
        sa.Column("created_at", TZ, nullable=False),
        sa.Column("reviewed_by", sa.String(128), nullable=True),
        sa.Column("reviewed_at", TZ, nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_facts_version")),
        sa.UniqueConstraint("version", name=op.f("uq_facts_version_version")),
        sa.CheckConstraint(_in("status", ("pending", "approved", "rejected")), name=op.f("ck_facts_version_status")),
        sa.CheckConstraint(
            "NOT second_review OR reviewed_by IS NULL OR reviewed_by <> created_by",
            name=op.f("ck_facts_version_second_reviewer"),
        ),
        sa.CheckConstraint(
            "(status = 'pending') = (reviewed_at IS NULL)", name=op.f("ck_facts_version_reviewed_when_decided")
        ),
        schema=S,
    )
    op.create_index(op.f("ix_facts_version_created_at"), "facts_version", ["created_at"], schema=S)

    op.create_table(
        "audit_log",
        _identity(),
        sa.Column("at", TZ, server_default=sa.func.now(), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("role", sa.String(16), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("object_type", sa.String(64), nullable=False),
        sa.Column("object_id", sa.String(128), nullable=True),
        sa.Column("details", JSON, server_default=EMPTY, nullable=False),
        sa.Column("client_ip", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_log")),
        schema=S,
    )
    op.create_index(op.f("ix_audit_log_at"), "audit_log", ["at"], schema=S)
    op.create_index(op.f("ix_audit_log_actor_at"), "audit_log", ["actor", "at"], schema=S)
    op.create_index(
        op.f("ix_audit_log_object_type_object_id_at"), "audit_log", ["object_type", "object_id", "at"], schema=S
    )

    op.execute(
        f"""
        CREATE FUNCTION {S}.reject_change() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION '{S}.% rows are written once; % refused', TG_TABLE_NAME, TG_OP
                USING ERRCODE = 'restrict_violation';
        END
        $$
        """
    )
    for table in APPEND_ONLY:
        op.execute(
            f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {S}.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION {S}.reject_change()"
        )


def downgrade() -> None:
    for table in APPEND_ONLY:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_append_only ON {S}.{table}")
    op.execute(f"DROP FUNCTION IF EXISTS {S}.reject_change()")
    for table in (
        "audit_log",
        "facts_version",
        "alarm_event",
        "shift_record",
        "plan_decision",
        "optimiser_run",
        "snapshot",
        "jobs",
    ):
        op.drop_table(table, schema=S)
