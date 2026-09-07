"""Contract tests for scheduled-job ORM metadata and Alembic hardening."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import CheckConstraint, DateTime, UniqueConstraint

from alembic.config import Config
from alembic.script import ScriptDirectory
from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobCommand,
    ScheduledJobOutput,
    ScheduledJobRevision,
    ScheduledJobRun,
    ScheduledJobRunLog,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION_DIRECTORY = _REPOSITORY_ROOT / "alembic" / "versions"


def _load_migration(filename: str, module_name: str) -> ModuleType:
    """Load one revision module whose numeric filename is not importable normally."""

    migration_path = _MIGRATION_DIRECTORY / filename
    specification = importlib.util.spec_from_file_location(module_name, migration_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _normalized_sql(statements: list[str]) -> str:
    """Return captured migration SQL with insignificant whitespace collapsed."""

    return "\n".join(re.sub(r"\s+", " ", statement).strip() for statement in statements)


def test_scheduled_job_metadata_enforces_phase_three_invariants() -> None:
    """ORM metadata must match the hardened PostgreSQL scheduling contract."""

    job_table = ScheduledJob.__table__
    run_table = ScheduledJobRun.__table__

    assert job_table.c.owner_user_id.nullable is False
    assert job_table.c.activated_at.nullable is True
    assert isinstance(job_table.c.activated_at.type, DateTime)
    assert job_table.c.activated_at.type.timezone is True
    assert run_table.c.scheduled_for.nullable is False

    constraints = {
        constraint.name: constraint
        for constraint in job_table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    activation_constraint = constraints[
        "ck_automation_scheduled_jobs_activation_consistency"
    ]
    activation_sql = str(activation_constraint.sqltext)
    assert "status = 'active'" in activation_sql
    assert "activated_at IS NOT NULL" in activation_sql
    assert "status <> 'active'" in activation_sql
    assert "activated_at IS NULL" in activation_sql
    job_indexes = {index.name for index in job_table.indexes}
    assert "ix_scheduled_jobs_target_created_job" in job_indexes
    assert job_table.c.activation_generation.nullable is False
    assert "completed" in str(
        constraints["ck_automation_scheduled_jobs_status"].sqltext
    )
    assert "execution_failed" in str(
        constraints["ck_automation_scheduled_jobs_status"].sqltext
    )
    assert {
        "schedule_cron",
        "timezone",
        "prompt_template",
        "tool_config",
        "next_run_at",
        "metadata",
    }.isdisjoint(job_table.c.keys())
    assert "metadata" not in run_table.c


def test_run_child_tables_index_or_constrain_their_foreign_keys() -> None:
    """Run children must support history reads and fast cascades by run_id."""

    log_indexes = {index.name for index in ScheduledJobRunLog.__table__.indexes}
    run_indexes = {index.name for index in ScheduledJobRun.__table__.indexes}

    assert "ix_job_run_logs_run_id" in log_indexes
    assert "ix_job_runs_job_created" in run_indexes
    running_index = next(
        index
        for index in ScheduledJobRun.__table__.indexes
        if index.name == "uq_job_runs_one_running_per_job"
    )
    assert running_index.unique is True
    assert str(running_index.dialect_options["postgresql"]["where"]) == (
        "status = 'running'"
    )
    assert str(running_index.dialect_options["sqlite"]["where"]) == (
        "status = 'running'"
    )
    output_constraints = {
        constraint.name
        for constraint in ScheduledJobOutput.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert "uq_job_outputs_run_id" in output_constraints

    run_constraints = {
        constraint.name: constraint
        for constraint in ScheduledJobRun.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    lease_sql = str(run_constraints["ck_job_runs_execution_lease_consistency"].sqltext)
    assert "lease_token IS NOT NULL" in lease_sql
    assert "lease_expires_at IS NOT NULL" in lease_sql
    assert "activation_generation > 0" in lease_sql
    assert "attempt_deadline_at IS NULL" in lease_sql


def test_storage_cleanup_revision_is_the_alembic_head() -> None:
    """The migration graph must place audited cleanup after the control bridge."""

    configuration = Config(str(_REPOSITORY_ROOT / "alembic.ini"))
    configuration.set_main_option("path_separator", "os")
    configuration.set_main_option("script_location", str(_REPOSITORY_ROOT / "alembic"))
    revisions = ScriptDirectory.from_config(configuration)

    assert revisions.get_current_head() == "20260907_0010"
    assert revisions.get_revision("20260907_0010").down_revision == "20260907_0009"
    assert revisions.get_revision("20260907_0009").down_revision == "20260905_0008"
    assert revisions.get_revision("20260905_0008").down_revision == "20260905_0007"
    assert revisions.get_revision("20260905_0007").down_revision == "20260905_0006"
    assert revisions.get_revision("20260905_0006").down_revision == "20260905_0005"


def test_control_queue_metadata_has_audit_and_claim_invariants() -> None:
    """ORM metadata must fence claims and retain immutable job revisions."""

    command_table = ScheduledJobCommand.__table__
    revision_table = ScheduledJobRevision.__table__
    command_constraints = {
        constraint.name: constraint
        for constraint in command_table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    revision_constraints = {
        constraint.name for constraint in revision_table.constraints
    }
    command_indexes = {index.name: index for index in command_table.indexes}

    assert "lease_token IS NOT NULL" in str(
        command_constraints["ck_scheduled_job_commands_lease_consistency"].sqltext
    )
    assert command_indexes["uq_scheduled_job_commands_one_open_per_job"].unique
    assert "ix_scheduled_job_commands_pending_target" in command_indexes
    assert "ix_scheduled_job_commands_expired_lease" in command_indexes
    assert "ix_scheduled_job_commands_requester" in command_indexes
    assert "ix_scheduled_job_revisions_changer" in {
        index.name for index in revision_table.indexes
    }
    assert "idempotency_key" not in command_table.c
    assert "uq_scheduled_job_revisions_job_number" in revision_constraints


def test_storage_cleanup_migration_guards_and_removes_only_audited_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revision 0010 must reject unexpected values before dropping redundancy."""

    migration = _load_migration(
        "20260907_0010_simplify_scheduled_job_storage.py",
        "scheduled_job_revision_0010",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    sql = _normalized_sql(statements)
    assert "tool_config <> '{}'::jsonb" in sql
    assert "next_run_at IS NOT NULL" in sql
    assert "job_runs WHERE metadata <> '{}'::jsonb" in sql
    assert "DROP COLUMN schedule_cron" in sql
    assert "DROP COLUMN idempotency_key" in sql
    assert "ALTER TABLE automation.job_runs DROP COLUMN metadata" in sql
    assert "ix_scheduled_job_commands_requester" in sql
    assert "ix_scheduled_job_revisions_changer" in sql
    assert sql.index("non-redundant legacy values exist") < sql.index(
        "DROP COLUMN schedule_cron"
    )


def test_control_bridge_migration_creates_queue_and_revision_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revision 0009 must backfill audit rows and add indexed command claims."""

    migration = _load_migration(
        "20260907_0009_add_scheduled_job_control_queue.py",
        "scheduled_job_revision_0009",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    sql = _normalized_sql(statements)
    assert "CREATE TABLE automation.scheduled_job_revisions" in sql
    assert "INSERT INTO automation.scheduled_job_revisions" in sql
    assert "CREATE TABLE automation.scheduled_job_commands" in sql
    assert "uq_scheduled_job_commands_one_open_per_job" in sql
    assert "WHERE status IN ('pending', 'processing')" in sql
    assert "ix_scheduled_job_commands_pending_target" in sql
    assert "ix_scheduled_job_commands_expired_lease" in sql

    with pytest.raises(RuntimeError, match="intentionally irreversible"):
        migration.downgrade()


def test_phase_four_reconciliation_migration_owns_only_its_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revision 0007 should add and remove only its target keyset index."""

    migration = _load_migration(
        "20260905_0007_index_scheduled_job_reconciliation.py",
        "scheduled_job_revision_0007",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()
    migration.downgrade()

    sql = _normalized_sql(statements)
    assert "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_target_created_job" in sql
    assert "(target_device_id, created_at, job_id)" in sql
    assert "DROP INDEX IF EXISTS automation.ix_scheduled_jobs_target_created_job" in sql


def test_phase_five_migration_guards_and_adds_fenced_execution_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revision 0008 must reject ambiguous rows before installing lease fencing."""

    migration = _load_migration(
        "20260905_0008_harden_scheduled_job_execution.py",
        "scheduled_job_revision_0008",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    sql = _normalized_sql(statements)
    assert "WHERE status = 'running'" in sql
    assert "HAVING count(*) > 1" in sql
    assert "ADD COLUMN activation_generation BIGINT" in sql
    assert "ADD COLUMN lease_token UUID" in sql
    assert "ADD COLUMN lease_expires_at TIMESTAMPTZ" in sql
    assert "ADD COLUMN attempt_deadline_at TIMESTAMPTZ" in sql
    assert "ADD COLUMN retry_not_before_at TIMESTAMPTZ" in sql
    assert "SET activation_generation = 1 WHERE status = 'active'" in sql
    assert "'completed', 'execution_failed'" in sql
    assert "ck_job_runs_execution_lease_consistency" in sql
    assert "attempt_deadline_at IS NULL" in sql
    assert "uq_job_outputs_run_id UNIQUE (run_id)" in sql
    assert "DROP INDEX IF EXISTS automation.ix_job_outputs_run_id" in sql
    assert sql.index("WHERE status = 'running'") < sql.index(
        "ADD COLUMN lease_token UUID"
    )


def test_phase_five_migration_refuses_to_remove_fencing() -> None:
    """Revision 0008 must not silently re-enable stale-worker writes."""

    migration = _load_migration(
        "20260905_0008_harden_scheduled_job_execution.py",
        "scheduled_job_revision_0008_downgrade",
    )

    with pytest.raises(RuntimeError, match="intentionally irreversible"):
        migration.downgrade()


def test_phase_three_hardening_migration_emits_guarded_schema_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revision 0006 must guard legacy data before tightening its columns."""

    migration = _load_migration(
        "20260905_0006_harden_scheduled_job_invariants.py",
        "scheduled_job_revision_0006",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    sql = _normalized_sql(statements)
    assert "ADD COLUMN activated_at TIMESTAMPTZ" in sql
    assert "WHERE owner_user_id IS NULL" in sql
    assert "WHERE scheduled_for IS NULL" in sql
    assert "HAVING count(*) > 1" in sql
    assert "ALTER COLUMN owner_user_id SET NOT NULL" in sql
    assert "ALTER COLUMN scheduled_for SET NOT NULL" in sql
    assert "fk_automation_scheduled_jobs_owner_user_id" in sql
    assert "to_regclass('identity.users') IS NOT NULL" in sql
    assert "ck_automation_scheduled_jobs_activation_consistency" in sql
    assert (
        "VALIDATE CONSTRAINT ck_automation_scheduled_jobs_activation_consistency" in sql
    )
    assert "CREATE INDEX IF NOT EXISTS ix_job_runs_job_created" in sql
    assert "CREATE INDEX IF NOT EXISTS ix_job_run_logs_run_id" in sql
    assert "CREATE INDEX IF NOT EXISTS ix_job_outputs_run_id" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_job_runs_one_running_per_job" in sql
    assert sql.index("WHERE owner_user_id IS NULL") < sql.index(
        "ALTER COLUMN owner_user_id SET NOT NULL"
    )
    assert sql.index("WHERE scheduled_for IS NULL") < sql.index(
        "ALTER COLUMN scheduled_for SET NOT NULL"
    )


def test_revision_0005_refuses_ambiguous_destructive_downgrade() -> None:
    """The deployment-table merge must fail closed instead of dropping data."""

    migration = _load_migration(
        "20260905_0005_extend_automation_scheduled_jobs.py",
        "scheduled_job_revision_0005",
    )

    with pytest.raises(RuntimeError, match="intentionally irreversible"):
        migration.downgrade()


def test_revision_0006_downgrade_preserves_ambiguous_legacy_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downgrade must not remove an owner FK or job index it may not own."""

    migration = _load_migration(
        "20260905_0006_harden_scheduled_job_invariants.py",
        "scheduled_job_revision_0006_downgrade",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.downgrade()

    sql = _normalized_sql(statements)
    assert "fk_automation_scheduled_jobs_owner_user_id" not in sql
    assert "ix_job_runs_job_created" not in sql
    assert "DROP COLUMN activated_at" in sql
