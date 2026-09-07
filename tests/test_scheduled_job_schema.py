"""Contract tests for the reduced scheduled-task storage schema."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, dialect

from alembic.config import Config
from alembic.script import ScriptDirectory
from furnace_data.relational.models import ScheduledTaskDefinitionRecord

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION_DIRECTORY = _REPOSITORY_ROOT / "alembic" / "versions"


def _load_migration(filename: str, module_name: str) -> ModuleType:
    """Load a numeric Alembic revision module from its file path."""

    migration_path = _MIGRATION_DIRECTORY / filename
    specification = importlib.util.spec_from_file_location(module_name, migration_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _normalized_sql(statements: list[str]) -> str:
    """Collapse insignificant whitespace in captured migration statements."""

    return "\n".join(re.sub(r"\s+", " ", statement).strip() for statement in statements)


def test_definition_model_contains_only_storage_and_ui_state() -> None:
    """Keep the ORM contract independent from runtime execution tables."""

    table = ScheduledTaskDefinitionRecord.__table__

    assert table.schema == "automation"
    assert set(table.c) == {
        table.c.job_id,
        table.c.job_name,
        table.c.job_type,
        table.c.schema_version,
        table.c.definition,
        table.c.status,
        table.c.owner_user_id,
        table.c.created_by_username,
        table.c.target_device_id,
        table.c.created_at,
        table.c.updated_at,
    }
    assert isinstance(table.c.definition.type.dialect_impl(dialect()), JSONB)
    assert table.c.owner_user_id.nullable is False
    constraints = {
        constraint.name: constraint
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "pending_provisioning" in str(
        constraints["ck_automation_scheduled_jobs_status"].sqltext
    )
    assert {index.name for index in table.indexes} == {
        "ix_scheduled_jobs_owner_created",
        "ix_scheduled_jobs_owner_status_created",
    }

    forbidden_tables = {
        "job_runs",
        "job_run_logs",
        "job_outputs",
        "scheduled_job_commands",
        "scheduled_job_revisions",
    }
    assert forbidden_tables.isdisjoint(
        table_name.rsplit(".", 1)[-1] for table_name in table.metadata.tables
    )


def test_reduced_storage_revision_is_the_alembic_head() -> None:
    """Keep a short migration graph ending at the legacy-schema cleanup."""

    configuration = Config(str(_REPOSITORY_ROOT / "alembic.ini"))
    configuration.set_main_option("path_separator", "os")
    configuration.set_main_option("script_location", str(_REPOSITORY_ROOT / "alembic"))
    revisions = ScriptDirectory.from_config(configuration)

    assert revisions.get_current_head() == "20260907_0011"
    assert revisions.get_revision("20260907_0011").down_revision == "20260907_0010"
    assert revisions.get_revision("20260907_0010").down_revision == "20260430_0004"
    assert revisions.get_revision("20260430_0004").down_revision == "20260427_0001"


def test_migration_creates_only_definition_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create the task table and owner indexes without runtime objects."""

    migration = _load_migration(
        "20260907_0010_create_scheduled_task_storage.py",
        "scheduled_task_storage_revision",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    sql = _normalized_sql(statements)
    assert "CREATE TABLE IF NOT EXISTS automation.scheduled_jobs" in sql
    assert "definition JSONB NOT NULL" in sql
    assert "owner_user_id UUID" in sql
    assert "target_device_id VARCHAR(128) NOT NULL" in sql
    assert "ix_scheduled_jobs_owner_status_created" in sql
    assert "REFERENCES identity.users(id)" in sql
    for forbidden in (
        "CREATE TABLE automation.job_runs",
        "CREATE TABLE automation.job_outputs",
        "CREATE TABLE automation.scheduled_job_commands",
        "CREATE TABLE automation.scheduled_job_revisions",
    ):
        assert forbidden not in sql


def test_storage_downgrade_refuses_to_delete_definitions() -> None:
    """Prevent automatic destructive removal of operator-authored JSON."""

    migration = _load_migration(
        "20260907_0010_create_scheduled_task_storage.py",
        "scheduled_task_storage_downgrade",
    )

    with pytest.raises(RuntimeError, match="intentionally irreversible"):
        migration.downgrade()


def test_cleanup_revision_removes_legacy_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove prototype activation fields while preserving JSON storage."""

    migration = _load_migration(
        "20260907_0011_align_scheduled_task_json_storage.py",
        "scheduled_task_storage_cleanup",
    )
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    sql = _normalized_sql(statements)
    assert (
        "DROP CONSTRAINT IF EXISTS ck_automation_scheduled_jobs_activation_consistency"
        in sql
    )
    assert (
        "DROP CONSTRAINT IF EXISTS ck_automation_scheduled_jobs_activation_generation"
        in sql
    )
    for column_name in (
        "is_active",
        "timer_unit_name",
        "provisioning_error",
        "activated_at",
        "activation_generation",
    ):
        assert f"DROP COLUMN IF EXISTS {column_name}" in sql
    assert "ix_scheduled_jobs_owner_created" in sql
    assert "ix_scheduled_jobs_owner_status_created" in sql
    assert "ON DELETE CASCADE" in sql


def test_cleanup_downgrade_refuses_to_fabricate_runtime_state() -> None:
    """Document that discarded prototype runtime metadata cannot be rebuilt."""

    migration = _load_migration(
        "20260907_0011_align_scheduled_task_json_storage.py",
        "scheduled_task_storage_cleanup_downgrade",
    )

    with pytest.raises(RuntimeError, match="intentionally irreversible"):
        migration.downgrade()
