"""Extend the existing automation tables for systemd scheduled tasks.

Revision ID: 20260905_0005
Revises: 20260430_0004
Create Date: 2026-09-05

The deployed database already owns the ``automation`` scheduling tables. This
migration preserves that structure, adds the validated JSON definition and
provisioning lifecycle required by the operator UI, and adds an idempotency key
for one-shot job execution. ``CREATE TABLE IF NOT EXISTS`` also keeps the
scheduled-task portion reproducible in an empty development database.
"""

from __future__ import annotations

from alembic import op

revision = "20260905_0005"
down_revision = "20260430_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Extend the automation job definition and run-lifecycle tables."""

    op.execute("CREATE SCHEMA IF NOT EXISTS automation")
    op.execute("""
        CREATE TABLE IF NOT EXISTS automation.scheduled_jobs (
            job_id TEXT PRIMARY KEY,
            job_name TEXT NOT NULL,
            job_type TEXT NOT NULL,
            schedule_cron TEXT,
            timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
            prompt_template TEXT,
            tool_config JSONB NOT NULL DEFAULT '{}'::jsonb,
            owner_user_id UUID,
            is_active BOOLEAN NOT NULL DEFAULT false,
            next_run_at TIMESTAMPTZ,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            schema_version VARCHAR(64) NOT NULL,
            definition JSONB NOT NULL,
            status VARCHAR(32) NOT NULL,
            created_by_username VARCHAR(128) NOT NULL,
            target_device_id VARCHAR(128) NOT NULL,
            timer_unit_name VARCHAR(255),
            provisioning_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN IF NOT EXISTS schema_version VARCHAR(64)"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN IF NOT EXISTS definition JSONB"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN IF NOT EXISTS status VARCHAR(32)"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN IF NOT EXISTS created_by_username VARCHAR(128)"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN IF NOT EXISTS target_device_id VARCHAR(128)"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN IF NOT EXISTS timer_unit_name VARCHAR(255)"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN IF NOT EXISTS provisioning_error TEXT"
    )
    op.execute(
        "UPDATE automation.scheduled_jobs SET "
        "schema_version = COALESCE(schema_version, 'legacy-automation/v0'), "
        "definition = COALESCE(definition, '{}'::jsonb), "
        "status = COALESCE(status, CASE WHEN is_active THEN 'active' ELSE 'paused' END), "
        "created_by_username = COALESCE(created_by_username, 'legacy'), "
        "target_device_id = COALESCE(target_device_id, 'bf2-jetson-01')"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ALTER COLUMN schema_version SET NOT NULL, "
        "ALTER COLUMN definition SET NOT NULL, "
        "ALTER COLUMN status SET NOT NULL, "
        "ALTER COLUMN created_by_username SET NOT NULL, "
        "ALTER COLUMN target_device_id SET NOT NULL"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "DROP CONSTRAINT IF EXISTS ck_automation_scheduled_jobs_status"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD CONSTRAINT ck_automation_scheduled_jobs_status CHECK "
        "(status IN ('pending_provisioning', 'active', 'paused', "
        "'provisioning_failed', 'deleted'))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_creator_created "
        "ON automation.scheduled_jobs (owner_user_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_status_created "
        "ON automation.scheduled_jobs (status, created_at)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS automation.job_runs (
            run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id TEXT NOT NULL REFERENCES automation.scheduled_jobs(job_id)
                ON DELETE CASCADE,
            status TEXT NOT NULL DEFAULT 'queued',
            scheduled_for TIMESTAMPTZ,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            triggered_by TEXT,
            error_message TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            attempt_number INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute(
        "ALTER TABLE automation.job_runs "
        "ADD COLUMN IF NOT EXISTS attempt_number INTEGER NOT NULL DEFAULT 1"
    )
    op.execute(
        "ALTER TABLE automation.job_runs "
        "DROP CONSTRAINT IF EXISTS job_runs_status_check"
    )
    op.execute(
        "ALTER TABLE automation.job_runs ADD CONSTRAINT job_runs_status_check "
        "CHECK (status IN ('queued', 'running', 'completed', 'failed', "
        "'timed_out', 'skipped', 'cancelled'))"
    )
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'automation.job_runs'::regclass
                  AND conname = 'uq_job_runs_job_scheduled_for'
            ) THEN
                ALTER TABLE automation.job_runs
                ADD CONSTRAINT uq_job_runs_job_scheduled_for
                UNIQUE (job_id, scheduled_for);
            END IF;
        END
        $$
        """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_job_runs_status_scheduled "
        "ON automation.job_runs (status, scheduled_for)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS automation.job_run_logs (
            id BIGSERIAL PRIMARY KEY,
            run_id UUID NOT NULL REFERENCES automation.job_runs(run_id)
                ON DELETE CASCADE,
            log_level TEXT NOT NULL DEFAULT 'info' CHECK
                (log_level IN ('debug', 'info', 'warning', 'error')),
            message TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS automation.job_outputs (
            output_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id UUID NOT NULL REFERENCES automation.job_runs(run_id)
                ON DELETE CASCADE,
            output_type TEXT NOT NULL,
            content TEXT,
            content_json JSONB,
            artifact_path TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)


def downgrade() -> None:
    """Refuse a destructive rollback across deployment-owned legacy tables."""

    raise RuntimeError(
        "Revision 20260905_0005 is intentionally irreversible: its upgrade "
        "merged scheduled-task fields into deployment-owned automation tables, "
        "so a generic downgrade cannot determine which columns and constraints "
        "are safe to remove. Restore from a verified database backup instead."
    )
