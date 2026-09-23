"""Create the reduced scheduled-task JSON definition store.

Revision ID: 20260907_0010
Revises: 20260430_0004
Create Date: 2026-09-07

This squashed revision is the published UI-and-JSON migration. It deliberately
keeps the revision identifier used by the unpublished full-system prototype so
developer databases already stamped at that revision remain readable. New
deployments receive only the definition table required by this milestone.
"""

from __future__ import annotations

from alembic import op

revision = "20260907_0010"
down_revision = "20260430_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create or extend the owner-scoped JSON definition table."""

    op.execute("CREATE SCHEMA IF NOT EXISTS automation")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS automation.scheduled_jobs (
            job_id TEXT PRIMARY KEY,
            job_name TEXT NOT NULL,
            job_type TEXT NOT NULL,
            schema_version VARCHAR(64) NOT NULL,
            definition JSONB NOT NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'pending_provisioning',
            owner_user_id UUID,
            created_by_username VARCHAR(128) NOT NULL,
            target_device_id VARCHAR(128) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
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
        "ADD COLUMN IF NOT EXISTS owner_user_id UUID"
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
        "UPDATE automation.scheduled_jobs SET "
        "schema_version = COALESCE(schema_version, 'legacy-automation/v0'), "
        "definition = COALESCE(definition, '{}'::jsonb), "
        "status = COALESCE(status, 'pending_provisioning'), "
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
        "'provisioning_failed', 'completed', 'execution_failed', 'deleted'))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_owner_created "
        "ON automation.scheduled_jobs (owner_user_id, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_owner_status_created "
        "ON automation.scheduled_jobs (owner_user_id, status, created_at DESC)"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('identity.users') IS NOT NULL
               AND NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'fk_scheduled_jobs_owner_user'
                      AND conrelid = 'automation.scheduled_jobs'::regclass
               ) THEN
                ALTER TABLE automation.scheduled_jobs
                    ADD CONSTRAINT fk_scheduled_jobs_owner_user
                    FOREIGN KEY (owner_user_id)
                    REFERENCES identity.users(id)
                    ON DELETE CASCADE
                    NOT VALID;
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    """Refuse to delete operator-authored definitions automatically."""

    raise RuntimeError(
        "The scheduled-task definition migration is intentionally irreversible. "
        "Export or back up automation.scheduled_jobs before removing it."
    )
