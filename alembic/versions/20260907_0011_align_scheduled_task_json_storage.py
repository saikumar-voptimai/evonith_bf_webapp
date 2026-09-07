"""Align legacy scheduled jobs with the UI-and-JSON storage milestone.

Revision ID: 20260907_0011
Revises: 20260907_0010
Create Date: 2026-09-07

Developer databases may have reached revision ``0010`` through the earlier
full scheduler prototype. Those databases still contain activation columns,
constraints, and polling indexes that are not part of the reduced milestone.
This migration removes that runtime-only state while preserving every task's
canonical JSON definition and owner metadata.
"""

from __future__ import annotations

from alembic import op

revision = "20260907_0011"
down_revision = "20260907_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove legacy runtime state and align owner-scoped JSON storage."""

    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "DROP CONSTRAINT IF EXISTS ck_automation_scheduled_jobs_activation_consistency"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "DROP CONSTRAINT IF EXISTS ck_automation_scheduled_jobs_activation_generation"
    )

    for index_name in (
        "ix_scheduled_jobs_creator_created",
        "ix_scheduled_jobs_status_created",
        "ix_scheduled_jobs_target_created_job",
    ):
        op.execute(f"DROP INDEX IF EXISTS automation.{index_name}")

    for column_name in (
        "is_active",
        "timer_unit_name",
        "provisioning_error",
        "activated_at",
        "activation_generation",
    ):
        op.execute(
            "ALTER TABLE automation.scheduled_jobs "
            f"DROP COLUMN IF EXISTS {column_name}"
        )

    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ALTER COLUMN status SET DEFAULT 'pending_provisioning'"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM automation.scheduled_jobs
                WHERE owner_user_id IS NULL
            ) THEN
                RAISE EXCEPTION
                    'automation.scheduled_jobs contains rows without an owner';
            END IF;

            ALTER TABLE automation.scheduled_jobs
                ALTER COLUMN owner_user_id SET NOT NULL;
        END
        $$
        """
    )

    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "DROP CONSTRAINT IF EXISTS scheduled_jobs_owner_user_id_fkey"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "DROP CONSTRAINT IF EXISTS fk_scheduled_jobs_owner_user"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('identity.users') IS NOT NULL THEN
                ALTER TABLE automation.scheduled_jobs
                    ADD CONSTRAINT fk_scheduled_jobs_owner_user
                    FOREIGN KEY (owner_user_id)
                    REFERENCES identity.users(id)
                    ON DELETE CASCADE
                    NOT VALID;

                ALTER TABLE automation.scheduled_jobs
                    VALIDATE CONSTRAINT fk_scheduled_jobs_owner_user;
            END IF;
        END
        $$
        """
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_owner_created "
        "ON automation.scheduled_jobs (owner_user_id, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_owner_status_created "
        "ON automation.scheduled_jobs (owner_user_id, status, created_at DESC)"
    )


def downgrade() -> None:
    """Refuse to recreate discarded runtime state with fabricated values."""

    raise RuntimeError(
        "The scheduled-task JSON-storage cleanup is intentionally irreversible. "
        "Restore a database backup to recover discarded runtime metadata."
    )
