"""Harden automation scheduled-job ownership and occurrence invariants.

Revision ID: 20260905_0006
Revises: 20260905_0005
Create Date: 2026-09-05

The preceding migration merged the Scheduled Tasks feature into tables owned by
the deployed database. This revision makes the Phase 3 runner assumptions
explicit in PostgreSQL: every job has an owner, every run has a logical
occurrence, activation fields move together, and run child tables can be
queried or cascaded efficiently by their foreign key.
"""

from __future__ import annotations

from alembic import op

revision = "20260905_0006"
down_revision = "20260905_0005"
branch_labels = None
depends_on = None


def _reject_rows_violating_hardened_invariants() -> None:
    """Abort without data loss when legacy rows cannot satisfy new invariants."""

    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM automation.scheduled_jobs
                WHERE owner_user_id IS NULL
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'Cannot require scheduled_jobs.owner_user_id: '
                              'legacy rows without an owner exist.',
                    HINT = 'Assign each legacy scheduled job to an identity.users row '
                           'before retrying this migration.';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM automation.job_runs
                WHERE scheduled_for IS NULL
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'Cannot require job_runs.scheduled_for: '
                              'legacy runs without an occurrence timestamp exist.',
                    HINT = 'Backfill or archive each legacy run before retrying this migration.';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM automation.job_runs
                WHERE status = 'running'
                GROUP BY job_id
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'Cannot enforce one running occurrence per job: '
                              'jobs with overlapping running rows exist.',
                    HINT = 'Resolve each duplicate running row before retrying this migration.';
            END IF;
        END
        $$
        """)


def _ensure_owner_foreign_key() -> None:
    """Add the owner foreign key only when its dependency exists and it is absent."""

    op.execute("""
        DO $$
        BEGIN
            IF to_regclass('identity.users') IS NOT NULL
               AND NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint AS constraint_row
                    WHERE constraint_row.conrelid =
                              'automation.scheduled_jobs'::regclass
                      AND constraint_row.contype = 'f'
                      AND constraint_row.conkey = ARRAY(
                            SELECT attribute_row.attnum
                            FROM pg_attribute AS attribute_row
                            WHERE attribute_row.attrelid =
                                      'automation.scheduled_jobs'::regclass
                              AND attribute_row.attname = 'owner_user_id'
                              AND NOT attribute_row.attisdropped
                        )
                      AND constraint_row.confrelid = to_regclass('identity.users')
                      AND constraint_row.confkey = ARRAY(
                            SELECT attribute_row.attnum
                            FROM pg_attribute AS attribute_row
                            WHERE attribute_row.attrelid =
                                      to_regclass('identity.users')
                              AND attribute_row.attname = 'id'
                              AND NOT attribute_row.attisdropped
                        )
                )
            THEN
                ALTER TABLE automation.scheduled_jobs
                    ADD CONSTRAINT fk_automation_scheduled_jobs_owner_user_id
                    FOREIGN KEY (owner_user_id)
                    REFERENCES identity.users(id)
                    NOT VALID;
            END IF;
        END
        $$
        """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conrelid = 'automation.scheduled_jobs'::regclass
                  AND conname = 'fk_automation_scheduled_jobs_owner_user_id'
            ) THEN
                ALTER TABLE automation.scheduled_jobs
                    VALIDATE CONSTRAINT
                        fk_automation_scheduled_jobs_owner_user_id;
            END IF;
        END
        $$
        """)


def upgrade() -> None:
    """Enforce runner invariants and index run child-table foreign keys."""

    op.execute(
        "ALTER TABLE automation.scheduled_jobs ADD COLUMN activated_at TIMESTAMPTZ"
    )
    _reject_rows_violating_hardened_invariants()

    op.execute("""
        UPDATE automation.scheduled_jobs
        SET
            is_active = (status = 'active'),
            activated_at = CASE
                WHEN status = 'active'
                    THEN COALESCE(activated_at, updated_at, created_at, now())
                ELSE NULL
            END
        WHERE
            is_active IS DISTINCT FROM (status = 'active')
            OR (
                status = 'active'
                AND activated_at IS NULL
            )
            OR (
                status <> 'active'
                AND activated_at IS NOT NULL
            )
        """)

    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ALTER COLUMN owner_user_id SET NOT NULL"
    )
    _ensure_owner_foreign_key()
    op.execute(
        "ALTER TABLE automation.job_runs ALTER COLUMN scheduled_for SET NOT NULL"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD CONSTRAINT ck_automation_scheduled_jobs_activation_consistency "
        "CHECK ("
        "(status = 'active' AND is_active AND activated_at IS NOT NULL) OR "
        "(status <> 'active' AND NOT is_active AND activated_at IS NULL)"
        ") NOT VALID"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "VALIDATE CONSTRAINT ck_automation_scheduled_jobs_activation_consistency"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_job_runs_job_created "
        "ON automation.job_runs (job_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_job_run_logs_run_id "
        "ON automation.job_run_logs (run_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_job_outputs_run_id "
        "ON automation.job_outputs (run_id)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_job_runs_one_running_per_job "
        "ON automation.job_runs (job_id) WHERE status = 'running'"
    )


def downgrade() -> None:
    """Remove owned hardening while preserving deployment-owned objects.

    The owner foreign key and ``ix_job_runs_job_created`` may have existed
    before this revision. They intentionally remain in place because the
    migration cannot safely establish their provenance during downgrade.
    """

    op.execute("DROP INDEX IF EXISTS automation.ix_job_outputs_run_id")
    op.execute("DROP INDEX IF EXISTS automation.ix_job_run_logs_run_id")
    op.execute("DROP INDEX IF EXISTS automation.uq_job_runs_one_running_per_job")
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "DROP CONSTRAINT ck_automation_scheduled_jobs_activation_consistency"
    )
    op.execute(
        "ALTER TABLE automation.job_runs ALTER COLUMN scheduled_for DROP NOT NULL"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ALTER COLUMN owner_user_id DROP NOT NULL"
    )
    op.execute("ALTER TABLE automation.scheduled_jobs DROP COLUMN activated_at")
