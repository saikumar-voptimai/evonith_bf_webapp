"""Add fenced execution leases and terminal one-time job states.

Revision ID: 20260905_0008
Revises: 20260905_0007
Create Date: 2026-09-05

Phase 5 executes operator-created jobs in a real headless agent process. This
revision adds the durable activation generation and renewable lease fields
needed to fence a process after another process reclaims its attempt. It also
adds explicit terminal states for one-time jobs and enforces the service's
one-output-per-run contract.
"""

from __future__ import annotations

from alembic import op

revision = "20260905_0008"
down_revision = "20260905_0007"
branch_labels = None
depends_on = None


def _reject_ambiguous_legacy_rows() -> None:
    """Abort when existing rows cannot be assigned trustworthy lease ownership."""

    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM automation.job_runs
                WHERE status = 'running'
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'Cannot add scheduled-job execution leases: '
                              'running rows already exist.',
                    HINT = 'Resolve each running job occurrence before retrying '
                           'the Phase 5 migration; lease ownership must not be invented.';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM automation.job_outputs
                GROUP BY run_id
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'Cannot require one scheduled-job output per run: '
                              'duplicate output rows exist.',
                    HINT = 'Review and resolve duplicate outputs before retrying '
                           'the Phase 5 migration.';
            END IF;
        END
        $$
        """)


def upgrade() -> None:
    """Install activation generations, fenced leases, and terminal job states."""

    _reject_ambiguous_legacy_rows()

    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD COLUMN activation_generation BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "UPDATE automation.scheduled_jobs SET activation_generation = 1 "
        "WHERE status = 'active'"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "DROP CONSTRAINT ck_automation_scheduled_jobs_status"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD CONSTRAINT ck_automation_scheduled_jobs_status CHECK "
        "(status IN ('pending_provisioning', 'active', 'paused', "
        "'provisioning_failed', 'completed', 'execution_failed', 'deleted')) "
        "NOT VALID"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "VALIDATE CONSTRAINT ck_automation_scheduled_jobs_status"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs "
        "ADD CONSTRAINT ck_automation_scheduled_jobs_activation_generation CHECK "
        "(activation_generation >= 0 AND "
        "(status <> 'active' OR activation_generation > 0)) NOT VALID"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_jobs VALIDATE CONSTRAINT "
        "ck_automation_scheduled_jobs_activation_generation"
    )

    op.execute(
        "ALTER TABLE automation.job_runs "
        "ADD COLUMN activation_generation BIGINT NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE automation.job_runs ADD COLUMN lease_token UUID")
    op.execute(
        "ALTER TABLE automation.job_runs ADD COLUMN lease_expires_at TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE automation.job_runs ADD COLUMN attempt_deadline_at TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE automation.job_runs ADD COLUMN retry_not_before_at TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE automation.job_runs "
        "ADD CONSTRAINT ck_job_runs_attempt_number_positive "
        "CHECK (attempt_number >= 1) NOT VALID"
    )
    op.execute(
        "ALTER TABLE automation.job_runs "
        "VALIDATE CONSTRAINT ck_job_runs_attempt_number_positive"
    )
    op.execute("""
        ALTER TABLE automation.job_runs
        ADD CONSTRAINT ck_job_runs_execution_lease_consistency CHECK (
            (
                status = 'running'
                AND completed_at IS NULL
                AND activation_generation > 0
                AND lease_token IS NOT NULL
                AND lease_expires_at IS NOT NULL
                AND (
                    (
                        retry_not_before_at IS NULL
                        AND attempt_deadline_at IS NOT NULL
                    )
                    OR (
                        retry_not_before_at IS NOT NULL
                        AND attempt_deadline_at IS NULL
                    )
                )
            )
            OR (
                status <> 'running'
                AND lease_token IS NULL
                AND lease_expires_at IS NULL
                AND attempt_deadline_at IS NULL
                AND retry_not_before_at IS NULL
            )
        ) NOT VALID
        """)
    op.execute(
        "ALTER TABLE automation.job_runs VALIDATE CONSTRAINT "
        "ck_job_runs_execution_lease_consistency"
    )

    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conrelid = 'automation.job_outputs'::regclass
                  AND conname = 'uq_job_outputs_run_id'
            ) THEN
                ALTER TABLE automation.job_outputs
                    ADD CONSTRAINT uq_job_outputs_run_id UNIQUE (run_id);
            END IF;
        END
        $$
        """)
    op.execute("DROP INDEX IF EXISTS automation.ix_job_outputs_run_id")


def downgrade() -> None:
    """Refuse to discard lease fencing or terminal execution history."""

    raise RuntimeError(
        "Revision 20260905_0008 is intentionally irreversible: removing its "
        "lease fencing can let stale workers commit, and its terminal job "
        "states cannot be represented by the preceding schema. Restore from "
        "a verified database backup instead."
    )
