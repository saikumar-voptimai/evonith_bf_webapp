"""Remove redundant scheduled-task storage and complete actor indexing.

Revision ID: 20260907_0010
Revises: 20260907_0009
Create Date: 2026-09-07

The JSON definition is the canonical source for schedule and instruction data.
This revision removes unused legacy projections, an empty run metadata field,
and an idempotency key that was never part of the application request contract.
It fails closed if another deployment contains non-redundant legacy values.
"""

from __future__ import annotations

from alembic import op

revision = "20260907_0010"
down_revision = "20260907_0009"
branch_labels = None
depends_on = None


def _reject_nonredundant_legacy_values() -> None:
    """Abort rather than discard legacy values not represented elsewhere."""

    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM automation.scheduled_jobs
                WHERE
                    schedule_cron IS DISTINCT FROM CASE
                        WHEN definition#>>'{schedule,trigger,type}' = 'cron'
                            THEN definition#>>'{schedule,trigger,expression}'
                        ELSE NULL
                    END
                    OR timezone IS DISTINCT FROM
                        definition#>>'{schedule,timezone}'
                    OR prompt_template IS DISTINCT FROM
                        definition->>'instructions'
                    OR tool_config <> '{}'::jsonb
                    OR next_run_at IS NOT NULL
                    OR metadata NOT IN (
                        '{}'::jsonb,
                        '{"source": "scheduled_tasks_ui"}'::jsonb
                    )
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'Cannot remove redundant scheduled-job columns: '
                              'non-redundant legacy values exist.',
                    HINT = 'Move the reported legacy values into the canonical '
                           'definition or another owned table before retrying.';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM automation.job_runs
                WHERE metadata <> '{}'::jsonb
            ) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'Cannot remove job_runs.metadata: non-empty values exist.',
                    HINT = 'Migrate required run metadata before retrying.';
            END IF;
        END
        $$
        """)


def upgrade() -> None:
    """Drop audited redundancy and index both user-reference foreign keys."""

    _reject_nonredundant_legacy_values()

    op.execute("DROP INDEX IF EXISTS automation.ix_scheduled_jobs_active_next_run")
    op.execute("DROP INDEX IF EXISTS automation.ix_scheduled_job_revisions_job_created")
    op.execute("""
        ALTER TABLE automation.scheduled_jobs
            DROP COLUMN schedule_cron,
            DROP COLUMN timezone,
            DROP COLUMN prompt_template,
            DROP COLUMN tool_config,
            DROP COLUMN next_run_at,
            DROP COLUMN metadata
        """)
    op.execute("""
        ALTER TABLE automation.scheduled_job_commands
            DROP CONSTRAINT uq_scheduled_job_commands_idempotency_key,
            DROP COLUMN idempotency_key
        """)
    op.execute("ALTER TABLE automation.job_runs DROP COLUMN metadata")

    op.execute(
        "CREATE INDEX ix_scheduled_job_commands_requester "
        "ON automation.scheduled_job_commands (requested_by_user_id)"
    )
    op.execute(
        "CREATE INDEX ix_scheduled_job_revisions_changer "
        "ON automation.scheduled_job_revisions (changed_by_user_id)"
    )


def downgrade() -> None:
    """Restore compatible legacy projections from canonical definitions."""

    op.execute("DROP INDEX IF EXISTS automation.ix_scheduled_job_revisions_changer")
    op.execute("DROP INDEX IF EXISTS automation.ix_scheduled_job_commands_requester")

    op.execute("""
        ALTER TABLE automation.scheduled_jobs
            ADD COLUMN schedule_cron TEXT,
            ADD COLUMN timezone TEXT,
            ADD COLUMN prompt_template TEXT,
            ADD COLUMN tool_config JSONB NOT NULL DEFAULT '{}'::jsonb,
            ADD COLUMN next_run_at TIMESTAMPTZ,
            ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        """)
    op.execute("""
        UPDATE automation.scheduled_jobs
        SET
            schedule_cron = CASE
                WHEN definition#>>'{schedule,trigger,type}' = 'cron'
                    THEN definition#>>'{schedule,trigger,expression}'
                ELSE NULL
            END,
            timezone = definition#>>'{schedule,timezone}',
            prompt_template = definition->>'instructions'
        """)
    op.execute("""
        ALTER TABLE automation.scheduled_jobs
            ALTER COLUMN timezone SET NOT NULL,
            ALTER COLUMN timezone SET DEFAULT 'Asia/Kolkata'
        """)
    op.execute(
        "ALTER TABLE automation.job_runs "
        "ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_job_commands "
        "ADD COLUMN idempotency_key UUID DEFAULT gen_random_uuid()"
    )
    op.execute(
        "ALTER TABLE automation.scheduled_job_commands "
        "ALTER COLUMN idempotency_key SET NOT NULL"
    )
    op.execute("""
        ALTER TABLE automation.scheduled_job_commands
            ADD CONSTRAINT uq_scheduled_job_commands_idempotency_key
            UNIQUE (idempotency_key)
        """)
    op.execute(
        "ALTER TABLE automation.scheduled_job_commands "
        "ALTER COLUMN idempotency_key DROP DEFAULT"
    )

    op.execute(
        "CREATE INDEX ix_scheduled_jobs_active_next_run "
        "ON automation.scheduled_jobs (is_active, next_run_at)"
    )
    op.execute(
        "CREATE INDEX ix_scheduled_job_revisions_job_created "
        "ON automation.scheduled_job_revisions (job_id, created_at)"
    )
