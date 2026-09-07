"""Add the Jetson control queue and immutable job-definition revisions.

Revision ID: 20260907_0009
Revises: 20260905_0008
Create Date: 2026-09-07

Streamlit Cloud cannot invoke systemd on the BF2 Jetson. This revision adds a
durable, target-scoped command queue for that boundary and an append-only
revision history for definitions changed through the queue.
"""

from __future__ import annotations

from alembic import op

revision = "20260907_0009"
down_revision = "20260905_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create revision and lifecycle-command persistence with bounded indexes."""

    op.execute("""
        CREATE TABLE automation.scheduled_job_revisions (
            revision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id TEXT NOT NULL REFERENCES automation.scheduled_jobs(job_id)
                ON DELETE CASCADE,
            revision_number BIGINT NOT NULL,
            schema_version VARCHAR(64) NOT NULL,
            definition JSONB NOT NULL,
            changed_by_user_id UUID NOT NULL REFERENCES identity.users(id),
            changed_by_username VARCHAR(128) NOT NULL,
            change_kind VARCHAR(16) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_scheduled_job_revisions_number_positive
                CHECK (revision_number >= 1),
            CONSTRAINT ck_scheduled_job_revisions_change_kind
                CHECK (change_kind IN ('created', 'edited', 'rollback')),
            CONSTRAINT uq_scheduled_job_revisions_job_number
                UNIQUE (job_id, revision_number)
        )
        """)
    op.execute("""
        INSERT INTO automation.scheduled_job_revisions (
            job_id,
            revision_number,
            schema_version,
            definition,
            changed_by_user_id,
            changed_by_username,
            change_kind,
            created_at
        )
        SELECT
            job_id,
            1,
            schema_version,
            definition,
            owner_user_id,
            created_by_username,
            'created',
            created_at
        FROM automation.scheduled_jobs
        """)
    op.execute(
        "CREATE INDEX ix_scheduled_job_revisions_job_created "
        "ON automation.scheduled_job_revisions (job_id, created_at)"
    )

    op.execute("""
        CREATE TABLE automation.scheduled_job_commands (
            command_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id TEXT NOT NULL REFERENCES automation.scheduled_jobs(job_id)
                ON DELETE CASCADE,
            action VARCHAR(16) NOT NULL,
            requested_job_status VARCHAR(32) NOT NULL,
            status VARCHAR(16) NOT NULL DEFAULT 'pending',
            target_device_id VARCHAR(128) NOT NULL,
            requested_by_user_id UUID NOT NULL REFERENCES identity.users(id),
            requested_by_username VARCHAR(128) NOT NULL,
            idempotency_key UUID NOT NULL,
            expected_job_updated_at TIMESTAMPTZ NOT NULL,
            definition JSONB,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            maximum_attempts INTEGER NOT NULL DEFAULT 3,
            available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            lease_token UUID,
            lease_expires_at TIMESTAMPTZ,
            worker_id VARCHAR(128),
            resulting_job_status VARCHAR(32),
            error_message TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            CONSTRAINT ck_scheduled_job_commands_action CHECK
                (action IN ('provision', 'pause', 'resume', 'update', 'archive')),
            CONSTRAINT ck_scheduled_job_commands_status CHECK
                (status IN ('pending', 'processing', 'succeeded', 'failed')),
            CONSTRAINT ck_scheduled_job_commands_requested_job_status CHECK
                (requested_job_status IN (
                    'pending_provisioning', 'active', 'paused',
                    'provisioning_failed', 'completed', 'execution_failed'
                )),
            CONSTRAINT ck_scheduled_job_commands_attempts CHECK
                (attempt_count >= 0 AND maximum_attempts BETWEEN 1 AND 10),
            CONSTRAINT ck_scheduled_job_commands_update_definition CHECK (
                (action = 'update' AND definition IS NOT NULL)
                OR (action <> 'update' AND definition IS NULL)
            ),
            CONSTRAINT ck_scheduled_job_commands_lease_consistency CHECK (
                (
                    status = 'processing'
                    AND lease_token IS NOT NULL
                    AND lease_expires_at IS NOT NULL
                    AND started_at IS NOT NULL
                )
                OR (
                    status <> 'processing'
                    AND lease_token IS NULL
                    AND lease_expires_at IS NULL
                )
            ),
            CONSTRAINT uq_scheduled_job_commands_idempotency_key
                UNIQUE (idempotency_key)
        )
        """)
    op.execute(
        "CREATE INDEX ix_scheduled_job_commands_job_created "
        "ON automation.scheduled_job_commands (job_id, created_at)"
    )
    op.execute("""
        CREATE INDEX ix_scheduled_job_commands_pending_target
        ON automation.scheduled_job_commands (
            target_device_id,
            available_at,
            created_at,
            command_id
        )
        WHERE status = 'pending'
        """)
    op.execute("""
        CREATE INDEX ix_scheduled_job_commands_expired_lease
        ON automation.scheduled_job_commands (
            target_device_id,
            lease_expires_at,
            created_at,
            command_id
        )
        WHERE status = 'processing'
        """)
    op.execute("""
        CREATE UNIQUE INDEX uq_scheduled_job_commands_one_open_per_job
        ON automation.scheduled_job_commands (job_id)
        WHERE status IN ('pending', 'processing')
        """)


def downgrade() -> None:
    """Refuse to discard queued commands or definition audit history."""

    raise RuntimeError(
        "Revision 20260907_0009 is intentionally irreversible because a "
        "downgrade would discard lifecycle requests and definition history. "
        "Restore from a verified database backup instead."
    )
