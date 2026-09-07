"""Index the bounded target-device scan used by systemd reconciliation.

Revision ID: 20260905_0007
Revises: 20260905_0006
Create Date: 2026-09-05
"""

from __future__ import annotations

from alembic import op

revision = "20260905_0007"
down_revision = "20260905_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the keyset index used by explicit boot/deployment reconciliation."""

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scheduled_jobs_target_created_job "
        "ON automation.scheduled_jobs (target_device_id, created_at, job_id)"
    )


def downgrade() -> None:
    """Remove only the reconciliation index owned by this revision."""

    op.execute("DROP INDEX IF EXISTS automation.ix_scheduled_jobs_target_created_job")
