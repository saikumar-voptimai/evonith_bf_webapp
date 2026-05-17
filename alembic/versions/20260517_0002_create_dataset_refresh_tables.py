"""Create ML dataset refresh tracking tables.

Revision ID: 20260517_0002
Revises: 20260427_0001
Create Date: 2026-05-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260517_0002"
down_revision = "20260427_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ML dataset version and refresh-run metadata tables."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE SCHEMA IF NOT EXISTS ml_dataset")

    op.create_table(
        "dataset_versions",
        sa.Column(
            "version_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("rm_choice", sa.Text(), nullable=False, server_default="Full"),
        sa.Column("storage_mode", sa.Text(), nullable=False),
        sa.Column("target_table", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("confirmed_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confirmed_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "status IN ('building', 'active', 'superseded', 'failed')",
            name="ck_dataset_versions_status",
        ),
        sa.CheckConstraint(
            "storage_mode IN ('table', 'csv', 'parquet')",
            name="ck_dataset_versions_storage_mode",
        ),
        sa.PrimaryKeyConstraint("version_id"),
        schema="ml_dataset",
    )
    op.create_index(
        "ix_dataset_versions_dataset_status",
        "dataset_versions",
        ["dataset_name", "rm_choice", "status"],
        schema="ml_dataset",
    )
    op.create_index(
        "uq_dataset_versions_active",
        "dataset_versions",
        ["dataset_name", "rm_choice"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        schema="ml_dataset",
    )

    op.create_table(
        "dataset_refresh_runs",
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("rm_choice", sa.Text(), nullable=False, server_default="Full"),
        sa.Column("trigger_type", sa.Text(), nullable=False),
        sa.Column("triggered_by", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "output_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_dataset.dataset_versions.version_id"),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "trigger_type IN ('page_hit', 'manual', 'schedule')",
            name="ck_dataset_refresh_runs_trigger_type",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'success', 'failed', 'skipped')",
            name="ck_dataset_refresh_runs_status",
        ),
        sa.PrimaryKeyConstraint("run_id"),
        schema="ml_dataset",
    )
    op.create_index(
        "ix_dataset_refresh_runs_dataset_created",
        "dataset_refresh_runs",
        ["dataset_name", "rm_choice", "started_at"],
        schema="ml_dataset",
    )
    op.create_index(
        "uq_dataset_refresh_running",
        "dataset_refresh_runs",
        ["dataset_name", "rm_choice"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
        schema="ml_dataset",
    )


def downgrade() -> None:
    """Drop ML dataset refresh metadata tables."""
    op.execute("DROP VIEW IF EXISTS ml_dataset.active_hourly")
    op.drop_index(
        "uq_dataset_refresh_running",
        table_name="dataset_refresh_runs",
        schema="ml_dataset",
    )
    op.drop_index(
        "ix_dataset_refresh_runs_dataset_created",
        table_name="dataset_refresh_runs",
        schema="ml_dataset",
    )
    op.drop_table("dataset_refresh_runs", schema="ml_dataset")

    op.drop_index(
        "uq_dataset_versions_active",
        table_name="dataset_versions",
        schema="ml_dataset",
    )
    op.drop_index(
        "ix_dataset_versions_dataset_status",
        table_name="dataset_versions",
        schema="ml_dataset",
    )
    op.drop_table("dataset_versions", schema="ml_dataset")
    op.execute("DROP SCHEMA IF EXISTS ml_dataset CASCADE")
