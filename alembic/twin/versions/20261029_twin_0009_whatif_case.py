"""Saved what-if cases of the admin space.

Revision ID: twin_0009
Revises: twin_0008
Create Date: 2026-10-29

SV-4.3 (#306). ``whatif_case``: an administrator's FORGE or STRATA what-if case (``inputs``) and the result of its
latest run (``result``, the job, the time and the fact-register version it used). Saves and deletes write their
``audit_log`` row (``whatif.save``, ``whatif.delete``) in the same transaction.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "twin_0009"
down_revision = "twin_0008"
branch_labels = None
depends_on = None

S = "twin"
TZ = sa.DateTime(timezone=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "whatif_case",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("inputs", JSONB, server_default=sa.text("'{}'"), nullable=False),
        sa.Column("created_at", TZ, nullable=False),
        sa.Column("updated_at", TZ, nullable=False),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("result_job_id", sa.String(64), nullable=True),
        sa.Column("result_at", TZ, nullable=True),
        sa.Column("facts_version", sa.String(32), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_whatif_case")),
        sa.CheckConstraint("kind IN ('forge', 'strata')", name=op.f("ck_whatif_case_kind")),
        schema=S,
    )
    op.create_index(op.f("ix_whatif_case_updated_at"), "whatif_case", ["updated_at"], schema=S)
    op.create_index(op.f("ix_whatif_case_owner_kind"), "whatif_case", ["owner", "kind"], schema=S)


def downgrade() -> None:
    op.drop_table("whatif_case", schema=S)
