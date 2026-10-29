"""Furnace commentary on optimiser runs.

Revision ID: twin_0007
Revises: twin_0006
Create Date: 2026-10-29

SV-3.2 (#299). ``optimiser_commentary``: page 11's furnace commentary on a stored optimiser run, written by the job
``optimiser.commentary``: the LLM's text or, after its retry, the errors (status ``failed``), the full context it was
given with its SHA-1, the SHA-1 of the optimiser notes, the fact register version and what the context lacked.
Deleting a run deletes its commentaries.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "twin_0007"
down_revision = "twin_0006"
branch_labels = None
depends_on = None

S = "twin"
TZ = sa.DateTime(timezone=True)
JSON = postgresql.JSONB(astext_type=sa.Text())
EMPTY_LIST = sa.text("'[]'::jsonb")


def upgrade() -> None:
    op.create_table(
        "optimiser_commentary",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.String(64), nullable=True),
        sa.Column("requested_by", sa.String(128), nullable=False),
        sa.Column("generated_at", TZ, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("tries", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("context", sa.Text(), server_default="", nullable=False),
        sa.Column("context_sha1", sa.String(40), nullable=False),
        sa.Column("notes_sha1", sa.String(40), nullable=True),
        sa.Column("facts_version", sa.String(16), nullable=True),
        sa.Column("missing", JSON, server_default=EMPTY_LIST, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_optimiser_commentary")),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{S}.optimiser_run.id"],
            name=op.f("fk_optimiser_commentary_run_id_optimiser_run"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], [f"{S}.jobs.id"], name=op.f("fk_optimiser_commentary_job_id_jobs"), ondelete="SET NULL"
        ),
        sa.CheckConstraint("status IN ('ok', 'failed')", name=op.f("ck_optimiser_commentary_status")),
        schema=S,
    )
    op.create_index(op.f("ix_optimiser_commentary_generated_at"), "optimiser_commentary", ["generated_at"], schema=S)
    op.create_index(
        op.f("ix_optimiser_commentary_run_id_generated_at"),
        "optimiser_commentary",
        ["run_id", "generated_at"],
        schema=S,
    )


def downgrade() -> None:
    op.drop_table("optimiser_commentary", schema=S)
