"""Feedback on FurnaceMind answers.

Revision ID: twin_0008
Revises: twin_0007
Create Date: 2026-10-29

SV-5.2 (#313). ``fm_feedback``: the conversation owner's rating (``up`` / ``down``, page 7's thumbs) and comment on
one assistant message (``conversation_id``, ``seq``), one row per answer, changed in place by a later save; each save
writes its ``audit_log`` row (``fm.feedback``) in the same transaction. Deleting a conversation deletes its feedback.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "twin_0008"
down_revision = "twin_0007"
branch_labels = None
depends_on = None

S = "twin"
TZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "fm_feedback",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("author", sa.String(128), nullable=False),
        sa.Column("author_id", sa.String(64), nullable=True),
        sa.Column("rating", sa.String(8), nullable=False),
        sa.Column("comment", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", TZ, nullable=False),
        sa.Column("updated_at", TZ, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fm_feedback")),
        sa.UniqueConstraint("conversation_id", "seq", name=op.f("uq_fm_feedback_conversation_id_seq")),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            [f"{S}.fm_conversation.id"],
            name=op.f("fk_fm_feedback_conversation_id_fm_conversation"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("rating IN ('up', 'down')", name=op.f("ck_fm_feedback_rating")),
        sa.CheckConstraint("seq >= 1", name=op.f("ck_fm_feedback_seq")),
        schema=S,
    )
    op.create_index(op.f("ix_fm_feedback_updated_at"), "fm_feedback", ["updated_at"], schema=S)


def downgrade() -> None:
    op.drop_table("fm_feedback", schema=S)
