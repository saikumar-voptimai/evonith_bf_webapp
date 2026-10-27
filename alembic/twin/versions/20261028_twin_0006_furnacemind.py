"""FurnaceMind conversations and messages of the agent service.

Revision ID: twin_0006
Revises: twin_0005
Create Date: 2026-10-28

SV-5.1 (#312). ``fm_conversation``: a conversation of the twin-api's FurnaceMind service, owned by the user who
created it, with its title, reasoning level and the ``seq`` of its last message (``message_count``).
``fm_message``: its messages numbered 1, 2, ... (unique per conversation): the text replayed to the model
(``content``) and what the answer showed (``parts``: tool calls and results, Plotly figure JSON, dataset previews).
Deleting a conversation deletes its messages.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "twin_0006"
down_revision = "twin_0005"
branch_labels = None
depends_on = None

S = "twin"
TZ = sa.DateTime(timezone=True)
JSON = postgresql.JSONB(astext_type=sa.Text())
EMPTY = sa.text("'{}'::jsonb")
EMPTY_LIST = sa.text("'[]'::jsonb")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    op.create_table(
        "fm_conversation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner", sa.String(128), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=True),
        sa.Column("title", sa.String(200), server_default="", nullable=False),
        sa.Column("reasoning_level", sa.String(16), nullable=False),
        sa.Column("created_at", TZ, nullable=False),
        sa.Column("updated_at", TZ, nullable=False),
        sa.Column("message_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("meta", JSON, server_default=EMPTY, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fm_conversation")),
        sa.CheckConstraint(
            _in("reasoning_level", ("Low", "Medium", "High")), name=op.f("ck_fm_conversation_reasoning_level")
        ),
        sa.CheckConstraint("message_count >= 0", name=op.f("ck_fm_conversation_message_count")),
        schema=S,
    )
    op.create_index(op.f("ix_fm_conversation_updated_at"), "fm_conversation", ["updated_at"], schema=S)
    op.create_index(op.f("ix_fm_conversation_owner_updated_at"), "fm_conversation", ["owner", "updated_at"], schema=S)

    op.create_table(
        "fm_message",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), server_default="", nullable=False),
        sa.Column("parts", JSON, server_default=EMPTY_LIST, nullable=False),
        sa.Column("status", sa.String(16), server_default="complete", nullable=False),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("turn_id", sa.String(64), nullable=True),
        sa.Column("created_at", TZ, nullable=False),
        sa.Column("meta", JSON, server_default=EMPTY, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fm_message")),
        sa.UniqueConstraint("conversation_id", "seq", name=op.f("uq_fm_message_conversation_id_seq")),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            [f"{S}.fm_conversation.id"],
            name=op.f("fk_fm_message_conversation_id_fm_conversation"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(_in("role", ("user", "assistant")), name=op.f("ck_fm_message_role")),
        sa.CheckConstraint(_in("status", ("complete", "failed")), name=op.f("ck_fm_message_status")),
        sa.CheckConstraint("seq >= 1", name=op.f("ck_fm_message_seq")),
        schema=S,
    )
    op.create_index(op.f("ix_fm_message_created_at"), "fm_message", ["created_at"], schema=S)


def downgrade() -> None:
    op.drop_table("fm_message", schema=S)
    op.drop_table("fm_conversation", schema=S)
