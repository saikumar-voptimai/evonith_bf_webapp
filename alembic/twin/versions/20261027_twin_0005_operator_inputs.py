"""Operator inputs: versioned (SCD-2) prices, stock, availability and bounds of the optimisers.

Revision ID: twin_0005
Revises: twin_0004
Create Date: 2026-10-27

SV-3.1 (#298). ``operator_inputs`` holds one row per version of the document that was ``src/config/
bmo_operator_inputs.yml`` (``services.operator_inputs``): valid in [valid_from, valid_upto), the current version without
``valid_upto`` (at most one: a unique partial index), numbered ``version`` 1, 2, ... A save closes the current version
and opens the next one in the same transaction as its ``audit_log`` row. The trigger ``operator_inputs_scd2`` keeps the
history: it lets an update only close the current version (set ``valid_upto``, nothing else) and refuses every delete
(SQLSTATE 23001, restrict_violation). No data is written here: the first save records the YAML as version 1.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "twin_0005"
down_revision = "twin_0004"
branch_labels = None
depends_on = None

S = "twin"
TZ = sa.DateTime(timezone=True)
CURRENT = sa.text("valid_upto IS NULL")


def upgrade() -> None:
    op.create_table(
        "operator_inputs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("valid_from", TZ, nullable=False),
        sa.Column("valid_upto", TZ, nullable=True),
        sa.Column("author", sa.String(128), nullable=False),
        sa.Column("origin", sa.String(16), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "inputs", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_operator_inputs")),
        sa.UniqueConstraint("version", name=op.f("uq_operator_inputs_version")),
        sa.CheckConstraint("origin IN ('seed', 'streamlit', 'api')", name=op.f("ck_operator_inputs_origin")),
        sa.CheckConstraint("valid_upto IS NULL OR valid_upto > valid_from", name=op.f("ck_operator_inputs_validity")),
        schema=S,
    )
    op.create_index(op.f("ix_operator_inputs_valid_from"), "operator_inputs", ["valid_from"], schema=S)
    op.create_index(
        op.f("uq_operator_inputs_current"),
        "operator_inputs",
        [sa.text("(valid_upto IS NULL)")],
        unique=True,
        schema=S,
        postgresql_where=CURRENT,
    )
    op.execute(
        f"""
        CREATE FUNCTION {S}.operator_inputs_scd2() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'UPDATE' AND OLD.valid_upto IS NULL AND NEW.valid_upto IS NOT NULL
               AND (NEW.id, NEW.version, NEW.valid_from, NEW.author, NEW.origin, NEW.note, NEW.inputs, NEW.content_hash)
                   IS NOT DISTINCT FROM
                   (OLD.id, OLD.version, OLD.valid_from, OLD.author, OLD.origin, OLD.note, OLD.inputs, OLD.content_hash)
            THEN
                RETURN NEW;
            END IF;
            RAISE EXCEPTION '{S}.operator_inputs versions are written once and only closed; % refused', TG_OP
                USING ERRCODE = 'restrict_violation';
        END
        $$
        """
    )
    op.execute(
        f"CREATE TRIGGER operator_inputs_scd2 BEFORE UPDATE OR DELETE ON {S}.operator_inputs "
        f"FOR EACH ROW EXECUTE FUNCTION {S}.operator_inputs_scd2()"
    )


def downgrade() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS operator_inputs_scd2 ON {S}.operator_inputs")
    op.execute(f"DROP FUNCTION IF EXISTS {S}.operator_inputs_scd2()")
    op.drop_index(op.f("uq_operator_inputs_current"), table_name="operator_inputs", schema=S)
    op.drop_index(op.f("ix_operator_inputs_valid_from"), table_name="operator_inputs", schema=S)
    op.drop_table("operator_inputs", schema=S)
