"""Create core relational BF2 tables.

Revision ID: 20260427_0001
Revises:
Create Date: 2026-04-27
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260427_0001"
down_revision = None
branch_labels = None
depends_on = None


user_role_enum = sa.Enum(
    "admin",
    "supervisor",
    "user",
    name="user_role",
    native_enum=False,
)


def upgrade() -> None:
    """Create users, hopper history, and burden history tables."""
    op.create_table(
        "users",
        sa.Column("username", sa.String(length=128), nullable=False),
        sa.Column("password_hash", sa.String(length=128), nullable=False),
        sa.Column("role", user_role_enum, nullable=False),
        sa.PrimaryKeyConstraint("username"),
    )

    op.create_table(
        "hopper_material_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("hopper", sa.String(length=128), nullable=False),
        sa.Column("material", sa.String(length=256), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=False), nullable=False),
        sa.Column("valid_upto", sa.DateTime(timezone=False), nullable=True),
        sa.Column(
            "modifier",
            sa.String(length=128),
            nullable=False,
            server_default="system",
        ),
        sa.Column("ip_address", sa.String(length=256), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_hopper_material_history_hopper",
        "hopper_material_history",
        ["hopper"],
        unique=False,
    )
    op.create_index(
        "ix_hopper_history_hopper_valid_from",
        "hopper_material_history",
        ["hopper", "valid_from"],
        unique=False,
    )
    op.create_index(
        "ix_hopper_history_hopper_valid_upto",
        "hopper_material_history",
        ["hopper", "valid_upto"],
        unique=False,
    )

    op.create_table(
        "burden_distribution_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("field_name", sa.String(length=256), nullable=False),
        sa.Column("field_value_float", sa.Float(), nullable=True),
        sa.Column("field_value_text", sa.Text(), nullable=True),
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("valid_upto", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "modifier",
            sa.String(length=128),
            nullable=False,
            server_default="system",
        ),
        sa.Column("ip_address", sa.String(length=256), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "field_name",
            "valid_upto",
            name="uq_burden_active_record",
        ),
    )
    op.create_index(
        "ix_burden_distribution_history_field_name",
        "burden_distribution_history",
        ["field_name"],
        unique=False,
    )
    op.create_index(
        "ix_burden_history_field_valid_from",
        "burden_distribution_history",
        ["field_name", "valid_from"],
        unique=False,
    )
    op.create_index(
        "ix_burden_history_field_valid_upto",
        "burden_distribution_history",
        ["field_name", "valid_upto"],
        unique=False,
    )


def downgrade() -> None:
    """Drop core relational BF2 tables."""
    op.drop_index(
        "ix_burden_history_field_valid_upto",
        table_name="burden_distribution_history",
    )
    op.drop_index(
        "ix_burden_history_field_valid_from",
        table_name="burden_distribution_history",
    )
    op.drop_index(
        "ix_burden_distribution_history_field_name",
        table_name="burden_distribution_history",
    )
    op.drop_table("burden_distribution_history")

    op.drop_index(
        "ix_hopper_history_hopper_valid_upto",
        table_name="hopper_material_history",
    )
    op.drop_index(
        "ix_hopper_history_hopper_valid_from",
        table_name="hopper_material_history",
    )
    op.drop_index(
        "ix_hopper_material_history_hopper",
        table_name="hopper_material_history",
    )
    op.drop_table("hopper_material_history")

    op.drop_table("users")
