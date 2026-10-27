"""Audit log: the values of the written object before and after an admin write.

Revision ID: twin_0003
Revises: twin_0002
Create Date: 2026-10-27

SV-4.1 (#304). ``audit_log.before`` and ``audit_log.after`` hold the object's fields before and after an admin write,
written in the transaction of the write by ``services.audit.admin_write``; the audit viewer of the admin space shows
their difference. ``before`` is null for a created object, ``after`` for a deleted one; entries that record only
``details`` (plan decisions, fact-register proposals and reviews, alarm acknowledgements) leave both null. Adding a
nullable column changes no row, so the append-only trigger of the table does not fire.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "twin_0003"
down_revision = "twin_0002"
branch_labels = None
depends_on = None

S = "twin"
JSON = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.add_column("audit_log", sa.Column("before", JSON, nullable=True), schema=S)
    op.add_column("audit_log", sa.Column("after", JSON, nullable=True), schema=S)


def downgrade() -> None:
    op.drop_column("audit_log", "after", schema=S)
    op.drop_column("audit_log", "before", schema=S)
