"""Represent the deployed relational database baseline.

Revision ID: 20260430_0004
Revises: 20260427_0001
Create Date: 2026-04-30

The target database was already stamped at this revision and contains schemas
created by the deployment database project, including ``automation``. Those
historical migration sources are not present in this webapp repository. This
marker reconnects the local migration graph to the deployed revision without
recreating or changing any database object.
"""

from __future__ import annotations

revision = "20260430_0004"
down_revision = "20260427_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Record the already-deployed database baseline without changing it."""


def downgrade() -> None:
    """Leave objects owned by the deployed database baseline unchanged."""
