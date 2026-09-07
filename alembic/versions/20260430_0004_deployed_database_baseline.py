"""Represent the deployed relational database baseline.

Revision ID: 20260430_0004
Revises: 20260427_0001
Create Date: 2026-04-30

The target database already contains schemas managed by the deployment
database project. This marker reconnects the webapp migration graph without
recreating those objects.
"""

from __future__ import annotations

revision = "20260430_0004"
down_revision = "20260427_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Record the deployed database baseline without changing it."""


def downgrade() -> None:
    """Leave deployment-owned baseline objects unchanged."""
