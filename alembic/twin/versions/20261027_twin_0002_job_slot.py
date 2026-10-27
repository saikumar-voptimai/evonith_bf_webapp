"""Jobs: the schedule slot a scheduler job serves, and the execution time over all tries.

Revision ID: twin_0002
Revises: twin_0001
Create Date: 2026-10-27

SV-1.7 (#287). ``jobs.scheduled_for`` holds the IST cron time (in UTC) of the slot a job of the scheduler serves: the
cron entry, the catch-up at worker start and the record of a skipped slot set it; jobs started by a user leave it null.
The schedule audit (``services.schedule.audit``) matches slots to jobs on this column. ``jobs.run_s`` is the execution
time summed over the tries, s, from which the runtime statistics per kind are computed (``services.jobstats``): the
time between start and finish of a job that waited between tries includes the waits.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "twin_0002"
down_revision = "twin_0001"
branch_labels = None
depends_on = None

S = "twin"


def upgrade() -> None:
    op.add_column("jobs", sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True), schema=S)
    op.add_column("jobs", sa.Column("run_s", sa.Float(), nullable=True), schema=S)
    op.create_index(op.f("ix_jobs_scheduled_for"), "jobs", ["scheduled_for"], schema=S)


def downgrade() -> None:
    op.drop_index(op.f("ix_jobs_scheduled_for"), table_name="jobs", schema=S)
    op.drop_column("jobs", "run_s", schema=S)
    op.drop_column("jobs", "scheduled_for", schema=S)
