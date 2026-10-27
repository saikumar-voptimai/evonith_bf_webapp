"""Shift reports: the FurnaceMind Reports tab's report of a closed shift or the running one, with its LLM narrative.

Revision ID: twin_0004
Revises: twin_0003
Create Date: 2026-10-27

SV-2.3 (#295). ``shift_report`` holds one report per row: the closed shift's (``report.shift`` at 06:20, 14:20, 22:20
IST) or the running shift's on demand (``report.live``). ``data`` is the report's ``ShiftReportData``, ``document`` its
tables as the Reports tab renders them, ``narrative`` the LLM's analysis (null when the LLM failed after the retry:
``narrative_status`` failed). Headline values have typed columns with the unit in the name. A re-run of a shift adds a
row; the latest applies.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "twin_0004"
down_revision = "twin_0003"
branch_labels = None
depends_on = None

S = "twin"
TZ = sa.DateTime(timezone=True)
JSON = postgresql.JSONB(astext_type=sa.Text())
EMPTY = sa.text("'{}'::jsonb")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    op.create_table(
        "shift_report",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("shift_key", sa.String(16), nullable=False),
        sa.Column("letter", sa.String(1), nullable=False),
        sa.Column("shift_start", TZ, nullable=False),
        sa.Column("shift_end", TZ, nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("trigger", sa.String(16), nullable=False),
        sa.Column("requested_by", sa.String(128), nullable=False),
        sa.Column("job_id", sa.String(64), nullable=True),
        sa.Column("generated_at", TZ, nullable=False),
        sa.Column("furnace_status", sa.String(24), nullable=False),
        *(
            sa.Column(name, sa.Float(), nullable=True)
            for name in (
                "production_rate_t_h",
                "fuel_rate_kg_thm",
                "coke_rate_kg_thm",
                "pci_rate_kg_thm",
                "etaco_pct",
                "hm_si_pct",
            )
        ),
        sa.Column("narrative_status", sa.String(16), nullable=False),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("narrative_error", sa.Text(), nullable=True),
        sa.Column("narrative_tries", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("narrative_model", sa.String(128), nullable=True),
        sa.Column("data", JSON, server_default=EMPTY, nullable=False),
        sa.Column("document", JSON, server_default=EMPTY, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_shift_report")),
        sa.ForeignKeyConstraint(
            ["job_id"], [f"{S}.jobs.id"], name=op.f("fk_shift_report_job_id_jobs"), ondelete="SET NULL"
        ),
        sa.CheckConstraint(_in("letter", ("A", "B", "C")), name=op.f("ck_shift_report_letter")),
        sa.CheckConstraint(_in("kind", ("closed", "live")), name=op.f("ck_shift_report_kind")),
        sa.CheckConstraint(_in("trigger", ("schedule", "user")), name=op.f("ck_shift_report_trigger")),
        sa.CheckConstraint(
            _in("narrative_status", ("ok", "failed", "skipped")), name=op.f("ck_shift_report_narrative_status")
        ),
        sa.CheckConstraint("shift_start < shift_end", name=op.f("ck_shift_report_times")),
        schema=S,
    )
    op.create_index(op.f("ix_shift_report_shift_start"), "shift_report", ["shift_start"], schema=S)
    op.create_index(op.f("ix_shift_report_generated_at"), "shift_report", ["generated_at"], schema=S)
    op.create_index(
        op.f("ix_shift_report_shift_key_generated_at"), "shift_report", ["shift_key", "generated_at"], schema=S
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_shift_report_shift_key_generated_at"), table_name="shift_report", schema=S)
    op.drop_index(op.f("ix_shift_report_generated_at"), table_name="shift_report", schema=S)
    op.drop_index(op.f("ix_shift_report_shift_start"), table_name="shift_report", schema=S)
    op.drop_table("shift_report", schema=S)
