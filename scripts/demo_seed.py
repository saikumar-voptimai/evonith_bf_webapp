"""Seed the demo's twin store (user decision 16 Nov 2026, demo branch).

The demo (``python scripts/dev.py --demo``) runs on its own twin store, a SQLite file, so nothing done in a demo
(optimiser runs, accepted plans, yard edits, FurnaceMind conversations) reaches the store the app normally uses. This
script fills it with recent records of the normal store, anonymised on the way (``utils.demo``: plant, site, supplier
and mine names replaced; usernames replaced by ``person`` of ``config/demo.yml``):

* the optimiser runs, their commentaries and plan decisions of the last ``--days`` days (Blend & Fuel Optimiser's
  runs),
* the twin's hourly windows, the shift reports and the shift records of the last ``--report-days`` days (Shift
  reports, Live model tab). A shift whose closing twin window did not evaluate (status other than ``complete``) is left
  out with its report and record, so the demo opens on a shift with its twin values; ``--keep-failed`` keeps them.

The source is read only. FurnaceMind conversations, operator inputs, fact-register versions and the audit log are not
copied: the demo starts with none of them (the yard falls back to the operator-input file, the register to its file).

    python scripts/demo_seed.py                  # into src/storage/demo/twin_demo.sqlite (replaced)
    python scripts/demo_seed.py --days 4 --report-days 10 [--keep-failed]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEMO_DB = ROOT / "src" / "storage" / "demo" / "twin_demo.sqlite"
DEMO_URL = f"sqlite:///{DEMO_DB.as_posix()}"

#: Table, time column, which window (``days`` / ``report_days``), username columns.
TABLES: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("snapshot", "window_end", "report_days", ()),
    ("optimiser_run", "finished_at", "days", ("requested_by",)),
    ("optimiser_commentary", "generated_at", "days", ("requested_by",)),
    ("plan_decision", "decided_at", "days", ("decided_by",)),
    ("shift_report", "generated_at", "report_days", ("requested_by",)),
    ("shift_record", "shift_end", "report_days", ()),
]


def head_revision() -> str:
    """The twin schema's head migration (``alembic/twin/versions``): the revision no other names as its parent."""
    import re

    revisions, parents = set(), set()
    for path in (ROOT / "alembic" / "twin" / "versions").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        revisions.update(re.findall(r'^revision\s*=\s*"([^"]+)"', text, re.M))
        parents.update(re.findall(r'^down_revision\s*=\s*"([^"]+)"', text, re.M))
    (head,) = revisions - parents
    return head


def _source_url() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    url = os.environ.get("BF_TWIN_DB_URL") or os.environ.get("DATABASE_URL") or ""
    if not url or url == "off" or url.startswith("sqlite"):
        sys.exit("No source twin store: set BF_TWIN_DB_URL (or DATABASE_URL) to the store the app normally uses.")
    return url


def _naive(at: Any) -> Any:
    """A timestamp as naive UTC, so PostgreSQL's aware and SQLite's naive values compare."""
    if isinstance(at, datetime) and at.tzinfo is not None:
        return at.astimezone(UTC).replace(tzinfo=None)
    return at


def _clean(row: dict[str, Any], people: tuple[str, ...]) -> dict[str, Any]:
    from utils import demo

    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in people:
            out[key] = demo.person(value)
        elif key == "job_id":
            out[key] = None  # the job records stay behind
        elif isinstance(value, (str, dict, list)):
            out[key] = demo.redact(value)
        else:
            out[key] = value
    return out


def seed(
    source_url: str,
    target_url: str = DEMO_URL,
    *,
    days: int = 4,
    report_days: int = 10,
    keep_failed: bool = False,
) -> dict[str, int]:
    """Copy the recent records of ``source_url`` into a fresh store at ``target_url``; rows copied per table."""
    from sqlalchemy import select, text

    from data.twin.engine import build_engine
    from data.twin.models import TwinBase

    source = build_engine(source_url, application_name="demo_seed")
    target = build_engine(target_url)
    TwinBase.metadata.drop_all(target)
    TwinBase.metadata.create_all(target)
    with target.begin() as con:  # stamped at the head, as `alembic -n twin upgrade head` would leave it
        con.execute(text("DROP TABLE IF EXISTS alembic_version"))
        con.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"))
        con.execute(text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": head_revision()})
    tables = {t.name: t for t in TwinBase.metadata.sorted_tables}
    now = datetime.now(UTC)
    windows = {"days": now - timedelta(days=days), "report_days": now - timedelta(days=report_days)}
    copied: dict[str, int] = {}
    run_ids: set[Any] = set()
    clean_ends: set[Any] = set()  # window ends of the twin windows that evaluated
    with source.connect() as src, target.begin() as dst:
        for name, time_col, window, people in TABLES:
            table = tables[name]
            rows = [dict(r) for r in src.execute(select(table).where(table.c[time_col] >= windows[window])).mappings()]
            if name in ("optimiser_commentary", "plan_decision"):
                rows = [r for r in rows if r["run_id"] in run_ids]
            if name == "snapshot":
                clean_ends = {_naive(r["window_end"]) for r in rows if r.get("status") == "complete"}
                if not keep_failed:
                    rows = [r for r in rows if r.get("status") == "complete"]
            if name in ("shift_report", "shift_record") and not keep_failed:
                rows = [r for r in rows if _naive(r["shift_end"]) in clean_ends]
            if name == "shift_record":
                for r in rows:
                    if r.get("plan_run_id") not in run_ids:
                        r["plan_run_id"] = None
            clean = [_clean(r, people) for r in rows]
            if clean:
                dst.execute(table.insert(), clean)
            if name == "optimiser_run":
                run_ids = {r["id"] for r in rows}
            copied[name] = len(clean)
    source.dispose()
    target.dispose()
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=4, help="twin windows and optimiser runs of the last N days")
    parser.add_argument(
        "--report-days", type=int, default=10, help="twin windows, shift reports and records of the last N days"
    )
    parser.add_argument(
        "--keep-failed", action="store_true", help="keep the shifts whose closing twin window did not evaluate"
    )
    args = parser.parse_args(argv)
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    copied = seed(_source_url(), days=args.days, report_days=args.report_days, keep_failed=args.keep_failed)
    for name, n in copied.items():
        print(f"{name:22} {n:5} rows")
    print(f"demo twin store: {DEMO_DB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
