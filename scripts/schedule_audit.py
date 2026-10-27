"""Audit of the worker schedule (SV-1.7, #287): check of "14 consecutive days without a missed scheduled run".

    python scripts/schedule_audit.py [--days 14] [--end 2026-11-10T00:00:00+05:30] [--kind twin.evaluate ...]
                                     [--strict] [--list 20]

Reads ``twin.jobs`` (``BF_TWIN_DB_URL``, else ``DATABASE_URL``) and lists, per scheduled kind, the slots of the
period and what became of each: done, failed, cancelled, still running, skipped (worker not running at the slot, or
the previous job still active; the record gives the reason) or missed (no record at all); then the runtime statistics
per kind (runs, p50 / p95 / max execution time over the tries, p95 start delay). Exit code 1 when a slot was missed
or skipped (``--strict``: also when a scheduled run failed).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]


def _fmt(value: float | None, unit: str = " s") -> str:
    return "-" if value is None else f"{value:,.1f}{unit}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=float, default=14.0, help="period length, days")
    ap.add_argument("--end", help="period end, ISO time (default: now)")
    ap.add_argument("--kind", action="append", help="only these kinds (repeatable)")
    ap.add_argument("--strict", action="store_true", help="a failed scheduled run also fails the check")
    ap.add_argument("--list", type=int, default=20, help="slots not done listed per kind")
    args = ap.parse_args(argv)

    from data.twin.repositories import TwinStore
    from services import jobstats, schedule
    from services.jobs import CATALOG

    store = TwinStore.from_url(application_name="schedule-audit")
    if store is None:
        print("No twin store: set BF_TWIN_DB_URL or DATABASE_URL.", file=sys.stderr)
        return 2
    now = datetime.now(UTC)
    end = min(schedule.utc(args.end), now) if args.end else now
    end = min(end, now - schedule.SETTLE)
    start = end - timedelta(days=args.days)
    kinds = [k for k in CATALOG.values() if k.schedule is not None and (not args.kind or k.name in args.kind)]
    rows = store.jobs.window(start, end)
    table = schedule.audit(rows, kinds, start, end)
    ist = schedule.IST
    print(f"Schedule {start.astimezone(ist):%d %b %Y %H:%M} to {end.astimezone(ist):%d %b %Y %H:%M} IST "
          f"({args.days:g} d), {len(rows)} job records")  # fmt: skip
    print(f"{'kind':<22} {'schedule':<24} {'slots':>6} {'done':>6} {'failed':>6} {'running':>7} {'skipped':>7} "
          f"{'missed':>6}")  # fmt: skip
    failed = False
    for kind in kinds:
        a = table[kind.name]
        c = a.counts
        print(f"{kind.name:<22} {a.schedule:<24} {a.expected:>6} {c['done']:>6} {c['failed']:>6} {c['running']:>7} "
              f"{c['skipped']:>7} {c['missed']:>6}")  # fmt: skip
        failed |= not a.clean or (args.strict and c["failed"] > 0)
    for kind in kinds:
        problems = table[kind.name].problems
        if not problems:
            continue
        print(
            f"\n{kind.name}: {len(problems)} slot(s) not done"
            + (f", the latest {args.list}" if len(problems) > args.list else "")
        )
        for p in problems[-args.list :]:
            reason = f" - {p['reason']}" if p.get("reason") else ""
            print(f"  {p['slot'].astimezone(ist):%d %b %H:%M} IST  {p['state']:<9}{reason}")
    stats = jobstats.stats(rows, CATALOG.values(), start=start, end=end)
    print(f"\n{'kind':<22} {'runs':>5} {'done':>5} {'failed':>6} {'p50':>9} {'p95':>9} {'max':>9} {'delay p95':>10}")
    for st in stats.values():
        if st.runs == 0 and st.skipped == 0:
            continue
        print(f"{st.kind:<22} {st.runs:>5} {st.done:>5} {st.failed:>6} {_fmt(st.runtime_p50_s):>9} "
              f"{_fmt(st.runtime_p95_s):>9} {_fmt(st.runtime_max_s):>9} {_fmt(st.start_delay_p95_s):>10}")  # fmt: skip
    store.dispose()
    print(
        "\nclean" if not failed else "\nnot clean: missed or skipped slots" + (" or failed runs" if args.strict else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
