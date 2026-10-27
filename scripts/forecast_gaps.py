"""Gaps in the live forecast ledgers (SV-1.5, #285): check of "no gap above 10 min in the Si ledger over 7 days".

    python scripts/forecast_gaps.py [--days 7] [--end 2026-10-27T12:00:00+05:30] [--si-limit-min 10]
                                    [--charged-limit-min 60] [--si-storage DIR] [--charged-storage DIR]

Lists, for the Si ledger (origins) and the charged-coke ledger (data rows), every interval of more than the limit
between consecutive records in the window, with the attempts the worker or page 9 made within it (from the attempt
log of each storage folder): no attempt means nothing ran (worker stopped); attempts with a status mean the forecast
ran but its inputs were not usable (e.g. stale_inputs: no recent raw Si sample) or the run failed. With the shipped
single-horizon Si bundle the origins are hourly, so its gaps are measured against max(limit, 60 min). Exit code 1
when a gap exceeds its limit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=float, default=7.0, help="window length, days")
    ap.add_argument("--end", help="window end (default: now)")
    ap.add_argument("--si-limit-min", type=float, default=10.0)
    ap.add_argument("--charged-limit-min", type=float, default=60.0)
    ap.add_argument("--si-storage", type=Path, help="Si ledger folder (default: setting_bmo.yml)")
    ap.add_argument("--charged-storage", type=Path, help="charged-coke ledger folder (default: setting_bmo.yml)")
    args = ap.parse_args(argv)

    import pandas as pd

    from services import forecasts

    end = forecasts._utc(args.end) if args.end else pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=args.days)
    si_storage = args.si_storage or forecasts.si_paths().storage_dir
    charged_storage = args.charged_storage or forecasts.charged_paths().storage_dir
    si_limit = pd.Timedelta(minutes=args.si_limit_min)
    if args.si_storage is None:  # an hourly bundle issues hourly origins
        bundle = forecasts.active_si_bundle(si_storage, forecasts.si_paths().default_bundle)
        si_limit = max(si_limit, pd.Timedelta(minutes=forecasts.si_cadence(bundle)))
    failed = False
    for name, storage, limit in (
        ("si", si_storage, si_limit),
        ("charged_coke", charged_storage, pd.Timedelta(minutes=args.charged_limit_min)),
    ):
        times = [t for t in forecasts.ledger_times(name, storage) if start <= t <= end]
        found = forecasts.gaps(name, storage, start=start, end=end, limit=limit)
        failed |= bool(found)
        print(
            f"{name}: {len(times)} records in {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC; "
            f"{len(found)} gap(s) above {limit.total_seconds() / 60:g} min"
        )
        for gap in found:
            ist = forecasts.PLANT_TZ
            statuses = ", ".join(f"{k} {v}" for k, v in sorted(gap.statuses.items())) or "no attempt"
            print(
                f"  {gap.start.tz_convert(ist):%d %b %H:%M} to {gap.end.tz_convert(ist):%d %b %H:%M} IST "
                f"({gap.minutes:.0f} min): {gap.attempts} attempt(s): {statuses}"
                + (f"; errors: {' | '.join(gap.errors)}" if gap.errors else "")
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
