"""Burden column at a time and the implied lag (SV-1.3, #283).

    python scripts/transit_report.py [--at "2026-10-09 04:00"] [--lag] [--days 14]

The column (descent times, charges and coke by zone, blend per band) comes from the hourly furnace dataset
(``src/assets/data/furnace_dataset.csv`` or the path in ``config/coke_energy.yml``); ``--at`` defaults to the end of
its last hour. ``--lag`` adds the implied lag between the coke content of the charges and K_TRACE of the windows
stored in the twin store (``BF_TWIN_DB_URL`` / ``DATABASE_URL``) over the ``--days`` before ``--at``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--at", help="IST time (default: end of the record's last hour)")
    ap.add_argument("--dataset", type=Path, help="hourly dataset CSV (default: the configured one)")
    ap.add_argument("--lag", action="store_true", help="implied lag from the stored windows")
    ap.add_argument("--days", type=float, help="days of the implied lag (default: the configured 14)")
    args = ap.parse_args(argv)

    import pandas as pd

    from data.coke_energy import sources
    from services import coke_energy as ce
    from services import transit

    cfg = ce.config()
    path = args.dataset or sources.static_dataset_status(cfg.get("data_sources", {}) or {})["path"]
    fd = sources.load_hourly_dataset(path)
    at = pd.Timestamp(args.at) if args.at else fd["time"].max() + pd.Timedelta(hours=1)
    st = transit.state(fd, at, cfg)
    q = transit.quantities(st)
    print(f"burden column at {at:%Y-%m-%d %H:%M} IST ({path})")
    print(
        f"  charging rate {st.charges_per_hour:.2f} charges/h; record complete {st.complete}, "
        f"missing hours {st.missing_hours}"
    )
    nan = float("nan")
    print(
        f"  descent, from the record: cohesive-zone top {q.get('tau_cz_h', nan):.2f} h, raceways "
        f"{q.get('tau_raceway_h', nan):.2f} h; at the current rate: {q.get('tau_cz_rate_h', nan):.2f} h, "
        f"{q.get('tau_raceway_rate_h', nan):.2f} h"
    )
    for b in st.bands:
        blend = ", ".join(
            f"{k.removesuffix('_CALC_MT').lower()} {v:.1f}" for k, v in b.blend_wet_pct.items() if v >= 0.05
        )
        print(
            f"  {b.zone:9s} {b.charges:5.2f} charges, {b.newest_h:4.2f}-{b.oldest_h:4.2f} h old, coke "
            f"{b.coke_inventory_t:5.1f} t ({b.coke_kg_per_t_ferrous:5.1f} kg/t ferrous, {b.coke_t_per_charge:4.2f} t "
            f"per charge); wet % of ferrous: {blend}"
        )
    print(f"  coke in transit {st.coke_in_transit_t:.1f} t")
    if args.lag:
        from datetime import UTC

        from data.twin.repositories import TwinStore

        store = TwinStore.from_url()
        if store is None:
            print("implied lag: no twin store (set BF_TWIN_DB_URL or DATABASE_URL)")
            return 1
        _, lag = transit.settings(cfg)
        days = args.days or lag.days
        end_utc = at.tz_localize("Asia/Kolkata").tz_convert(UTC).to_pydatetime()
        rows = store.snapshots.series(end_utc - pd.Timedelta(days=days + 1).to_pytimedelta(), end_utc)
        if args.days:
            cfg = {
                **cfg,
                "twin": {
                    **cfg.get("twin", {}),
                    "transit": {**cfg.get("twin", {}).get("transit", {}), "implied_lag": {**vars(lag), "days": days}},
                },
            }
        res = transit.implied_lag(fd, rows, at, cfg)
        print(
            f"implied lag over {days:g} days: {res.lag_h:.2f} h (95 % interval {res.ci[0]:.2f}-{res.ci[1]:.2f} h), "
            f"peak r {res.r_peak:.2f}, {res.n_hours} hourly windows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
