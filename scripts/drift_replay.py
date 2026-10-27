"""Replay of the implied-omega drift alarm (SV-1.2, #282): alarm dates and in-control run length.

    python scripts/drift_replay.py [--source evidence|snapshots] [--from 2026-01-01] [--to 2026-10-07] [--out r.json]

``evidence`` (default): the daily series of the STRATA-Learn evidence (``src/assets/burden_engine/
strata_evidence_seed.csv``, 191 stable days, 15 Dec 2025 - 7 Oct 2026): r = (omega of the day - its 14-day anchor) -
STRATA 1.5 posterior prediction from the day's network channels; lag 1 day, gaps up to 7 days bridged, clear after
3 days. The posterior was fitted on these days, so the replay is in sample; rho and sigma_e are re-estimated on the
series (``--posterior`` uses the posterior's 0.682 / 0.0172 instead).

``snapshots``: the hourly windows stored in the twin store (``BF_TWIN_DB_URL`` / ``DATABASE_URL``) with the live
settings of ``config/coke_energy.yml`` (``twin.drift``): lag 8 h, gaps up to 3 windows, clear after 24 h.

Reports the AR(1) parameters, every alarm episode (raised, cleared, direction, signals, peak CUSUM), the largest sums
and their dates, the run lengths of the chains in evaluations (completed and censored), the estimated in-control ARL
and Siegmund's nominal ARL for k and h.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]

EVIDENCE = ROOT / "src" / "assets" / "burden_engine" / "strata_evidence_seed.csv"
ASSET = ROOT / "src" / "assets" / "burden_engine" / "omega_model.json"


def evidence_series(posterior: bool):
    import pandas as pd

    from utils.coke_energy import drift

    cs = json.loads(ASSET.read_text(encoding="utf-8"))["channel_scales"]
    r = drift.evidence_residuals(pd.read_csv(EVIDENCE, parse_dates=["date"]), cs)
    settings = drift.DriftSettings(
        lag=pd.Timedelta(days=1),
        max_gap=7,
        min_pairs=30,
        rho=cs["rho"] if posterior else None,
        sigma_e=cs["sigma_e"] if posterior else None,
    )
    return r, settings


def snapshot_series():
    from data.twin.repositories import TwinStore
    from services import coke_energy as ce
    from services import drift as service

    store = TwinStore.from_url()
    if store is None:
        raise SystemExit("no twin store: set BF_TWIN_DB_URL or DATABASE_URL")
    rows = store.snapshots.series(*_span(store))
    return service.residual_series(rows), service.DriftConfig.from_config(ce.config()).settings


def _span(store):
    from datetime import UTC, datetime

    first = store.snapshots.series(datetime(2000, 1, 1, tzinfo=UTC), datetime.now(UTC))
    if not first:
        raise SystemExit("the twin store holds no snapshots")
    return first[0].window_end.replace(tzinfo=UTC), datetime.now(UTC)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=("evidence", "snapshots"), default="evidence")
    ap.add_argument("--from", dest="start", default="2026-01-01", help="first date reported (IST date)")
    ap.add_argument("--to", dest="end", default="2026-10-07", help="last date reported")
    ap.add_argument("--posterior", action="store_true", help="evidence: rho and sigma_e of the posterior")
    ap.add_argument("--out", type=Path, help="write the report as JSON")
    args = ap.parse_args(argv)

    import pandas as pd

    from utils.coke_energy import drift

    r, settings = evidence_series(args.posterior) if args.source == "evidence" else snapshot_series()
    report = drift.evaluate(r, settings)
    if report.ar1 is None:
        print(f"no evaluation: {report.note}")
        return 1
    ar = report.ar1
    tz = r.index.tz
    lo, hi = pd.Timestamp(args.start, tz=tz), pd.Timestamp(args.end, tz=tz) + pd.Timedelta(days=1)
    pts = report.points[(report.points.index >= lo) & (report.points.index < hi)]
    eps = [e for e in report.episodes if lo <= e.raised_at < hi]
    done, censored = drift.run_lengths(pts, settings.lag)
    arl = drift.arl_estimate(done, censored)
    nominal = drift.siegmund_arl(settings.k, settings.h)
    hours = settings.lag / pd.Timedelta(hours=1)
    lag = f"{hours / 24:g} day" if hours % 24 == 0 else f"{hours:g} h"
    print(
        f"source {args.source}: {len(pts)} evaluations {pts.index.min():%Y-%m-%d} to {pts.index.max():%Y-%m-%d}, "
        f"lag {lag}"
    )
    print(
        f"AR(1) ({ar.source}): rho {ar.rho:.3f}, sigma_e {ar.sigma_e:.4f}, mean of r {ar.mu_hat:+.4f} "
        f"(in-control mean {ar.mu:g}), {ar.n_pairs} pairs; CUSUM k {settings.k:g}, h {settings.h:g}"
    )
    print(
        f"largest S+ {pts.s_pos.max():.2f} on {pts.s_pos.idxmax():%Y-%m-%d %H:%M}, "
        f"largest S- {pts.s_neg.max():.2f} on {pts.s_neg.idxmax():%Y-%m-%d %H:%M}"
    )
    print(f"alarm episodes: {len(eps)}")
    for e in eps:
        cleared = f"{e.cleared_at:%Y-%m-%d %H:%M}" if e.cleared_at is not None else "open"
        print(
            f"  {e.direction:4s} raised {e.raised_at:%Y-%m-%d %H:%M}, cleared {cleared}, {e.signals} signals, "
            f"peak {e.peak:.1f}"
        )
    total = sum(done) + sum(censored)
    arl_text = f"{arl:.0f}" if arl is not None else f"> {total} (no signal)"
    print(
        f"run lengths (evaluations): completed {done or '-'}, censored {censored}; in-control ARL {arl_text}; "
        f"nominal ARL for iid N(0, 1) innovations {nominal:.0f}"
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "source": args.source,
                    "ar1": vars(ar),
                    "settings": {
                        "lag": str(settings.lag),
                        "k": settings.k,
                        "h": settings.h,
                        "mu": settings.mu,
                        "clear_after": str(settings.clear),
                        "max_gap": settings.max_gap,
                    },
                    "episodes": [
                        {
                            "direction": e.direction,
                            "raised_at": str(e.raised_at),
                            "cleared_at": str(e.cleared_at),
                            "signals": e.signals,
                            "peak": e.peak,
                        }
                        for e in eps
                    ],
                    "run_lengths": {"completed": done, "censored": censored, "arl": arl, "nominal": nominal},
                    "points": [
                        {"t": str(t), **{k: (None if pd.isna(v) else v) for k, v in row.items()}}
                        for t, row in pts.iterrows()
                    ],
                },
                indent=1,
                default=str,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
