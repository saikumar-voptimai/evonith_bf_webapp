"""Plant-data parity of the twin window evaluation with the Coke Energy pipeline (SV-1.1, #281; gate G1).

    python scripts/twin_parity.py [--days 14] [--strata] [--out report.json]

Loads the Coke Energy sources as page 10 does (``services.coke_energy``: hourly dataset, 10-min InfluxDB data, Neon
HM/slag and DPR), runs the pipeline, evaluates the window of every balanced shift dated in the last ``--days`` days
(``Window.of_shift``, ``utils.coke_energy.twin.evaluate_window``) and compares K_TRACE, K_FORGE, omega_imp, K_charged
and omega_cal with the pipeline's shift table (``K_burnt``, ``K_energy``, ``omega_shift``, ``K_charged``,
``omega_fixed``). Exit status 1 when a difference exceeds 0.1 kg/tHM or 0.0005 in omega, or a value exists on one
side only (acceptance criterion of #281). ``--strata`` also evaluates K_STRATA for every window and reports the
network solves and the time of each evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]

TOL_K = 0.1  # kg/tHM
TOL_OMEGA = 0.0005
PAIRS = (  # (twin attribute, pipeline column, tolerance)
    ("k_trace", "K_burnt", TOL_K),
    ("k_forge", "K_energy", TOL_K),
    ("omega_imp", "omega_shift", TOL_OMEGA),
    ("k_charged", "K_charged", TOL_K),
    ("omega_cal", "omega_fixed", TOL_OMEGA),
)


def _value(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=14, help="shifts dated in the last DAYS days (default 14)")
    ap.add_argument("--strata", action="store_true", help="also evaluate K_STRATA (network solves)")
    ap.add_argument("--out", type=Path, help="write the per-shift comparison as JSON")
    args = ap.parse_args(argv)

    import pandas as pd

    from services import coke_energy as ce
    from services.twin import load_frames
    from utils.coke_energy.pipeline import PipelineSettings
    from utils.coke_energy.twin import Window, evaluate_window

    cfg = ce.config()
    settings = PipelineSettings.from_config(cfg)
    keys = ce.data_keys(cfg, 0)
    t0 = time.perf_counter()
    frames = load_frames(keys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pool = ce.page_data(keys, settings).result
    print(f"sources and pipeline: {time.perf_counter() - t0:.0f} s, {len(pool.shifts)} balanced shifts")

    T = pool.shifts
    last = T.sdate.max()
    T = T[T.sdate > last - pd.Timedelta(days=args.days)]
    worst = {name: 0.0 for name, _, _ in PAIRS}
    failures: list[str] = []
    rows: list[dict] = []
    seconds: list[float] = []
    for skey, row in T.iterrows():
        t1 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = evaluate_window(
                frames.fd,
                frames.m10,
                frames.hm,
                frames.dpr,
                Window.of_shift(str(skey)),
                pool,
                settings,
                frames.provisional,
                strata=args.strata,
            )
        seconds.append(time.perf_counter() - t1)
        out = {"shift": str(skey), "status": ev.status, "seconds": round(seconds[-1], 2), "notes": ev.notes}
        for name, col, tol in PAIRS:
            a, b = getattr(ev, name), _value(row[col])
            out[name], out[col] = a, b
            if (a is None) != (b is None):
                failures.append(f"{skey}: {name} {a} against {col} {b}")
            elif a is not None:
                d = abs(a - b)
                worst[name] = max(worst[name], d)
                if d > tol:
                    failures.append(f"{skey}: {name} differs by {d:.6f} (tolerance {tol})")
        if args.strata:
            out.update(
                k_strata=ev.k_strata,
                k_strata_sd=ev.k_strata_sd,
                omega_strata=ev.omega_strata,
                network_solves=(ev.flags.get("strata") or {}).get("network_solves"),
            )
        rows.append(out)

    print(f"{len(rows)} shifts from {T.sdate.min():%Y-%m-%d} to {last:%Y-%m-%d}")
    for name, col, tol in PAIRS:
        print(f"  {name:10s} vs {col:12s} largest difference {worst[name]:.6f} (tolerance {tol})")
    if seconds:
        print(
            f"  evaluation time per window: median {sorted(seconds)[len(seconds) // 2]:.1f} s, max {max(seconds):.1f} s"
        )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    for line in failures:
        print("FAIL", line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
