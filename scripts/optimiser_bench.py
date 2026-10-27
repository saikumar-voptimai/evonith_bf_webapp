"""Runtime of the scheduled optimiser runs on this host (SV-1.7, #287): check of "Step 2 < 90 s on the AGX Orin".

    python scripts/optimiser_bench.py [--mode step2|step1] [--calibration day|live] [--fresh-jacobian]
                                      [--limit-s 90]

Builds page 11's default cases as the worker does (live yard and calibration when Neon and InfluxDB answer, else the
reference yard and the shipped calibration, as stated in the output), then solves them as the scheduled job does:
Step 2 with STRATA-ω and Fixed-ω in parallel in a 2-process forkserver pool, Step 1 with STRATA-ω. The network
Jacobian of the calibration is read from its disk cache, as after ``calibration.nightly``; ``--fresh-jacobian``
computes it again first (17 network solves) and reports that time separately. Exit code 1 when the wall time of the
solves exceeds ``--limit-s``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]


def _solve(args: tuple) -> tuple[dict, float, tuple[int, float]]:
    """Runs in a pool process: one case, its time and the network solves it made."""
    from services import optimiser
    from utils.burden_engine.strata_network import solve_stats

    n0, s0 = solve_stats()
    t0 = time.perf_counter()
    res = optimiser.compute_mode(*args)
    n1, s1 = solve_stats()
    return res, time.perf_counter() - t0, (n1 - n0, s1 - s0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("step2", "step1"), default="step2")
    ap.add_argument("--calibration", choices=("day", "live"), default="day")
    ap.add_argument("--fresh-jacobian", action="store_true", help="compute the network Jacobian again first")
    ap.add_argument("--limit-s", type=float, default=90.0, help="wall-time limit of the solves, s")
    args = ap.parse_args(argv)

    from services import day_calibration, optimiser

    t0 = time.perf_counter()
    modes = ("strata", "fixed") if args.mode == "step2" else ("strata",)
    cases = [optimiser.default_case(args.mode, omega_mode=m, calibration_basis=args.calibration) for m in modes]
    first = cases[0]
    print(f"cases built in {time.perf_counter() - t0:.1f} s: yard {first.yard_source}, calibration "
          f"{first.calibration.get('source')} ({first.calibration.get('basis')}, {first.calibration.get('window')}), "
          f"cpus {os.cpu_count()}")  # fmt: skip
    if args.mode == "step2":
        print(f"production {first.summary()['production_target_t_d']:.0f} t/d ({first.production_source})")
    cal_json = first.args[2]
    stats = day_calibration.network_jacobian(cal_json)
    source = str(stats.get("jacobian", ""))
    if args.fresh_jacobian and source.startswith("cache "):
        from utils.burden_engine import strata_network

        path = strata_network.CACHE_DIR / source.split()[1]
        backup = path.with_name(path.name + ".bak")
        path.replace(backup)
        strata_network.clear_pool()
        try:
            stats = day_calibration.network_jacobian(cal_json)
        except BaseException:
            backup.replace(path)
            raise
        same = json.loads(backup.read_text())["J"] == json.loads(path.read_text())["J"]
        backup.unlink()
        print(f"network Jacobian computed again: {stats['jacobian']}, equal to the cached one: {same}")
    else:
        print(f"network Jacobian: {source}")

    ctx = multiprocessing.get_context("forkserver" if os.name == "posix" else "spawn")
    with ProcessPoolExecutor(max_workers=len(cases), mp_context=ctx) as pool:
        list(pool.map(abs, range(len(cases))))  # start the processes before the clock
        t1 = time.perf_counter()
        futures = [pool.submit(_solve, case.args) for case in cases]
        results = [f.result() for f in futures]
        wall = time.perf_counter() - t1
    for case, (res, seconds, (solves, solve_s)) in zip(cases, results, strict=True):
        plan = optimiser.plan_of(res) or {}
        out, cost = plan.get("out") or {}, plan.get("cost") or {}
        print(f"{args.mode} {case.omega_mode:>6}: {seconds:6.1f} s, {solves} network solves ({solve_s:.1f} s); "
              f"{plan.get('P_tday', 0):.0f} t/d, coke {out.get('K', float('nan')):.1f} kg/tHM, "
              f"cost {cost.get('total', float('nan')):,.0f} Rs/tHM")  # fmt: skip
    ok = wall <= args.limit_s
    print(f"wall time of the solves {wall:.1f} s ({'within' if ok else 'above'} {args.limit_s:g} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
