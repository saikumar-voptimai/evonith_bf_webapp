"""Runtime of the optimiser runs and network solves on this host (SV-1.7 #287, SV-1.8 #288).

    python scripts/optimiser_bench.py [--mode step2|step1|max_production|jacobian|solve] [--calibration day|live]
                                      [--solvers N] [--compare] [--fresh-jacobian] [--limit-s 90]

Builds page 11's default cases as the worker does (live yard and calibration when Neon and InfluxDB answer, else the
reference yard and the shipped calibration, as stated in the output) and solves them as the scheduled job does:
Step 2 with STRATA-ω and Fixed-ω in parallel in a 2-process pool, Step 1 and maximum production with STRATA-ω, each
case with its network solves in a pool of ``--solvers`` processes (``BF_STRATA_SOLVERS``, else min(8, cpus); 0 or 1
= one after the other). ``--compare`` runs the cases serially as well and checks that the plans and the network solve
counts are identical. ``--mode jacobian`` times the 17-solve Jacobian of the calibration, serial and on started solver
processes (their start timed apart: it is spent once per run); ``--mode solve`` one network solve. The Jacobian of
the calibration is read from its disk cache, as after ``calibration.nightly``; ``--fresh-jacobian`` computes it again
first. Exit code 1 when the wall time of the solves exceeds ``--limit-s`` (or a comparison differs).
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


def _solve(args: tuple, solvers: int | None) -> tuple[dict, float, tuple[int, float]]:
    """Runs in a pool process: one case (its network solves in a solver pool), its time and the solves made."""
    from services import optimiser
    from utils.burden_engine import strata_network

    strata_network.clear_pool()  # every case starts from the calibration anchor, as in a fresh worker process
    n0, s0 = strata_network.solve_stats()
    t0 = time.perf_counter()
    res = optimiser.solve_case(*args, solvers=solvers)
    n1, s1 = strata_network.solve_stats()
    return res, time.perf_counter() - t0, (n1 - n0, s1 - s0)


def _cached_jacobian(cal_json: str) -> tuple[str, Path | None]:
    from services import day_calibration
    from utils.burden_engine import strata_network

    source = str(day_calibration.network_jacobian(cal_json, solvers=0).get("jacobian", ""))
    path = strata_network.CACHE_DIR / source.split()[1] if source.startswith("cache ") else None
    return source, path


def _time_jacobian(cal_json: str, solvers: int) -> float:
    """Seconds for the 17 solves of the calibration's Jacobian, its cache file moved aside meanwhile."""
    from services import day_calibration
    from utils.burden_engine import strata_network

    _, path = _cached_jacobian(cal_json)
    backup = path.with_name(path.name + ".bak") if path is not None else None
    if path is not None:
        path.replace(backup)
    strata_network.clear_pool()
    try:
        t0 = time.perf_counter()
        day_calibration.network_jacobian(cal_json, solvers=solvers)
        return time.perf_counter() - t0
    finally:
        if backup is not None:
            backup.replace(path)  # the result is the same; keep the original file
        strata_network.clear_pool()


def _plan_line(mode: str, case, res: dict, seconds: float, solves: tuple[int, float]) -> str:
    from services import optimiser

    plan = optimiser.plan_of(res) or {}
    out, cost = plan.get("out") or {}, plan.get("cost") or {}
    return (f"{mode} {case.omega_mode:>6}: {seconds:6.1f} s, {solves[0]} network solves ({solves[1]:.1f} s solving); "
            f"{plan.get('P_tday', 0):.0f} t/d, coke {out.get('K', float('nan')):.2f} kg/tHM, "
            f"cost {cost.get('total', float('nan')):,.1f} Rs/tHM")  # fmt: skip


def _same(a: dict, b: dict) -> bool:
    from services import optimiser

    pa, pb = optimiser.plan_of(a) or {}, optimiser.plan_of(b) or {}
    keys = ("P_tday",)
    return (
        all(pa.get(k) == pb.get(k) for k in keys)
        and (pa.get("out") or {}).get("K") == (pb.get("out") or {}).get("K")
        and (pa.get("cost") or {}).get("total") == (pb.get("cost") or {}).get("total")
        and json.dumps(pa.get("share_wet"), sort_keys=True) == json.dumps(pb.get("share_wet"), sort_keys=True)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("step2", "step1", "max_production", "jacobian", "solve"), default="step2")
    ap.add_argument("--calibration", choices=("day", "live"), default="day")
    ap.add_argument("--solvers", type=int, default=None, help="solver processes per case (default: BF_STRATA_SOLVERS)")
    ap.add_argument("--compare", action="store_true", help="also solve serially and compare plans and solve counts")
    ap.add_argument("--fresh-jacobian", action="store_true", help="compute the network Jacobian again first")
    ap.add_argument("--limit-s", type=float, default=90.0, help="wall-time limit of the solves, s")
    args = ap.parse_args(argv)

    from services import optimiser
    from utils.burden_engine import strata_network

    solvers = strata_network.solver_count() if args.solvers is None else args.solvers
    t0 = time.perf_counter()
    mode = "step2" if args.mode in ("jacobian", "solve") else args.mode
    modes = ("strata", "fixed") if mode == "step2" else ("strata",)
    cases = [optimiser.default_case(mode, omega_mode=m, calibration_basis=args.calibration) for m in modes]
    first = cases[0]
    print(f"cases built in {time.perf_counter() - t0:.1f} s: yard {first.yard_source}, calibration "
          f"{first.calibration.get('source')} ({first.calibration.get('basis')}, {first.calibration.get('window')}), "
          f"cpus {os.cpu_count()}, solver processes {solvers}")  # fmt: skip
    cal_json = first.args[2]

    if args.mode == "solve":
        from utils.burden_engine.engine import Engine

        net = Engine(cal=json.loads(cal_json)).omega_model().net
        times = []
        for _ in range(4):
            t1 = time.perf_counter()
            net.solve(net.U[0] + 0.0)
            times.append(time.perf_counter() - t1)
        print(
            f"one network solve: {times[0]:.2f} s the first in this process (it builds the STRATA-Flow context), "
            f"then {min(times[1:]):.2f}-{max(times[1:]):.2f} s"
        )
        return 0
    if args.mode == "jacobian":
        serial = _time_jacobian(cal_json, 0)
        t1 = time.perf_counter()
        pool = strata_network.start_solvers(solvers)  # started once per run; timed apart from the Jacobian
        started = time.perf_counter() - t1
        try:
            with strata_network.parallel_solves(pool):
                parallel = _time_jacobian(cal_json, None)  # on the started processes
        finally:
            strata_network.stop_solvers(pool)
        pre = strata_network.LAST_START.get("preloaded", 0) if pool is not None else 0
        print(
            f"network Jacobian (17 solves): serial {serial:.1f} s; {solvers} solver processes started in "
            f"{started:.1f} s ({pre} with the STRATA-Flow context from the forkserver), then the Jacobian "
            f"{parallel:.1f} s"
        )
        return 0 if parallel <= args.limit_s else 1

    source, _ = _cached_jacobian(cal_json)
    if args.fresh_jacobian:
        print(f"network Jacobian computed again: {_time_jacobian(cal_json, solvers):.1f} s")
    else:
        print(f"network Jacobian: {source}")

    ctx = multiprocessing.get_context("forkserver" if os.name == "posix" else "spawn")
    ok = True
    runs = [("parallel", solvers)] + ([("serial", 0)] if args.compare else [])
    results: dict[str, list] = {}
    for label, n in runs:
        with ProcessPoolExecutor(max_workers=len(cases), mp_context=ctx) as pool:
            list(pool.map(abs, range(len(cases))))  # start the processes before the clock
            t1 = time.perf_counter()
            futures = [pool.submit(_solve, case.args, n) for case in cases]
            results[label] = [f.result() for f in futures]
            wall = time.perf_counter() - t1
        for case, (res, seconds, solves) in zip(cases, results[label], strict=True):
            print(f"[{label}] " + _plan_line(mode, case, res, seconds, solves))
        print(f"[{label}] wall time of the solves {wall:.1f} s")
        if label == "parallel":
            ok &= wall <= args.limit_s
    if args.compare:
        for case, (a, _, sa), (b, _, sb) in zip(cases, results["parallel"], results["serial"], strict=True):
            same = _same(a, b) and sa[0] == sb[0]
            ok &= same
            verdict = "identical" if same else "DIFFER"
            print(f"{case.omega_mode}: plan and solve count {verdict} (solves {sa[0]} / {sb[0]})")
    print("within the limit" if ok else "above the limit or different")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
