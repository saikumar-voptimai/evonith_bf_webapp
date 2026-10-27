"""Parallel STRATA-Flow network solves (SV-1.8, #288): solved together, the states give the same anchors, the same
ring quantities and the same solve counts as one after the other. Fast tests replace the network by a deterministic
function and run the "parallel" solves in a thread pool; the slow tests run the network in two solver processes."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

import numpy as np
import pytest

from utils.burden_engine import strata_network as sn

CTX = dict(noncoke_t_per_charge=40.0, limestone_kg_thm=20.0, hbt_c=1150.0, blast_moisture=15.0, top_eta_co=48.0,
           top_h2=3.0, top_gas_T_c=140.0, matrix={})  # fmt: skip


@dataclass
class State:
    """The attributes ``NetworkParts.vec`` reads."""

    shares: dict[str, float] = field(default_factory=lambda: {"pellet": 0.2})
    coke: float = 300.0
    nut: float = 70.0
    pci: float = 190.0
    prod: float = 95.0
    blast_t: float = 1100.0
    o2: float = 0.28
    rdi: float = 25.0
    fer: float = 1600.0
    humidity: float = 15.0


def fake_solve(state):
    """A deterministic stand-in for the network: y from the state's numbers."""
    x = np.array([state["coke_kg_thm"], state["production_t_h"], state["enrichment_true"], state["pci_kg_thm"]])
    y = np.concatenate([np.sin(x * (k + 1) / 100.0) for k in range(5)])[:18]
    return [float(v) for v in y], 0.01


@pytest.fixture()
def fake_network(monkeypatch):
    monkeypatch.setattr(sn, "solve_state", fake_solve)
    yield
    sn.set_executor(None)


def states():
    """Ten states: repeats, near repeats (within 1 scaled unit) and new ones, in an order that mixes them."""
    return [
        State(), State(coke=302.0), State(prod=97.5), State(coke=301.0), State(), State(o2=0.29), State(prod=97.0),
        State(coke=320.0), State(coke=319.0, prod=95.4), State(pci=180.0),
    ]  # fmt: skip


def serial_and_parallel(tol: float):
    serial = sn.NetworkParts(CTX, cache_dir=None)
    ys_serial = [serial.anchor(st, tol) for st in states()]
    batch = sn.NetworkParts(CTX, cache_dir=None)
    with sn.parallel_solves(ThreadPoolExecutor(4)):
        ys_batch = batch.anchor_many(states(), tol)
    return serial, ys_serial, batch, ys_batch


@pytest.mark.parametrize("tol", [1e-9, 1.0, 3.0])
def test_states_anchored_together_equal_one_after_the_other(fake_network, tol) -> None:
    serial, ys_serial, batch, ys_batch = serial_and_parallel(tol)
    assert batch.n_solves == serial.n_solves
    np.testing.assert_array_equal(batch.U, serial.U)
    np.testing.assert_array_equal(batch.Y, serial.Y)
    for a, b in zip(ys_batch, ys_serial, strict=True):
        np.testing.assert_array_equal(a, b)


def test_reuse_within_the_tolerance_counts_as_one_solve(fake_network) -> None:
    exact = serial_and_parallel(1e-9)[2]
    coarse = serial_and_parallel(1.0)[2]
    assert exact.n_solves == 9  # the repeat of the first state reuses its solve
    assert coarse.n_solves < exact.n_solves  # the near repeats reuse too


def test_the_jacobian_is_the_same_in_parallel(fake_network) -> None:
    serial = sn.NetworkParts(CTX, cache_dir=None).set_reference(State())
    with sn.parallel_solves(ThreadPoolExecutor(4)):
        parallel = sn.NetworkParts(CTX, cache_dir=None).set_reference(State())
    np.testing.assert_array_equal(parallel.J, serial.J)
    assert parallel.n_solves == serial.n_solves == len(sn.INPUTS) + 1


def test_without_an_executor_states_are_solved_in_this_process(fake_network, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(sn, "solve_state", lambda st: calls.append(st) or fake_solve(st))
    net = sn.NetworkParts(CTX, cache_dir=None)
    net.anchor_many([State(), State(coke=330.0)])
    assert len(calls) == 2 and net.n_solves == 2


def test_the_executor_holds_for_its_own_thread_or_task() -> None:
    """Set in one thread (a Streamlit session) or asyncio task (a worker job), the executor is not that of another
    thread; the threads ``asyncio.to_thread`` starts from the task carry it."""
    ex = ThreadPoolExecutor(1)
    seen: dict[str, object] = {}

    async def job() -> None:
        with sn.parallel_solves(ex):
            seen["to_thread"] = await asyncio.to_thread(sn.executor)
            other = threading.Thread(target=lambda: seen.__setitem__("thread", sn.executor()))
            other.start()
            other.join()

    asyncio.run(job())
    assert seen == {"to_thread": ex, "thread": None}
    assert sn.executor() is None
    ex.shutdown()


def test_solver_processes_inside_a_block_with_an_executor_use_it() -> None:
    """A run started inside a block that has solver processes (``solve_case`` in a sweep) starts no second pool."""
    ex = ThreadPoolExecutor(1)
    with sn.parallel_solves(ex):
        with sn.solver_processes(4) as inner:
            assert inner is ex and sn.executor() is ex
        assert sn.executor() is ex and not ex._shutdown  # neither replaced nor stopped
    ex.shutdown()


class BrokenPool:
    """An executor whose processes have died."""

    def map(self, fn, items):
        raise BrokenProcessPool("a solver process was killed")


def test_a_broken_pool_leaves_the_solves_to_this_process(fake_network, caplog) -> None:
    serial = sn.NetworkParts(CTX, cache_dir=None)
    ys_serial = serial.anchor_many(states())
    net = sn.NetworkParts(CTX, cache_dir=None)
    with caplog.at_level(logging.WARNING, logger=sn.__name__), sn.parallel_solves(BrokenPool()):
        ys = net.anchor_many(states())
        assert sn.executor() is None  # left for the rest of the block
    assert "solving one after the other" in caplog.text
    assert net.n_solves == serial.n_solves
    for a, b in zip(ys, ys_serial, strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_main_module_is_restored_unless_another_thread_set_its_own(monkeypatch) -> None:
    main, other = types.ModuleType("__main__"), types.ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", main)
    with sn._main_without_script():
        assert sys.modules["__main__"] is not main and not hasattr(sys.modules["__main__"], "__file__")
    assert sys.modules["__main__"] is main
    with sn._main_without_script():
        sys.modules["__main__"] = other  # a Streamlit script run of another session meanwhile
    assert sys.modules["__main__"] is other


def test_solver_processes_start_under_a_main_script_without_a_guard(monkeypatch, tmp_path) -> None:
    """A Streamlit page or an app test is ``__main__`` without a main guard; the solver processes must not run it."""
    script = tmp_path / "page.py"
    script.write_text("raise SystemExit('the main script ran in a solver process')\n")
    main = types.ModuleType("__main__")
    main.__file__ = str(script)
    monkeypatch.setitem(sys.modules, "__main__", main)
    pool = sn.start_solvers(2)
    try:
        assert pool is not None and sn.LAST_START["processes"] == 2
        pids = set(pool.map(_pid, range(4)))
        assert pids and os.getpid() not in pids
    finally:
        sn.stop_solvers(pool)
    assert sys.modules["__main__"] is main


def _pid(_: int) -> int:
    return os.getpid()


@pytest.mark.skipif(sys.platform == "win32", reason="the forkserver start method is POSIX only")
def test_the_forkserver_builds_the_context_once_for_its_solver_processes() -> None:
    """With ``src`` on PYTHONPATH (as the worker runs) the forkserver imports ``strata_preload``: the solver processes
    of every pool start with the STRATA-Flow context built."""
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import json\n"
        "from utils.burden_engine import strata_network as sn\n"
        "out = []\n"
        "for _ in range(2):\n"
        "    pool = sn.start_solvers(2)\n"
        "    out.append(dict(sn.LAST_START))\n"
        "    sn.stop_solvers(pool)\n"
        "print(json.dumps(out))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "furnace_data")]))
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300, check=True)
    starts = json.loads(run.stdout.strip().splitlines()[-1])
    assert [s["processes"] for s in starts] == [2, 2]
    assert [s["preloaded"] for s in starts] == [2, 2]
    assert starts[1]["seconds"] < starts[0]["seconds"]  # the second pool forks from the running forkserver


def _failing_warm(_: int) -> int:
    raise RuntimeError("no STRATA-Flow context")


def test_a_pool_that_does_not_start_gives_serial_solves(monkeypatch, caplog) -> None:
    monkeypatch.setattr(sn, "_warm", _failing_warm)
    with caplog.at_level(logging.WARNING, logger=sn.__name__):
        assert sn.start_solvers(2) is None
    assert "did not start" in caplog.text
    assert sn.start_solvers(1) is None and sn.start_solvers(0) is None  # fewer than 2: serial, nothing started


@pytest.mark.skipif(os.environ.get("BURDEN_ENGINE_FAST") == "1", reason="network solves (BURDEN_ENGINE_FAST)")
def test_a_plans_three_states_solved_in_two_processes_equal_serial() -> None:
    """The real network: a plan's blend, production and operation states solved in two solver processes give the
    same ring quantities and anchors as solved in this process one after the other."""
    from utils.burden_engine.engine import Engine
    from utils.burden_engine.omega_model import PlanOmega

    eng = Engine()
    model = eng.omega_model()
    ref = model.ref
    plan_state = type(ref)(dict(ref.shares) | {"pellet": ref.shares.get("pellet", 0.0) + 0.05}, ref.fe_frac, ref.cls,
                           ref.coke + 8.0, ref.nut, ref.pci, ref.prod + 3.0, ref.blast_t, ref.o2, ref.rdi, ref.fer,
                           ref.humidity)  # fmt: skip
    todo = PlanOmega.states(model, plan_state, plan_state.coke)

    def run(executor):
        net = sn.NetworkParts(model.net.ctx, cache_dir=None, jacobian=False)
        net.U, net.Y = model.net.U[:1].copy(), model.net.Y[:1].copy()  # the calibration anchor only
        with sn.parallel_solves(executor):
            ys = net.anchor_many(todo)
        return net, ys

    serial, ys_serial = run(None)
    pool = sn.start_solvers(2)
    assert pool is not None
    try:
        parallel, ys_parallel = run(pool)
    finally:
        sn.stop_solvers(pool)
    assert parallel.n_solves == serial.n_solves == 3
    np.testing.assert_allclose(parallel.Y, serial.Y, rtol=0, atol=1e-12)
    for a, b in zip(ys_parallel, ys_serial, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------------- independent items
def add_anchor(eng, k: int) -> int:
    """An item that adds one anchor of its own (a stand-in for a search that solves the network)."""
    net = eng.omega_model().net
    u = net.U[0].copy()
    u[-2] += 10.0 * (k + 1)  # ferrous rate: a state of its own per item
    net._add(u, net.Y[0] + k)
    net.n_solves += 1
    return len(net.U)


def test_independent_items_start_from_the_same_anchors_and_merge_in_order(monkeypatch) -> None:
    from utils.burden_engine.engine import Engine

    eng = Engine()
    net = eng.omega_model().net
    n_anchors, n_solves = len(net.U), net.n_solves
    seen = eng.run_items(add_anchor, [(k,) for k in range(4)], network=True)
    assert seen == [n_anchors + 1] * 4  # every item saw only the anchors present before the items, plus its own
    assert len(net.U) == n_anchors + 4 and net.n_solves == n_solves + 4
    np.testing.assert_array_equal(net.Y[n_anchors:, 0] - net.Y[0, 0], [0.0, 1.0, 2.0, 3.0])  # in item order
    assert eng.run_items(add_anchor, [(0,)], network=False) == [n_anchors + 5]  # Fixed-ω: one after the other


@pytest.mark.skipif(os.environ.get("BURDEN_ENGINE_FAST") == "1", reason="network solves (BURDEN_ENGINE_FAST)")
def test_independent_items_in_solver_processes_equal_serial() -> None:
    """The real network: the maximum production at two oxygen levels with STRATA-ω, in two solver processes and in
    this process one after the other: the same plans, anchors and solve counts."""
    from dataclasses import replace

    from utils.burden_engine.engine import Engine, Limits, _item_max_prod_at

    def run(executor):
        sn.clear_pool()
        eng = Engine()
        s = replace(eng.cal_settings(), omega_mode="blend")
        L = replace(Limits(), process=True)
        levels = [s.O2_per_air, s.O2_per_air + 0.006]
        items = [(eng.process_settings(L, r, s), L, True, False, None) for r in levels]
        with sn.parallel_solves(executor), eng.network_tolerance(eng.SEARCH_TOL):
            plans = eng.run_items(_item_max_prod_at, items, network=True)
        net = eng.omega_model().net
        return plans, net.U.copy(), net.n_solves

    serial = run(None)
    pool = sn.start_solvers(2)
    assert pool is not None
    try:
        parallel = run(pool)
    finally:
        sn.stop_solvers(pool)
    assert parallel[2] == serial[2]
    np.testing.assert_array_equal(parallel[1], serial[1])
    for a, b in zip(parallel[0], serial[0], strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert a["P_th"] == b["P_th"] and a["cost"]["total"] == b["cost"]["total"]
            np.testing.assert_array_equal(a["z"], b["z"])
    sn.clear_pool()
