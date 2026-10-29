"""STRATA-Flow network inside the optimiser: ring quantities f, o, t from the network itself, no surrogate (STRATA 1.3).

STRATA 1.2 took the ring gas shares f_i, Fe shares o_i and thermal-reserve-zone residence times t_i from a quadratic
surrogate fitted offline to 520 network runs. STRATA 1.3 runs the network (``utils.strata_flow``: burden profile,
layered Ergun resistance, floating cohesive zone; ~1.1 s per solve) during the optimisation:

  exact(st)          one network solve for an omega state (blend, coke, nut coke, PCI, production, blast, O2, RDI,
                     ferrous rate, blast moisture) with the calibration window's charging matrix and top gas.
  anchor(st)         exact solve, stored as an anchor (u_a, f_a, o_a, t_a).
  parts(st)          ring quantities of a state u: the nearest anchor plus the network Jacobian,
                       (f, o, t)(u) = (f, o, t)(u_a) + J (u - u_a)
                     J is the network's own first derivative (forward differences, one solve per input) at the
                     calibration state, cached on disk per calibration. Exact at every anchor.

The engine anchors the current iterate's blend state at every SLP iteration (one solve), the plan's three states
(blend, blend + production, full operation) at convergence before one more LP step, and the three states of the
reported result, so every reported omega comes from network solves at the plan itself; the Jacobian is used only
between anchors (inside one iteration's finite differences and the coke fixed point of the balance) and the distance
from the nearest anchor is reported. Production searches run with a reuse tolerance (``Engine.network_tolerance``)
and re-solve the plan they return exactly (``Engine.refine``).

Exact mode (``jacobian=False``, the twin's window evaluation, SV-1.1): no Jacobian; ``parts(st)`` solves every state
that has no solve yet (one solve per distinct state, kept as an anchor). Used where the states are known in advance
and few (the anchor, then blend, blend + production and full operation of one window), so that the 17 solves of a
Jacobian per anchor are not spent.

Parallel solves (SV-1.8): states that do not depend on each other are solved together (``solve_many``,
``anchor_many``, the 17 solves of a Jacobian) in the executor of the current thread or task (``parallel_solves``;
solver processes from ``start_solvers`` / ``solver_processes``), with the same anchors and solve counts as one after
the other. The anchors of a NetworkParts are shared by the engines of its calibration in the process and locked
(``NetworkParts.lock``).

Inputs u (network units): wet share of each ferrous material except sinter (the sinter share closes the sum), coke,
nut coke, PCI (kg/tHM), production (t/h), blast per tonne (Nm3/tHM), O2 fraction of the blast, sinter RDI (%),
ferrous rate (kg/tHM wet), blast moisture (g/Nm3). Pure numerical code; reads and writes one JSON cache file.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import math
import threading
import time
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

SRC = Path(__file__).resolve().parents[2]
CACHE_DIR = SRC / "storage" / "strata_flow"
FERROUS = ("sinter", "pellet", "lloyds_clo", "geomin", "nmdc_rom", "donimalai", "lloyds_oversize", "gomti")
SHARE_INPUTS = FERROUS[1:]
SCALAR_INPUTS = ("coke", "nut", "pci", "prod", "blast_t", "o2", "rdi", "fer", "humidity")
INPUTS = tuple("sh_" + m for m in SHARE_INPUTS) + SCALAR_INPUTS
# forward-difference step per input (network units) and the distance scale of the nearest-anchor search
STEP = dict(sh=0.02, coke=10.0, nut=5.0, pci=10.0, prod=2.0, blast_t=20.0, o2=0.005, rdi=1.0, fer=20.0, humidity=2.0)
SCALE = dict(sh=0.01, coke=5.0, nut=5.0, pci=5.0, prod=1.0, blast_t=10.0, o2=0.002, rdi=0.5, fer=10.0, humidity=0.5)
NET_VERSION = "STRATA-Flow v0.4.1 (plant RI / RDI catalogue, 23 Oct 2026)"
_STATS = {"solves": 0, "seconds": 0.0}            # network solves of this process (all NetworkParts)
# Executor of the parallel network solves (SV-1.8); None: one after the other. A context variable, so that it holds
# for the thread or asyncio task that sets it (a Streamlit session, a worker job) and for the threads that
# ``asyncio.to_thread`` starts from there, not for the other sessions or jobs of the process.
_EXECUTOR: contextvars.ContextVar[Executor | None] = contextvars.ContextVar("strata_executor", default=None)


def solve_stats() -> tuple[int, float]:
    """(network solves, seconds spent in them) of this process so far; differences give the cost of one step."""
    return int(_STATS["solves"]), float(_STATS["seconds"])


def executor() -> Executor | None:
    """The executor of the parallel network solves in this context (None: one after the other)."""
    return _EXECUTOR.get()


def set_executor(executor: Executor | None) -> Executor | None:
    """Solve independent network states in ``executor`` (a process pool's ``map``) in the current context; None
    solves them one after the other. Returns the previous executor. The results, the anchors and the solve counts are
    the same either way."""
    old = _EXECUTOR.get()
    _EXECUTOR.set(executor)
    return old


@contextlib.contextmanager
def parallel_solves(executor: Executor | None) -> Iterator[None]:
    """``executor`` for the network solves of the block (the current thread or task and what it starts)."""
    token = _EXECUTOR.set(executor)
    try:
        yield
    finally:
        _EXECUTOR.reset(token)


def solve_state(state: Mapping[str, Any]) -> tuple[list[float], float]:
    """One network solve of a network state (top level, so a process pool can run it): y = f(6) + o(6) + t(6) and
    the seconds it took."""
    from utils.strata_flow.omega_eval import evaluate
    t0 = time.perf_counter()
    r = evaluate(dict(state))
    return [float(x) for x in (*r["f"], *r["o"], *r["t"])], time.perf_counter() - t0


def _per_input(table: Mapping[str, float]) -> np.ndarray:
    return np.array([table["sh"] if n.startswith("sh_") else table[n] for n in INPUTS], float)


class NetworkParts:
    """Ring quantities from the STRATA-Flow network, anchored per state; see the module docstring."""

    def __init__(self, context: Mapping[str, Any], cache_dir: str | Path | None = CACHE_DIR, jacobian: bool = True):
        self.ctx = dict(context)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.jacobian = bool(jacobian)                              # False: exact mode, a solve per distinct state
        self.U = np.zeros((0, len(INPUTS)))                         # anchor inputs
        self.Y = np.zeros((0, 18))                                  # anchor outputs y = f(6) + o(6) + t(6)
        self.J: np.ndarray | None = None
        self.u_ref: np.ndarray | None = None
        self.n_solves = 0
        self.sec_solves = 0.0
        self.last_distance = 0.0
        self.step = _per_input(STEP)
        self.scale = _per_input(SCALE)
        # The anchors are shared by the engines of a calibration in the process (``network_parts``), also across
        # threads (Streamlit sessions, the worker's evaluation threads): reads and additions under this lock;
        # ``Engine.run_items`` holds it while its items run.
        self.lock = threading.RLock()

    # ------------------------------------------------------------------------------------------------ states
    @staticmethod
    def vec(st) -> np.ndarray:
        hum = float(st.humidity) if math.isfinite(float(st.humidity)) else float("nan")
        return np.array([float(st.shares.get(m, 0.0)) for m in SHARE_INPUTS]
                        + [st.coke, st.nut, st.pci, st.prod, st.blast_t, st.o2, st.rdi, st.fer, hum], float)

    def network_state(self, u: np.ndarray) -> dict[str, Any]:
        c = self.ctx
        sh = {m: float(max(v, 0.0)) for m, v in zip(SHARE_INPUTS, u[:len(SHARE_INPUTS)])}
        sh["sinter"] = max(0.0, 1.0 - sum(sh.values()))
        sh = {m: v for m, v in sh.items() if v > 1e-9}
        coke, nut, pci, prod, blast_t, o2, rdi, fer, hum = (float(x) for x in u[len(SHARE_INPUTS):])
        thm_per_charge = float(c["noncoke_t_per_charge"]) * 1000.0 / (fer + nut + float(c["limestone_kg_thm"]))
        return dict(blend_wet=sh, coke_kg_thm=coke, nut_coke_kg_thm=nut, pci_kg_thm=pci, production_t_h=prod,
                    blast_nm3_h=prod * blast_t, enrichment_true=o2, hbt_c=float(c["hbt_c"]),
                    blast_moisture=hum if math.isfinite(hum) else float(c["blast_moisture"]),
                    charges_per_hour=prod / thm_per_charge, noncoke_t_per_charge=float(c["noncoke_t_per_charge"]),
                    limestone_kg_thm=float(c["limestone_kg_thm"]), top_eta_co=float(c["top_eta_co"]),
                    top_h2=float(c["top_h2"]), top_gas_T_c=float(c["top_gas_T_c"]), sinter_rdi_pct=rdi,
                    matrix=c["matrix"])

    def solve(self, u: np.ndarray) -> np.ndarray:
        """One network solve: y = f(6) + o(6) + t(6)."""
        return self.solve_many([u])[0]

    def solve_many(self, us: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Network solves of several states, in parallel when an executor is set (``set_executor``); y per state, in
        order. Solve counts and seconds are those of the solves themselves, as when run one after the other."""
        states = [self.network_state(u) for u in us]
        results = map_or_serial(solve_state, states)
        out = []
        for y, dt in results:
            self.n_solves += 1
            self.sec_solves += dt
            _STATS["solves"] += 1
            _STATS["seconds"] += dt
            out.append(np.asarray(y, float))
        return out

    @staticmethod
    def _split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.clip(y[0:6], 1e-4, None); f = f / f.sum()
        o = np.clip(y[6:12], 1e-4, None); o = o / o.sum()
        t = np.clip(y[12:18], 0.05, None)
        return f, o, t

    def _fill(self, u: np.ndarray) -> np.ndarray:
        u = np.array(u, float)
        if not math.isfinite(u[-1]):
            u[-1] = float(self.ctx["blast_moisture"])
        return u

    # ------------------------------------------------------------------------------------------------ anchors
    def exact(self, st) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._split(self.anchor(st))

    @property
    def anchors(self) -> list[tuple[np.ndarray, np.ndarray]]:
        with self.lock:
            return list(zip(self.U, self.Y))

    def _add(self, u: np.ndarray, y: np.ndarray) -> None:
        with self.lock:
            self.U = np.vstack([self.U, u[None, :]])
            self.Y = np.vstack([self.Y, y[None, :]])

    def anchor(self, st, tol: float = 1e-9) -> np.ndarray:
        """Exact solve at the state (unless an anchor sits within ``tol`` in scaled units); returns y."""
        return self.anchor_many([st], tol)[0]

    def anchor_many(self, states: Sequence[Any], tol: float = 1e-9) -> list[np.ndarray]:
        """``anchor`` of each state in order, as one after the other would do it (a state reuses an anchor within
        ``tol``, also one that an earlier state of the list adds), with the new solves run together (``solve_many``).
        Returns y per state."""
        us = [self._fill(self.vec(st)) for st in states]
        with self.lock:
            new: list[np.ndarray] = []                              # inputs to solve, in the order they are added
            refs: list[tuple[str, Any]] = []                        # ("old", y) | ("new", index into new)
            for u in us:
                d_old, i_old = (math.inf, -1)
                if len(self.U):
                    d = np.max(np.abs(self.U - u[None, :]) / self.scale[None, :], axis=1)
                    i_old = int(np.argmin(d)); d_old = float(d[i_old])
                d_new, i_new = (math.inf, -1)
                if new:
                    d = np.max(np.abs(np.asarray(new) - u[None, :]) / self.scale[None, :], axis=1)
                    i_new = int(np.argmin(d)); d_new = float(d[i_new])
                if min(d_old, d_new) <= tol:                        # the nearest anchor; ties: the older one
                    refs.append(("old", self.Y[i_old]) if d_old <= d_new else ("new", i_new))
                else:
                    refs.append(("new", len(new)))
                    new.append(u)
            ys = self.solve_many(new) if new else []
            for u, y in zip(new, ys):
                self._add(u, y)
        return [ref if kind == "old" else ys[ref] for kind, ref in refs]

    def nearest(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        with self.lock:
            d = np.max(np.abs(self.U - u[None, :]) / self.scale[None, :], axis=1)
            i = int(np.argmin(d))
            return self.U[i], self.Y[i], float(d[i])

    def parts(self, st) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = self._fill(self.vec(st))
        with self.lock:
            if not len(self.U) or (self.jacobian and self.J is None):
                raise RuntimeError("NetworkParts.set_reference() first")
            ua, ya, d = self.nearest(u)
            if d <= 1e-9:
                self.last_distance = d
                return self._split(ya)
            if not self.jacobian:                                   # exact mode: solve the state itself
                y = self.solve(u)
                self._add(u, y)
                self.last_distance = 0.0
                return self._split(y)
            self.last_distance = d
            return self._split(ya + self.J @ (u - ua))

    def distance(self, st) -> float:
        """Largest scaled distance of the state from its nearest anchor (0 = network solve at the state itself);
        scale: 1 pt of a ferrous share, 5 kg coke / nut / PCI, 1 t/h, 10 Nm3/t, 0.2 pt O2, 0.5 pt RDI, 10 kg ferrous,
        0.5 g/Nm3."""
        with self.lock:
            return self.nearest(self._fill(self.vec(st)))[2] if len(self.U) else math.inf

    # ------------------------------------------------------------------------------------------------ reference
    def _key(self, u: np.ndarray, tag: str) -> str:
        from utils.strata_flow.materials import MEASURED_SIEVE, RDI_PCT, RI_PCT
        ctx = {k: v for k, v in self.ctx.items()}
        cat = dict(ri=RI_PCT, rdi=RDI_PCT, sieve=MEASURED_SIEVE)          # a catalogue change invalidates the cache
        payload = json.dumps(dict(u=[round(float(x), 6) for x in u], ctx=ctx, tag=tag, net=NET_VERSION, cat=cat),
                             sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def set_reference(self, ref, tag: str = "") -> "NetworkParts":
        """Exact solve at the calibration state and the network Jacobian there (disk cache per calibration); in exact
        mode the solve only."""
        u0 = self._fill(self.vec(ref))
        self.u_ref = u0
        if not self.jacobian:
            self.J = None
            self.U, self.Y = u0[None, :].copy(), self.solve(u0)[None, :]
            self.jacobian_source = "none (exact mode: a network solve at every state)"
            self.cache_file = None
            return self
        key = self._key(u0, tag)
        path = self.cache_dir / f"jacobian_{key}.json" if self.cache_dir else None
        self.cache_file = path.name if path is not None else None      # the calibration console checks it (SV-4.6)
        if path is not None and path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                if d.get("inputs") == list(INPUTS):
                    y0 = np.asarray(d["y0"], float)
                    self.J = np.asarray(d["J"], float)
                    self.U, self.Y = u0[None, :].copy(), y0[None, :].copy()
                    self.jacobian_source = f"cache {path.name}"
                    return self
            except (ValueError, KeyError, OSError):
                pass
        t0 = time.perf_counter()
        us = [u0]
        for j in range(len(INPUTS)):
            u = u0.copy()
            u[j] += self.step[j]
            if INPUTS[j].startswith("sh_") and 1.0 - u[:len(SHARE_INPUTS)].sum() < 0:   # no sinter left to replace
                u[j] -= 2 * self.step[j]
            us.append(u)
        ys = self.solve_many(us)                                    # in parallel when an executor is set
        y0 = ys[0]
        J = np.zeros((18, len(INPUTS)))
        for j in range(len(INPUTS)):
            J[:, j] = (ys[j + 1] - y0) / (us[j + 1][j] - u0[j])
        self.J = J
        self.U, self.Y = u0[None, :].copy(), y0[None, :].copy()
        self.jacobian_source = f"computed ({len(INPUTS) + 1} network solves, {time.perf_counter() - t0:.0f} s)"
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(inputs=list(INPUTS), u0=u0.tolist(), y0=y0.tolist(), J=J.tolist(),
                                                ctx=self.ctx, net=NET_VERSION), default=str), encoding="utf-8")
            except OSError:
                pass
        return self

    def stats(self) -> dict[str, Any]:
        return dict(network=NET_VERSION, solves=self.n_solves, seconds=round(self.sec_solves, 1), anchors=len(self.U),
                    jacobian=getattr(self, "jacobian_source", ""), cache_file=getattr(self, "cache_file", None))


_POOL: dict[str, NetworkParts] = {}
_POOL_LOCK = threading.RLock()    # one reference solve (and Jacobian) per calibration, also with concurrent threads
POOL_MAX = 32                     # calibrations kept per process (oldest dropped; one per day in a long-lived worker)


def network_parts(context: Mapping[str, Any], ref, tag: str = "", jacobian: bool = True) -> NetworkParts:
    """One NetworkParts per (network context, calibration state, mode) in the process, so that engines built on the
    same calibration (a page rerun, today's blend, the scenario runs) share the network solves already made."""
    net = NetworkParts(context, jacobian=jacobian)
    key = net._key(net._fill(net.vec(ref)), tag if jacobian else f"{tag}|exact")
    with _POOL_LOCK:
        if key not in _POOL:
            while len(_POOL) >= POOL_MAX:
                _POOL.pop(next(iter(_POOL)))
            _POOL[key] = net.set_reference(ref, tag)
        return _POOL[key]


def clear_pool() -> None:
    with _POOL_LOCK:
        _POOL.clear()


# ----------------------------------------------------------------------------------------------- solver processes
def _warm(_: int) -> tuple[int, bool]:
    """Build the STRATA-Flow context in a solver process (its first solve would otherwise pay for it). Returns the
    process id and whether the context was built already (in the forkserver, ``strata_preload``)."""
    import os

    from utils.strata_flow.omega_eval import context
    preloaded = context.cache_info().currsize > 0
    context()
    return os.getpid(), preloaded


#: The last ``start_solvers`` of this process: processes started, seconds to start them (warm), and how many found the
#: STRATA-Flow context built in the forkserver.
LAST_START: dict[str, Any] = {}


def solver_count(default: int | None = None) -> int:
    """Processes for parallel network solves: ``BF_STRATA_SOLVERS``, else ``default``, else min(8, cpus)."""
    import os

    value = os.environ.get("BF_STRATA_SOLVERS")
    if value:
        return max(int(value), 0)
    return default if default is not None else min(8, os.cpu_count() or 1)


@contextlib.contextmanager
def _main_without_script() -> Iterator[None]:
    """While solver processes start: ``__main__`` replaced by an empty module. A forkserver or spawn child runs the
    parent's main script (or main module) again before its task; a Streamlit session, an app test or a ``python -m``
    module without a main guard would run there. The solver processes run library functions only."""
    import sys
    import types

    main = sys.modules.get("__main__")
    stand_in = types.ModuleType("__main__")
    sys.modules["__main__"] = stand_in
    try:
        yield
    finally:
        if sys.modules.get("__main__") is stand_in:  # another thread (a Streamlit run) may have set its own meanwhile
            if main is None:
                sys.modules.pop("__main__", None)
            else:
                sys.modules["__main__"] = main


def start_solvers(n: int | None = None) -> Executor | None:
    """``n`` network-solver processes (``solver_count``), started and warm (STRATA-Flow context built), or None for
    fewer than 2. A pool that fails to start gives None with a warning: the solves then run one after the other, with
    the same results. Stop it with ``stop_solvers``. Python 3.12 pool workers are not daemonic, so a process of the
    worker's model-solve pool can start one.

    On POSIX the processes fork from this process's forkserver, which builds the context once when it can import the
    code (``strata_preload``; Python 3.12 starts the forkserver with the interpreter's own path, so ``src`` must be on
    ``PYTHONPATH``, as the worker runs): the first pool of a process starts in about 2.7 s here, later ones in 0.03 s.
    Otherwise every process builds its own context (about 2.6 s, in parallel). ``LAST_START`` records the start."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    n = solver_count() if n is None else n
    if n < 2:
        return None
    method = "forkserver" if os.name == "posix" else "spawn"
    mp = multiprocessing.get_context(method)
    if method == "forkserver":
        # The forkserver of this process builds the STRATA-Flow context once (``strata_preload``) and the solver
        # processes fork from it with the context built; no effect when the forkserver runs already.
        mp.set_forkserver_preload(["utils.burden_engine.strata_preload"])
    t0 = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=n, mp_context=mp)
    try:
        with _main_without_script():  # the processes start in submit (one per task while none is idle)
            futures = [pool.submit(_warm, i) for i in range(n)]
        warm = [f.result() for f in futures]
        LAST_START.clear()
        LAST_START.update(processes=n, seconds=round(time.perf_counter() - t0, 2),
                          preloaded=sum(1 for _, pre in warm if pre))
    except Exception as exc:  # noqa: BLE001 - any failure to start: serial solves, same results
        log.warning("STRATA-Flow solver processes did not start (%s: %s); solving one after the other",
                    type(exc).__name__, exc)
        pool.shutdown(wait=True, cancel_futures=True)
        return None
    return pool


def stop_solvers(pool: Executor | None) -> None:
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


@contextlib.contextmanager
def solver_processes(n: int | None = None) -> Iterator[Executor | None]:
    """``start_solvers(n)`` as the executor of the block's network solves (``parallel_solves``), stopped afterwards
    (no idle solvers stay behind a model-solve process of the worker); fewer than 2 processes: serial. Inside a block
    that has an executor already, that one is used (no second pool, no second start)."""
    outer = executor()
    if outer is not None:
        yield outer
        return
    pool = start_solvers(n)
    try:
        with parallel_solves(pool):
            yield pool
    finally:
        stop_solvers(pool)


def drop_executor(exc: BaseException) -> None:
    """A broken pool (a solver process killed): no executor for the rest of the context, with a warning."""
    log.warning("STRATA-Flow solver processes failed (%s: %s); solving one after the other", type(exc).__name__, exc)
    _EXECUTOR.set(None)


def map_or_serial(fn, items: Sequence[Any]) -> list[Any]:
    """``fn`` over the items in the context's executor (in order), else here one after the other. When the pool
    breaks, the items run here (``drop_executor``): the results are the same."""
    from concurrent.futures import BrokenExecutor

    ex = executor()
    if ex is not None and len(items) > 1:
        try:
            return list(ex.map(fn, items))
        except BrokenExecutor as exc:
            drop_executor(exc)
    return [fn(item) for item in items]
