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

Inputs u (network units): wet share of each ferrous material except sinter (the sinter share closes the sum), coke,
nut coke, PCI (kg/tHM), production (t/h), blast per tonne (Nm3/tHM), O2 fraction of the blast, sinter RDI (%),
ferrous rate (kg/tHM wet), blast moisture (g/Nm3). Pure numerical code; reads and writes one JSON cache file.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

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


def _per_input(table: Mapping[str, float]) -> np.ndarray:
    return np.array([table["sh"] if n.startswith("sh_") else table[n] for n in INPUTS], float)


class NetworkParts:
    """Ring quantities from the STRATA-Flow network, anchored per state; see the module docstring."""

    def __init__(self, context: Mapping[str, Any], cache_dir: str | Path | None = CACHE_DIR):
        self.ctx = dict(context)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.U = np.zeros((0, len(INPUTS)))                         # anchor inputs
        self.Y = np.zeros((0, 18))                                  # anchor outputs y = f(6) + o(6) + t(6)
        self.J: np.ndarray | None = None
        self.u_ref: np.ndarray | None = None
        self.n_solves = 0
        self.sec_solves = 0.0
        self.last_distance = 0.0
        self.step = _per_input(STEP)
        self.scale = _per_input(SCALE)

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
        from utils.strata_flow.omega_eval import evaluate
        t0 = time.perf_counter()
        r = evaluate(self.network_state(u))
        self.n_solves += 1
        self.sec_solves += time.perf_counter() - t0
        return np.r_[np.asarray(r["f"], float), np.asarray(r["o"], float), np.asarray(r["t"], float)]

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
        return list(zip(self.U, self.Y))

    def _add(self, u: np.ndarray, y: np.ndarray) -> None:
        self.U = np.vstack([self.U, u[None, :]])
        self.Y = np.vstack([self.Y, y[None, :]])

    def anchor(self, st, tol: float = 1e-9) -> np.ndarray:
        """Exact solve at the state (unless an anchor sits within ``tol`` in scaled units); returns y."""
        u = self._fill(self.vec(st))
        if len(self.U):
            ua, ya, d = self.nearest(u)
            if d <= tol:
                return ya
        y = self.solve(u)
        self._add(u, y)
        return y

    def nearest(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        d = np.max(np.abs(self.U - u[None, :]) / self.scale[None, :], axis=1)
        i = int(np.argmin(d))
        return self.U[i], self.Y[i], float(d[i])

    def parts(self, st) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.J is None or not len(self.U):
            raise RuntimeError("NetworkParts.set_reference() first")
        u = self._fill(self.vec(st))
        ua, ya, d = self.nearest(u)
        self.last_distance = d
        if d <= 1e-9:
            return self._split(ya)
        return self._split(ya + self.J @ (u - ua))

    def distance(self, st) -> float:
        """Largest scaled distance of the state from its nearest anchor (0 = network solve at the state itself);
        scale: 1 pt of a ferrous share, 5 kg coke / nut / PCI, 1 t/h, 10 Nm3/t, 0.2 pt O2, 0.5 pt RDI, 10 kg ferrous,
        0.5 g/Nm3."""
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
        """Exact solve at the calibration state and the network Jacobian there (disk cache per calibration)."""
        u0 = self._fill(self.vec(ref))
        self.u_ref = u0
        key = self._key(u0, tag)
        path = self.cache_dir / f"jacobian_{key}.json" if self.cache_dir else None
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
        y0 = self.solve(u0)
        J = np.zeros((18, len(INPUTS)))
        for j in range(len(INPUTS)):
            u = u0.copy()
            u[j] += self.step[j]
            if INPUTS[j].startswith("sh_") and 1.0 - u[:len(SHARE_INPUTS)].sum() < 0:   # no sinter left to replace
                u[j] -= 2 * self.step[j]
            J[:, j] = (self.solve(u) - y0) / (u[j] - u0[j])
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
                    jacobian=getattr(self, "jacobian_source", ""))


_POOL: dict[str, NetworkParts] = {}


def network_parts(context: Mapping[str, Any], ref, tag: str = "") -> NetworkParts:
    """One NetworkParts per (network context, calibration state) in the process, so that engines built on the same
    calibration (a page rerun, today's blend, the scenario runs) share the network solves already made."""
    net = NetworkParts(context)
    key = net._key(net._fill(net.vec(ref)), tag)
    if key not in _POOL:
        _POOL[key] = net.set_reference(ref, tag)
    return _POOL[key]


def clear_pool() -> None:
    _POOL.clear()
