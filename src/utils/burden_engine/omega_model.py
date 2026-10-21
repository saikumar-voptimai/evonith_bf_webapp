"""Blend-dependent shaft efficiency (omega) for the burden engine (fact register GAS-13..GAS-20, ASM-18).

Why
  With omega held at the calibration (ASM-01) a blend changes the coke rate only through its chemistry: Fe units,
  slag, flux and LOI. It also changes how well the gas is used in the thermal reserve zone (TRZ): reducibility and
  size of each ferrous, where the burden puts ore and coke across the radius, and how long the ore stays in the TRZ.
  This module supplies omega as a function of the blend, the operating point and the blast moisture, with a 1-sigma
  uncertainty and a certainty class for every plan.

Model (constants in ``assets/burden_engine/omega_model.json``; built from the BF-02 layered gas-flow model v0.4,
scaled and tested out of sample on 191 days of BF-02 data, Dec 2025 - Oct 2026; OMEGA v1.1)

  w(state, R)   ring counter-current model of the TRZ. Six rings, i = 1 (centre) .. 6 (wall):
                  C_g,i = R f_i        oxygen the ring's gas can take at the W point (f_i: gas share of ring i)
                  C_s,i = o_i          wustite oxygen of the ring's ore (o_i: Fe share of ring i)
                  NTU_i = kappa k t_i C_s,i / C_min,i,   C_r,i = C_min,i / C_max,i
                  eps_i = (1 - exp(-NTU (1 - C_r))) / (1 - C_r exp(-NTU (1 - C_r)))
                  w     = sum eps_i C_min,i / sum C_g,i
                f, o, t (TRZ residence, h) come from a quadratic surrogate of the gas-flow network (burden profile,
                layered Ergun resistance, floating cohesive zone); k is the Fe-weighted reduction rate of the blend,
                r_m (d_test / d_m)^1.5 with r_m = ln(1 - RI_m) / ln(1 - RI_sinter) from the ISO 7215 index;
                kappa is re-fitted at start-up so that w(calibration state) equals the calibrated omega.
  omega(plan) = omega_cal + [w(blend_plan, op_cal) - w(cal)] + sum_c b_c (s_c,plan - s_c,cal)      blend channel
                          + [w(blend_plan, op_plan) - w(blend_plan, op_cal)]                        operating channel
                          + a_h (moisture_plan - moisture_cal)                                      blast moisture
  The blend channel is the physics with a plant-data scale per material group: b_c = (lam_g - 1) g_c, where g_c is
  the physics change of omega per wet pt of class c replacing sinter and lam_g the scale of its group (pellet; lump
  ore = Lloyds CLO, Geomin, NMDC ROM, Donimalai, Gomti, oversize). lam_pellet, lam_lump, the operating scale and a_h
  are the AR(1) posterior of the daily change of omega against its 14-day anchor (GAS-14, GAS-19); the posterior is
  stored as the mean and covariance of (x_blend, b_c) so that the code below is unchanged from v1.0.
  The gas-capacity channel of v1.0 (R from the carbon gas of the balance, GAS-17) is retired: out of sample it gave
  no skill and its data scale was negative. ``gas_capacity_channel: false`` holds R at the calibration.

Uncertainty of omega(plan) relative to the calibration state (1 sigma): posterior of the blend scales; the operating
scale's posterior (oper_rel_sd x |operating channel|); the moisture coefficient's standard error x |moisture change|;
surrogate error. Converted to coke with the loop-inclusive dK/domega of the engine. Reported apart (not in the plan
comparison): the calibration's own standard error (common to every plan) and the day-to-day spread of omega.

Certainty classes: High = sigma(coke) <= 2 kg/tHM and blend and production inside the range BF-02 ran (1st-99th
percentile of days); Medium = sigma(coke) <= 5 kg/tHM and within the margin of that range; Low otherwise.

Pure numerical code: no Streamlit, no I/O except reading the JSON constants.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "burden_engine" / "omega_model.json"
CERTAINTY_LEVELS = ("any", "medium", "high")


def counter_current_eff(ntu: np.ndarray, cr: np.ndarray) -> np.ndarray:
    ntu = np.asarray(ntu, float)
    cr = np.clip(np.asarray(cr, float), 0.0, 1.0)
    out = np.empty_like(ntu)
    one = np.isclose(cr, 1.0, atol=1e-6)
    out[one] = ntu[one] / (1.0 + ntu[one])
    k = ~one
    e = np.exp(-ntu[k] * (1.0 - cr[k]))
    out[k] = (1.0 - e) / (1.0 - cr[k] * e)
    return out


def _phi(u: np.ndarray) -> np.ndarray:
    """Quadratic features 1, u_i, u_i u_j (i <= j) of a standardised input vector (as fitted)."""
    p = len(u)
    quad = [u[i] * u[j] for i in range(p) for j in range(i, p)]
    return np.concatenate(([1.0], u, quad))


@dataclass
class OmegaState:
    """What the omega model needs about a plan (or the calibration point)."""

    shares: dict[str, float]          # wet share of the ferrous charge by model material (sums to 1)
    fe_frac: dict[str, float]         # share of the charge's Fe by model material
    cls: dict[str, float]             # wet % of ferrous by data class (sinter, pellet, clo, rom, gomti, oversize)
    coke: float                       # dry coke, kg/tHM
    nut: float                        # nut coke as charged, kg/tHM
    pci: float                        # PCI as injected, kg/tHM
    prod: float                       # t/h
    blast_t: float                    # hot blast incl. O2, Nm3/tHM (meter basis)
    o2: float                         # O2 fraction of the hot blast (oxygen added / total)
    rdi: float                        # sinter RDI, %
    fer: float                        # wet ferrous, kg/tHM
    humidity: float = float("nan")    # blast moisture, g/Nm3 (DATA-11); nan: no moisture term

    def inputs(self, order: list[str]) -> np.ndarray:
        vals = dict(coke_kg_thm=self.coke, nut_coke_kg_thm=self.nut, pci_kg_thm=self.pci, production_t_h=self.prod,
                    blast_t=self.blast_t, enrichment_true=self.o2, sinter_rdi_pct=self.rdi, fer_kgthm=self.fer)
        return np.array([self.shares.get(k[3:], 0.0) if k.startswith("sh_") else float(vals[k]) for k in order])


@dataclass
class OmegaTerms:
    """Omega of one plan and its parts (all in omega units)."""

    omega: float
    d_blend: float                   # data-scaled blend channel
    d_blend_phys: float              # the model's blend channel before scaling
    d_disc: float                    # class discrepancy
    d_oper: float                    # operating channel at the calibration gas capacity
    d_R: float                       # gas-capacity channel (0 when retired, GAS-17)
    sigma: float                     # 1 sigma, relative to the calibration state
    sigma_parts: dict[str, float]
    R: float
    k_blend: float
    support: str                     # inside / margin / outside
    outside: list[str] = field(default_factory=list)
    gas_in_excess: float = float("nan")
    wall_gas_share: float = float("nan")
    g: list[float] = field(default_factory=list)      # gradient of omega in the posterior parameters (lam_b, b)
    d_moist: float = 0.0             # blast-moisture term (GAS-19)


class OmegaModel:
    """Load once (``load``), then ``calibrate`` to an engine calibration, then ``plan`` per evaluation."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = dict(spec)
        self.version = str(spec.get("version", ""))
        mats = spec["materials"]
        self.materials = list(mats)
        self.rate = {m: self._rate(v) for m, v in mats.items()}
        self.fe_pct = {m: float(v["fe_pct"]) for m, v in mats.items()}
        self.cls_of = {m: v["class"] for m, v in mats.items()}
        self.engine_map = dict(spec.get("engine_map", {}))
        self.kind_default = dict(spec.get("kind_default", {}))
        sur = spec["surrogate"]
        self.inputs = list(sur["inputs"])
        self.mu = np.asarray(sur["mu"], float)
        self.sd = np.asarray(sur["sd"], float)
        self.W = np.asarray(sur["W"], float)
        self.sigma_sur = float(sur.get("pair_rmse", 0.0))
        po = spec["posterior"]
        self.post_names = list(po["names"])
        self.post_mean = np.asarray(po["mean"], float)
        self.post_cov = np.asarray(po["cov"], float)
        self.classes = [n[3:] for n in self.post_names if n.startswith("sh_")]
        self.op_rel_sd = float(spec["priors"]["oper_rel_sd"])
        self.R_rel_sd = float(spec["priors"]["gas_capacity_rel_sd"])
        self.gas_capacity = bool(spec.get("gas_capacity_channel", True))
        mo = spec.get("moisture") or {}
        self.a_moist = float(mo.get("coef", 0.0))
        self.a_moist_se = float(mo.get("se", 0.0))
        self.support = {k: tuple(v) for k, v in spec["support"]["classes"].items()}
        self.support_prod = tuple(spec["support"]["production_t_h"])
        self.margin_pt = float(spec["support"]["margin_pt"])
        self.margin_prod = float(spec["support"]["margin_t_h"])
        cert = spec["certainty"]
        self.sig_high, self.sig_medium = float(cert["high_sigma_kg"]), float(cert["medium_sigma_kg"])
        self.cal_se = float(spec["sigma"]["omega_cal_se"])
        self.day_sd = float(spec["sigma"]["omega_day_sd"])
        self.kappa = float(spec["kinetics"]["kappa"])
        self.rdi = float(spec["reference"]["sinter_rdi_pct"])
        self.ref: OmegaState | None = None
        self.R_ref = float(spec["reference"]["R"])
        self.omega_ref = float(spec["reference"]["omega"])
        self.w_ref = float("nan")

    # ---------------------------------------------------------------------------------------------- construction
    @classmethod
    def load(cls, path: str | Path | None = None) -> "OmegaModel":
        return cls(_load_spec(str(Path(path or MODEL_PATH).resolve())))

    def _rate(self, v: Mapping[str, Any]) -> float:
        k = self.spec["kinetics"]
        r = math.log(1 - float(v["ri_pct"]) / 100) / math.log(1 - float(k["ri_ref_pct"]) / 100)
        return r * (float(k["d_test_mm"]) / float(v["sauter_mm"])) ** float(k["n_size"])

    def model_material(self, var_id: str, kind: str) -> str:
        """Model material for an engine material: by id, else by kind (lump ore -> generic CLO properties)."""
        return self.engine_map.get(var_id) or self.kind_default.get(kind, self.kind_default.get("ore", "lloyds_clo"))

    # ---------------------------------------------------------------------------------------------- states
    def blend_parts(self, wet_by_material: Mapping[str, float]) -> tuple[dict, dict, dict]:
        """(wet shares, Fe fractions, class wet %) of a ferrous blend given as wet kg (or shares) by model material."""
        tot = sum(max(v, 0.0) for v in wet_by_material.values())
        if tot <= 0:
            raise ValueError("empty ferrous blend")
        sh = {m: max(v, 0.0) / tot for m, v in wet_by_material.items() if v > 0}
        fe = {m: v * self.fe_pct[m] for m, v in sh.items()}
        sfe = sum(fe.values())
        fe = {m: v / sfe for m, v in fe.items()}
        cls: dict[str, float] = {}
        for m, v in sh.items():
            c = self.cls_of[m]
            cls[c] = cls.get(c, 0.0) + 100.0 * v
        return sh, fe, cls

    def k_blend(self, fe_frac: Mapping[str, float]) -> float:
        return float(sum(v * self.rate[m] for m, v in fe_frac.items()))

    def parts(self, st: OmegaState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = (st.inputs(self.inputs) - self.mu) / self.sd
        y = _phi(u) @ self.W
        f = np.clip(y[0:6], 1e-4, None); f = f / f.sum()
        o = np.clip(y[6:12], 1e-4, None); o = o / o.sum()
        t = np.clip(y[12:18], 0.05, None)
        return f, o, t

    def w(self, st: OmegaState, R: float, kappa: float | None = None, parts=None) -> float:
        f, o, t = parts if parts is not None else self.parts(st)
        k = self.k_blend(st.fe_frac)
        cg = R * f
        cmin = np.minimum(cg, o); cmax = np.maximum(cg, o)
        cr = cmin / np.maximum(cmax, 1e-12)
        eff = counter_current_eff((self.kappa if kappa is None else kappa) * k * t * o / np.maximum(cmin, 1e-12), cr)
        return float(np.sum(eff * cmin) / np.sum(cg))

    # ---------------------------------------------------------------------------------------------- calibration
    def calibrate(self, ref: OmegaState, omega_cal: float, R_cal: float) -> "OmegaModel":
        """Anchor on the engine's calibration point: kappa so that w(ref, R_cal) = omega_cal."""
        self.ref, self.R_ref, self.omega_ref = ref, float(R_cal), float(omega_cal)
        pr = self.parts(ref)
        lo, hi = 1e-3, 1e3
        for _ in range(70):
            mid = math.sqrt(lo * hi)
            if self.w(ref, R_cal, mid, pr) < omega_cal:
                lo = mid
            else:
                hi = mid
        self.kappa = math.sqrt(lo * hi)
        self.w_ref = self.w(ref, R_cal, parts=pr)
        return self

    # ---------------------------------------------------------------------------------------------- prediction
    def lam_b(self) -> float:
        return float(self.post_mean[self.post_names.index("x_blend")])

    def plan(self, blend_state: OmegaState) -> "PlanOmega":
        """Per-plan object: the blend channel and discrepancy are fixed by the blend; ``omega(K, R)`` adds the
        operating and gas-capacity channels for the dry coke K and capacity ratio R of the balance."""
        if self.ref is None:
            raise RuntimeError("OmegaModel.calibrate() first")
        return PlanOmega(self, blend_state)

    def support_status(self, cls: Mapping[str, float], prod: float) -> tuple[str, list[str]]:
        status, outside = "inside", []
        for c, (lo, hi) in self.support.items():
            v = float(cls.get(c, 0.0))
            if v < lo - 1e-6 or v > hi + 1e-6:
                far = v < lo - self.margin_pt or v > hi + self.margin_pt
                outside.append(f"{c} {v:.1f} % (BF-02 ran {lo:.1f}-{hi:.1f} %)")
                status = "outside" if far or status == "outside" else "margin"
        lo, hi = self.support_prod
        if prod < lo - 1e-6 or prod > hi + 1e-6:
            far = prod < lo - self.margin_prod or prod > hi + self.margin_prod
            outside.append(f"production {prod:.1f} t/h (BF-02 ran {lo:.1f}-{hi:.1f} t/h)")
            status = "outside" if far or status == "outside" else "margin"
        return status, outside

    def certainty(self, sigma_coke: float, support: str) -> str:
        if sigma_coke <= self.sig_high and support == "inside":
            return "High"
        if sigma_coke <= self.sig_medium and support in ("inside", "margin"):
            return "Medium"
        return "Low"

    def limits_for(self, level: str) -> dict[str, Any] | None:
        """Constraints a certainty filter puts on the optimiser: sigma(coke) cap and the class / production range."""
        if level not in ("medium", "high"):
            return None
        m, mp = (0.0, 0.0) if level == "high" else (self.margin_pt, self.margin_prod)
        return dict(sigma_max=self.sig_high if level == "high" else self.sig_medium,
                    classes={c: (max(0.0, lo - m), min(100.0, hi + m)) for c, (lo, hi) in self.support.items()},
                    production=(self.support_prod[0] - mp, self.support_prod[1] + mp))


class PlanOmega:
    """Omega of one blend as a function of the balance's coke rate K and capacity ratio R."""

    def __init__(self, model: OmegaModel, st: OmegaState):
        self.m, self.st = model, st
        ref = model.ref
        # blend channel: the plan's blend and ferrous per tHM at the calibration's operation
        self.st_b = OmegaState(st.shares, st.fe_frac, st.cls, ref.coke, ref.nut, ref.pci, ref.prod, ref.blast_t, ref.o2,
                               ref.rdi, st.fer, ref.humidity)
        self.pb = model.parts(self.st_b)
        self.w_b = model.w(self.st_b, model.R_ref, parts=self.pb)
        self.d_blend_phys = self.w_b - model.w_ref
        names = model.post_names
        self.ds = np.array([st.cls.get(c, 0.0) - ref.cls.get(c, 0.0) for c in model.classes])
        b = np.array([model.post_mean[names.index("sh_" + c)] for c in model.classes])
        self.d_blend = model.lam_b() * self.d_blend_phys
        self.d_disc = float(b @ self.ds)
        self.g = np.r_[self.d_blend_phys, self.ds]
        idx = [names.index("x_blend")] + [names.index("sh_" + c) for c in model.classes]
        self.V = model.post_cov[np.ix_(idx, idx)]
        dh = float(st.humidity) - float(ref.humidity)
        self.d_moist = model.a_moist * dh if math.isfinite(dh) else 0.0
        self.s_moist = model.a_moist_se * abs(dh) if math.isfinite(dh) else 0.0
        self.base = model.omega_ref + self.d_blend + self.d_disc + self.d_moist
        self._last: tuple[float, float, float, float] | None = None

    def _state_at(self, K: float) -> OmegaState:
        s = self.st
        return OmegaState(s.shares, s.fe_frac, s.cls, float(K), s.nut, s.pci, s.prod, s.blast_t, s.o2, s.rdi, s.fer,
                          s.humidity)

    def omega(self, K: float, R: float) -> float:
        if not self.m.gas_capacity:
            R = self.m.R_ref
        st = self._state_at(K)
        p = self.m.parts(st)
        w_op = self.m.w(st, self.m.R_ref, parts=p)
        w_R = self.m.w(st, R, parts=p)
        d_oper, d_R = w_op - self.w_b, w_R - w_op
        self._last = (float(K), float(R), d_oper, d_R)
        return self.base + d_oper + d_R

    def terms(self, K: float, R: float) -> OmegaTerms:
        om = self.omega(K, R)
        _, _, d_oper, d_R = self._last
        m = self.m
        s_post = float(math.sqrt(max(self.g @ self.V @ self.g, 0.0)))
        s_oper = m.op_rel_sd * abs(d_oper)
        s_R = m.R_rel_sd * abs(d_R)
        du = (self.st.inputs(m.inputs) - m.ref.inputs(m.inputs)) / m.sd
        s_sur = m.sigma_sur * min(1.0, float(np.linalg.norm(du)))
        sig = math.sqrt(s_post ** 2 + s_oper ** 2 + s_R ** 2 + s_sur ** 2 + self.s_moist ** 2)
        support, outside = m.support_status(self.st.cls, self.st.prod)
        st = self._state_at(K)
        f, o, t = m.parts(st)
        excess = float(f[(R * f) >= o].sum())
        return OmegaTerms(omega=om, d_blend=self.d_blend, d_blend_phys=self.d_blend_phys, d_disc=self.d_disc,
                          d_oper=d_oper, d_R=d_R, sigma=sig,
                          sigma_parts=dict(data=s_post, operation=s_oper, gas_capacity=s_R, surrogate=s_sur,
                                           moisture=self.s_moist),
                          R=float(R), k_blend=m.k_blend(self.st.fe_frac), support=support, outside=outside,
                          gas_in_excess=excess, wall_gas_share=float(f[-1]), g=[float(x) for x in self.g],
                          d_moist=self.d_moist)

    def sigma(self, K: float, R: float) -> float:
        """1-sigma of omega without the support and ring diagnostics (cheap; used inside the optimiser)."""
        self.omega(K, R)
        _, _, d_oper, d_R = self._last
        m = self.m
        s_post2 = max(float(self.g @ self.V @ self.g), 0.0)
        du = (self.st.inputs(m.inputs) - m.ref.inputs(m.inputs)) / m.sd
        s_sur = m.sigma_sur * min(1.0, float(np.linalg.norm(du)))
        return math.sqrt(s_post2 + (m.op_rel_sd * d_oper) ** 2 + (m.R_rel_sd * d_R) ** 2 + s_sur ** 2 + self.s_moist ** 2)


def difference_sigma(model: OmegaModel, a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """1-sigma of omega(a) - omega(b) for two evaluated plans (``OmegaTerms`` as dicts). The data part uses the
    difference of the gradients (shared parameters cancel); operating and gas-capacity parts are taken as fully
    correlated between the plans (same physics), the surrogate part as independent."""
    names = model.post_names
    idx = [names.index("x_blend")] + [names.index("sh_" + c) for c in model.classes]
    V = model.post_cov[np.ix_(idx, idx)]
    dg = np.asarray(a["g"], float) - np.asarray(b["g"], float)
    s2 = max(float(dg @ V @ dg), 0.0)
    s2 += (model.op_rel_sd * (a["d_oper"] - b["d_oper"])) ** 2 + (model.R_rel_sd * (a["d_R"] - b["d_R"])) ** 2
    s2 += a["sigma_parts"]["surrogate"] ** 2 + b["sigma_parts"]["surrogate"] ** 2
    # moisture: same coefficient for both plans (fully correlated); zero when both plans run at the same moisture
    s2 += (a["sigma_parts"].get("moisture", 0.0) - b["sigma_parts"].get("moisture", 0.0)) ** 2
    return math.sqrt(s2)


@lru_cache(maxsize=4)
def _load_spec(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
