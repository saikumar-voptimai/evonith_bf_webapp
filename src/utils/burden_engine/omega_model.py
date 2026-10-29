"""Blend-dependent shaft efficiency (omega) for the burden engine (fact register GAS-13..GAS-20, ASM-18).

Why
  With omega held at the calibration (ASM-01) a blend changes the coke rate only through its chemistry: Fe units,
  slag, flux and LOI. It also changes how well the gas is used in the thermal reserve zone (TRZ): reducibility and
  size of each ferrous, where the burden puts ore and coke across the radius, and how long the ore stays in the TRZ.
  This module supplies omega as a function of the blend, the operating point and the blast moisture, with a 1-sigma
  uncertainty and a certainty class for every plan.

Model (constants in ``assets/burden_engine/omega_model.json``; STRATA 1.5 = STRATA-omega on STRATA-Flow, the Stratified
Transport & Reduction Analytics model of BF-02 (layered gas-flow model v0.4), scaled and tested out of sample on 191 days
of BF-02 data, Dec 2025 - Oct 2026, and on 411 days of 2023-24 data with frozen parameters; OMEGA v1.0 / v1.1,
STRATA 1.2, 1.3 and 1.4 were earlier versions)

  w(state, R)   ring counter-current model of the TRZ. Six rings, i = 1 (centre) .. 6 (wall):
                  C_g,i = R f_i        oxygen the ring's gas can take at the W point (f_i: gas share of ring i)
                  C_s,i = o_i          wustite oxygen of the ring's ore (o_i: Fe share of ring i)
                  NTU_i = kappa k t_i C_s,i / C_min,i,   C_r,i = C_min,i / C_max,i
                  eps_i = (1 - exp(-NTU (1 - C_r))) / (1 - C_r exp(-NTU (1 - C_r)))
                  w     = sum eps_i C_min,i / sum C_g,i
                f, o, t (TRZ residence, h) come from the STRATA-Flow network itself (burden profile, layered Ergun
                resistance, floating cohesive zone; ``strata_network.NetworkParts``: network solves at the calibration
                and at every plan, the network Jacobian between solves; STRATA 1.2 used a quadratic surrogate, which
                remains selectable with ``parts_backend: surrogate`` but is not used by the engine or the page);
                k is the Fe-weighted reduction rate of the blend,
                r_m^beta (d_test / d_m)^n with r_m = ln(1 - RI_m) / ln(1 - RI_sinter) from the laboratory RI
                (IS 8167 / ISO 7215); STRATA 1.3: beta = 0, n = 1.0 calibrated on the plant data (the measured RI
                differences show no effect on the plant's omega; size does), STRATA 1.2: beta = 1, n = 1.5;
                kappa is re-fitted at start-up so that w(calibration state) equals the calibrated omega.
  omega(plan) = omega_cal + [w(blend_plan, op_cal) - w(cal)] + sum_c b_c (s_c,plan - s_c,cal)      blend channel
                          + lam_rest [w(blend_plan, op_plan) - w(blend_plan, op_cal, prod_plan)]    other operation
                          + lam_prod [w(blend_plan, op_cal, prod_plan) - w(blend_plan, op_cal)]     production (1.4: 0)
                          + a_h (moisture_plan - moisture_cal)                          blast moisture (1.5: 0)
                          + a_n (nut_plan - nut_cal)                                                nut coke (1.3)
  The blend channel is the physics with a plant-data scale per material group: b_c = (lam_g - 1) g_c, where g_c is
  the physics change of omega per wet pt of class c replacing sinter and lam_g the scale of its group (pellet; lump
  ore = Lloyds CLO, Geomin, NMDC ROM, Donimalai, Gomti, oversize). lam_pellet, lam_lump, the operating scale and a_h
  are the AR(1) posterior of the daily change of omega against its 14-day anchor (GAS-14, GAS-19); the posterior is
  stored as the mean and covariance of (x_blend, b_c) so that the code below is unchanged from v1.0.

STRATA 1.5 (asset ``version``)
  STRATA 1.4 on the Open-Meteo daily blast moisture (DATA-11, ``utils/coke_energy/weather.py`` source ``open_meteo``):
  the day's moisture enters the coke-energy pipeline, the calibration and the balance of every plan, and the data term
  a_h is dropped (``moisture.coef`` 0, se 0; GAS-19 retired): with the moisture in the balance it is 0.0000 +/- 0.0009
  and lowers out-of-sample skill. Posterior on the Open-Meteo run: lam_pellet 1.42 +/- 0.45, lam_lump 0.92 +/- 0.46,
  lam_rest 0.80 +/- 0.28, a_n 0.0010 +/- 0.0003 per kg/tHM. 2026 expanding test RMSE 5.99 against Fixed-omega 6.39
  kg/tHM (STRATA 1.4 on the RH-scaled record: 6.44 against 6.94).

STRATA 1.4
  STRATA 1.3 without the production channel: ``operating_scales.production_channel`` false, lam_prod = 0 with zero
  uncertainty. The production part of the physics (TRZ residence) is still solved and reported (``d_prod_phys``), and
  the other-operation channel stays defined against the plan production, w(op_plan) - w(op_cal, prod_plan), as in the
  fit. Reason: the production scale was not replicated on 2023-24 data (-0.42 +/- 0.28 against 0.59 +/- 0.30 on 2026)
  and the channel lowered out-of-sample coke skill in both periods (GAS-16, GAS-25). Production changes the coke rate
  through the balance (heat loss per tHM, GAS-22). Posterior: lam_pellet 1.39 +/- 0.46, lam_lump 0.97 +/- 0.46,
  lam_rest 0.83 +/- 0.29, a_h 0.0016 +/- 0.0024 per g/Nm3, a_n 0.00093 +/- 0.00034 per kg/tHM.

STRATA 1.3
  Ring quantities from network solves, not from the surrogate. RI set S9 = S8 with Gomti 65 %; the Lloyds fines
  oversize (ore_6) reduces at the Lloyds CLO size (no size credit for its 8-16 mm sizing); the Donimalai oversize
  return (ore_11) is Donimalai. Blast moisture from the smoothed weather-record RH (DATA-11, no step changes): the data
  moisture coefficient is small and uncertain (GAS-19). Nut-coke data term a_n (GAS-23): omega rises with the nut
  coke charged in the ore layer, which the network's layer model does not represent. PCI acts through the balance
  (carbon, energy, H2) and the network; no additional data term (it lowered out-of-sample skill).

STRATA 1.2 (``operating_scales`` in the asset)
  The operating channel is split: production part = w(plan blend, calibration operation, plan production) minus
  w(plan blend, calibration operation), scaled by lam_prod (TRZ residence; 0.68 +/- 0.29), and the rest (coke layers,
  fuels, oxygen) scaled by lam_rest (1.37 +/- 0.28), correlated. RI set S8 (RI review, Oct 2026). Certainty tier
  Investigate: a class beyond ``certainty.investigate.classes_max`` (pellet > 25 % of ferrous) or sigma(coke) above
  ``certainty.investigate.sigma_kg``; ``limits_for("low")`` keeps every plan except Investigate ones.
  ``assumption_sensitivity`` gives the omega change when one critical assumption takes another credible value.
  The gas-capacity channel of v1.0 (R from the carbon gas of the balance, GAS-17) is retired: out of sample it gave
  no skill and its data scale was negative. ``gas_capacity_channel: false`` holds R at the calibration.

Uncertainty of omega(plan) relative to the calibration state (1 sigma): posterior of the blend scales; the operating
scales' posterior; the moisture and nut-coke coefficients' standard errors x |change|; with the surrogate backend
only, the surrogate error. Converted to coke with the loop-inclusive dK/domega of the engine. Reported apart (not in the plan
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

from utils.burden_engine.strata_network import NetworkParts, network_parts

MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "burden_engine" / "omega_model.json"
CERTAINTY_LEVELS = ("any", "low", "medium", "high")      # "low" = everything except Investigate
TIERS = ("High", "Medium", "Low", "Investigate")


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
    d_moist: float = 0.0             # blast-moisture term (GAS-19; 0 since STRATA 1.5)
    d_prod: float = 0.0              # operating channel: production part (TRZ residence), data-scaled (STRATA 1.2)
    d_rest: float = 0.0              # operating channel: coke layers, fuels, oxygen, data-scaled
    d_prod_phys: float = 0.0
    d_rest_phys: float = 0.0
    investigate: list[str] = field(default_factory=list)   # reasons for the Investigate tier (beyond common practice)
    d_nut: float = 0.0               # nut-coke data term (STRATA 1.3, GAS-23)
    network: dict = field(default_factory=dict)            # network backend: solves and distance from the anchors


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
        self.backend = str(spec.get("parts_backend", "surrogate"))
        if self.backend not in ("network", "surrogate"):
            raise ValueError(f"parts_backend must be 'network' or 'surrogate', not {self.backend!r}")
        sur = spec.get("surrogate") or {}
        if self.backend == "surrogate" and not sur:
            raise ValueError("parts_backend 'surrogate' needs the surrogate block in the asset")
        self.inputs = list(sur.get("inputs", []))
        self.mu = np.asarray(sur.get("mu", []), float)
        self.sd = np.asarray(sur.get("sd", []), float)
        self.W = np.asarray(sur.get("W", []), float)
        self.sigma_sur = float(sur.get("pair_rmse", 0.0)) if self.backend == "surrogate" else 0.0
        self.net_ctx = dict((spec.get("network") or {}).get("context") or {})
        self.net: NetworkParts | None = None
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
        nc = spec.get("nut_coke") or {}
        self.a_nut = float(nc.get("coef", 0.0))
        self.a_nut_se = float(nc.get("se", 0.0))
        ops = spec.get("operating_scales")
        self.split_oper = bool(ops)                     # STRATA 1.2: production and other operation scaled apart
        ops = ops or {}
        self.lam_prod, self.sd_prod = float(ops.get("production", 1.0)), float(ops.get("production_sd", self.op_rel_sd))
        self.lam_rest, self.sd_rest = float(ops.get("other", 1.0)), float(ops.get("other_sd", self.op_rel_sd))
        self.corr_oper = float(ops.get("corr", 0.0))
        self.prod_channel = bool(ops.get("production_channel", True)) and (self.lam_prod != 0.0 or self.sd_prod > 0.0)
        self.support = {k: tuple(v) for k, v in spec["support"]["classes"].items()}
        self.support_prod = tuple(spec["support"]["production_t_h"])
        self.margin_pt = float(spec["support"]["margin_pt"])
        self.margin_prod = float(spec["support"]["margin_t_h"])
        cert = spec["certainty"]
        self.sig_high, self.sig_medium = float(cert["high_sigma_kg"]), float(cert["medium_sigma_kg"])
        inv = cert.get("investigate") or {}
        self.inv_sigma = float(inv.get("sigma_kg") or math.inf)
        self.inv_max = {c: float(v) for c, v in (inv.get("classes_max") or {}).items()}
        self.inv_min = {c: float(v) for c, v in (inv.get("classes_min") or {}).items()}
        self.inv_prod = tuple(inv["production_t_h"]) if inv.get("production_t_h") else None
        self.inv_note = str(inv.get("note", "beyond common practice"))
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
    def load(cls, path: str | Path | None = None, backend: str | None = None) -> "OmegaModel":
        spec = dict(_load_spec(str(Path(path or MODEL_PATH).resolve())))
        if backend:
            spec["parts_backend"] = backend
        return cls(spec)

    def set_sinter_strength(self, ss: Mapping[str, Any] | None) -> "OmegaModel":
        """Sinter RI (reference of the rate law) and RDI (network shaft fines) from the latest raw-material strength
        data (calibration ``sinter_strength``); without it the asset's values (the latest test when the asset was
        built). Call before ``calibrate``."""
        if not ss:
            return self
        if ss.get("ri_pct"):
            mats = {k: dict(v) for k, v in self.spec["materials"].items()}
            mats["sinter"]["ri_pct"] = float(ss["ri_pct"])
            self.spec = dict(self.spec, materials=mats, kinetics=dict(self.spec["kinetics"], ri_ref_pct=float(ss["ri_pct"])))
            self.rate = {m: self._rate(v) for m, v in mats.items()}
        if ss.get("rdi_pct"):
            self.rdi = float(ss["rdi_pct"])
        return self

    def set_network_context(self, override: Mapping[str, Any] | None) -> "OmegaModel":
        """Charging matrix, top gas, HBT etc. of the calibration window for the network (default: the asset's)."""
        if override:
            self.net_ctx = {**self.net_ctx, **{k: v for k, v in override.items() if v is not None}}
        return self

    def _rate(self, v: Mapping[str, Any], beta: float | None = None, n: float | None = None) -> float:
        """Relative reduction rate r = [ln(1 - RI) / ln(1 - RI_sinter)]^beta x (d_test / d)^n (STRATA 1.3: beta and n
        calibrated on the plant data with the laboratory RI as given; beta = 1, n = 1.5 is the first-order link)."""
        k = self.spec["kinetics"]
        beta = float(k.get("ri_exponent", 1.0)) if beta is None else float(beta)
        n = float(k["n_size"]) if n is None else float(n)
        r = math.log(1 - float(v["ri_pct"]) / 100) / math.log(1 - float(k["ri_ref_pct"]) / 100)
        return r ** beta * (float(k["d_test_mm"]) / float(v["sauter_mm"])) ** n

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
        if self.backend == "network":
            if self.net is None:
                raise RuntimeError("OmegaModel.calibrate() first (network backend)")
            return self.net.parts(st)
        u = (st.inputs(self.inputs) - self.mu) / self.sd
        y = _phi(u) @ self.W
        f = np.clip(y[0:6], 1e-4, None); f = f / f.sum()
        o = np.clip(y[6:12], 1e-4, None); o = o / o.sum()
        t = np.clip(y[12:18], 0.05, None)
        return f, o, t

    def rings(self, st: OmegaState, R: float, kappa: float | None = None, parts=None) -> dict[str, Any]:
        """The ring counter-current model of a state, ring by ring (1 centre .. 6 wall): f, o, t, gas capacity
        C_g = R f, ore oxygen C_s = o, NTU, exchanger efficiency eps, each ring's share of w (eps C_min / sum C_g; the
        shares sum to w), the blend rate k and w itself."""
        f, o, t = parts if parts is not None else self.parts(st)
        k = self.k_blend(st.fe_frac)
        cg = R * f
        cmin = np.minimum(cg, o); cmax = np.maximum(cg, o)
        cr = cmin / np.maximum(cmax, 1e-12)
        ntu = (self.kappa if kappa is None else kappa) * k * t * o / np.maximum(cmin, 1e-12)
        eff = counter_current_eff(ntu, cr)
        total = np.sum(cg)
        return dict(f=f, o=o, t=t, c_gas=cg, c_ore=o, ntu=ntu, eff=eff, share=eff * cmin / total, k=k,
                    w=float(np.sum(eff * cmin) / total))

    def w(self, st: OmegaState, R: float, kappa: float | None = None, parts=None) -> float:
        return self.rings(st, R, kappa, parts)["w"]

    # ---------------------------------------------------------------------------------------------- calibration
    def calibrate(self, ref: OmegaState, omega_cal: float, R_cal: float, jacobian: bool = True) -> "OmegaModel":
        """Anchor on the engine's calibration point: kappa so that w(ref, R_cal) = omega_cal.

        Network backend: ``jacobian=False`` skips the network Jacobian at the anchor (17 solves) and solves every state
        on first use instead (exact mode of ``strata_network``; the twin's window evaluation)."""
        self.ref, self.R_ref, self.omega_ref = ref, float(R_cal), float(omega_cal)
        if self.backend == "network":
            self.net = network_parts(self.net_ctx, ref, jacobian=jacobian)
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
    def anchor(self, st: OmegaState, tol: float = 1e-9) -> None:
        """Network backend: one network solve at this state unless a solve lies within ``tol`` scaled units (1 unit =
        1 pt of a ferrous share, 5 kg/tHM coke, 1 t/h ...); no-op with the surrogate."""
        self.anchor_many([st], tol)

    def anchor_many(self, states: list[OmegaState], tol: float = 1e-9) -> None:
        """``anchor`` of each state in order, the new solves run together (in parallel with an executor,
        ``strata_network.parallel_solves``); the same anchors as one after the other."""
        if self.backend == "network" and self.net is not None:
            self.net.anchor_many(states, tol=max(float(tol), 1e-9))

    def network_stats(self) -> dict[str, Any]:
        return self.net.stats() if (self.backend == "network" and self.net is not None) else dict(backend=self.backend)

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

    def investigate_reasons(self, cls: Mapping[str, float], prod: float | None = None) -> list[str]:
        """Why a plan falls in the Investigate tier (GAS-18): a class or production beyond the configured envelope of
        common practice (e.g. pellet above 25 % of ferrous). Empty list: not Investigate on these grounds."""
        out = []
        for c, hi in self.inv_max.items():
            v = float(cls.get(c, 0.0))
            if v > hi + 1e-6:
                out.append(f"{c} {v:.1f} % > {hi:.0f} % ({self.inv_note})")
        for c, lo in self.inv_min.items():
            v = float(cls.get(c, 0.0))
            if v < lo - 1e-6:
                out.append(f"{c} {v:.1f} % < {lo:.0f} % ({self.inv_note})")
        if self.inv_prod and prod is not None and not self.inv_prod[0] - 1e-6 <= prod <= self.inv_prod[1] + 1e-6:
            out.append(f"production {prod:.1f} t/h outside {self.inv_prod[0]:.0f}-{self.inv_prod[1]:.0f} t/h ({self.inv_note})")
        return out

    def certainty(self, sigma_coke: float, support: str, investigate: list[str] | None = None) -> str:
        """High / Medium / Low / Investigate (GAS-18)."""
        if investigate or sigma_coke > self.inv_sigma:
            return "Investigate"
        if sigma_coke <= self.sig_high and support == "inside":
            return "High"
        if sigma_coke <= self.sig_medium and support in ("inside", "margin"):
            return "Medium"
        return "Low"

    def limits_for(self, level: str) -> dict[str, Any] | None:
        """Constraints a certainty filter puts on the optimiser: sigma(coke) cap and the class / production range.
        "low" keeps every plan except Investigate ones; "medium" / "high" as before."""
        if level == "low":
            classes = {c: (self.inv_min.get(c, 0.0), self.inv_max.get(c, 100.0)) for c in set(self.inv_max) | set(self.inv_min)}
            return dict(sigma_max=self.inv_sigma, classes=classes, production=self.inv_prod)
        if level not in ("medium", "high"):
            return None
        m, mp = (0.0, 0.0) if level == "high" else (self.margin_pt, self.margin_prod)
        return dict(sigma_max=self.sig_high if level == "high" else self.sig_medium,
                    classes={c: (max(0.0, lo - m), min(100.0, hi + m)) for c, (lo, hi) in self.support.items()},
                    production=(self.support_prod[0] - mp, self.support_prod[1] + mp))


class PlanOmega:
    """Omega of one blend as a function of the balance's coke rate K and capacity ratio R."""

    @staticmethod
    def blend_state(model: OmegaModel, st: OmegaState) -> OmegaState:
        """The plan's blend and ferrous per tHM at the calibration's operation (state B)."""
        ref = model.ref
        return OmegaState(st.shares, st.fe_frac, st.cls, ref.coke, ref.nut, ref.pci, ref.prod, ref.blast_t, ref.o2,
                          ref.rdi, st.fer, ref.humidity)

    @staticmethod
    def production_state(model: OmegaModel, st: OmegaState) -> OmegaState:
        """State B at the plan's production (state P)."""
        ref = model.ref
        return OmegaState(st.shares, st.fe_frac, st.cls, ref.coke, ref.nut, ref.pci, st.prod, ref.blast_t, ref.o2,
                          ref.rdi, st.fer, ref.humidity)

    @staticmethod
    def states(model: OmegaModel, st: OmegaState, K: float) -> list[OmegaState]:
        """The states a plan's omega at coke rate K solves: B, P (when the operating channel is split) and the full
        operation at K, in the order the plan solves them (SV-1.8: solved together by ``OmegaModel.anchor_many``)."""
        full = OmegaState(st.shares, st.fe_frac, st.cls, float(K), st.nut, st.pci, st.prod, st.blast_t, st.o2, st.rdi,
                          st.fer, st.humidity)
        prod = [PlanOmega.production_state(model, st)] if model.split_oper else []
        return [PlanOmega.blend_state(model, st), *prod, full]

    def __init__(self, model: OmegaModel, st: OmegaState):
        self.m, self.st = model, st
        ref = model.ref
        # blend channel: the plan's blend and ferrous per tHM at the calibration's operation
        self.st_b = self.blend_state(model, st)
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
        dn = float(st.nut) - float(ref.nut)
        self.d_nut = model.a_nut * dn
        self.s_nut = model.a_nut_se * abs(dn)
        self.offset = 0.0                # explicit omega shift of the calibration anchor (engine Settings.omega)
        self.base = model.omega_ref + self.d_blend + self.d_disc + self.d_moist + self.d_nut
        # operating channel, production part: the plan's blend at the calibration's operation but the plan's production
        if model.split_oper:
            self.st_bp = self.production_state(model, st)
            self.w_bp = model.w(self.st_bp, model.R_ref)
        else:
            self.w_bp = self.w_b
        self.d_prod_phys = self.w_bp - self.w_b
        self._last: tuple[float, float, float, float, float] | None = None

    def anchor_blend(self, tol: float = 0.0) -> None:
        """Network backend: solve the network at the plan's blend state (one solve per SLP iterate)."""
        self.m.anchor(self.st_b, tol)
        self._refresh()

    def anchor_all(self, K: float, tol: float = 0.0) -> None:
        """Network backend: solve the network at the blend, blend + production and full-operation states of the plan
        (at coke rate K), so that its omega comes from network solves at the plan itself (``tol`` > 0: reuse a solve
        within that many scaled units)."""
        self.m.anchor_many(self.states(self.m, self.st, K), tol)   # the three solves together (SV-1.8)
        self._refresh()

    def _refresh(self) -> None:
        m = self.m
        self.pb = m.parts(self.st_b)
        self.w_b = m.w(self.st_b, m.R_ref, parts=self.pb)
        self.d_blend_phys = self.w_b - m.w_ref
        self.d_blend = m.lam_b() * self.d_blend_phys
        self.g = np.r_[self.d_blend_phys, self.ds]
        self.base = m.omega_ref + self.d_blend + self.d_disc + self.d_moist + self.d_nut + self.offset
        self.w_bp = m.w(self.st_bp, m.R_ref) if m.split_oper else self.w_b
        self.d_prod_phys = self.w_bp - self.w_b

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
        d_rest_phys = w_op - self.w_bp
        if self.m.split_oper:
            d_oper = self.m.lam_prod * self.d_prod_phys + self.m.lam_rest * d_rest_phys
        else:
            d_oper = w_op - self.w_b
        d_R = w_R - w_op
        self._last = (float(K), float(R), d_oper, d_R, d_rest_phys)
        return self.base + d_oper + d_R

    def _s_oper(self, d_oper: float, d_rest_phys: float) -> float:
        m = self.m
        if m.split_oper:
            return oper_sigma(m, self.d_prod_phys, d_rest_phys)
        return m.op_rel_sd * abs(d_oper)

    def terms(self, K: float, R: float) -> OmegaTerms:
        om = self.omega(K, R)
        _, _, d_oper, d_R, d_rest_phys = self._last
        m = self.m
        s_post = float(math.sqrt(max(self.g @ self.V @ self.g, 0.0)))
        s_oper = self._s_oper(d_oper, d_rest_phys)
        s_R = m.R_rel_sd * abs(d_R)
        s_sur = 0.0
        if m.backend == "surrogate":
            du = (self.st.inputs(m.inputs) - m.ref.inputs(m.inputs)) / m.sd
            s_sur = m.sigma_sur * min(1.0, float(np.linalg.norm(du)))
        sig = math.sqrt(s_post ** 2 + s_oper ** 2 + s_R ** 2 + s_sur ** 2 + self.s_moist ** 2 + self.s_nut ** 2)
        support, outside = m.support_status(self.st.cls, self.st.prod)
        st = self._state_at(K)
        f, o, t = m.parts(st)
        excess = float(f[(R * f) >= o].sum())
        return OmegaTerms(omega=om, d_blend=self.d_blend, d_blend_phys=self.d_blend_phys, d_disc=self.d_disc,
                          d_oper=d_oper, d_R=d_R, sigma=sig,
                          sigma_parts=dict(data=s_post, operation=s_oper, gas_capacity=s_R, surrogate=s_sur,
                                           moisture=self.s_moist, nut_coke=self.s_nut),
                          R=float(R), k_blend=m.k_blend(self.st.fe_frac), support=support, outside=outside,
                          gas_in_excess=excess, wall_gas_share=float(f[-1]), g=[float(x) for x in self.g],
                          d_moist=self.d_moist,
                          d_prod=(m.lam_prod * self.d_prod_phys if m.split_oper else 0.0),
                          d_rest=(m.lam_rest * d_rest_phys if m.split_oper else d_oper),
                          d_prod_phys=self.d_prod_phys, d_rest_phys=d_rest_phys,
                          investigate=m.investigate_reasons(self.st.cls, self.st.prod), d_nut=self.d_nut,
                          network=self._network_info(st))

    def _network_info(self, st_op: OmegaState) -> dict[str, Any]:
        m = self.m
        if m.backend != "network" or m.net is None:
            return dict(backend=m.backend)
        net = m.net
        d = dict(blend=net.distance(self.st_b), operation=net.distance(st_op))
        if m.split_oper:
            d["production"] = net.distance(self.st_bp)
        return dict(backend="network", distance=d, exact=bool(max(d.values()) <= 1e-9), **net.stats())

    def sigma(self, K: float, R: float) -> float:
        """1-sigma of omega without the support and ring diagnostics (cheap; used inside the optimiser)."""
        self.omega(K, R)
        _, _, d_oper, d_R, d_rest_phys = self._last
        m = self.m
        s_post2 = max(float(self.g @ self.V @ self.g), 0.0)
        s_sur = 0.0
        if m.backend == "surrogate":
            du = (self.st.inputs(m.inputs) - m.ref.inputs(m.inputs)) / m.sd
            s_sur = m.sigma_sur * min(1.0, float(np.linalg.norm(du)))
        return math.sqrt(s_post2 + self._s_oper(d_oper, d_rest_phys) ** 2 + (m.R_rel_sd * d_R) ** 2 + s_sur ** 2
                         + self.s_moist ** 2 + self.s_nut ** 2)


def difference_sigma(model: OmegaModel, a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """1-sigma of omega(a) - omega(b) for two evaluated plans (``OmegaTerms`` as dicts). The data part uses the
    difference of the gradients (shared parameters cancel); operating and gas-capacity parts are taken as fully
    correlated between the plans (same physics), the surrogate part (surrogate backend only) as independent."""
    names = model.post_names
    idx = [names.index("x_blend")] + [names.index("sh_" + c) for c in model.classes]
    V = model.post_cov[np.ix_(idx, idx)]
    dg = np.asarray(a["g"], float) - np.asarray(b["g"], float)
    s2 = max(float(dg @ V @ dg), 0.0)
    if model.split_oper:
        s2 += oper_sigma(model, a.get("d_prod_phys", 0.0) - b.get("d_prod_phys", 0.0),
                         a.get("d_rest_phys", 0.0) - b.get("d_rest_phys", 0.0)) ** 2
    else:
        s2 += (model.op_rel_sd * (a["d_oper"] - b["d_oper"])) ** 2
    s2 += (model.R_rel_sd * (a["d_R"] - b["d_R"])) ** 2
    s2 += a["sigma_parts"].get("surrogate", 0.0) ** 2 + b["sigma_parts"].get("surrogate", 0.0) ** 2
    # moisture and nut coke: same coefficient for both plans (fully correlated); zero at the same moisture / nut coke
    s2 += (a["sigma_parts"].get("moisture", 0.0) - b["sigma_parts"].get("moisture", 0.0)) ** 2
    s2 += (a["sigma_parts"].get("nut_coke", 0.0) - b["sigma_parts"].get("nut_coke", 0.0)) ** 2
    return math.sqrt(s2)


def oper_sigma(model: OmegaModel, d_prod_phys: float, d_rest_phys: float) -> float:
    """1-sigma of the scaled operating channel: production and other parts with correlated scales (STRATA 1.2, 1.3;
    STRATA 1.4: production sd 0, so only the other-operation part)."""
    a, b = model.sd_prod * d_prod_phys, model.sd_rest * d_rest_phys
    return math.sqrt(max(a * a + b * b + 2 * model.corr_oper * a * b, 0.0))


def assumption_sensitivity(model: OmegaModel, st: OmegaState, K: float, R: float, ri_step: float = 5.0,
                           n_alt: float = 1.5, min_fe_share: float = 0.005, beta_alt: float = 0.25) -> list[dict[str, Any]]:
    """Change of a plan's omega (against the calibration) when one critical assumption takes another credible value.

    Each alternative re-anchors kappa on the calibration (so the calibration's omega is unchanged) and re-evaluates the
    plan at the same coke rate and capacity ratio. Covered: the reducibility (IS 8167 / ISO 7215) of every material in
    the plan or the calibration (+/- the asset's ri_sd of the material, else ri_step), the size exponent of the
    reduction rate (n -> n_alt), the exponent of the RI in the rate (beta -> beta_alt; with beta = 0 the RI rows are zero
    and omitted), the plant-data scales of the pellet and lump-ore channels and of the production channel where the
    asset has one (+/- 1 sd; STRATA 1.4 has none),
    the blast-moisture and nut-coke coefficients (+/- 1 se).
    Returns rows sorted by the size of the effect; omega units (multiply by dK/domega for coke)."""
    import copy
    base = PlanOmega(model, st).omega(K, R)
    ref = model.ref
    rows: list[dict[str, Any]] = []

    def alt(rates: Mapping[str, float]) -> float:
        m2 = copy.copy(model)
        m2.rate = dict(rates)
        m2.calibrate(ref, model.omega_ref, model.R_ref)
        return PlanOmega(m2, st).omega(K, R) - base

    mats = {mm for mm, v in st.fe_frac.items() if v >= min_fe_share} | {mm for mm, v in ref.fe_frac.items() if v >= min_fe_share}
    kin = model.spec["kinetics"]
    beta0 = float(kin.get("ri_exponent", 1.0))
    if beta0 == 0.0:
        mats = set()                      # the RI does not enter the rate: no RI rows (see the beta row below)
    for mm in sorted(mats):
        info = model.spec["materials"].get(mm, {})
        ri = float(info.get("ri_pct", float("nan")))
        step = float(info.get("ri_sd", ri_step))              # the test's uncertainty where the asset carries one
        effs = []
        for dri in (-step, step):
            v = dict(info, ri_pct=min(95.0, max(30.0, ri + dri)))
            r2 = dict(model.rate); r2[mm] = model._rate(v)
            effs.append(alt(r2))
        worst = max(effs, key=abs)
        rows.append(dict(assumption=f"RI {mm}", value=f"{ri:.0f} %", alternative=f"{ri - step:.0f} / {ri + step:.0f} %",
                         basis=str(info.get("ri_basis", "literature")), d_omega=float(worst),
                         d_omega_lo=float(effs[0]), d_omega_hi=float(effs[1]), kind="RI"))
    n0 = float(kin["n_size"])
    if abs(n_alt - n0) > 1e-9:
        r2 = {mm: model._rate(v, n=n_alt) for mm, v in model.spec["materials"].items()}
        rows.append(dict(assumption="size exponent of the reduction rate", value=f"{n0:g}", alternative=f"{n_alt:g}",
                         basis="calibrated on the plant data (n 0.5 / 1.0 / 1.5 tested out of sample; GAS-21)",
                         d_omega=float(alt(r2)), kind="size"))
    if abs(beta_alt - beta0) > 1e-9:
        r2 = {mm: model._rate(v, beta=beta_alt) for mm, v in model.spec["materials"].items()}
        rows.append(dict(assumption="RI exponent of the reduction rate (laboratory RI weight)", value=f"{beta0:g}",
                         alternative=f"{beta_alt:g}", basis="calibrated on the plant data with the laboratory RI as given "
                         "(GAS-21)", d_omega=float(alt(r2)), kind="RI"))
    cs = model.spec.get("channel_scales") or {}
    g = cs.get("g_per_pt") or {}
    if g:
        ds = {c: st.cls.get(c[3:], 0.0) - ref.cls.get(c[3:], 0.0) for c in g}
        pel = g.get("sh_pellet", 0.0) * ds.get("sh_pellet", 0.0)
        lump = sum(g[c] * ds[c] for c in g if c != "sh_pellet")
        rows.append(dict(assumption="plant-data scale, pellet channel", value=f"{cs.get('lam_pellet', 1):.2f}",
                         alternative=f"+/- {cs.get('sd_pellet', 0):.2f}", basis="AR(1) posterior, GAS-14",
                         d_omega=float(cs.get("sd_pellet", 0.0) * abs(pel)), kind="scale", symmetric=True))
        rows.append(dict(assumption="plant-data scale, lump-ore channel", value=f"{cs.get('lam_lump', 1):.2f}",
                         alternative=f"+/- {cs.get('sd_lump', 0):.2f}", basis="AR(1) posterior, GAS-14",
                         d_omega=float(cs.get("sd_lump", 0.0) * abs(lump)), kind="scale", symmetric=True))
    if model.split_oper and model.prod_channel:
        po = PlanOmega(model, st)
        rows.append(dict(assumption="plant-data scale, production channel", value=f"{model.lam_prod:.2f}",
                         alternative=f"+/- {model.sd_prod:.2f}", basis="AR(1) posterior, GAS-16",
                         d_omega=float(model.sd_prod * abs(po.d_prod_phys)), kind="scale", symmetric=True))
    dh = float(st.humidity) - float(ref.humidity)
    if math.isfinite(dh) and abs(dh) > 0 and model.a_moist_se > 0:      # STRATA 1.5: no moisture term (GAS-19 retired)
        rows.append(dict(assumption="blast-moisture coefficient", value=f"{model.a_moist:.4f}/g",
                         alternative=f"+/- {model.a_moist_se:.4f}", basis="GAS-19 (DATA-11)",
                         d_omega=float(model.a_moist_se * abs(dh)), kind="moisture", symmetric=True))
    dn = float(st.nut) - float(ref.nut)
    if abs(dn) > 0 and model.a_nut_se > 0:
        rows.append(dict(assumption="nut-coke coefficient", value=f"{model.a_nut:.4f}/kg",
                         alternative=f"+/- {model.a_nut_se:.4f}", basis="GAS-23 (AR(1) posterior)",
                         d_omega=float(model.a_nut_se * abs(dn)), kind="nut", symmetric=True))
    rows.sort(key=lambda r: -abs(r["d_omega"]))
    return rows


@lru_cache(maxsize=4)
def _load_spec(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
