"""BF-02 burden engine 2.0: the ferrous blend and flux that minimise ore + flux + fuel cost, or maximise production,
with the coke rate from the two-zone (Rist) heat and mass balance and the blast limit from BF-02's burden-resistance
rules.

What it adds to the Blend Mix Optimiser LP (``utils.bmo.lp_solver.run_lp_baseline``)
  * fuel cost in the objective: the coke rate (kg/tHM) is an output of the two-zone balance
    (``utils.coke_energy.rist.solve_2z``) for every blend, with PCI and nut coke fixed. Coke is linearised around the
    current blend and the LP re-solved until the blend stops moving (sequential LP). ``fuel=False`` gives the
    ore + flux objective (the page-9 objective) from the same rows.
  * production limits (``Limits.process``): hot-blast pressure, top pressure, pressure drop, blower volume, oxygen
    flow and enrichment, hot-blast temperature, charging rate. Max blast comes from (Pb^2 - Pt^2) = K V^1.8.
  * three modes: :meth:`Engine.min_cost` at a fixed production; :meth:`Engine.max_production`;
    :meth:`Engine.min_total_cost` with the production level chosen by the engine (plus the cost frontier).

Engine 2.0 (fact register v1.1)
  * coefficients outside the balance come from ``facts_register.json`` (``utils.burden_engine.facts``);
  * RAFT is the plant workbook at the input blast moisture (DATA-03/04), reported and checked against the band;
  * plausibility bands (BAND-01..03): fuel rate above 600 kg/tHM is an elastic limit row in every mode and below
    520 a validity check, so such a result comes back infeasible and is not reported; HMT and HM Si inputs outside
    1,440-1,520 degC and 0.20-0.80 % are refused;
  * the webapp hand coefficients (COKE-17) are gone.

Shaft efficiency (``Settings.omega_mode``)
  * "fixed" (default, ASM-01): omega held at the calibration for every blend; a blend moves the coke rate through its
    chemistry only.
  * "blend" (ASM-18, GAS-13..20): STRATA-omega from ``utils.burden_engine.omega_model`` (STRATA 1.2) - the STRATA-Flow
    gas-flow model's ring counter-current description of the thermal reserve zone, scaled by the plant data and tested
    out of sample (GAS-20) - as a function of the blend, the coke rate, production, oxygen and the blast moisture
    (solved inside the two-zone balance). Every result then carries sigma(coke), a certainty class (High / Medium /
    Low / Investigate) and the parts of omega; ``Engine.critical_assumptions`` gives the coke effect of each critical
    assumption (RI, size exponent, plant-data scales, moisture coefficient);
    ``Settings.risk_k`` adds risk_k x sigma(cost) to the objective and ``Settings.certainty`` ("low" = all but
    Investigate / "medium" / "high") restricts the optimiser to plans of that certainty.

Units: decision variables are WET kg/tHM of each material; shares are wet % of the ferrous charge; prices Rs per wet
tonne. Pure numerical code: no Streamlit, no I/O except reading the JSON inputs when given paths.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from utils.burden_engine.facts import ENGINE_FACTS, FactRegister, load_register
from utils.burden_engine.hydraulics import Hydraulics, raft_c
from utils.burden_engine.omega_model import (CERTAINTY_LEVELS, OmegaModel, OmegaState, assumption_sensitivity,
                                             difference_sigma)
from utils.coke_energy.rist import solve_2z

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "burden_engine"
DEFAULT_CAL_PATH = ASSET_DIR / "cal_2z_jul_oct_2026.json"
REFERENCE_INPUTS_PATH = ASSET_DIR / "inputs_uat_2026-10-16.json"

OVERSIZE_RULE = "oversize"
GOMTI_RULE = "gomti"


class Infeasible(Exception):
    """The LP itself failed (not a limit breach: those come back as ``feasible=False``)."""


class BandError(ValueError):
    """An input lies outside its plausibility band (BAND-02 / BAND-03)."""


@dataclass
class Settings:
    """Furnace controls for one evaluation. Defaults = Jul-Oct 2026 calibration, client thermal basis (ASM-08)."""

    HBT: float = 1188.4
    O2_per_air: float = 0.05209          # Nm3 O2 per Nm3 air (model basis); see Hydraulics.enrich_tag()
    humidity_g: float = 15.0             # ambient blast moisture, g/Nm3 (DATA-04: not measured)
    HMT: float = 1470.0
    hm_Si: float = 0.3836
    pci_plant: float = 190.0             # kg/tHM as injected
    nut_plant: float = 70.0              # kg/tHM as charged
    Pb: float = 2.79                     # hot-blast pressure, bar g
    Pt: float = 1.42                     # top pressure, bar g
    omega: float | None = None           # shaft efficiency; None = calibrated value (fixed mode)
    nut_size_delta_mm: float = 0.0       # nut coke mean size vs the base point (FLOW-07)
    omega_mode: str = "fixed"            # "fixed" (ASM-01) or "blend" (ASM-18: omega_model)
    risk_k: float = 0.0                  # blend mode: objective = cost + risk_k x sigma(cost)
    certainty: str = "any"               # blend mode: "any", "low" (all but Investigate), "medium", "high"


@dataclass
class Limits:
    """Operator limits. Slag, charging and fuel-band rows are always on; process rows only when process=True.
    Defaults equal the register (ASM-11/12/13, FLOW-12); :meth:`from_register` builds them from it."""

    slag_max: float = 340.0              # kg/tHM
    B2_min: float = 1.05
    B2_max: float = 1.15
    Al2O3_max: float = 20.0              # % of slag
    MgO_min: float = 6.5                 # % of slag
    MgO_Al2O3_min: float = 0.345
    charge_t: float = 26.46              # non-coke t per charge (ferrous + flux + nut coke)
    charges_per_h: float = 6.5
    T_top_min: float = 120.0             # degC, upper-zone heat check (ASM-05/13)
    fuel_min: float = 520.0              # kg/tHM, coke wet + nut + PCI (BAND-01)
    fuel_max: float = 600.0
    process: bool = False
    Pb_max: float = 2.85                 # bar g
    dP_max: float = 1.45                 # bar
    Pt_min: float = 1.30
    Pt_max: float = 1.50
    V_max: float = 115000.0              # Nm3/h hot blast (blower / stoves)
    O2_flow_max: float = 6500.0          # Nm3/h
    enrich_max: float = 5.5              # plant enrichment tag, %
    HBT_max: float = 1230.0              # degC
    RAFT_max: float = 2350.0
    RAFT_min: float = 2100.0

    @classmethod
    def from_register(cls, reg: FactRegister | None = None, **overrides: Any) -> "Limits":
        reg = reg or load_register()
        d = {k: v for k, v in reg.default_limits().items() if k in cls.__dataclass_fields__}
        lo, hi = reg.band("BAND-01")
        d.update(fuel_min=lo, fuel_max=hi)
        d.update(overrides)
        return cls(**d)


@dataclass
class Var:
    id: str
    name: str
    kind: str                            # sinter / ore / pellet / flux
    price: float
    chem: dict
    moist: float
    lo: float = 0.0                      # wet % of ferrous (ferrous only)
    hi: float = 100.0
    max_t_day: float | None = None       # wet t/day available (stock over the plan horizon); None = no cap
    k_rule: str | None = None            # burden-resistance rule: 'gomti' (FLOW-03) or 'oversize' (FLOW-04)
    loi_type: str = "H2O"                # what the LOI leaves as: combined water or carbonate CO2


def _load(obj: str | Path | Mapping[str, Any]) -> dict:
    if isinstance(obj, (str, Path)):
        with open(obj, encoding="utf-8") as fh:
            return json.load(fh)
    return copy.deepcopy(dict(obj))


def _k_rule(m: Mapping[str, Any]) -> str | None:
    if m.get("k_rule"):
        return m["k_rule"]
    mid = m["id"]
    if mid == "gomti_clo":
        return GOMTI_RULE
    if mid == "lloyds_oversize_clo":
        return OVERSIZE_RULE
    return None


class Engine:
    OUT = ["K", "K_wet", "hb", "O2", "T_top", "slag", "CaO", "SiO2", "MgO", "Al2O3", "Fe_res", "eta", "sig_K"]

    def __init__(self, inputs: str | Path | Mapping[str, Any] | None = None,
                 cal: str | Path | Mapping[str, Any] | None = None,
                 facts: FactRegister | None = None, materials: Sequence[str] | None = None,
                 prices: Mapping[str, float] | None = None,
                 bounds: Mapping[str, tuple[float, float]] | None = None):
        self.inp = _load(inputs if inputs is not None else REFERENCE_INPUTS_PATH)
        self.cal = _load(cal if cal is not None else DEFAULT_CAL_PATH)
        self.facts = facts or load_register()
        self.th = self.facts.thumb_rules()
        # A live calibration carries its own packed-bed anchor (same shifts as omega and the losses); the shipped
        # Jul-Oct 2026 calibration uses the register's base point (FLOW-01).
        anchor = (self.cal.get("anchor") or {}) if isinstance(self.cal, Mapping) else {}
        self.th["base"] = {**self.th["base"], **{k: float(v) for k, v in anchor.items()}}
        self.hy = Hydraulics.from_rules(self.th)
        self.band_hmt = self.facts.band("BAND-02")
        self.band_si = self.facts.band("BAND-03")
        mats = self.inp["materials"]
        if materials is None:
            materials = [m["id"] for m in mats if m.get("selected", m.get("selected_in_uat", True))]
        self.fer: list[Var] = []
        for mid in materials:
            m = next(x for x in mats if x["id"] == mid)
            price = (prices or {}).get(mid, m["price"])
            lo, hi = (bounds or {}).get(mid, (m["lo"], m["hi"]))
            self.fer.append(Var(m["id"], m.get("name", mid), m["kind"], float(price), m["chem"], float(m["moist"]),
                                float(lo), float(hi), max_t_day=m.get("max_t_day"), k_rule=_k_rule(m),
                                loi_type=m.get("loi_type", "H2O")))
        if not self.fer:
            raise ValueError("no ferrous materials selected")
        self.flux = [Var(f["id"], f.get("name", f["id"]), "flux", float((prices or {}).get(f["id"], f["price"])), f["chem"],
                         float(f["moist"]), max_t_day=f.get("max_t_day"),
                         loi_type=f.get("loi_type", "H2O" if f["id"] in ("quartz", "quartzite") else "CO2"))
                     for f in self.inp["fluxes"]]
        self.vars = self.fer + self.flux
        self.nf, self.nx = len(self.fer), len(self.flux)
        self.n = self.nf + self.nx
        fu = self.inp["fuel"]
        self.prices_fuel = dict(coke=float(fu["coke_price"]), nut=float(fu["nut_price"]), pci=float(fu["pci_price"]))
        c = self.cal
        self.case0 = copy.deepcopy(c["case"])
        self.p = c["p"]
        du = self.case0["dust"]
        self.fe_dust = du["dc_kg"] * du["dc_Fe_pct"] / 100 + du["gcp_kg"] * du["gcp_Fe_pct"] / 100
        # Hot-blast meter vs model: at the calibration point the model's blast + O2 per tHM x production reproduces the
        # meter's average to ~1 % (DATA-01); this factor removes the remainder.
        hb_meter = c.get("hb_meter_nm3h") or self.hy.base["V"]
        self.c_hb = hb_meter / (c["case"]["prod_th"] * c["hb_per_t_model_cal"])
        self.K_base = c["coke_model_cal"]
        self._x0 = (c["K"], c["V"], c["y_d"])
        self.fe_wet = np.array([v.chem["Fe"] / 100 * (1 - v.moist / 100) for v in self.vars])
        self.is_pel = np.array([v.kind == "pellet" for v in self.fer], float)
        self.is_gom = np.array([v.k_rule == GOMTI_RULE for v in self.fer], float)
        self.is_o6 = np.array([v.k_rule == OVERSIZE_RULE for v in self.fer], float)
        self.n_evals = 0
        self._om: OmegaModel | None = None
        self._dKdw = float("nan")

    # ------------------------------------------------------------------------------------------------ omega model
    def omega_model(self) -> OmegaModel:
        """The blend-dependent omega model, anchored on this engine's calibration (built on first use)."""
        if self._om is None:
            m = OmegaModel.load()
            self.om_mat = [m.model_material(v.id, v.kind) for v in self.fer]
            self.om_cls = {c: np.array([m.cls_of[mm] == c for mm in self.om_mat], float) for c in m.support}
            c, P = self._cal_case(), self.case0["prod_th"]
            cal = self.cal
            r = solve_2z(c, self.p, cal["omega"], cal["Q_LZ_GJh"] * 1000 / P, cal["Q_UZ_GJh"] * 1000 / P, cal["r_H"],
                         T_R=cal["T_R"], x0=self._x0)
            m.calibrate(self._omega_ref_state(m), cal["omega"], r["R_cap"])
            self._RC_cal = float(r["R_cap_C"])
            # loop-inclusive dK/domega: the same blend with omega shifted by d (coke and gas capacity respond)
            po = m.plan(m.ref)
            d = 0.005
            Ks = []
            for off in (0.0, d):
                fn = (lambda K, z, off=off: po.omega(K, self.R_eff(z)) + off)
                Ks.append(solve_2z(c, self.p, cal["omega"], cal["Q_LZ_GJh"] * 1000 / P, cal["Q_UZ_GJh"] * 1000 / P,
                                   cal["r_H"], T_R=cal["T_R"], x0=self._x0, omega_fn=fn)["K"])
            self._dKdw = (Ks[1] - Ks[0]) / d
            self._om = m
        return self._om

    def R_eff(self, z: Mapping[str, Any]) -> float:
        """Capacity ratio the omega model sees. STRATA 1.1+ (``gas_capacity_channel: false``): the calibration's R for
        every plan - the gas-capacity channel (GAS-17, retired) gave no out-of-sample skill and a negative data scale.
        With the channel on (v1.0): the calibration's total R moved by the change of its carbon-gas part only."""
        from utils.coke_energy.rist import Y_W
        if self._om is not None and not self._om.gas_capacity:
            return self._om.R_ref
        return self._om.R_ref + (z["etaWc"] * z["nCg"] / (Y_W * z["nFe"]) - self._RC_cal) if self._om is not None \
            else z["O_max_TRZ"] / (Y_W * z["nFe"])

    def _cal_case(self):
        """The calibration point as a two-zone case (class-level burden of the calibration window)."""
        c = copy.deepcopy(self.case0)
        c["burden"] = [dict(b, LOI_type=b.get("LOI_type", "CO2" if b["name"] == "flux" else "H2O"))
                       for b in self.cal["burden_cal"]]
        return c

    def _omega_ref_state(self, m: OmegaModel) -> OmegaState:
        """The calibration window as an omega state: class shares of the calibration, the lump-ore split of the model's
        reference window (the calibration carries classes only), the calibration's coke, fuels, production, blast."""
        cal, case = self.cal, self.case0
        wet: dict[str, float] = {}
        split = m.spec["reference"]["lump_split"]
        for b in cal["burden_cal"]:
            kg = float(b["dry_kg"]) / (1 - float(b.get("moist_pct", 0.0)) / 100)
            if b["name"] == "sinter":
                wet["sinter"] = wet.get("sinter", 0.0) + kg
            elif b["name"] == "pellet":
                wet["pellet"] = wet.get("pellet", 0.0) + kg
            elif b["name"] == "ore":
                for mm, f in split.items():
                    wet[mm] = wet.get(mm, 0.0) + kg * float(f)
        sh, fe, cls = m.blend_parts(wet)
        r_o2 = float(case["O2_per_air"])
        return OmegaState(sh, fe, cls, coke=float(cal["K"]),
                          nut=float(case["nut_kg"]) / (1 - float(case["nut"]["moist_pct"]) / 100),
                          pci=float(case["pci_kg"]) / (1 - float(case["pci"]["moist_pct"]) / 100),
                          prod=float(case["prod_th"]), blast_t=self.c_hb * float(cal["hb_per_t_model_cal"]),
                          o2=r_o2 / (1 + r_o2), rdi=m.rdi, fer=sum(wet.values()),
                          humidity=float(case.get("humidity_g", float("nan"))))

    def plan_state(self, z, s: Settings, P) -> OmegaState:
        m = self.omega_model()
        wet: dict[str, float] = {}
        for mm, w in zip(self.om_mat, z[:self.nf]):
            wet[mm] = wet.get(mm, 0.0) + max(float(w), 0.0)
        sh, fe, cls = m.blend_parts(wet)
        return OmegaState(sh, fe, cls, coke=float(m.ref.coke), nut=float(s.nut_plant), pci=float(s.pci_plant),
                          prod=float(P), blast_t=m.ref.blast_t, o2=s.O2_per_air / (1 + s.O2_per_air), rdi=m.rdi,
                          fer=float(np.sum(np.maximum(z[:self.nf], 0.0))), humidity=float(s.humidity_g))

    def omega_limits(self, s: Settings) -> dict[str, Any] | None:
        """The certainty filter's constraints (blend mode only)."""
        if s.omega_mode != "blend" or s.certainty not in CERTAINTY_LEVELS or s.certainty == "any":
            return None
        return self.omega_model().limits_for(s.certainty)

    def p_range(self, s: Settings, lo: float, hi: float) -> tuple[float, float]:
        """Production search range, narrowed to the certainty filter's production range."""
        lim = self.omega_limits(s)
        if lim and lim.get("production"):
            lo, hi = max(lo, lim["production"][0]), min(hi, lim["production"][1])
        return lo, max(lo, hi)

    # ------------------------------------------------------------------------------------------------ checks
    def check_settings(self, s: Settings) -> None:
        """BAND-02 / BAND-03: refuse thermal inputs outside what BF-02 runs; known omega mode and certainty level."""
        if s.omega_mode not in ("fixed", "blend"):
            raise ValueError(f"omega_mode must be 'fixed' or 'blend', not {s.omega_mode!r}")
        if s.certainty not in CERTAINTY_LEVELS:
            raise ValueError(f"certainty must be one of {CERTAINTY_LEVELS}, not {s.certainty!r}")
        lo, hi = self.band_hmt
        if not lo <= s.HMT <= hi:
            raise BandError(f"HMT {s.HMT:.0f} degC is outside BAND-02 ({lo:.0f}-{hi:.0f} degC)")
        lo, hi = self.band_si
        if not lo <= s.hm_Si <= hi:
            raise BandError(f"HM Si {s.hm_Si:.2f} % is outside BAND-03 ({lo:.2f}-{hi:.2f} %)")

    # ------------------------------------------------------------------------------------------------ model
    def make_case(self, z, s: Settings, P):
        c = copy.deepcopy(self.case0)
        c["burden"] = []
        for v, w in zip(self.vars, z):
            dry = max(w, 0.0) * (1 - v.moist / 100)
            c["burden"].append(dict(name=v.id, dry_kg=dry, comp=v.chem, moist_pct=v.moist, LOI_type=v.loi_type))
        c["pci_kg"] = s.pci_plant * (1 - c["pci"]["moist_pct"] / 100)
        c["nut_kg"] = s.nut_plant * (1 - c["nut"]["moist_pct"] / 100)
        c["hm"] = dict(c["hm"]); c["hm"]["Si"] = s.hm_Si
        c["HMT"], c["HBT"], c["O2_per_air"], c["humidity_g"] = s.HMT, s.HBT, s.O2_per_air, s.humidity_g
        c["prod_th"] = P
        return c

    def evaluate(self, z, s: Settings, P, full=False):
        """Two-zone solve for an explicit burden z (wet kg/tHM). The Fe balance is NOT forced here (see size_fe)."""
        c = self.make_case(z, s, P)
        cal = self.cal
        om = cal["omega"] if s.omega is None else s.omega
        po = None
        if s.omega_mode == "blend":
            po = self.omega_model().plan(self.plan_state(z, s, P))
            if s.omega is not None:                       # an explicit omega shifts the calibration anchor
                po.base += s.omega - cal["omega"]
        fn = (lambda K, zz: po.omega(K, self.R_eff(zz))) if po is not None else None
        r = solve_2z(c, self.p, om, cal["Q_LZ_GJh"] * 1000 / P, cal["Q_UZ_GJh"] * 1000 / P, cal["r_H"],
                     T_R=cal["T_R"], x0=self._x0, omega_fn=fn)
        self.n_evals += 1
        ro = r["ro"]; sl = ro["slag"]
        K = r["K"]; wet = 1 - c["coke"]["moist_pct"] / 100; Kw = K / wet
        R_eff = self.R_eff(r["z"]) if po is not None else r["R_cap"]
        sig_w = po.sigma(K, R_eff) if po is not None else 0.0
        out = dict(K=K, K_wet=Kw, hb=r["V"] + ro["V_O2"], O2=ro["V_O2"], T_top=r["T_top"], slag=ro["slag_kg"],
                   CaO=sl["CaO"], SiO2=sl["SiO2"], MgO=sl["MgO"], Al2O3=sl["Al2O3"],
                   Fe_res=ro["Fe_burden"] - ro["Fe_HM"] - ro["Fe_slag"] - self.fe_dust, eta=100 * r["eta"],
                   sig_K=abs(self._dKdw) * sig_w / wet if po is not None else 0.0)
        if full:
            out.update(air=r["V"], y_d=r["y_d"], r_d=r["r_d"], topgas=ro["V_tg"], B2=sl["B2"], Al2O3_pct=sl["Al2O3_pct"],
                       MgO_pct=sl["MgO_pct"], O_err=r["O_err"], E_err=r["E_err"], omega=r["omega"], R_cap=r["R_cap"],
                       fuel_dry=K + c["nut_kg"] + c["pci_kg"], fuel_rate=Kw + s.nut_plant + s.pci_plant,
                       nut_dry=c["nut_kg"], pci_dry=c["pci_kg"], CO2_carb=ro["CO2_carb_kg"])
            if po is not None:
                out["omega_terms"] = asdict(po.terms(K, R_eff))
        return out

    def phi(self, z, s, P):
        o = self.evaluate(z, s, P)
        return np.array([o[k] for k in self.OUT]), o

    def size_fe(self, z, s, P, tol=1e-6):
        """Scale the ferrous part of z so the Fe balance closes exactly (fluxes unchanged)."""
        z = np.array(z, float)
        a = 1.0
        zz = z.copy()
        for _ in range(8):
            zz = z.copy(); zz[:self.nf] *= a
            o = self.evaluate(zz, s, P)
            if abs(o["Fe_res"]) < tol:
                break
            fe = float(self.fe_wet[:self.nf] @ zz[:self.nf])
            a *= 1 - o["Fe_res"] / fe
        return zz

    # ------------------------------------------------------------------------------------------------ helpers
    def shares(self, z) -> dict[str, float]:
        Y = float(z[:self.nf].sum())
        return dict(pellet=100 * float(self.is_pel @ z[:self.nf]) / Y, gomti=100 * float(self.is_gom @ z[:self.nf]) / Y,
                    oversize=100 * float(self.is_o6 @ z[:self.nf]) / Y)

    def dlnK(self, z, K, s: Settings, ore6_on=False):
        """ln(K / K_base): burden resistance relative to the register's base point (FLOW-02..07)."""
        th, b = self.th, self.hy.base
        sh = self.shares(z)
        o2 = self.hy.o2_pct_of_blast(s.O2_per_air)
        return (th["beta_pellet"] * (sh["pellet"] - b["pellet"]) + th["beta_gomti"] * (sh["gomti"] - b["gomti"])
                + th["beta_ore6_on"] * (float(ore6_on) - b["ore6_on"]) + th["beta_o2"] * (o2 - b["o2_pct"])
                + th["beta_coke"] * (K - self.K_base) + th["beta_nut_size"] * s.nut_size_delta_mm)

    def grad_dlnK(self, z, JK):
        th = self.th
        Y = z[:self.nf].sum()
        g = np.zeros(self.n)
        for arr, beta in [(self.is_pel, th["beta_pellet"]), (self.is_gom, th["beta_gomti"])]:
            S = float(arr @ z[:self.nf])
            g[:self.nf] += beta * 100 * (arr * Y - S) / Y ** 2
        return g + th["beta_coke"] * JK

    def raft(self, s: Settings, hb_per_t: float | None = None, O2_per_t: float | None = None) -> float:
        """Plant-workbook RAFT at the blast per tonne (metered basis). Without a result, the calibration's blast per
        tonne is used, which is how the control search sets HBT before the balance is solved."""
        if hb_per_t is None:
            hb_per_t = self.c_hb * self.cal["hb_per_t_model_cal"]
        if O2_per_t is None:
            O2_per_t = hb_per_t * s.O2_per_air / (1 + s.O2_per_air)
        return raft_c(s.HBT, hb_per_t, O2_per_t, s.pci_plant, s.humidity_g)

    def cost(self, z, o, fuel=True, s: Settings | None = None):
        pr = np.array([v.price for v in self.vars])
        ore = float(pr[:self.nf] @ z[:self.nf]) / 1000
        flx = float(pr[self.nf:] @ z[self.nf:]) / 1000
        s = s or Settings()
        coke = o["K_wet"] * self.prices_fuel["coke"] / 1000
        nut = s.nut_plant * self.prices_fuel["nut"] / 1000
        pci = s.pci_plant * self.prices_fuel["pci"] / 1000
        sigma = float(o.get("sig_K", 0.0)) * self.prices_fuel["coke"] / 1000          # 1 sigma of the coke cost
        risk = s.risk_k * sigma if fuel else 0.0
        obj = ore + flx + (coke if fuel else 0.0) + risk
        total = ore + flx + coke + nut + pci
        return dict(ore=ore, flux=flx, coke=coke, nut=nut, pci=pci, fuel=coke + nut + pci,
                    total=total, objective=obj, sigma=sigma, risk_adjusted=total + s.risk_k * sigma)

    def var_cap(self, P):
        """Upper bound on each variable in wet kg/tHM from its t/day cap (stock, flux limit) at production P t/h."""
        return np.array([v.max_t_day * 1000 / (P * 24) if v.max_t_day is not None else np.inf for v in self.vars])

    def _try(self, *args, **kw):
        """slp() that returns None when the LP itself has no solution (e.g. stock caps against share minimums)."""
        try:
            return self.slp(*args, **kw)
        except Infeasible:
            return None

    def start_point(self, s, P):
        lo = np.array([v.lo for v in self.fer]); hi = np.array([v.hi for v in self.fer])
        sh = np.clip((lo + hi) / 2, lo, hi)
        if sh.sum() <= 0:
            sh = np.ones(self.nf)
        sh = sh / sh.sum()
        fe_blend = float(sh @ self.fe_wet[:self.nf])
        z = np.zeros(self.n); z[:self.nf] = sh * 960.0 / fe_blend
        return self.size_fe(z, s, P)

    # ------------------------------------------------------------------------------------------------ constraint rows
    def rows(self, z0, Phi0, J, s: Settings, P, L: Limits, ore6_on=False):
        """Linearised rows a.z <= b (ineq) and a.z = b (eq), each with a name and a scale for the elastic penalty."""
        ix = {k: i for i, k in enumerate(self.OUT)}
        lin = lambda k: (J[ix[k]], Phi0[ix[k]] - J[ix[k]] @ z0)      # value(z) ~ a.z + c
        ub, eq = [], []

        def le(name, a, c, rhs, scale, soft=True):        # a.z + c <= rhs
            ub.append((name, a, rhs - c, scale, soft))

        aS, cS = lin("slag"); aCa, cCa = lin("CaO"); aSi, cSi = lin("SiO2"); aMg, cMg = lin("MgO"); aAl, cAl = lin("Al2O3")
        aFe, cFe = lin("Fe_res"); aT, cT = lin("T_top"); aKw, cKw = lin("K_wet")
        eq.append(("Fe balance", aFe, -cFe, 1.0))
        le("slag cap", aS, cS, L.slag_max, 1.0)
        le("basicity min", L.B2_min * aSi - aCa, L.B2_min * cSi - cCa, 0.0, 1.0)
        le("basicity max", aCa - L.B2_max * aSi, cCa - L.B2_max * cSi, 0.0, 1.0)
        le("Al2O3 max", aAl - L.Al2O3_max / 100 * aS, cAl - L.Al2O3_max / 100 * cS, 0.0, 1.0)
        le("MgO min", L.MgO_min / 100 * aS - aMg, L.MgO_min / 100 * cS - cMg, 0.0, 1.0)
        le("MgO/Al2O3 min", L.MgO_Al2O3_min * aAl - aMg, L.MgO_Al2O3_min * cAl - cMg, 0.0, 1.0)
        le("top gas temp min", -aT, -cT, -L.T_top_min, 1.0)
        fixed_fuel = s.nut_plant + s.pci_plant
        le("fuel rate max", aKw, cKw + fixed_fuel, L.fuel_max, 1.0)
        # The band's lower end is a validity check only (see violation): a row would make the optimiser burn coke.
        cap = L.charge_t * L.charges_per_h
        a_ch = np.full(self.n, P / 1000)
        le("charging", a_ch, P / 1000 * s.nut_plant, cap, 1.0)
        lim = self.omega_limits(s)
        if lim:                                              # certainty filter (blend mode)
            if math.isfinite(lim["sigma_max"]):
                aSg, cSg = lin("sig_K")
                le("certainty sigma", aSg, cSg, lim["sigma_max"], 1.0)
            for cname, (lo_c, hi_c) in lim["classes"].items():
                mask = self.om_cls.get(cname)
                if mask is None or not mask.any():
                    continue
                a = np.zeros(self.n); a[:self.nf] = mask - hi_c / 100
                le(f"certainty {cname} max", a, 0.0, 0.0, 1.0)
                if lo_c > 0:
                    a = np.zeros(self.n); a[:self.nf] = lo_c / 100 - mask
                    le(f"certainty {cname} min", a, 0.0, 0.0, 1.0)
        if L.process:
            aH, cH = lin("hb"); aO, cO = lin("O2")
            K0 = Phi0[ix["K"]]
            d0 = self.dlnK(z0, K0, s, ore6_on)
            R0 = self.hy.v_ceiling(s.Pb, s.Pt, d0)
            gd = self.grad_dlnK(z0, J[ix["K"]])
            n = self.hy.n_flow
            # P c hb(z) <= R0 (1 - (dlnK(z) - d0)/n)
            a = P * self.c_hb * aH + R0 / n * gd
            c = P * self.c_hb * cH - R0 / n * (gd @ z0)
            le("blast pressure", a, c, R0, 1000.0)
            le("blast volume", P * self.c_hb * aH, P * self.c_hb * cH, L.V_max, 1000.0)
            le("oxygen flow", P * aO, P * cO, L.O2_flow_max, 100.0)
        hard = []
        for i, v in enumerate(self.fer):
            lo, hi = v.lo, v.hi
            if v.k_rule == OVERSIZE_RULE and L.process:
                lo, hi = (max(lo, 1.0), hi) if ore6_on else (0.0, 0.0)
            a = np.zeros(self.n); a[:self.nf] = lo / 100; a[i] -= 1
            hard.append((f"min share {v.id}", a, 0.0))
            a = np.zeros(self.n); a[:self.nf] = -hi / 100; a[i] += 1
            hard.append((f"max share {v.id}", a, 0.0))
        return ub, eq, hard

    # ------------------------------------------------------------------------------------------------ SLP
    def violation(self, z, o, s, P, L, ore6_on=False, checks=True):
        """Limit breaches of an evaluated blend. ``checks`` adds the validity checks that are not optimisation rows
        (fuel rate below BAND-01): they mark a result as not reportable but must not steer the SLP's merit."""
        v = {}
        v["Fe balance"] = abs(o["Fe_res"])
        v["slag cap"] = max(0.0, o["slag"] - L.slag_max)
        v["basicity min"] = max(0.0, L.B2_min * o["SiO2"] - o["CaO"])
        v["basicity max"] = max(0.0, o["CaO"] - L.B2_max * o["SiO2"])
        v["Al2O3 max"] = max(0.0, o["Al2O3"] - L.Al2O3_max / 100 * o["slag"])
        v["MgO min"] = max(0.0, L.MgO_min / 100 * o["slag"] - o["MgO"])
        v["MgO/Al2O3 min"] = max(0.0, L.MgO_Al2O3_min * o["Al2O3"] - o["MgO"])
        v["top gas temp min"] = max(0.0, L.T_top_min - o["T_top"])
        fuel = o["K_wet"] + s.nut_plant + s.pci_plant
        v["fuel rate max"] = max(0.0, fuel - L.fuel_max)
        if checks:
            v["fuel rate min"] = max(0.0, L.fuel_min - fuel)
        v["charging"] = max(0.0, P / 1000 * (z.sum() + s.nut_plant) - L.charge_t * L.charges_per_h)
        lim = self.omega_limits(s)
        if lim:
            if math.isfinite(lim["sigma_max"]):
                v["certainty sigma"] = max(0.0, float(o.get("sig_K", 0.0)) - lim["sigma_max"])
            Y = float(np.sum(z[:self.nf]))
            for cname, (lo_c, hi_c) in lim["classes"].items():
                mask = self.om_cls.get(cname)
                if mask is None or not mask.any():
                    continue
                sc = float(mask @ z[:self.nf])
                v[f"certainty {cname} max"] = max(0.0, sc - hi_c / 100 * Y)
                if lo_c > 0:
                    v[f"certainty {cname} min"] = max(0.0, lo_c / 100 * Y - sc)
        if L.process:
            V = P * self.c_hb * o["hb"]
            v["blast pressure"] = max(0.0, V - self.hy.v_ceiling(s.Pb, s.Pt, self.dlnK(z, o["K"], s, ore6_on))) / 1000
            v["blast volume"] = max(0.0, V - L.V_max) / 1000
            v["oxygen flow"] = max(0.0, P * o["O2"] - L.O2_flow_max) / 100
        return v

    def slp(self, s: Settings, P, L: Limits, fuel=True, z0=None, ore6_on=False, max_iter=40, tol=0.02, verbose=False):
        """Sequential LP with an elastic (penalised) copy of every physics row, so it always returns a point;
        'feasible' is True when no penalty is used. Returns the blend, outputs, costs and LP duals."""
        self.check_settings(s)
        M = 1e5
        z0 = self.start_point(s, P) if z0 is None else np.array(z0, float)
        vcap = self.var_cap(P)
        dmax = np.r_[np.full(self.nf, 400.0), np.full(self.nx, 40.0)]
        delta = dmax.copy()
        Phi0, o0 = self.phi(z0, s, P)
        merit = lambda z, o: self.cost(z, o, fuel, s)["objective"] + M * sum(self.violation(z, o, s, P, L, ore6_on, checks=False).values())
        m0 = merit(z0, o0)
        last_lp = None
        lp = soft_ub = eq = hard = n_s = J = None
        for it in range(max_iter):
            J = np.zeros((len(self.OUT), self.n))
            for j in range(self.n):
                h = 5.0 if j < self.nf else 2.0
                zz = z0.copy(); zz[j] += h
                J[:, j] = (self.phi(zz, s, P)[0] - Phi0) / h
            ub, eq, hard = self.rows(z0, Phi0, J, s, P, L, ore6_on)
            soft_ub = [r for r in ub if r[4]]
            n_s = len(soft_ub) + 2 * len(eq)
            pr = np.array([v.price for v in self.vars]) / 1000
            cz = pr + (self.prices_fuel["coke"] / 1000 * (J[self.OUT.index("K_wet")] + s.risk_k * J[self.OUT.index("sig_K")])
                       if fuel else 0.0)
            c = np.r_[cz, np.full(n_s, M) / np.r_[[r[3] for r in soft_ub], [r[3] for r in eq], [r[3] for r in eq]]]
            A_ub, b_ub = [], []
            for k, (name, a, b, scale, soft) in enumerate(soft_ub):
                row = np.zeros(self.n + n_s); row[:self.n] = a; row[self.n + k] = -1.0
                A_ub.append(row); b_ub.append(b)
            for name, a, b in hard:
                row = np.zeros(self.n + n_s); row[:self.n] = a
                A_ub.append(row); b_ub.append(b)
            A_eq, b_eq = [], []
            for k, (name, a, b, scale) in enumerate(eq):
                row = np.zeros(self.n + n_s); row[:self.n] = a
                row[self.n + len(soft_ub) + 2 * k] = 1.0; row[self.n + len(soft_ub) + 2 * k + 1] = -1.0
                A_eq.append(row); b_eq.append(b)
            lo = np.maximum(0.0, z0 - delta); hi = z0 + delta
            hi = np.minimum(hi, vcap)
            lo = np.minimum(lo, hi)
            bounds = [(l, h) for l, h in zip(lo, hi)] + [(0, None)] * n_s
            lp = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                         bounds=bounds, method="highs")
            if not lp.success:
                raise Infeasible(lp.message)
            z1 = lp.x[:self.n]
            Phi1, o1 = self.phi(z1, s, P)
            m1 = merit(z1, o1)
            step = np.max(np.abs(z1 - z0))
            if verbose:
                print(f"  it {it:2d} merit {m0:10.2f} -> {m1:10.2f} step {step:8.3f} delta {delta[0]:7.2f}")
            if m1 <= m0 + 1e-6 or step < tol:
                last_lp = dict(lp=lp, ub=soft_ub, eq=eq, hard=hard, n_s=n_s, J=J, z0=z0, iters=it + 1)
                z0, Phi0, o0, m0 = z1, Phi1, o1, m1
                delta = np.minimum(delta * 1.5, dmax)
                if step < tol:
                    break
            else:
                delta *= 0.4
                if delta[0] < 0.01:
                    break
        if last_lp is None:
            last_lp = dict(lp=lp, ub=soft_ub, eq=eq, hard=hard, n_s=n_s, J=J, z0=z0, iters=max_iter)
        return self._result(z0, s, P, L, fuel, last_lp, ore6_on)

    def _result(self, z, s, P, L, fuel, last, ore6_on):
        zs = self.size_fe(z, s, P)
        o = self.evaluate(zs, s, P, full=True)
        viol = self.violation(zs, o, s, P, L, ore6_on)
        lp = last["lp"]
        names_ub = [r[0] for r in last["ub"]] + [h[0] for h in last["hard"]]
        duals = {}
        try:
            mu = lp.ineqlin.marginals
            for k, nm in enumerate(names_ub):
                if abs(mu[k]) > 1e-9:
                    duals[nm] = float(mu[k])
            duals["Fe balance"] = float(lp.eqlin.marginals[0])
        except Exception:
            pass
        slack_used = float(np.sum(lp.x[self.n:])) if lp is not None else 0.0
        Y = zs[:self.nf].sum()
        V = P * self.c_hb * o["hb"]
        if not L.process:      # page mode: no blast rows, but report the pressure the blend needs with oversize as used
            ore6_on = bool(self.is_o6.any() and 100 * float(self.is_o6 @ zs[:self.nf]) / Y > 1.0)
        d = self.dlnK(zs, o["K"], s, ore6_on)
        raft = self.raft(s, self.c_hb * o["hb"], o["O2"])
        sh = self.shares(zs)
        warnings = []
        vm = self.th["valid_max"]
        if sh["pellet"] > vm["pellet"] + 1e-6:
            warnings.append(f"Pellet {sh['pellet']:.1f} % of ferrous is beyond the {vm['pellet']:.1f} % the pellet "
                            "burden-resistance rule was fitted on (FLOW-02, ASM-09): extrapolated.")
        if sh["gomti"] > vm["gomti"] + 1e-6:
            warnings.append(f"Gomti {sh['gomti']:.1f} % of ferrous is beyond the {vm['gomti']:.1f} % its rule was "
                            "fitted on (FLOW-03, ASM-09): extrapolated.")
        if not L.RAFT_min <= raft <= L.RAFT_max:
            warnings.append(f"RAFT {raft:.0f} degC is outside the {L.RAFT_min:.0f}-{L.RAFT_max:.0f} degC band at "
                            f"{s.humidity_g:.0f} g/Nm3 blast moisture (DATA-03/04).")
        om_info = self.omega_info(o, s)
        if om_info["mode"] == "blend" and om_info["outside"]:
            warnings.append("Shaft efficiency extrapolated beyond what BF-02 ran: " + "; ".join(om_info["outside"])
                            + f" (certainty {om_info['certainty']}, GAS-18).")
        if om_info["mode"] == "blend" and om_info["certainty"] == "Investigate":
            why = "; ".join(om_info["investigate"]) or f"sigma(coke) {om_info['sigma_K']:.1f} kg/tHM above the Investigate limit"
            warnings.append(f"Investigate tier: {why}. The coke prediction rests on the RI, size and channel-scale "
                            "assumptions listed under critical assumptions; trial before adoption (GAS-18).")
        out = dict(P_th=P, P_tday=P * 24, settings=asdict(s), limits=asdict(L), fuel_in_objective=fuel, ore6_on=ore6_on,
                   z=zs, wet_kg={v.id: float(w) for v, w in zip(self.vars, zs)},
                   dry_kg={v.id: float(w * (1 - v.moist / 100)) for v, w in zip(self.vars, zs)},
                   share_wet={v.id: float(100 * w / Y) for v, w in zip(self.fer, zs[:self.nf])},
                   t_day={v.id: float(w * P * 24 / 1000) for v, w in zip(self.vars, zs)},
                   ferrous_wet=float(Y), ferrous_dry=float(sum(w * (1 - v.moist / 100) for v, w in zip(self.fer, zs[:self.nf]))),
                   fe_wet_pct=float(100 * (self.fe_wet[:self.nf] @ zs[:self.nf]) / Y),
                   out=o, cost=self.cost(zs, o, fuel, s), viol=viol, feasible=(sum(viol.values()) < 1e-3),
                   binding=sorted(k for k, x in duals.items() if k != "Fe balance"),
                   duals=duals, slack_used=slack_used, lp_iterations=last.get("iters"),
                   blast_nm3h=V, O2_nm3h=P * o["O2"], enrich_tag=self.hy.enrich_tag(s.O2_per_air), raft=raft,
                   dlnK=d, V_ceiling=self.hy.v_ceiling(s.Pb, s.Pt, d), Pb_needed=self.hy.pb_needed(V, s.Pt, d),
                   charging_th=P / 1000 * (zs.sum() + s.nut_plant), charges_per_h=P / 1000 * (zs.sum() + s.nut_plant) / L.charge_t,
                   charging_cap_th=L.charge_t * L.charges_per_h, warnings=warnings, omega=om_info,
                   facts_version=self.facts.version, facts=list(ENGINE_FACTS))
        out["dP_needed"] = out["Pb_needed"] - s.Pt
        return out

    def omega_info(self, o: Mapping[str, Any], s: Settings) -> dict[str, Any]:
        """Shaft efficiency of an evaluated plan (``evaluate(full=True)``): value, parts, sigma and certainty."""
        if s.omega_mode != "blend":
            return dict(mode="fixed", value=float(o["omega"]), omega_cal=float(self.cal["omega"]), sigma_K=0.0,
                        certainty="Held", support="", outside=[])
        m = self.omega_model()
        t = o["omega_terms"]
        wet = 1 - self.case0["coke"]["moist_pct"] / 100
        to_K = abs(self._dKdw) / wet
        sigK = float(o["sig_K"])
        inv = list(t.get("investigate") or [])
        return dict(mode="blend", value=float(o["omega"]), omega_cal=float(self.cal["omega"]), sigma_omega=t["sigma"],
                    sigma_K=sigK, sigma_parts_K={k: to_K * v for k, v in t["sigma_parts"].items()},
                    certainty=m.certainty(sigK, t["support"], inv), support=t["support"], outside=list(t["outside"]),
                    investigate=inv,
                    d_blend=t["d_blend"], d_blend_phys=t["d_blend_phys"], d_disc=t["d_disc"], d_oper=t["d_oper"],
                    d_prod=t.get("d_prod", 0.0), d_rest=t.get("d_rest", t["d_oper"]),
                    d_R=t["d_R"], d_moist=t.get("d_moist", 0.0), R=t["R"], k_blend=t["k_blend"],
                    gas_in_excess=t["gas_in_excess"],
                    wall_gas_share=t["wall_gas_share"], cal_se_K=to_K * m.cal_se, day_sd_K=to_K * m.day_sd,
                    dK_domega=self._dKdw, risk_k=s.risk_k, filter=s.certainty, version=m.version, terms=t)

    def critical_assumptions(self, r: Mapping[str, Any], s: Settings, ri_step: float = 5.0,
                             n_alt: float = 1.0) -> list[dict[str, Any]]:
        """Coke effect (kg/tHM wet) of each critical assumption of a STRATA-omega result taking another credible value
        (``omega_model.assumption_sensitivity``): RI of each material +/- ri_step, size exponent n -> n_alt, plant-data
        scales +/- 1 sd, moisture coefficient +/- 1 se. Linear in omega through dK/domega; sorted by size."""
        om = r.get("omega") or {}
        if om.get("mode") != "blend":
            return []
        m = self.omega_model()
        z, P = np.asarray(r["z"], float), float(r["P_th"])
        st = self.plan_state(z, s, P)
        o = r["out"]
        R = om["R"]
        rows = assumption_sensitivity(m, st, float(o["K"]), float(R), ri_step=ri_step, n_alt=n_alt)
        wet = 1 - self.case0["coke"]["moist_pct"] / 100
        to_K = self._dKdw / wet
        for row in rows:
            row["d_coke_kg"] = abs(to_K * row["d_omega"]) if row.get("symmetric") else to_K * row["d_omega"]
            if "d_omega_lo" in row:
                row["d_coke_lo_kg"], row["d_coke_hi_kg"] = to_K * row["d_omega_lo"], to_K * row["d_omega_hi"]
        return rows

    def saving_sigma(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float] | None:
        """1-sigma of the coke (kg/tHM wet) and cost (Rs/tHM) difference between two blend-mode results of this
        engine's calibration (e.g. the plan and today's charged blend)."""
        oa, ob = a.get("omega") or {}, b.get("omega") or {}
        if oa.get("mode") != "blend" or ob.get("mode") != "blend":
            return None
        m = self.omega_model()
        sw = difference_sigma(m, oa["terms"], ob["terms"])
        wet = 1 - self.case0["coke"]["moist_pct"] / 100
        sk = abs(self._dKdw) * sw / wet
        return dict(sigma_omega=sw, sigma_coke=sk, sigma_cost=sk * self.prices_fuel["coke"] / 1000)

    # ------------------------------------------------------------------------------------------------ modes
    def min_cost(self, prod_t_day=None, P=None, s: Settings | None = None, L: Limits | None = None, fuel=True, z0=None,
                 ore6_on=False, **kw):
        """Cheapest blend at a fixed production (t/d or t/h). ``fuel=False`` = ore + flux objective."""
        P = P if P is not None else prod_t_day / 24
        return self.slp(s or Settings(), P, L or Limits(), fuel=fuel, z0=z0, ore6_on=ore6_on, **kw)

    def process_settings(self, L: Limits, O2_per_air=None, base: Settings | None = None):
        """Controls at their production limits: HBT max (trimmed to RAFT max), Pb max, Pt = Pb max - dP max within
        [Pt_min, Pt_max]; O2 as given (default: the base settings')."""
        s = replace(base or Settings())
        s.O2_per_air = s.O2_per_air if O2_per_air is None else O2_per_air
        s.HBT = L.HBT_max
        while self.raft(s) > L.RAFT_max and s.HBT > 1000:
            s.HBT -= 1.0
        s.Pb = L.Pb_max
        s.Pt = min(max(L.Pb_max - L.dP_max, L.Pt_min), L.Pt_max)
        return s

    def max_prod_at(self, s: Settings, L: Limits, fuel=True, lo=80.0, hi=115.0, tol=0.01, ore6_on=False, z0=None):
        """Highest production (t/h) with a feasible blend at controls s. Bisection on production."""
        lo, hi = self.p_range(s, lo, hi)
        r = self._try(s, lo, L, fuel, z0=z0, ore6_on=ore6_on)
        if r is None or not r["feasible"]:
            return None
        best = r; z0 = r["z"]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            r = self._try(s, mid, L, fuel, z0=z0, ore6_on=ore6_on)
            if r is not None and r["feasible"]:
                lo, best, z0 = mid, r, r["z"]
            else:
                hi = mid
        return best

    def _ore6_options(self):
        return [False, True] if any(v.k_rule == OVERSIZE_RULE and v.hi > 0 for v in self.fer) else [False]

    def max_production(self, L: Limits, fuel=True, base: Settings | None = None, o2_grid=None, verbose=False, tie=0.02):
        """Blend and controls that give the most hot metal per hour within the limits. Oxygen enrichment is searched
        (more O2 lowers blast per tonne but raises burden resistance and is capped by flow and enrichment limits);
        among settings within ``tie`` t/h of the best, the one with the least oxygen is returned."""
        L = replace(L, process=True)
        r_max = self.hy.o2_per_air_from_tag(L.enrich_max)
        r_now = (base or Settings()).O2_per_air
        grid = list(o2_grid) if o2_grid is not None else list(np.linspace(r_now, r_max, 6))
        found = []
        for on6 in self._ore6_options():
            for r in grid:
                res = self.max_prod_at(self.process_settings(L, r, base), L, fuel, ore6_on=on6)
                if res is not None:
                    found.append(res)
                    if verbose:
                        print(f"   ore6 {on6} enrich {self.hy.enrich_tag(r):.2f} -> {res['P_th']:.2f} t/h")
        if not found:
            return None
        top = max(found, key=lambda x: x["P_th"])
        if len(grid) > 2:                                   # refine oxygen around the best grid point
            r_best = top["settings"]["O2_per_air"]; step = grid[1] - grid[0]
            for r in np.linspace(max(r_now, r_best - step), min(r_max, r_best + step), 7):
                res = self.max_prod_at(self.process_settings(L, r, base), L, fuel, ore6_on=top["ore6_on"], z0=top["z"])
                if res is not None:
                    found.append(res)
        P_best = max(x["P_th"] for x in found)
        best = min([x for x in found if x["P_th"] >= P_best - tie], key=lambda x: (x["settings"]["O2_per_air"], -x["P_th"]))
        best["o2_trace"] = sorted((bool(x["ore6_on"]), x["settings"]["O2_per_air"], self.hy.enrich_tag(x["settings"]["O2_per_air"]),
                                   x["P_th"]) for x in found)
        return best

    def frontier(self, s: Settings, L: Limits, P_list, fuel=True, ore6_on=False):
        rows = []; z0 = None
        for P in P_list:
            r = self._try(s, P, L, fuel, z0=z0, ore6_on=ore6_on)
            if r is None:
                r = dict(P_th=P, P_tday=P * 24, feasible=False, cost=dict(total=np.inf), lp_failed=True)
            rows.append(r)
            if r["feasible"]:
                z0 = r["z"]
        return rows

    def min_total_cost(self, L: Limits, base: Settings | None = None, P_lo=85.0, step=0.5, o2_levels=None, tol_P=0.02,
                       tol_r=1e-4, curves=True):
        """Production level, blend and oxygen with the lowest ore + flux + fuel cost per tHM within the limits.
        Controls: HBT and pressures at their limits (they cost nothing in Rs/tHM). Oxygen: the base settings' unless
        o2_levels (O2/air ratios, a grid from the base up) is given; then oxygen is searched too. Oxygen has no price,
        and at a fixed production more oxygen adds a little coke, so it is used only to reach a higher production.
        Cost against production is single-dipped (falls while the cheap blend fits, rises once a limit forces dearer
        ore) -> golden-section on production; over oxygen, the best grid level is refined between its neighbours.
        Also returns 'curves': the frontier on a production grid (one per oversize option)."""
        L = replace(L, process=True)
        r_now = (base or Settings()).O2_per_air
        P_lo = self.p_range(base or Settings(), P_lo, 1e9)[0]
        levels = [r_now] if o2_levels is None else sorted(set([r_now] + [r for r in o2_levels if r >= r_now - 1e-12]))
        cost = lambda x: x["cost"]["total"] if (x is not None and x["feasible"]) else np.inf
        gr = (5 ** 0.5 - 1) / 2
        best = None; out_curves = {}
        for on6 in self._ore6_options():
            memo, tops = {}, {}

            def at_r(r):
                key = round(float(r), 7)
                if key in memo:
                    return memo[key]
                s_r = self.process_settings(L, r, base)
                top = self.max_prod_at(s_r, L, True, lo=P_lo, ore6_on=on6)
                if top is None:
                    memo[key] = None
                    return None
                tops[key] = top["P_th"]
                seen = [top]
                f = lambda P: seen.append(self._try(s_r, P, L, True, z0=top["z"], ore6_on=on6)) or seen[-1]
                a, b = P_lo, top["P_th"]
                c, d = b - gr * (b - a), a + gr * (b - a)
                fc, fd = f(c), f(d)
                while b - a > tol_P:
                    if cost(fc) <= cost(fd):
                        b, d, fd = d, c, fc; c = b - gr * (b - a); fc = f(c)
                    else:
                        a, c, fc = c, d, fd; d = a + gr * (b - a); fd = f(d)
                res = min([x for x in seen if x is not None], key=cost)
                memo[key] = res if np.isfinite(cost(res)) else None
                return memo[key]

            grid = [(r, at_r(r)) for r in levels]
            ok = [(r, x) for r, x in grid if x is not None]
            if not ok:
                continue
            i = min(range(len(ok)), key=lambda k: cost(ok[k][1]))
            cand = ok[i][1]
            if len(ok) > 1:
                a = ok[max(i - 1, 0)][0]; b = ok[min(i + 1, len(ok) - 1)][0]
                c, d = b - gr * (b - a), a + gr * (b - a)
                fc, fd = at_r(c), at_r(d)
                while b - a > tol_r:
                    if cost(fc) <= cost(fd):
                        b, d, fd = d, c, fc; c = b - gr * (b - a); fc = at_r(c)
                    else:
                        a, c, fc = c, d, fd; d = a + gr * (b - a); fd = at_r(d)
                for x in memo.values():
                    if cost(x) < cost(cand):
                        cand = x
            if best is None or cost(cand) < cost(best):
                best = cand
            if curves:
                k_top = max(tops, key=tops.get); P_top = tops[k_top]
                r_curve = sorted(set([r for r, _ in ok] + [cand["settings"]["O2_per_air"], k_top]))
                Ps = list(np.arange(P_lo, P_top, step)) + [P_top]
                per_level = [self.frontier(self.process_settings(L, r, base), L, Ps, True, on6) for r in r_curve]
                rows = []
                for k, P in enumerate(Ps):
                    feas = [lvl[k] for lvl in per_level if lvl[k]["feasible"]]
                    if feas:
                        rows.append(min(feas, key=lambda x: x["cost"]["total"]))
                out_curves[on6] = rows
        if best is not None:
            best["curves"] = out_curves
        return best


def value_in_use(eng: Engine, z, s: Settings, P, B2=1.10, balancer="limestone"):
    """Cost of one tonne of Fe from each ferrous material: purchase + flux to bring its own gangue to basicity B2 +
    coke for its slag and the flux CO2 (two-zone response at blend z). Surplus CaO (sinter) is credited at the
    balancer's cost and coke. Differences between materials are what the optimiser trades."""
    _, o0 = eng.phi(z, s, P)
    dK, dS = {}, {}
    for j, v in enumerate(eng.vars):
        zz = np.array(z, float); zz[j] += 10.0
        o1 = eng.evaluate(zz, s, P)
        dK[v.id] = 100 * (o1["K_wet"] - o0["K_wet"]); dS[v.id] = 100 * (o1["slag"] - o0["slag"])     # per wet tonne
    b = next(v for v in eng.flux if v.id == balancer); bd = 1 - b.moist / 100
    bCaO, bSiO2 = b.chem["CaO"] * bd * 10, b.chem["SiO2"] * bd * 10                                      # kg per wet t
    cp = eng.prices_fuel["coke"] / 1000
    rows = []
    for v in eng.fer:
        d = 1 - v.moist / 100
        fe = v.chem["Fe"] * d * 10; sio2 = v.chem["SiO2"] * d * 10; cao = v.chem["CaO"] * d * 10
        Lb = (B2 * sio2 - cao) / (bCaO - B2 * bSiO2)                    # t balancer per t material (negative = credit)
        coke = dK[v.id] + Lb * dK[balancer]; slag = dS[v.id] + Lb * dS[balancer]
        buy = v.price; flux = Lb * b.price; fuel = coke * cp
        rows.append(dict(id=v.id, name=v.name, price=v.price, fe_wet_pct=fe / 10, buy_per_tFe=1000 * buy / fe,
                         flux_kg_per_t=1000 * Lb, flux_per_tFe=1000 * flux / fe, coke_kg_per_tFe=1000 * coke / fe,
                         fuel_per_tFe=1000 * fuel / fe, slag_kg_per_tFe=1000 * slag / fe,
                         viu_per_tFe=1000 * (buy + flux + fuel) / fe))
    return rows, dK, dS


def summary(r, eng: Engine) -> dict[str, Any]:
    """Flat dict for tables."""
    o = r["out"]; c = r["cost"]; s = r["settings"]
    d = dict(prod_t_day=r["P_tday"], prod_th=r["P_th"])
    for v in eng.fer:
        d["sh_" + v.id] = r["share_wet"][v.id]
    for v in eng.flux:
        d["kg_" + v.id] = r["wet_kg"][v.id]
    d.update(ferrous_wet=r["ferrous_wet"], fe_pct=r["fe_wet_pct"], coke_dry=o["K"], coke_wet=o["K_wet"],
             fuel_rate=o["fuel_rate"], slag=o["slag"], B2=o["B2"], Al2O3_pct=o["Al2O3_pct"], MgO_pct=o["MgO_pct"],
             MgO_Al2O3=o["MgO"] / o["Al2O3"], eta_CO=o["eta"], T_top=o["T_top"], HBT=s["HBT"], enrich=r["enrich_tag"],
             raft=r["raft"], blast=r["blast_nm3h"], O2=r["O2_nm3h"], Pb=s["Pb"], Pt=s["Pt"], Pb_needed=r["Pb_needed"],
             dP_needed=r["dP_needed"], V_ceiling=r["V_ceiling"], charges_h=r["charges_per_h"], ore_cost=c["ore"],
             flux_cost=c["flux"], coke_cost=c["coke"], nut_cost=c["nut"], pci_cost=c["pci"], fuel_cost=c["fuel"],
             total=c["total"], total_day=c["total"] * r["P_tday"], feasible=r["feasible"],
             binding=", ".join(r.get("binding", [])))
    om = r.get("omega") or {}
    d.update(omega=om.get("value", o.get("omega")), omega_mode=om.get("mode", "fixed"), sigma_coke=om.get("sigma_K", 0.0),
             sigma_cost=c.get("sigma", 0.0), certainty=om.get("certainty", "Held"))
    return d
