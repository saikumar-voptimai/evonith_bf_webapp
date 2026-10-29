"""What-if analysis of the admin space (SV-4.3, #306): FORGE and STRATA what-if cases, saved per administrator.

FORGE what-if (job ``whatif.forge``)
    The calibration point of a what-if day (the calibration ``calibration.nightly`` stored for that IST day,
    ``services.day_calibration``) or of the shipped Jul-Oct 2026 calibration, changed by the levers of
    ``coke_energy.whatif.run_scenario`` (coke chemistry, PCI, nut coke, ηCO, HM Si, HMT, HBT, O₂, blast moisture,
    production) and the shaft efficiency. ``run_scenario`` applies the levers; each case is then solved by
    * the two-zone (Rist) balance with ω, Q_LZ and Q_UZ (GJ/h) and r_H held at the calibration (FORGE's model, ηCO
      and the top-gas temperature are outputs; ``rist.solve_2z``), slag CaO/SiO₂ held at the calibration's by
      limestone or quartzite charged to the furnace (``bf_balance.flux_needed``): the basis of the what-if table of
      "BF-02 coke model: what the latest results mean" (15 Oct 2026, page 10) and of the register's COKE-01..08;
    * the oxygen-balance model with ηCO held (``run_scenario`` itself: heat loss calibrated on the calibration point,
      ``bf_balance.calibrate_loss``, constant in GJ/h; limestone holds the basicity), the table's "fixed ηCO" column.
STRATA plan what-if (job ``whatif.strata``)
    Burden & Fuel Optimiser plans on one basis: the calibration of the what-if day (blast moisture: that day's
    Open-Meteo value) or the shipped one (its own moisture), the live or reference yard, a production and STRATA-ω or
    Fixed-ω (``services.optimiser.default_case``). A case forces a ferrous blend (``scenarios.run_today``: the shares
    pinned, flux to the basicity band, slag-quality and charging limits lifted and reported) or asks for the cheapest
    blend (``Engine.min_cost`` with fuel, Step 2), with its own controls.
Cases
    ``twin.whatif_case``: an administrator's saved cases (FORGE levers, or a blend and controls) with the result of
    their latest run. Saves and deletes go through ``services.audit.admin_write`` (``whatif.save``, ``whatif.delete``).
    The page compares up to four cases side by side; every result states the calibration, the what-if day and the
    fact-register version (``FactRegister.label``) it used.

Coke rates are kg/tHM dry unless named wet (as charged); costs Rs/tHM; production t/h (FORGE) or t/d (STRATA). No
Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from data.twin.models import WhatIfCase
from data.twin.repositories import Actor, NotFound, TwinStore
from services import audit
from services.jobs import (
    WHATIF_MAX_CASES,
    ForgeCaseParams,
    ForgeLevers,
    OptimiserSettings,
    StrataCaseParams,
    WhatIfForgeParams,
    WhatIfStrataParams,
)
from services.twin_views import utc

KINDS = ("forge", "strata")
JOB_KINDS = {"forge": "whatif.forge", "strata": "whatif.strata"}
MAX_CASES_PER_KIND = 50
#: Fields of ``twin.whatif_case`` the audit rows record.
FIELDS = ("id", "owner", "kind", "name", "inputs", "created_at", "updated_at")

#: ForgeLevers field -> key of ``coke_energy.whatif.run_scenario``'s scenario (the coke chemistry goes into ``coke``).
COKE_LEVERS = {"coke_fc_pct": "FC", "coke_ash_pct": "ash", "coke_vm_pct": "VM", "coke_moisture_pct": "moist_pct"}
SCENARIO_LEVERS = {
    "pci_kg_thm": "pci_kg_plant",
    "nut_coke_kg_thm": "nut_kg_plant",
    "eta_co_pct": "eta_CO",
    "hm_si_pct": "hm_Si",
    "hmt_c": "HMT",
    "hbt_c": "HBT",
    "o2_per_air": "O2_per_air",
    "blast_moisture_g_nm3": "humidity_g",
    "production_t_h": "prod_th",
}
TWO_ZONE_NOTE = (
    "Two-zone (Rist) balance, FORGE's model: shaft efficiency ω, the lower- and upper-zone heat losses (GJ/h) and r_H "
    "held at the calibration; ηCO and the top-gas temperature are outputs; slag CaO/SiO₂ held at the calibration's by "
    "limestone or quartzite in the furnace."
)
FIXED_ETA_NOTE = (
    "Oxygen-balance model (coke_energy.whatif.run_scenario): ηCO and the top-gas temperature held, the heat loss "
    "calibrated on the calibration point and held in GJ/h; limestone holds the slag basicity."
)


class WhatIfInputError(ValueError):
    """A what-if request that cannot be solved as given (no calibration for the day, unknown materials)."""


# ------------------------------------------------------------------------------------------------------- views


class WhatIfBasis(BaseModel):
    calibration: Literal["day", "reference"] = Field(
        ..., description="day = calibration.nightly's for the what-if day; reference = the shipped Jul-Oct 2026 one"
    )
    day: date | None = Field(None, description="What-if day (IST) of a day calibration")
    window: str = Field(..., description="Shifts the calibration was made on (IST dates)")
    n_shifts: int | None = Field(None, description="Stable shifts of the calibration, count")
    omega: float | None = Field(None, description="Shaft efficiency of the calibration, dimensionless")
    q_lz_gj_h: float | None = Field(None, description="Lower-zone heat loss of the calibration, GJ/h")
    q_uz_gj_h: float | None = Field(None, description="Upper-zone heat loss of the calibration, GJ/h")
    r_h: float | None = Field(
        None, description="Hydrogen share of the reduction work of the calibration, dimensionless"
    )
    coke_kg_thm: float | None = Field(None, description="Coke of the calibration point, kg/tHM dry")
    blast_moisture_g_nm3: float | None = Field(None, description="Blast moisture of the calibration point, g/Nm³")


class ForgeInputs(BaseModel):
    """The values of every FORGE lever in a case (the calibration point's where the case leaves a lever unset)."""

    coke_fc_pct: float = Field(..., description="Coke fixed carbon, dry mass %")
    coke_ash_pct: float = Field(..., description="Coke ash, dry mass %")
    coke_vm_pct: float = Field(..., description="Coke volatile matter, dry mass %")
    coke_moisture_pct: float = Field(..., description="Coke moisture as charged, mass %")
    pci_kg_thm: float = Field(..., description="PCI as injected, kg/tHM")
    nut_coke_kg_thm: float = Field(..., description="Nut coke as charged, kg/tHM")
    eta_co_pct: float = Field(..., description="ηCO the oxygen-balance model holds, %")
    hm_si_pct: float = Field(..., description="Hot-metal Si, mass %")
    hmt_c: float = Field(..., description="Hot-metal temperature, °C")
    hbt_c: float = Field(..., description="Hot-blast temperature, °C")
    o2_per_air: float = Field(..., description="Oxygen enrichment on the model basis, Nm³ O₂ per Nm³ air")
    blast_moisture_g_nm3: float = Field(..., description="Blast moisture, g/Nm³")
    production_t_h: float = Field(..., description="Hot-metal production, t/h")
    shaft_efficiency: float = Field(..., description="Shaft efficiency the two-zone balance holds, dimensionless")


class ForgeModelResult(BaseModel):
    """One model's answer for a FORGE case."""

    coke_kg_thm: float = Field(..., description="Coke rate, kg/tHM dry")
    coke_wet_kg_thm: float = Field(..., description="Coke rate as charged, kg/tHM")
    d_coke_kg_thm: float = Field(..., description="Coke rate minus the calibration point's (same model), kg/tHM dry")
    fuel_rate_kg_thm: float = Field(..., description="Coke as charged + nut coke + PCI, kg/tHM")
    eta_co_pct: float = Field(..., description="Top-gas CO utilisation, % (an input of the oxygen-balance model)")
    top_gas_c: float = Field(..., description="Top-gas temperature, °C (an input of the oxygen-balance model)")
    blast_air_nm3_thm: float = Field(..., description="Blast air, Nm³/tHM")
    o2_nm3_thm: float = Field(..., description="Enrichment oxygen, Nm³/tHM")
    top_gas_nm3_thm: float = Field(..., description="Top gas, Nm³/tHM")
    slag_kg_thm: float = Field(..., description="Slag, kg/tHM")
    slag_cao_sio2: float = Field(..., description="Slag CaO/SiO₂, dimensionless")
    slag_al2o3_pct: float = Field(..., description="Slag Al₂O₃, mass %")
    slag_mgo_pct: float = Field(..., description="Slag MgO, mass %")
    limestone_kg_thm: float = Field(..., description="Limestone added to hold the basicity, kg/tHM")
    quartzite_kg_thm: float | None = Field(None, description="Quartzite added to hold the basicity, kg/tHM (two-zone)")
    shaft_efficiency: float | None = Field(None, description="Shaft efficiency, dimensionless (two-zone)")
    direct_reduction_pct: float | None = Field(None, description="Degree of direct reduction, % (two-zone)")


class ForgeBase(BaseModel):
    inputs: ForgeInputs = Field(..., description="The calibration point's lever values")
    two_zone: ForgeModelResult = Field(..., description="Two-zone balance at the calibration point")
    fixed_eta: ForgeModelResult = Field(..., description="Oxygen-balance model at the calibration point")


class ForgeCaseResult(BaseModel):
    case_id: str | None = Field(None, description="Saved case (UUID)")
    name: str
    status: Literal["ok", "failed"]
    error: str | None = Field(None, description="Why the case failed")
    levers: ForgeLevers = Field(..., description="The levers the case set")
    inputs: ForgeInputs = Field(..., description="Every lever's value in the case")
    two_zone: ForgeModelResult | None = Field(None, description="Two-zone balance (ω held), FORGE's what-if model")
    fixed_eta: ForgeModelResult | None = Field(None, description="Oxygen-balance model (ηCO held)")
    warnings: list[str] = Field(default_factory=list, description="Plausibility bands breached (register BAND-01..03)")
    notes: list[str] = Field(default_factory=list, description="Levers one of the two models does not take")


class StrataSettingsUsed(BaseModel):
    hmt_c: float = Field(..., description="Hot-metal temperature target, °C")
    hm_si_pct: float = Field(..., description="Hot-metal Si target, mass %")
    pci_kg_thm: float = Field(..., description="PCI as injected, kg/tHM")
    nut_coke_kg_thm: float = Field(..., description="Nut coke as charged, kg/tHM")
    blast_moisture_g_nm3: float = Field(..., description="Blast moisture, g/Nm³")
    hbt_c: float = Field(..., description="Hot-blast temperature, °C")
    o2_per_air: float = Field(..., description="Oxygen enrichment on the model basis, Nm³ O₂ per Nm³ air")


class WhatIfMaterialRow(BaseModel):
    id: str = Field(..., description="Material id")
    name: str
    share_pct: float | None = Field(None, description="Share of the wet ferrous burden, % (ferrous only)")
    kg_thm: float = Field(..., description="Wet mass, kg/tHM")


class StrataCaseResult(BaseModel):
    case_id: str | None = Field(None, description="Saved case (UUID)")
    name: str
    status: Literal["ok", "infeasible", "failed"] = Field(
        ..., description="infeasible: the LP has no solution, a band refused the inputs, or a limit stays breached"
    )
    error: str | None = Field(None, description="Why the case is infeasible or failed")
    forced: bool = Field(..., description="The ferrous blend was forced (else the cheapest blend)")
    requested_blend: dict[str, float] | None = Field(None, description="The forced shares as given, % of wet ferrous")
    production_t_d: float | None = Field(None, description="Hot-metal production, t/d")
    cost_rs_thm: float | None = Field(None, description="Ore + flux + fuel − slag credit, Rs/tHM")
    cost_gross_rs_thm: float | None = Field(None, description="Ore + flux + fuel, Rs/tHM")
    ore_rs_thm: float | None = Field(None, description="Ferrous burden, Rs/tHM")
    flux_rs_thm: float | None = Field(None, description="Flux, Rs/tHM")
    fuel_rs_thm: float | None = Field(None, description="Coke + nut coke + PCI, Rs/tHM")
    slag_credit_rs_thm: float | None = Field(None, description="Slag sold, Rs/tHM")
    coke_kg_thm: float | None = Field(None, description="Coke rate, kg/tHM dry")
    coke_wet_kg_thm: float | None = Field(None, description="Coke rate as charged, kg/tHM")
    coke_sd_kg_thm: float | None = Field(None, description="1 sigma of the coke rate (STRATA-ω), kg/tHM dry")
    fuel_rate_kg_thm: float | None = Field(None, description="Coke as charged + nut coke + PCI, kg/tHM")
    eta_co_pct: float | None = Field(None, description="Top-gas CO utilisation, %")
    top_gas_c: float | None = Field(None, description="Top-gas temperature, °C")
    shaft_efficiency: float | None = Field(None, description="Shaft efficiency, dimensionless")
    certainty: str | None = Field(None, description="High, Medium, Low or Investigate (STRATA-ω); Held (Fixed-ω)")
    slag_kg_thm: float | None = Field(None, description="Slag, kg/tHM")
    slag_cao_sio2: float | None = Field(None, description="Slag CaO/SiO₂, dimensionless")
    slag_al2o3_pct: float | None = Field(None, description="Slag Al₂O₃, mass %")
    slag_mgo_pct: float | None = Field(None, description="Slag MgO, mass %")
    raft_c: float | None = Field(None, description="RAFT (plant workbook), °C")
    blast_nm3_h: float | None = Field(None, description="Hot blast (meter basis), Nm³/h")
    o2_nm3_h: float | None = Field(None, description="Oxygen flow, Nm³/h")
    charges_per_h: float | None = Field(None, description="Charging rate, charges/h")
    blend: list[WhatIfMaterialRow] = Field(default_factory=list, description="Ferrous materials of the plan")
    flux: list[WhatIfMaterialRow] = Field(default_factory=list, description="Flux of the plan")
    settings: StrataSettingsUsed | None = Field(None, description="Controls and targets of the plan")
    warnings: list[str] = Field(default_factory=list, description="The engine's warnings")
    violations: list[str] = Field(default_factory=list, description="Limits the plan breaches")
    facts_version: str | None = Field(None, description="Fact-register version the engine used")


class StrataBasis(WhatIfBasis):
    omega_mode: Literal["strata", "fixed"] = Field(..., description="STRATA-ω (blend-dependent) or Fixed-ω")
    yard: str = Field(..., description="Yard the plans used: live or reference")
    yard_message: str = Field("", description="Why the yard fell back to the reference, if it did")
    production_source: str = Field("", description="Where the production came from")


class ForgeCaseRun(BaseModel):
    """A FORGE case's latest result, as stored with the case."""

    computed_at: datetime = Field(..., description="UTC")
    job_id: str | None = None
    facts_version: str = Field(..., description="Fact-register version used (e.g. 1.8, 1.8 + overlay 2)")
    basis: WhatIfBasis
    base: ForgeBase = Field(..., description="The calibration point the case changes")
    result: ForgeCaseResult


class StrataCaseRun(BaseModel):
    """A STRATA case's latest result, as stored with the case."""

    computed_at: datetime = Field(..., description="UTC")
    job_id: str | None = None
    facts_version: str = Field(..., description="Fact-register version used (e.g. 1.8, 1.8 + overlay 2)")
    basis: StrataBasis
    result: StrataCaseResult


class StrataCaseInputs(BaseModel):
    model_config = {"extra": "forbid"}

    blend: dict[str, float] | None = Field(
        None, description="Forced ferrous blend: material id → share of the wet ferrous burden, %; null = cheapest"
    )
    settings: OptimiserSettings | None = Field(None, description="Controls and targets instead of the basis")


class WhatIfCaseView(BaseModel):
    id: str = Field(..., description="Case id (UUID)")
    kind: Literal["forge", "strata"]
    name: str
    owner: str
    created_at: datetime = Field(..., description="UTC")
    updated_at: datetime = Field(..., description="UTC")
    forge: ForgeLevers | None = Field(None, description="FORGE levers (kind forge)")
    strata: StrataCaseInputs | None = Field(None, description="Forced blend and controls (kind strata)")
    forge_run: ForgeCaseRun | None = Field(None, description="Latest FORGE result; null before the first run")
    strata_run: StrataCaseRun | None = Field(None, description="Latest STRATA result; null before the first run")


# ------------------------------------------------------------------------------------------------- numbers


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _finite(doc: Any) -> Any:
    """A JSON document with non-finite numbers as None and numpy values as numbers."""
    if isinstance(doc, Mapping):
        return {str(k): _finite(v) for k, v in doc.items()}
    if isinstance(doc, (list, tuple)):
        return [_finite(v) for v in doc]
    if hasattr(doc, "item") and not isinstance(doc, (str, bytes)):
        doc = doc.item()
    if isinstance(doc, float):
        return doc if math.isfinite(doc) else None
    return doc


# ------------------------------------------------------------------------------------------------ calibration


def stored_days(directory: Any = None) -> list[date]:
    """IST days with a stored day calibration, newest first (``calibration.nightly`` keeps the last 30)."""
    from services import day_calibration

    out = []
    for path in day_calibration.folder(directory).glob("????-??-??.json"):
        try:
            out.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(out, reverse=True)


def resolve_calibration(
    basis: str, day: date | None = None, *, directory: Any = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(calibration, basis fields of ``WhatIfBasis``). ``day`` (default: the latest stored day) needs a stored day
    calibration (``WhatIfInputError`` otherwise; the what-if never falls back to another calibration)."""
    from services import day_calibration, optimiser

    if basis == "reference":
        cfg = optimiser.page_config(optimiser.config_mtime("burden_optimiser.yml"))
        info, _ = optimiser.case_calibration(cfg, 0, "reference")
        cal = info.cal
        window = " to ".join(str(x) for x in cal.get("period", ["", ""]))
        return cal, _basis_fields(cal, "reference", None, window, cal.get("n_shifts"))
    if basis != "day":
        raise WhatIfInputError(f"calibration must be day or reference, not {basis!r}")
    if day is None:
        days = stored_days(directory)
        if not days:
            raise WhatIfInputError(
                "No day calibration is stored yet (calibration.nightly at 02:30 IST); use the reference calibration."
            )
        day = days[0]
    info = day_calibration.load(day, directory)
    if info is None:
        raise WhatIfInputError(
            f"No calibration is stored for {day.isoformat()} (calibration.nightly keeps the last "
            f"{day_calibration.KEEP} days)."
        )
    window = " to ".join(str(x) for x in info.window if x) or "unknown window"
    return info.cal, _basis_fields(info.cal, "day", day, window, info.n_shifts)


def _basis_fields(cal: Mapping[str, Any], kind: str, day: date | None, window: str, n: Any) -> dict[str, Any]:
    case = cal.get("case") or {}
    return dict(
        calibration=kind,
        day=day,
        window=window,
        n_shifts=int(n) if _num(n) is not None else None,
        omega=_num(cal.get("omega")),
        q_lz_gj_h=_num(cal.get("Q_LZ_GJh")),
        q_uz_gj_h=_num(cal.get("Q_UZ_GJh")),
        r_h=_num(cal.get("r_H")),
        coke_kg_thm=_num(cal.get("K")),
        blast_moisture_g_nm3=_num(case.get("humidity_g")),
    )


# --------------------------------------------------------------------------------------------------- FORGE


def forge_scenario(levers: Mapping[str, Any]) -> dict[str, Any]:
    """The scenario of ``coke_energy.whatif.run_scenario`` for a case's levers (unset levers left out)."""
    lv = {k: float(v) for k, v in levers.items() if v is not None}
    scn: dict[str, Any] = {}
    coke = {key: lv[name] for name, key in COKE_LEVERS.items() if name in lv}
    if coke:
        scn["coke"] = coke
    scn.update({key: lv[name] for name, key in SCENARIO_LEVERS.items() if name in lv})
    return scn


def _forge_point(cal: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
    """(calibration case with its burden, model parameters, slag CaO/SiO₂ the balances hold)."""
    from utils.burden_engine.engine import Engine
    from utils.coke_energy.bf_balance import DEFAULTS, slag_and_flux

    case = Engine(cal=dict(cal))._cal_case()
    p = {**DEFAULTS, **dict(cal.get("p") or {})}
    b2 = _num(cal.get("B2"))
    if b2 is None:
        b2 = float(slag_and_flux(float(cal["K"]), case, p)["B2"])
    return case, p, b2


def _two_zone(case: Mapping[str, Any], cal: Mapping[str, Any], p: Mapping[str, Any], prod_th: float, omega: float,
              b2: float) -> dict[str, Any]:  # fmt: skip
    from utils.coke_energy.bf_balance import flux_needed
    from utils.coke_energy.rist import solve_2z

    return solve_2z(
        case,
        p,
        omega,
        float(cal["Q_LZ_GJh"]) * 1000.0 / prod_th,
        float(cal["Q_UZ_GJh"]) * 1000.0 / prod_th,
        float(cal["r_H"]),
        T_R=float(cal.get("T_R", 950.0)),
        flux_fn=lambda K, c: flux_needed(K, c, p, b2),
        x0=(float(cal["K"]), float(cal["V"]), float(cal["y_d"])),
    )


def _two_zone_out(r: Mapping[str, Any], case: Mapping[str, Any], inputs: Mapping[str, float], k0: float) -> dict:
    ro, sl = r["ro"], r["ro"]["slag"]
    k = float(r["K"])
    wet = k / (1 - float(case["coke"]["moist_pct"]) / 100)
    return dict(
        coke_kg_thm=k,
        coke_wet_kg_thm=wet,
        d_coke_kg_thm=k - k0,
        fuel_rate_kg_thm=wet + inputs["nut_coke_kg_thm"] + inputs["pci_kg_thm"],
        eta_co_pct=100 * float(r["eta"]),
        top_gas_c=float(r["T_top"]),
        blast_air_nm3_thm=float(r["V"]),
        o2_nm3_thm=float(ro["V_O2"]),
        top_gas_nm3_thm=float(ro["V_tg"]),
        slag_kg_thm=float(ro["slag_kg"]),
        slag_cao_sio2=float(sl["B2"]),
        slag_al2o3_pct=float(sl["Al2O3_pct"]),
        slag_mgo_pct=float(sl["MgO_pct"]),
        limestone_kg_thm=float(r["lime"]) + 0.0,
        quartzite_kg_thm=float(r["qtz"]) + 0.0,
        shaft_efficiency=float(r["omega"]),
        direct_reduction_pct=100 * float(r["r_d"]),
    )


def _fixed_eta_out(r: Mapping[str, Any], inputs: Mapping[str, float], k0: float) -> dict:
    from utils.coke_energy import whatif

    s = whatif.summarize(r)
    wet = float(s["coke_asCharged_kgthm"])
    return dict(
        coke_kg_thm=float(s["coke_dry_kgthm"]),
        coke_wet_kg_thm=wet,
        d_coke_kg_thm=float(s["coke_dry_kgthm"]) - k0,
        fuel_rate_kg_thm=wet + inputs["nut_coke_kg_thm"] + inputs["pci_kg_thm"],
        eta_co_pct=float(s["eta_CO_pct"]),
        top_gas_c=float(r["case"]["T_top"]),
        blast_air_nm3_thm=float(s["blast_air_nm3thm"]),
        o2_nm3_thm=float(s["O2_nm3thm"]),
        top_gas_nm3_thm=float(s["topgas_nm3thm"]),
        slag_kg_thm=float(s["slag_kgthm"]),
        slag_cao_sio2=float(s["slag_B2"]),
        slag_al2o3_pct=float(s["slag_Al2O3_pct"]),
        slag_mgo_pct=float(s["slag_MgO_pct"]),
        limestone_kg_thm=float(s["limestone_added_kgthm"]),
    )


def _converged(r: Mapping[str, Any], two_zone: bool) -> bool:
    if _num(r.get("K")) is None:
        return False
    if two_zone:
        return abs(float(r.get("O_err", 0.0))) < 1e-3 and abs(float(r.get("E_err", 0.0))) < 1.0
    return abs(float(r.get("O_err", 0.0))) < 1e-3


def forge_inputs(case: Mapping[str, Any], omega: float, eta_pct: float) -> dict[str, float]:
    """Every lever's value at a calibration case."""
    coke = case["coke"]
    return dict(
        coke_fc_pct=float(coke["FC"]),
        coke_ash_pct=float(coke["ash"]),
        coke_vm_pct=float(coke["VM"]),
        coke_moisture_pct=float(coke["moist_pct"]),
        pci_kg_thm=float(case["pci_kg"]) / (1 - float(case["pci"]["moist_pct"]) / 100),
        nut_coke_kg_thm=float(case["nut_kg"]) / (1 - float(case["nut"]["moist_pct"]) / 100),
        eta_co_pct=float(eta_pct),
        hm_si_pct=float(case["hm"]["Si"]),
        hmt_c=float(case["HMT"]),
        hbt_c=float(case["HBT"]),
        o2_per_air=float(case["O2_per_air"]),
        blast_moisture_g_nm3=float(case["humidity_g"]),
        production_t_h=float(case["prod_th"]),
        shaft_efficiency=float(omega),
    )


def _band_warnings(reg: Any, inputs: Mapping[str, float], fuel_rate: float | None) -> list[str]:
    out = []
    lo, hi = reg.band("BAND-02")
    if not lo <= inputs["hmt_c"] <= hi:
        out.append(f"HMT {inputs['hmt_c']:,.0f} °C is outside BAND-02 ({lo:,.0f}–{hi:,.0f} °C).")
    lo, hi = reg.band("BAND-03")
    if not lo <= inputs["hm_si_pct"] <= hi:
        out.append(f"HM Si {inputs['hm_si_pct']:.2f} % is outside BAND-03 ({lo:.2f}–{hi:.2f} %).")
    lo, hi = reg.band("BAND-01")
    if fuel_rate is not None and not lo <= fuel_rate <= hi:
        out.append(f"Fuel rate {fuel_rate:,.1f} kg/tHM (two-zone) is outside BAND-01 ({lo:,.0f}–{hi:,.0f} kg/tHM).")
    total = inputs["coke_fc_pct"] + inputs["coke_ash_pct"] + inputs["coke_vm_pct"]
    if abs(total - 100.0) > 0.5:
        out.append(f"Coke FC + ash + VM = {total:.1f} % on dry basis: the proximate analysis does not add up to 100 %.")
    return out


def forge_compute(cal: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], reg: Any = None) -> dict[str, Any]:
    """FORGE what-if of ``cases`` (``ForgeCaseParams`` documents) at the calibration point of ``cal``: the base and
    every case by the two-zone balance and the oxygen-balance model (JSON; ``ForgeBase``, ``ForgeCaseResult``)."""
    from utils.burden_engine.facts import load_register
    from utils.coke_energy import whatif
    from utils.coke_energy.bf_balance import calibrate_loss

    reg = reg or load_register()
    case0, p, b2 = _forge_point(cal)
    omega0 = float(cal["omega"])
    k0 = float(cal["K"])
    base_tz = _two_zone(case0, cal, p, float(case0["prod_th"]), omega0, b2)
    eta0 = float(base_tz["eta"])
    q_mj, _, _ = calibrate_loss(case0, k0, p, eta_CO=eta0)
    q_gj_h = float(q_mj) * float(case0["prod_th"]) / 1000.0
    base_fx = whatif.run_scenario(case0, k0, eta0, q_gj_h, {}, p=cal.get("p"))
    base_inputs = forge_inputs(case0, omega0, 100 * eta0)
    k_tz, k_fx = float(base_tz["K"]), float(base_fx["K"])
    base = dict(
        inputs=base_inputs,
        two_zone=_two_zone_out(base_tz, case0, base_inputs, k_tz),
        fixed_eta=_fixed_eta_out(base_fx, base_inputs, k_fx),
    )
    results = []
    for spec in cases:
        levers = ForgeLevers.model_validate(spec.get("levers") or {}).model_dump(exclude_none=True)
        inputs = {**base_inputs, **levers}
        out: dict[str, Any] = dict(
            case_id=spec.get("case_id"),
            name=spec["name"],
            status="ok",
            error=None,
            levers=levers,
            inputs=inputs,
            two_zone=None,
            fixed_eta=None,
            warnings=[],
            notes=[],
        )
        if "eta_co_pct" in levers:
            out["notes"].append("ηCO applies to the oxygen-balance model; the two-zone balance computes it from ω.")
        if "shaft_efficiency" in levers:
            out["notes"].append("Shaft efficiency applies to the two-zone balance; the oxygen-balance model holds ηCO.")
        try:
            fx = whatif.run_scenario(case0, k0, eta0, q_gj_h, forge_scenario(levers), p=cal.get("p"))
            case = copy.deepcopy(fx["case"])
            case["prod_th"] = inputs["production_t_h"]
            tz = _two_zone(case, cal, p, inputs["production_t_h"], inputs["shaft_efficiency"], b2)
        except Exception as exc:  # noqa: BLE001 - a case that does not solve is reported, the others still run
            out.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400])
            results.append(out)
            continue
        if _converged(tz, True):
            out["two_zone"] = _two_zone_out(tz, case, inputs, k_tz)
        if _converged(fx, False):
            out["fixed_eta"] = _fixed_eta_out(fx, inputs, k_fx)
        if out["two_zone"] is None and out["fixed_eta"] is None:
            out.update(status="failed", error="Neither balance converged for these inputs.")
        elif out["two_zone"] is None or out["fixed_eta"] is None:
            which = "two-zone balance" if out["two_zone"] is None else "oxygen-balance model"
            out["warnings"].append(f"The {which} did not converge for these inputs.")
        out["warnings"] = _band_warnings(reg, inputs, (out["two_zone"] or {}).get("fuel_rate_kg_thm")) + out["warnings"]
        results.append(out)
    return _finite(dict(facts_version=reg.label, base=base, cases=results, notes=[TWO_ZONE_NOTE, FIXED_ETA_NOTE]))


def forge_compute_json(cal_json: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """``forge_compute`` for the worker's process pool (picklable arguments)."""
    return forge_compute(json.loads(cal_json), cases)


# -------------------------------------------------------------------------------------------------- STRATA


def strata_case_result(spec: Mapping[str, Any], r: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, Any]:
    """The figures of one plan (``Engine`` result) for the side-by-side table."""
    o, c, s = r["out"], r["cost"], r["settings"]
    om = r.get("omega") or {}
    blend_mode = om.get("mode") == "blend"
    ferrous = dict(r.get("share_wet") or {})
    wet = dict(r.get("wet_kg") or {})
    violations = sorted(k for k, v in dict(r.get("viol") or {}).items() if _num(v) is not None and float(v) > 1e-3)
    feasible = bool(r.get("feasible", True))
    return dict(
        case_id=spec.get("case_id"),
        name=spec["name"],
        status="ok" if feasible else "infeasible",
        error=None if feasible else "The plan breaches: " + ", ".join(violations or ["a limit"]),
        forced=bool(spec.get("blend")),
        requested_blend=spec.get("blend"),
        production_t_d=_num(r.get("P_tday")),
        cost_rs_thm=_num(c.get("total")),
        cost_gross_rs_thm=_num(c.get("gross")),
        ore_rs_thm=_num(c.get("ore")),
        flux_rs_thm=_num(c.get("flux")),
        fuel_rs_thm=_num(c.get("fuel")),
        slag_credit_rs_thm=_num(c.get("slag_credit")),
        coke_kg_thm=_num(o.get("K")),
        coke_wet_kg_thm=_num(o.get("K_wet")),
        coke_sd_kg_thm=_num(om.get("sigma_K")) if blend_mode else None,
        fuel_rate_kg_thm=_num(o.get("fuel_rate")),
        eta_co_pct=_num(o.get("eta")),
        top_gas_c=_num(o.get("T_top")),
        shaft_efficiency=_num(o.get("omega")),
        certainty=om.get("certainty"),
        slag_kg_thm=_num(o.get("slag")),
        slag_cao_sio2=_num(o.get("B2")),
        slag_al2o3_pct=_num(o.get("Al2O3_pct")),
        slag_mgo_pct=_num(o.get("MgO_pct")),
        raft_c=_num(r.get("raft")),
        blast_nm3_h=_num(r.get("blast_nm3h")),
        o2_nm3_h=_num(r.get("O2_nm3h")),
        charges_per_h=_num(r.get("charges_per_h")),
        blend=[
            dict(id=k, name=names.get(k, k), share_pct=float(v), kg_thm=float(wet.get(k, 0.0)))
            for k, v in sorted(ferrous.items(), key=lambda kv: -kv[1])
            if float(v) > 1e-6
        ],
        flux=[
            dict(id=k, name=names.get(k, k), share_pct=None, kg_thm=float(v))
            for k, v in wet.items()
            if k not in ferrous and float(v) > 1e-6
        ],
        settings=dict(
            hmt_c=float(s["HMT"]),
            hm_si_pct=float(s["hm_Si"]),
            pci_kg_thm=float(s["pci_plant"]),
            nut_coke_kg_thm=float(s["nut_plant"]),
            blast_moisture_g_nm3=float(s["humidity_g"]),
            hbt_c=float(s["HBT"]),
            o2_per_air=float(s["O2_per_air"]),
        ),
        warnings=list(r.get("warnings") or []),
        violations=violations,
        facts_version=r.get("facts_version"),
    )


def _strata_refused(spec: Mapping[str, Any], status: str, error: str) -> dict[str, Any]:
    return dict(
        case_id=spec.get("case_id"),
        name=spec["name"],
        status=status,
        error=error[:400],
        forced=bool(spec.get("blend")),
        requested_blend=spec.get("blend"),
    )


def _strata_cases(inputs: Mapping[str, Any], cal: Mapping[str, Any], settings: Mapping[str, Any],
                  limits: Mapping[str, Any], prod_t_d: float, cases: Sequence[Mapping[str, Any]],
                  names: Mapping[str, str]) -> list[dict[str, Any]]:  # fmt: skip
    from services.optimiser import SETTINGS_FIELDS, engine_overrides
    from utils.burden_engine import BandError, Infeasible
    from utils.burden_engine.engine import Engine, Limits, Settings
    from utils.burden_engine.scenarios import run_today

    known = {m["id"] for m in inputs["materials"]}
    engine: Engine | None = None
    out = []
    for spec in cases:
        try:
            s = Settings(**{**dict(settings), **engine_overrides(spec.get("settings"), SETTINGS_FIELDS, "setting")})
            lim = Limits(**dict(limits))
            blend = spec.get("blend")
            if blend:
                unknown = sorted(set(blend) - known)
                if unknown:
                    raise WhatIfInputError(f"not in the yard: {', '.join(unknown)}")
                r = run_today(inputs, cal, blend, prod_t_d, s, lim, tol_pct=0.0)
                if r is None:
                    out.append(_strata_refused(spec, "infeasible", "The LP has no solution for this blend."))
                    continue
            else:
                engine = engine or Engine(dict(inputs), dict(cal))
                r = engine.min_cost(prod_t_day=prod_t_d, s=s, L=lim, fuel=True)
            out.append(strata_case_result(spec, r, names))
        except (BandError, Infeasible, WhatIfInputError) as exc:
            out.append(_strata_refused(spec, "infeasible", f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001 - one case's failure is reported, the others still run
            out.append(_strata_refused(spec, "failed", f"{type(exc).__name__}: {exc}"))
    return _finite(out)


def strata_compute(inputs_json: str, cal_json: str, settings: dict[str, Any], limits: dict[str, Any],
                   prod_t_d: float, cases: list[dict[str, Any]], names: dict[str, str],
                   solvers: int | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """The STRATA what-if cases in the worker's process pool: with STRATA-ω the network solves of the plans run in
    a pool of ``solvers`` processes (``strata_network.solver_processes``, as ``optimiser.solve_case``)."""
    from utils.burden_engine import strata_network

    inputs, cal = json.loads(inputs_json), json.loads(cal_json)
    if settings.get("omega_mode") != "blend":
        return _strata_cases(inputs, cal, settings, limits, prod_t_d, cases, names)
    with strata_network.solver_processes(solvers):
        return _strata_cases(inputs, cal, settings, limits, prod_t_d, cases, names)


# ---------------------------------------------------------------------------------------------------- cases


def case_inputs(kind: str, forge: Any = None, strata: Any = None) -> dict[str, Any]:
    """The stored ``inputs`` of a case (the same document the run parameters carry)."""
    if kind == "forge":
        levers = ForgeLevers.model_validate(forge or {}).model_dump(exclude_none=True)
        return {"levers": levers}
    if kind == "strata":
        doc = StrataCaseInputs.model_validate(strata or {})
        settings = doc.settings.model_dump(exclude_none=True) if doc.settings is not None else None
        blend = StrataCaseParams(name="x", blend=doc.blend).blend
        return {"blend": blend, "settings": settings or None}
    raise WhatIfInputError(f"kind must be forge or strata, not {kind!r}")


def case_view(row: WhatIfCase) -> dict[str, Any]:
    inputs = dict(row.inputs or {})
    view: dict[str, Any] = dict(
        id=str(row.id),
        kind=row.kind,
        name=row.name,
        owner=row.owner,
        created_at=utc(row.created_at),
        updated_at=utc(row.updated_at),
        forge=inputs.get("levers") if row.kind == "forge" else None,
        strata=dict(blend=inputs.get("blend"), settings=inputs.get("settings")) if row.kind == "strata" else None,
    )
    if row.result:
        view["forge_run" if row.kind == "forge" else "strata_run"] = row.result
    return view


def list_cases(store: TwinStore, owner: str, kind: str | None = None) -> list[dict[str, Any]]:
    return [case_view(row) for row in store.whatif.list(owner, kind, limit=MAX_CASES_PER_KIND * len(KINDS))]


def _owned(store: TwinStore, owner: str, case_id: str) -> WhatIfCase:
    row = store.whatif.get(case_id)
    if row is None or row.owner != owner:
        raise NotFound(f"no what-if case {case_id} of {owner}")
    return row


def save_case(
    store: TwinStore,
    actor: Actor,
    *,
    kind: str,
    name: str,
    inputs: Mapping[str, Any],
    case_id: str | None = None,
    at: datetime | None = None,
) -> dict[str, Any]:
    """Create the actor's case, or change one of theirs (``NotFound`` otherwise), with its audit row
    (``whatif.save``). A change of the inputs clears the stored result (it no longer belongs to the case)."""
    import uuid

    clean = name.strip()
    if not clean or len(clean) > 80:
        raise WhatIfInputError("a case name has 1-80 characters")
    when = at or datetime.now(UTC)
    if case_id is None:
        if len(store.whatif.list(actor.name, kind, limit=MAX_CASES_PER_KIND + 1)) >= MAX_CASES_PER_KIND:
            raise WhatIfInputError(f"at most {MAX_CASES_PER_KIND} saved {kind} cases per user: delete one first")
        key = uuid.uuid4()
    else:
        key = _owned(store, actor.name, case_id).id
    with audit.admin_write(store, actor, "whatif.save", "whatif_case", key, details={"kind": kind}, at=at) as write:
        row = write.session.get(WhatIfCase, key, with_for_update=True)
        if case_id is not None and (row is None or row.owner != actor.name):
            raise NotFound(f"no what-if case {case_id} of {actor.name}")
        write.before = audit.row_values(row, include=FIELDS) if row is not None else None
        if row is None:
            row = WhatIfCase(id=key, owner=actor.name, kind=kind, created_at=when)
            write.session.add(row)
        elif row.kind != kind:
            raise WhatIfInputError(f"case {case_id} is a {row.kind} case, not {kind}")
        if row.inputs is None or dict(row.inputs) != dict(inputs):
            row.result = None
            row.result_job_id = None
            row.result_at = None
            row.facts_version = None
        row.name = clean
        row.inputs = dict(inputs)
        row.updated_at = when
        write.session.flush()
        write.after = audit.row_values(row, include=FIELDS)
    saved = store.whatif.get(key)
    assert saved is not None
    return case_view(saved)


def delete_case(store: TwinStore, actor: Actor, case_id: str, *, at: datetime | None = None) -> None:
    """Delete one of the actor's cases (``NotFound`` otherwise) with its audit row (``whatif.delete``)."""
    key = _owned(store, actor.name, case_id).id
    with audit.admin_write(store, actor, "whatif.delete", "whatif_case", key, at=at) as write:
        row = write.session.get(WhatIfCase, key, with_for_update=True)
        if row is None or row.owner != actor.name:
            raise NotFound(f"no what-if case {case_id} of {actor.name}")
        write.before = audit.row_values(row, include=FIELDS)
        write.after = None
        write.session.delete(row)


def run_params(
    store: TwinStore,
    owner: str,
    kind: str,
    case_ids: Sequence[str],
    **basis: Any,
) -> WhatIfForgeParams | WhatIfStrataParams:
    """The job parameters of a run of the owner's saved cases (in the order given, at most four)."""
    if not 1 <= len(case_ids) <= WHATIF_MAX_CASES:
        raise WhatIfInputError(f"a run compares 1-{WHATIF_MAX_CASES} cases")
    if len(set(case_ids)) != len(case_ids):
        raise WhatIfInputError("a case is named twice")
    rows = [_owned(store, owner, cid) for cid in case_ids]
    wrong = [r.name for r in rows if r.kind != kind]
    if wrong:
        raise WhatIfInputError(f"not {kind} cases: {', '.join(wrong)}")
    if kind == "forge":
        cases = [
            ForgeCaseParams(case_id=str(r.id), name=r.name, levers=dict(r.inputs or {}).get("levers") or {})
            for r in rows
        ]
        return WhatIfForgeParams(cases=cases, **basis)
    cases_s = [
        StrataCaseParams(
            case_id=str(r.id),
            name=r.name,
            blend=dict(r.inputs or {}).get("blend"),
            settings=dict(r.inputs or {}).get("settings"),
        )
        for r in rows
    ]
    return WhatIfStrataParams(cases=cases_s, **basis)


def spec_inputs(kind: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """The stored ``inputs`` of the case a run's case parameters were built from."""
    if kind == "forge":
        return case_inputs("forge", forge=spec.get("levers") or {})
    return case_inputs("strata", strata={"blend": spec.get("blend"), "settings": spec.get("settings")})


def store_results(
    store: TwinStore | None,
    kind: str,
    owner: str,
    run: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Store each case's result with its saved case (the owner's, unchanged since the run started); returns the ids
    of the cases written."""
    if store is None:
        return []
    written = []
    for spec, result in zip(specs, results, strict=True):
        cid = spec.get("case_id")
        if not cid:
            continue
        doc = _finite({**run, "result": result})
        if store.whatif.store_result(
            cid,
            owner=owner,
            inputs=spec_inputs(kind, spec),
            result=json.loads(json.dumps(doc, default=str)),
            job_id=run.get("job_id"),
            facts_version=(str(run.get("facts_version") or "")[:32] or None),
        ):
            written.append(str(cid))
    return written


def materials(store: TwinStore | None = None) -> list[dict[str, str]]:
    """Ferrous materials a forced blend can name: the latest optimiser run's yard (live), else the reference yard."""
    from utils.burden_engine.engine import REFERENCE_INPUTS_PATH

    with open(REFERENCE_INPUTS_PATH, encoding="utf-8") as fh:
        ref = json.load(fh)
    out = {m["id"]: dict(id=m["id"], name=m.get("name", m["id"]), source="reference") for m in ref["materials"]}
    fluxes = {f["id"] for f in ref.get("fluxes", [])}
    if store is not None:
        try:
            latest = store.optimiser.latest()
        except Exception:  # noqa: BLE001 - the reference list still serves the editor
            latest = None
        names = dict(((latest.inputs or {}).get("materials") or {}) if latest is not None else {})
        flux_ids = (
            {f.get("id") for f in (((latest.inputs or {}).get("yard") or {}).get("fluxes") or [])} if latest else set()
        )
        for mid, name in names.items():
            if mid not in fluxes and mid not in flux_ids:
                out[mid] = dict(id=mid, name=str(name), source="live")
    return sorted(out.values(), key=lambda m: m["name"].lower())
