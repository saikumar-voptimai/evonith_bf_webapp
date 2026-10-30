"""Typed view of a stored Burden & Fuel Optimiser run (SV-3.2, #299): what page 11 shows under a plan, with units.

``result_view`` turns a ``twin.optimiser_run`` row (the case in ``inputs``, the engine's result in ``result``) into
``OptimiserResult``: the plan (Step 2 with fuel in the objective, Step 1 the cheapest level, else the maximum
production), Step 2's ore + flux-only plan (page 9's objective) and today's charged blend modelled at the plan's
production, Step 1's highest output and frontier, the value of limits, the critical assumptions, the certainty tier
and the STRATA-Flow network solves behind the plan. The panel figures come from ``utils.burden_engine.scenarios``
(``plan_tiles``, ``binding_table``, ``limit_label``, ``plan_sentence``), the functions page 11 renders, so no quantity
is composed twice. SV-3.3 (#300) adds what the SPA's Burden & Fuel Optimiser page shows next to the plan: each plan's
limits and the material prices, the lead line, the case the run was solved for (settings and limits in the API's
names, what the request set, today's operating point) and the yard it read.

Units: costs Rs/tHM; burden, flux, coke and fuel kg/tHM (coke dry unless named wet); production and tonnages t/d;
temperatures °C; pressures bar g (pressure drop bar); gas flows Nm³/h.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from services.live_inputs import LiveInputsRef
from utils.burden_engine.scenarios import binding_table, limit_label, limit_labels, plan_sentence, plan_tiles


class BlendRow(BaseModel):
    id: str
    name: str
    kind: str | None = Field(None, description="sinter, ore or pellet (from the yard of the case)")
    share_wet_pct: float = Field(..., description="Share of the ferrous burden, wet, %")
    wet_kg_thm: float = Field(..., description="Charged wet, kg/tHM")
    dry_kg_thm: float | None = Field(None, description="Charged dry, kg/tHM")
    t_d: float | None = Field(None, description="Charged wet per day, t/d")
    price_rs_t: float | None = Field(None, description="Price of the wet material in the case's yard, Rs/t")


class FluxRow(BaseModel):
    id: str
    name: str
    wet_kg_thm: float = Field(..., description="Charged wet, kg/tHM")
    t_d: float | None = Field(None, description="Charged wet per day, t/d")
    price_rs_t: float | None = Field(None, description="Price of the wet flux in the case's yard, Rs/t")


class CostSplit(BaseModel):
    ore_rs_thm: float = Field(..., description="Ferrous materials, Rs/tHM")
    flux_rs_thm: float = Field(..., description="Flux, Rs/tHM")
    fuel_rs_thm: float = Field(..., description="Coke + nut coke + PCI, Rs/tHM")
    coke_rs_thm: float = Field(..., description="Coke, Rs/tHM")
    nut_coke_rs_thm: float = Field(..., description="Nut coke, Rs/tHM")
    pci_rs_thm: float = Field(..., description="PCI, Rs/tHM")
    gross_rs_thm: float = Field(..., description="Ore + flux + fuel before the slag credit, Rs/tHM")
    slag_credit_rs_thm: float = Field(..., description="Granulated slag sold (ASM-20), Rs/tHM")
    total_rs_thm: float = Field(..., description="Ore + flux + fuel − slag credit, Rs/tHM")
    total_sigma_rs_thm: float | None = Field(None, description="1 sigma of the total (STRATA-ω coke), Rs/tHM")
    total_lakh_rs_d: float = Field(..., description="Total at the plan's production, lakh Rs/d")


class FuelSplit(BaseModel):
    coke_dry_kg_thm: float = Field(..., description="Coke rate, dry, kg/tHM")
    coke_wet_kg_thm: float = Field(..., description="Coke rate as charged, kg/tHM")
    coke_sigma_kg_thm: float | None = Field(None, description="1 sigma of the dry coke rate (STRATA-ω), kg/tHM")
    nut_coke_kg_thm: float = Field(..., description="Nut coke as charged (fixed input), kg/tHM")
    pci_kg_thm: float = Field(..., description="PCI as injected (fixed input), kg/tHM")
    fuel_rate_kg_thm: float = Field(..., description="Coke wet + nut coke + PCI (BAND-01 520-600), kg/tHM")


class SlagView(BaseModel):
    rate_kg_thm: float = Field(..., description="Slag rate, kg/tHM")
    t_d: float = Field(..., description="Slag per day, t/d")
    al2o3_pct: float | None = Field(None, description="Al₂O₃ of the slag, %")
    mgo_pct: float | None = Field(None, description="MgO of the slag, %")
    cao_sio2: float | None = Field(None, description="CaO/SiO₂ (B2), dimensionless")
    t_basicity: float | None = Field(None, description="(CaO + MgO)/SiO₂, dimensionless")
    ib4: float | None = Field(None, description="(CaO + MgO)/(SiO₂ + Al₂O₃), dimensionless")
    mgo_al2o3: float | None = Field(None, description="MgO/Al₂O₃, dimensionless")


class ChargingView(BaseModel):
    coke_t_d: float = Field(..., description="Coke as charged, t/d")
    nut_coke_t_d: float = Field(..., description="Nut coke, t/d")
    pci_t_d: float = Field(..., description="PCI, t/d")
    ferrous_t_d: float = Field(..., description="Ferrous burden wet, t/d")
    flux_kg_thm: float = Field(..., description="Flux wet, kg/tHM")
    charges_per_h: float | None = Field(None, description="Charges needed, charges/h")
    hm_per_charge_t: float | None = Field(None, description="Hot metal per charge, t")
    charging_t_h: float | None = Field(None, description="Non-coke burden charged, t/h")
    charging_cap_t_h: float | None = Field(None, description="Charging limit (charges/h × t per charge), t/h")


class ProcessView(BaseModel):
    hmt_c: float | None = Field(None, description="Hot-metal temperature target, °C")
    hm_si_pct: float | None = Field(None, description="Hot-metal Si target, %")
    hbt_c: float | None = Field(None, description="Hot-blast temperature, °C")
    blast_moisture_g_nm3: float | None = Field(None, description="Blast moisture, g/Nm³")
    blast_nm3h: float | None = Field(None, description="Hot blast, meter basis, Nm³/h")
    o2_nm3h: float | None = Field(None, description="Oxygen, Nm³/h")
    enrichment_tag_pct: float | None = Field(None, description="Plant enrichment tag, %")
    raft_c: float | None = Field(None, description="RAFT (plant workbook), °C")
    eta_co_pct: float | None = Field(None, description="Gas utilisation ηCO, %")
    top_gas_c: float | None = Field(None, description="Top-gas temperature, °C")
    hot_blast_pressure_needed_bar_g: float | None = Field(None, description="Hot-blast pressure needed, bar g")
    top_pressure_bar_g: float | None = Field(None, description="Top pressure, bar g")
    dp_needed_bar: float | None = Field(None, description="Pressure drop needed, bar")
    blast_ceiling_nm3h: float | None = Field(None, description="Blast at the set hot-blast pressure, Nm³/h")


class PlanLimits(BaseModel):
    """The limits a plan or a case was solved under, in the API's names (``services.optimiser.LIMITS_FIELDS``); the
    fuel-rate band is the register's BAND-01; the process rows apply when ``process`` is true."""

    slag_max_kg_thm: float | None = Field(None, description="Slag rate cap, kg/tHM")
    cao_sio2_min: float | None = Field(None, description="CaO/SiO₂ of the slag, lower limit, dimensionless")
    cao_sio2_max: float | None = Field(None, description="CaO/SiO₂ of the slag, upper limit, dimensionless")
    al2o3_max_pct: float | None = Field(None, description="Al₂O₃ of the slag, upper limit, %")
    mgo_min_pct: float | None = Field(None, description="MgO of the slag, lower limit, %")
    mgo_al2o3_min: float | None = Field(None, description="MgO/Al₂O₃ of the slag, lower limit, dimensionless")
    charge_t: float | None = Field(None, description="Non-coke burden per charge (ferrous + flux + nut coke), t")
    charges_per_h: float | None = Field(None, description="Charging rate limit, charges/h")
    top_gas_min_c: float | None = Field(None, description="Top-gas temperature, lower limit, °C")
    fuel_min_kg_thm: float | None = Field(None, description="Fuel rate, lower edge of BAND-01, kg/tHM")
    fuel_max_kg_thm: float | None = Field(None, description="Fuel rate, upper edge of BAND-01, kg/tHM")
    process: bool = Field(False, description="The process rows below apply (Step 1, maximum production)")
    hot_blast_pressure_max_bar_g: float | None = Field(None, description="Hot-blast pressure limit, bar g")
    dp_max_bar: float | None = Field(None, description="Pressure drop limit, bar")
    top_pressure_min_bar_g: float | None = Field(None, description="Top pressure, lower limit, bar g")
    top_pressure_max_bar_g: float | None = Field(None, description="Top pressure, upper limit, bar g")
    blast_max_nm3h: float | None = Field(None, description="Hot-blast volume limit, Nm³/h")
    o2_flow_max_nm3h: float | None = Field(None, description="Oxygen flow limit, Nm³/h")
    enrichment_max_pct: float | None = Field(None, description="Plant enrichment tag limit, %")
    hbt_max_c: float | None = Field(None, description="Hot-blast temperature limit, °C")
    raft_min_c: float | None = Field(None, description="RAFT, lower limit, °C")
    raft_max_c: float | None = Field(None, description="RAFT, upper limit, °C")


class NetworkSolves(BaseModel):
    """STRATA-Flow network solves behind a STRATA-ω plan."""

    solves: int | None = Field(None, description="Network solves of this calibration in the run's process, count")
    seconds: float | None = Field(None, description="Time of those solves, s")
    anchors: int | None = Field(None, description="Network states solved exactly (anchors), count")
    jacobian: str | None = Field(None, description="Where the network Jacobian came from (disk cache or solved)")
    max_distance: float | None = Field(
        None, description="Largest distance of the plan's states from a solve, scaled units, dimensionless"
    )
    exact: bool | None = Field(None, description="Every state of the plan is a network solve")


class ShaftEfficiency(BaseModel):
    mode: Literal["fixed", "blend"] = Field(..., description="fixed: Fixed-ω (ASM-01); blend: STRATA-ω (ASM-18)")
    omega: float | None = Field(None, description="Shaft efficiency of the plan, dimensionless")
    omega_cal: float | None = Field(None, description="Shaft efficiency of the calibration, dimensionless")
    omega_sigma: float | None = Field(None, description="1 sigma of the shaft efficiency, dimensionless")
    certainty: str | None = Field(None, description="High, Medium, Low, Investigate (STRATA-ω); Held (Fixed-ω)")
    outside: list[str] = Field(default_factory=list, description="Where the plan is outside what BF-2 has run")
    investigate: list[str] = Field(default_factory=list, description="Why the plan is in the Investigate tier")
    calibration_sigma_kg_thm: float | None = Field(None, description="Calibration error, not in sigma, kg/tHM")
    day_sd_kg_thm: float | None = Field(None, description="Day-to-day spread, not in sigma, kg/tHM")
    network: NetworkSolves | None = Field(None, description="Network solves behind the plan (STRATA-ω)")


class BindingLimit(BaseModel):
    limit: str = Field(..., description="Engine name of the limit")
    label: str = Field(..., description="Operator name of the limit")
    rs_thm_per_unit: float = Field(..., description="Shadow price: saving from relaxing one unit, Rs/tHM")
    unit: str = Field(..., description="Unit the shadow price is per")


class Violation(BaseModel):
    limit: str
    label: str
    amount: float = Field(..., description="By how much the plan breaks the limit, in the limit's unit")


class CriticalAssumption(BaseModel):
    assumption: str
    value: str = Field(..., description="Value the model uses")
    alternative: str = Field(..., description="Another credible value")
    coke_effect_kg_thm: float | None = Field(None, description="Coke effect of the alternative, wet, kg/tHM")
    coke_effect_lo_kg_thm: float | None = Field(None, description="Coke effect, lower alternative, wet, kg/tHM")
    coke_effect_hi_kg_thm: float | None = Field(None, description="Coke effect, upper alternative, wet, kg/tHM")
    symmetric: bool = Field(False, description="The effect is ± coke_effect_kg_thm")
    basis: str = ""


class PlanView(BaseModel):
    label: str = Field(..., description="What the plan is")
    feasible: bool
    violations: list[Violation] = Field(default_factory=list, description="Limits the plan breaks")
    production_t_d: float = Field(..., description="Hot metal, t/d")
    production_t_h: float = Field(..., description="Hot metal, t/h")
    burden_fe_wet_pct: float | None = Field(None, description="Fe of the ferrous burden, wet, %")
    ferrous_kg_thm: float | None = Field(None, description="Ferrous burden, wet, kg/tHM")
    blend: list[BlendRow] = Field(..., description="Ferrous blend, largest share first")
    flux: list[FluxRow] = Field(..., description="Flux (above 0.5 kg/tHM)")
    cost: CostSplit
    fuel: FuelSplit
    slag: SlagView
    charging: ChargingView
    process: ProcessView
    shaft_efficiency: ShaftEfficiency
    limits: PlanLimits | None = Field(None, description="The limits the plan was solved under")
    binding: list[BindingLimit] = Field(default_factory=list, description="Binding limits, largest shadow price first")
    warnings: list[str] = Field(default_factory=list)
    limited_by: str = Field("", description="Operator names of the binding limits (share bounds left out)")
    facts_version: str | None = Field(None, description="Fact register version of the engine")


class FrontierPoint(BaseModel):
    production_t_d: float = Field(..., description="Hot metal, t/d")
    cost_rs_thm: float = Field(..., description="Ore + flux + fuel − slag credit, Rs/tHM")
    ore_flux_rs_thm: float | None = Field(None, description="Ore + flux, Rs/tHM")
    fuel_rs_thm: float | None = Field(None, description="Fuel, Rs/tHM")
    coke_dry_kg_thm: float | None = Field(None, description="Coke rate, dry, kg/tHM")
    fuel_rate_kg_thm: float | None = Field(None, description="Fuel rate, kg/tHM")
    slag_kg_thm: float | None = Field(None, description="Slag rate, kg/tHM")
    hbt_c: float | None = Field(None, description="Hot-blast temperature, °C")
    enrichment_tag_pct: float | None = Field(None, description="Plant enrichment tag, %")
    blast_nm3h: float | None = Field(None, description="Hot blast, Nm³/h")
    o2_nm3h: float | None = Field(None, description="Oxygen, Nm³/h")
    hot_blast_pressure_needed_bar_g: float | None = Field(None, description="Hot-blast pressure needed, bar g")
    charges_per_h: float | None = Field(None, description="Charges needed, charges/h")
    marginal_rs_per_t: float | None = Field(None, description="Cost of each extra tonne of hot metal, Rs/t")
    binding: str = Field("", description="Binding limits (share bounds left out)")
    binding_label: str = Field("", description="Operator names of the binding limits")


class ValueOfLimit(BaseModel):
    change: str = Field(..., description="The limit relaxed one step")
    saving_rs_thm: float | None = Field(None, description="Saving at the same production (null: infeasible), Rs/tHM")
    binds_after: str = Field("", description="What binds after the change")
    binds_after_label: str = Field("", description="Operator names of what binds after the change")


class PlanLead(BaseModel):
    """Page 11's line over a plan: what to charge, what it costs and, in Step 2, what it saves against today's blend
    (``scenarios.plan_sentence``); maximum production names its binding limits."""

    kind: Literal["saving", "dearer", "plain"] = Field(..., description="saving / dearer against today's blend")
    text: str = Field(..., description="Markdown (bold spans only)")


class RunSettings(BaseModel):
    """The controls and thermal targets a run was solved at, in the API's names
    (``services.optimiser.SETTINGS_FIELDS``): page 11's defaults with the request's values applied last."""

    hmt_c: float | None = Field(None, description="Hot-metal temperature target (BAND-02), °C")
    hm_si_pct: float | None = Field(None, description="Hot-metal Si target (BAND-03), %")
    pci_kg_thm: float | None = Field(None, description="PCI as injected (fixed input), kg/tHM")
    nut_coke_kg_thm: float | None = Field(None, description="Nut coke as charged (fixed input), kg/tHM")
    blast_moisture_g_nm3: float | None = Field(None, description="Blast moisture (DATA-11), g/Nm³")
    hbt_c: float | None = Field(None, description="Hot-blast temperature, °C")
    o2_per_air: float | None = Field(None, description="Oxygen enrichment on the model basis, Nm³ O₂ per Nm³ air")
    hot_blast_pressure_bar_g: float | None = Field(None, description="Hot-blast pressure, bar g")
    top_pressure_bar_g: float | None = Field(None, description="Top pressure, bar g")


class TodayPoint(BaseModel):
    """Today's operating point: medians of the regime window of the calibration (page 11's "Today" hints)."""

    period: list[str] = Field(default_factory=list, description="First and last day of the window (IST dates)")
    n_shifts: int | None = Field(None, description="Stable shifts in the window, count")
    production_t_d: float | None = Field(None, description="Hot metal, t/d")
    hot_blast_nm3h: float | None = Field(None, description="Hot blast, meter basis, Nm³/h")
    o2_nm3h: float | None = Field(None, description="Oxygen, Nm³/h")
    enrichment_tag_pct: float | None = Field(None, description="Plant enrichment tag, %")
    hbt_c: float | None = Field(None, description="Hot-blast temperature, °C")
    hot_blast_pressure_bar_g: float | None = Field(None, description="Hot-blast pressure, bar g")
    top_pressure_bar_g: float | None = Field(None, description="Top pressure, bar g")
    dp_bar: float | None = Field(None, description="Pressure drop, bar")
    raft_c: float | None = Field(None, description="RAFT (DCS), °C")


class RunCase(BaseModel):
    """The case a run was solved for (``Case.summary``), in the API's names."""

    settings: RunSettings = Field(..., description="Controls and thermal targets used")
    limits: PlanLimits = Field(..., description="Limits used")
    settings_set: list[str] = Field(default_factory=list, description="Settings the request set (others: defaults)")
    limits_set: list[str] = Field(default_factory=list, description="Limits the request set (others: defaults)")
    certainty_filter: str | None = Field(None, description="STRATA-ω certainty filter: any, low, medium, high")
    risk_k: float | None = Field(None, description="STRATA-ω risk aversion k (cost + k × σ), dimensionless")
    search_o2: bool = Field(False, description="Oxygen searched up to its limit (Step 1, maximum production)")
    auto: bool = Field(False, description="Auto mode's three decision settings were run as well")
    production_source: str | None = Field(None, description="Where Step 2's production target came from")
    today_window: str | None = Field(None, description="When today's charged blend was charged (IST)")
    yard_message: str | None = Field(None, description="Why the reference yard stands in (empty: the live yard)")
    operator_inputs: dict[str, Any] = Field(
        default_factory=dict, description="Version of the operator inputs the yard was built with (source, version)"
    )
    today: TodayPoint | None = Field(None, description="Today's operating point (none on the shipped calibration)")
    live_inputs: LiveInputsRef | None = Field(
        None,
        description=(
            "The live-inputs snapshot the live yard was built from (SV-7.3, GET /api/inputs/live); null on the "
            "reference yard or for a run stored before the snapshot existed"
        ),
    )


class YardMaterial(BaseModel):
    id: str
    name: str
    kind: str | None = Field(None, description="sinter, ore or pellet")
    selected: bool = Field(True, description="Offered to the optimiser (operator's availability)")
    carried: bool = Field(True, description="Has lab chemistry and is carried by the engine")
    reason: str | None = Field(None, description="Why the material is not carried")
    fe_pct: float | None = Field(None, description="Fe of the dry material, %")
    sio2_pct: float | None = Field(None, description="SiO₂ of the dry material, %")
    al2o3_pct: float | None = Field(None, description="Al₂O₃ of the dry material, %")
    cao_pct: float | None = Field(None, description="CaO of the dry material, %")
    loi_pct: float | None = Field(None, description="Loss on ignition, %")
    moisture_pct: float | None = Field(None, description="Moisture, %")
    price_rs_t: float | None = Field(None, description="Price per wet tonne, Rs/t")
    min_share_pct: float | None = Field(None, description="Lower bound of the wet share of the ferrous charge, %")
    max_share_pct: float | None = Field(None, description="Upper bound of the wet share of the ferrous charge, %")
    stock_t: float | None = Field(None, description="Stock in the yard, t")
    chem_source: str | None = Field(None, description="Where the chemistry came from")


class YardFlux(BaseModel):
    id: str
    name: str
    price_rs_t: float | None = Field(None, description="Price per wet tonne, Rs/t")
    max_t_d: float | None = Field(None, description="Upper bound of the flux charged, t/d")
    stock_t: float | None = Field(None, description="Stock in the yard, t")
    cao_pct: float | None = Field(None, description="CaO, %")
    mgo_pct: float | None = Field(None, description="MgO, %")
    sio2_pct: float | None = Field(None, description="SiO₂, %")
    loi_pct: float | None = Field(None, description="Loss on ignition, %")


class YardFuel(BaseModel):
    coke_rs_t: float | None = Field(None, description="Coke price, Rs/t")
    nut_coke_rs_t: float | None = Field(None, description="Nut coke price, Rs/t")
    pci_rs_t: float | None = Field(None, description="PCI coal price, Rs/t")
    slag_rs_t: float | None = Field(None, description="Granulated slag sale price (ASM-20), Rs/t")


class RunYard(BaseModel):
    """The yard the run read: ferrous materials (carried or not), fluxes and fuel prices."""

    materials: list[YardMaterial] = Field(default_factory=list)
    fluxes: list[YardFlux] = Field(default_factory=list)
    fuel: YardFuel = Field(default_factory=YardFuel)


class OptimiserResult(BaseModel):
    run_id: str = Field(..., description="Run id (UUID)")
    mode: Literal["step1", "step2", "max_production"]
    omega_mode: Literal["strata", "fixed"]
    status: Literal["ok", "infeasible", "failed"]
    error: str | None = Field(None, description="Why there is no plan (infeasible or failed)")
    started_at: datetime | None = Field(None, description="UTC")
    finished_at: datetime | None = Field(None, description="UTC")
    run_s: float | None = Field(None, description="Duration of the run, s")
    production_target_t_d: float | None = Field(None, description="Step 2 target, t/d")
    yard_source: str | None = Field(None, description="live or reference")
    calibration: dict[str, Any] = Field(default_factory=dict, description="Calibration source, window, ω and basis")
    certainty: str | None = Field(None, description="Certainty tier of the plan: High, Medium, Low, Investigate")
    plan: PlanView | None = Field(None, description="The plan the run reports")
    ore_flux_plan: PlanView | None = Field(None, description="Step 2: ore + flux objective (page 9), same limits")
    today_plan: PlanView | None = Field(None, description="Step 2: today's charged blend at the plan's production")
    saving_rs_thm: float | None = Field(None, description="Today's charged blend minus the plan, Rs/tHM")
    saving_lakh_rs_d: float | None = Field(None, description="The saving at the plan's production, lakh Rs/d")
    highest_output_plan: PlanView | None = Field(None, description="Step 1: the highest production that is feasible")
    frontier: list[FrontierPoint] = Field(default_factory=list, description="Step 1: cheapest plan per level")
    value_of_limits: list[ValueOfLimit] = Field(default_factory=list, description="Step 2: each limit one step")
    critical_assumptions: list[CriticalAssumption] = Field(
        default_factory=list, description="Coke effect of each critical assumption of a STRATA-ω plan, largest first"
    )
    critical_assumptions_error: str | None = Field(None, description="Why the sensitivity is missing")
    solves: NetworkSolves | None = Field(None, description="STRATA-Flow network solves behind the plan")
    lead: PlanLead | None = Field(None, description="Page 11's line over the plan (Step 2, maximum production)")
    case: RunCase | None = Field(None, description="The case the run was solved for")
    yard: RunYard | None = Field(None, description="The yard the run read")


# ------------------------------------------------------------------------------------------------------- helpers
def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _get(row: Any, name: str, default: Any = None) -> Any:
    return row.get(name, default) if isinstance(row, Mapping) else getattr(row, name, default)


def flux_ids_of(plan: Mapping[str, Any]) -> list[str]:
    """Flux ids of a plan: charged ids that are not ferrous (``wet_kg`` covers both, ``share_wet`` the ferrous)."""
    ferrous = set(plan.get("share_wet") or {})
    return [i for i in (plan.get("wet_kg") or {}) if i not in ferrous]


def _api_names(fields: Mapping[str, str], values: Mapping[str, Any]) -> dict[str, float | None]:
    """Engine field values (``Settings`` / ``Limits``) under the API's names (``fields``: API name -> engine)."""
    return {api: _num(values.get(engine)) for api, engine in fields.items()}


def limits_view(limits: Mapping[str, Any] | None) -> PlanLimits | None:
    """``Limits`` fields of a plan or case under the API's names."""
    from services.optimiser import LIMITS_FIELDS  # the API process imports the optimiser service lazily

    if not isinstance(limits, Mapping) or not limits:
        return None
    return PlanLimits(
        **_api_names(LIMITS_FIELDS, limits),
        fuel_min_kg_thm=_num(limits.get("fuel_min")),
        fuel_max_kg_thm=_num(limits.get("fuel_max")),
        process=bool(limits.get("process", False)),
    )


def _text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _today(regime: Any) -> TodayPoint | None:
    if not isinstance(regime, Mapping):
        return None
    prod_th = _num(regime.get("prod_th"))
    n = regime.get("n_shifts")
    return TodayPoint(
        period=[str(x) for x in regime.get("period") or []],
        n_shifts=int(n) if isinstance(n, (int, float)) and math.isfinite(n) else None,
        production_t_d=prod_th * 24 if prod_th is not None else None,
        hot_blast_nm3h=_num(regime.get("hot_blast_nm3h")),
        o2_nm3h=_num(regime.get("O2_nm3h")),
        enrichment_tag_pct=_num(regime.get("enrich_tag")),
        hbt_c=_num(regime.get("HBT")),
        hot_blast_pressure_bar_g=_num(regime.get("Pb")),
        top_pressure_bar_g=_num(regime.get("Pt")),
        dp_bar=_num(regime.get("dP")),
        raft_c=_num(regime.get("raft_dcs")),
    )


def case_view(inputs: Mapping[str, Any]) -> RunCase | None:
    """The stored case (``optimiser_run.inputs``) under the API's names; None for a run stored without one."""
    from services.optimiser import SETTINGS_FIELDS

    settings, limits = inputs.get("settings"), inputs.get("limits")
    if not isinstance(settings, Mapping) or not isinstance(limits, Mapping):
        return None
    over = dict(inputs.get("overrides") or {})
    blend = settings.get("omega_mode") == "blend"
    return RunCase(
        settings=RunSettings(**_api_names(SETTINGS_FIELDS, settings)),
        limits=limits_view(limits) or PlanLimits(),
        settings_set=sorted(dict(over.get("settings") or {})),
        limits_set=sorted(dict(over.get("limits") or {})),
        certainty_filter=_text(settings.get("certainty")) if blend else None,
        risk_k=_num(settings.get("risk_k")) if blend else None,
        search_o2=bool(inputs.get("search_o2", False)),
        auto=bool(inputs.get("auto", False)),
        production_source=_text(inputs.get("production_source")),
        today_window=_text(inputs.get("today_window")),
        yard_message=_text(inputs.get("yard_message")),
        operator_inputs=dict(inputs.get("operator_inputs") or {}),
        today=_today(inputs.get("regime")),
        live_inputs=(
            LiveInputsRef.model_validate(inputs["live_inputs"])
            if isinstance(inputs.get("live_inputs"), Mapping)
            else None
        ),
    )


def _chem(row: Mapping[str, Any], key: str) -> float | None:
    return _num(dict(row.get("chem") or {}).get(key))


def yard_view(yard: Mapping[str, Any] | None) -> RunYard | None:
    """The stored yard rows (``Case.yard``); None for a run stored without them."""
    if not isinstance(yard, Mapping) or not yard:
        return None
    materials = [
        YardMaterial(
            id=str(m["id"]),
            name=str(m.get("name") or m["id"]),
            kind=_text(m.get("kind")),
            selected=bool(m.get("selected", True)),
            carried=bool(m.get("carried", True)),
            reason=_text(m.get("reason")),
            fe_pct=_chem(m, "Fe"),
            sio2_pct=_chem(m, "SiO2"),
            al2o3_pct=_chem(m, "Al2O3"),
            cao_pct=_chem(m, "CaO"),
            loi_pct=_chem(m, "LOI"),
            moisture_pct=_num(m.get("moist")),
            price_rs_t=_num(m.get("price")),
            min_share_pct=_num(m.get("lo")),
            max_share_pct=_num(m.get("hi")),
            stock_t=_num(m.get("stock_mt")),
            chem_source=_text(m.get("chem_source")),
        )
        for m in yard.get("materials") or []
        if isinstance(m, Mapping) and "id" in m
    ]
    fluxes = [
        YardFlux(
            id=str(f["id"]),
            name=str(f.get("name") or f["id"]),
            price_rs_t=_num(f.get("price")),
            max_t_d=_num(f.get("max_qty_mt", f.get("max_t_day"))),
            stock_t=_num(f.get("stock_mt")),
            cao_pct=_chem(f, "CaO"),
            mgo_pct=_chem(f, "MgO"),
            sio2_pct=_chem(f, "SiO2"),
            loi_pct=_chem(f, "LOI"),
        )
        for f in yard.get("fluxes") or []
        if isinstance(f, Mapping) and "id" in f
    ]
    fu = dict(yard.get("fuel") or {})
    fuel = YardFuel(
        coke_rs_t=_num(fu.get("coke_price")),
        nut_coke_rs_t=_num(fu.get("nut_price")),
        pci_rs_t=_num(fu.get("pci_price")),
        slag_rs_t=_num(fu.get("slag_price")),
    )
    return RunYard(materials=materials, fluxes=fluxes, fuel=fuel)


def lead_view(engine_mode: Any, plan: Mapping[str, Any] | None, today: Mapping[str, Any] | None,
              names: Mapping[str, str]) -> PlanLead | None:  # fmt: skip
    """Page 11's line over the plan: Step 2 ``plan_sentence`` against today's blend, maximum production its binding
    limits; Step 1 shows none."""
    if plan is None:
        return None
    if engine_mode == "fixed":
        kind, text = plan_sentence(plan, names, flux_ids_of(plan), today)
        return PlanLead(kind=kind, text=text)
    if engine_mode == "max":
        text = (
            "Limited by: "
            + (limit_labels(plan.get("binding", []), names) or "–")
            + ". Controls at their limits; oxygen searched within its flow and enrichment limits, least oxygen among "
            "equal outputs."
        )
        return PlanLead(kind="plain", text=text)
    return None


def _network(om: Mapping[str, Any]) -> NetworkSolves | None:
    net = om.get("network") or {}
    if net.get("backend") != "network":
        return None
    dist = [_num(v) for v in dict(net.get("distance") or {}).values()]
    dist = [d for d in dist if d is not None]
    return NetworkSolves(
        solves=net.get("solves"),
        seconds=_num(net.get("seconds")),
        anchors=net.get("anchors"),
        jacobian=str(net.get("jacobian") or "") or None,
        max_distance=max(dist) if dist else None,
        exact=net.get("exact"),
    )


def plan_view(
    r: Mapping[str, Any],
    label: str,
    names: Mapping[str, str],
    kinds: Mapping[str, str] | None = None,
    prices: Mapping[str, float] | None = None,
) -> PlanView:
    """One plan as page 11's panels show it (``plan_tiles``; binding rows from ``binding_table``)."""
    flux_ids = flux_ids_of(r)
    tiles = plan_tiles(r, flux_ids)
    c, f, s, g, ov = tiles["cost"], tiles["fuel"], tiles["slag"], tiles["charging"], tiles["overview"]
    out, st, om = r.get("out") or {}, r.get("settings") or {}, r.get("omega") or {}
    wet, dry, t_day = r.get("wet_kg") or {}, r.get("dry_kg") or {}, r.get("t_day") or {}
    blend = sorted(
        (
            BlendRow(
                id=i,
                name=names.get(i, i),
                kind=(kinds or {}).get(i),
                share_wet_pct=float(share),
                wet_kg_thm=float(wet.get(i, 0.0) or 0.0),
                dry_kg_thm=_num(dry.get(i)),
                t_d=_num(t_day.get(i)),
                price_rs_t=_num((prices or {}).get(i)),
            )
            for i, share in (r.get("share_wet") or {}).items()
            if share is not None and float(share) > 0.05
        ),
        key=lambda b: -b.share_wet_pct,
    )
    flux = [
        FluxRow(
            id=i,
            name=names.get(i, i),
            wet_kg_thm=float(wet.get(i, 0.0) or 0.0),
            t_d=_num(t_day.get(i)),
            price_rs_t=_num((prices or {}).get(i)),
        )  # fmt: skip
        for i in flux_ids
        if float(wet.get(i, 0.0) or 0.0) >= 0.5
    ]
    sigma_cost = _num((r.get("cost") or {}).get("sigma"))
    blend_mode = om.get("mode") == "blend"
    bt = binding_table(r)
    return PlanView(
        label=label,
        feasible=bool(r.get("feasible", True)),
        violations=[
            Violation(limit=k, label=limit_label(k, names), amount=float(v))
            for k, v in (r.get("viol") or {}).items()
            if v is not None and float(v) > 1e-3
        ],
        production_t_d=float(ov["P_tday"]),
        production_t_h=float(ov["P_th"]),
        burden_fe_wet_pct=_num(ov["fe_wet_pct"]),
        ferrous_kg_thm=_num(ov["ferrous_kg"]),
        blend=blend,
        flux=flux,
        cost=CostSplit(
            ore_rs_thm=c["ore"],
            flux_rs_thm=c["flux"],
            fuel_rs_thm=c["fuel"],
            coke_rs_thm=c["coke"],
            nut_coke_rs_thm=c["nut"],
            pci_rs_thm=c["pci"],
            gross_rs_thm=c["gross"],
            slag_credit_rs_thm=c["slag_credit"],
            total_rs_thm=c["total"],
            total_sigma_rs_thm=sigma_cost if blend_mode else None,
            total_lakh_rs_d=c["total"] * ov["P_tday"] / 1e5,
        ),
        fuel=FuelSplit(
            coke_dry_kg_thm=f["coke_dry"],
            coke_wet_kg_thm=f["coke_wet"],
            coke_sigma_kg_thm=_num(om.get("sigma_K")) if blend_mode else None,
            nut_coke_kg_thm=f["nut"],
            pci_kg_thm=f["pci"],
            fuel_rate_kg_thm=f["total"],
        ),
        slag=SlagView(
            rate_kg_thm=s["rate"],
            t_d=s["t_day"],
            al2o3_pct=_num(s["Al2O3_pct"]),
            mgo_pct=_num(s["MgO_pct"]),
            cao_sio2=_num(s["b2"]),
            t_basicity=_num(s["t_basicity"]),
            ib4=_num(s["ib4"]),
            mgo_al2o3=_num(s["mgo_al2o3"]),
        ),
        charging=ChargingView(
            coke_t_d=g["coke_t"],
            nut_coke_t_d=g["nut_t"],
            pci_t_d=g["pci_t"],
            ferrous_t_d=g["ferrous_t"],
            flux_kg_thm=g["flux_kg"],
            charges_per_h=_num(r.get("charges_per_h")),
            hm_per_charge_t=_num(g["hm_per_charge"]),
            charging_t_h=_num(r.get("charging_th")),
            charging_cap_t_h=_num(r.get("charging_cap_th")),
        ),
        process=ProcessView(
            hmt_c=_num(st.get("HMT")),
            hm_si_pct=_num(st.get("hm_Si")),
            hbt_c=_num(st.get("HBT")),
            blast_moisture_g_nm3=_num(st.get("humidity_g")),
            blast_nm3h=_num(r.get("blast_nm3h")),
            o2_nm3h=_num(r.get("O2_nm3h")),
            enrichment_tag_pct=_num(r.get("enrich_tag")),
            raft_c=_num(r.get("raft")),
            eta_co_pct=_num(out.get("eta")),
            top_gas_c=_num(out.get("T_top")),
            hot_blast_pressure_needed_bar_g=_num(r.get("Pb_needed")),
            top_pressure_bar_g=_num(st.get("Pt")),
            dp_needed_bar=_num(r.get("dP_needed")),
            blast_ceiling_nm3h=_num(r.get("V_ceiling")),
        ),
        shaft_efficiency=ShaftEfficiency(
            mode="blend" if blend_mode else "fixed",
            omega=_num(om.get("value", out.get("omega"))),
            omega_cal=_num(om.get("omega_cal")),
            omega_sigma=_num(om.get("sigma_omega")) if blend_mode else None,
            certainty=om.get("certainty"),
            outside=[str(x) for x in om.get("outside") or []],
            investigate=[str(x) for x in om.get("investigate") or []],
            calibration_sigma_kg_thm=_num(om.get("cal_se_K")) if blend_mode else None,
            day_sd_kg_thm=_num(om.get("day_sd_K")) if blend_mode else None,
            network=_network(om) if blend_mode else None,
        ),
        limits=limits_view(r.get("limits")),
        binding=[
            BindingLimit(
                limit=str(x.limit),
                label=limit_label(str(x.limit), names),
                rs_thm_per_unit=float(x.rs_thm_per_unit),
                unit=str(x.unit),
            )  # fmt: skip
            for x in bt.itertuples()
        ]
        if not bt.empty
        else [],
        warnings=[str(w) for w in r.get("warnings") or []],
        limited_by=limit_labels([str(b) for b in r.get("binding") or []], names),
        facts_version=r.get("facts_version"),
    )


def _labels(text: Any, names: Mapping[str, str] | None = None) -> str:
    """Operator names of a comma-joined list of engine limits (page 11's frontier and value-of-limits tables)."""
    value = str(text or "")
    if value == "infeasible":
        return value
    return limit_labels([x.strip() for x in value.split(",") if x.strip()], names)


def _frontier(rows: Sequence[Mapping[str, Any]] | None) -> list[FrontierPoint]:
    return [
        FrontierPoint(
            production_t_d=float(x["prod_t_day"]),
            cost_rs_thm=float(x["cost_rs_thm"]),
            ore_flux_rs_thm=_num(x.get("ore_flux_rs_thm")),
            fuel_rs_thm=_num(x.get("fuel_rs_thm")),
            coke_dry_kg_thm=_num(x.get("coke_dry")),
            fuel_rate_kg_thm=_num(x.get("fuel_rate")),
            slag_kg_thm=_num(x.get("slag")),
            hbt_c=_num(x.get("HBT")),
            enrichment_tag_pct=_num(x.get("enrich_tag")),
            blast_nm3h=_num(x.get("blast_nm3h")),
            o2_nm3h=_num(x.get("O2_nm3h")),
            hot_blast_pressure_needed_bar_g=_num(x.get("Pb_needed")),
            charges_per_h=_num(x.get("charges_per_h")),
            marginal_rs_per_t=_num(x.get("marginal_rs_per_t")),
            binding=str(x.get("binding") or ""),
            binding_label=_labels(x.get("binding")),
        )
        for x in rows or []
        if isinstance(x, Mapping)
    ]


def _critical(rows: Any) -> tuple[list[CriticalAssumption], str | None]:
    if isinstance(rows, Mapping):
        return [], str(rows.get("error") or "the sensitivity failed")
    out = []
    for x in rows or []:
        out.append(
            CriticalAssumption(
                assumption=str(x.get("assumption", "")),
                value=str(x.get("value", "")),
                alternative=str(x.get("alternative", "")),
                coke_effect_kg_thm=_num(x.get("d_coke_kg")),
                coke_effect_lo_kg_thm=_num(x.get("d_coke_lo_kg")),
                coke_effect_hi_kg_thm=_num(x.get("d_coke_hi_kg")),
                symmetric=bool(x.get("symmetric", False)),
                basis=str(x.get("basis", "")),
            )
        )
    return out, None


def result_view(row: Any) -> OptimiserResult:
    """The typed result of a stored run (a ``twin.optimiser_run`` row or a mapping with its columns)."""
    inputs = dict(_get(row, "inputs") or {})
    res = dict(_get(row, "result") or {})
    names = dict(inputs.get("materials") or {})
    yard = dict(inputs.get("yard") or {})
    kinds = {m["id"]: m.get("kind") for m in yard.get("materials") or [] if isinstance(m, Mapping) and "id" in m}
    prices = {
        str(m["id"]): m.get("price")
        for m in [*(yard.get("materials") or []), *(yard.get("fluxes") or [])]
        if isinstance(m, Mapping) and "id" in m
    }
    mode = str(_get(row, "mode"))
    engine_mode = res.get("mode")
    started, finished = _get(row, "started_at"), _get(row, "finished_at")
    view: dict[str, Any] = dict(
        run_id=str(_get(row, "id")),
        mode=mode,
        omega_mode=str(_get(row, "omega_mode")),
        status=str(_get(row, "status")),
        error=res.get("error"),
        started_at=started,
        finished_at=finished,
        run_s=(finished - started).total_seconds() if started is not None and finished is not None else None,
        production_target_t_d=_num(inputs.get("production_target_t_d")),
        yard_source=inputs.get("yard_source"),
        calibration={k: v for k, v in dict(inputs.get("calibration") or {}).items()},
        certainty=_get(row, "certainty"),
        saving_rs_thm=_num(_get(row, "saving_rs_thm")),
    )
    plan = None
    if engine_mode == "fixed":
        plan = res.get("with_fuel")
        if plan is not None:
            view["plan"] = plan_view(plan, f"Cheapest blend at {plan['P_tday']:,.0f} t/d, fuel in the objective",
                                     names, kinds, prices)  # fmt: skip
        if res.get("ore_only") is not None:
            view["ore_flux_plan"] = plan_view(res["ore_only"], "Ore + flux objective (page 9)", names, kinds, prices)
        if res.get("today") is not None:
            label = "Today's charged blend, modelled" + (
                f" ({inputs['today_window']})" if inputs.get("today_window") else ""
            )
            view["today_plan"] = plan_view(res["today"], label, names, kinds, prices)
        view["value_of_limits"] = [
            ValueOfLimit(
                change=str(x.get("change", "")),
                saving_rs_thm=_num(x.get("saving_rs_thm")),
                binds_after=str(x.get("note") or ""),
                binds_after_label=_labels(x.get("note"), names),
            )
            for x in res.get("value_of_limits") or []
            if isinstance(x, Mapping)
        ]
    elif engine_mode == "cheapest":
        plan = res.get("best")
        if plan is not None:
            view["plan"] = plan_view(plan, f"Cheapest output on the Step 1 frontier, {plan['P_tday']:,.0f} t/d",
                                     names, kinds, prices)  # fmt: skip
        if res.get("top") is not None:
            top = res["top"]
            view["highest_output_plan"] = plan_view(
                top, f"Highest output, {top['P_tday']:,.0f} t/d", names, kinds, prices
            )
        view["frontier"] = _frontier(res.get("frontier"))
    elif engine_mode == "max":
        plan = res.get("result")
        if plan is not None:
            view["plan"] = plan_view(plan, f"Maximum production, {plan['P_tday']:,.0f} t/d", names, kinds, prices)
    if view.get("plan") is not None and view["saving_rs_thm"] is not None:
        view["saving_lakh_rs_d"] = view["saving_rs_thm"] * view["plan"].production_t_d / 1e5
    view["critical_assumptions"], view["critical_assumptions_error"] = _critical(res.get("critical_assumptions"))
    view["lead"] = lead_view(engine_mode, plan, res.get("today") if engine_mode == "fixed" else None, names)
    view["case"] = case_view(inputs)
    view["yard"] = yard_view(yard)
    if view.get("plan") is not None:
        view["solves"] = view["plan"].shaft_efficiency.network
    return OptimiserResult(**view)
