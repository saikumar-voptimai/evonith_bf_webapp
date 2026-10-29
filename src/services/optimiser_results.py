"""Typed view of a stored Burden & Fuel Optimiser run (SV-3.2, #299): what page 11 shows under a plan, with units.

``result_view`` turns a ``twin.optimiser_run`` row (the case in ``inputs``, the engine's result in ``result``) into
``OptimiserResult``: the plan (Step 2 with fuel in the objective, Step 1 the cheapest level, else the maximum
production), Step 2's ore + flux-only plan (page 9's objective) and today's charged blend modelled at the plan's
production, Step 1's highest output and frontier, the value of limits, the critical assumptions, the certainty tier
and the STRATA-Flow network solves behind the plan. The panel figures come from ``utils.burden_engine.scenarios``
(``plan_tiles``, ``binding_table``, ``limit_label``), the functions page 11 renders, so no quantity is composed twice.

Units: costs Rs/tHM; burden, flux, coke and fuel kg/tHM (coke dry unless named wet); production and tonnages t/d;
temperatures °C; pressures bar g (pressure drop bar); gas flows Nm³/h.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from utils.burden_engine.scenarios import binding_table, limit_label, plan_tiles


class BlendRow(BaseModel):
    id: str
    name: str
    kind: str | None = Field(None, description="sinter, ore or pellet (from the yard of the case)")
    share_wet_pct: float = Field(..., description="Share of the ferrous burden, wet, %")
    wet_kg_thm: float = Field(..., description="Charged wet, kg/tHM")
    dry_kg_thm: float | None = Field(None, description="Charged dry, kg/tHM")
    t_d: float | None = Field(None, description="Charged wet per day, t/d")


class FluxRow(BaseModel):
    id: str
    name: str
    wet_kg_thm: float = Field(..., description="Charged wet, kg/tHM")
    t_d: float | None = Field(None, description="Charged wet per day, t/d")


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
    binding: list[BindingLimit] = Field(default_factory=list, description="Binding limits, largest shadow price first")
    warnings: list[str] = Field(default_factory=list)
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


class ValueOfLimit(BaseModel):
    change: str = Field(..., description="The limit relaxed one step")
    saving_rs_thm: float | None = Field(None, description="Saving at the same production (null: infeasible), Rs/tHM")
    binds_after: str = Field("", description="What binds after the change")


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
    r: Mapping[str, Any], label: str, names: Mapping[str, str], kinds: Mapping[str, str] | None = None
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
            )
            for i, share in (r.get("share_wet") or {}).items()
            if share is not None and float(share) > 0.05
        ),
        key=lambda b: -b.share_wet_pct,
    )
    flux = [
        FluxRow(id=i, name=names.get(i, i), wet_kg_thm=float(wet.get(i, 0.0) or 0.0), t_d=_num(t_day.get(i)))
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
        facts_version=r.get("facts_version"),
    )


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
                                     names, kinds)  # fmt: skip
        if res.get("ore_only") is not None:
            view["ore_flux_plan"] = plan_view(res["ore_only"], "Ore + flux objective (page 9)", names, kinds)
        if res.get("today") is not None:
            label = "Today's charged blend, modelled" + (
                f" ({inputs['today_window']})" if inputs.get("today_window") else ""
            )
            view["today_plan"] = plan_view(res["today"], label, names, kinds)
        view["value_of_limits"] = [
            ValueOfLimit(
                change=str(x.get("change", "")),
                saving_rs_thm=_num(x.get("saving_rs_thm")),
                binds_after=str(x.get("note") or ""),
            )
            for x in res.get("value_of_limits") or []
            if isinstance(x, Mapping)
        ]
    elif engine_mode == "cheapest":
        plan = res.get("best")
        if plan is not None:
            view["plan"] = plan_view(plan, f"Cheapest output on the Step 1 frontier, {plan['P_tday']:,.0f} t/d",
                                     names, kinds)  # fmt: skip
        if res.get("top") is not None:
            top = res["top"]
            view["highest_output_plan"] = plan_view(top, f"Highest output, {top['P_tday']:,.0f} t/d", names, kinds)
        view["frontier"] = _frontier(res.get("frontier"))
    elif engine_mode == "max":
        plan = res.get("result")
        if plan is not None:
            view["plan"] = plan_view(plan, f"Maximum production, {plan['P_tday']:,.0f} t/d", names, kinds)
    if view.get("plan") is not None and view["saving_rs_thm"] is not None:
        view["saving_lakh_rs_d"] = view["saving_rs_thm"] * view["plan"].production_t_d / 1e5
    view["critical_assumptions"], view["critical_assumptions_error"] = _critical(res.get("critical_assumptions"))
    if view.get("plan") is not None:
        view["solves"] = view["plan"].shaft_efficiency.network
    return OptimiserResult(**view)
