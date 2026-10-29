"""Blend Mix Optimiser API views (SV-3.5, #302): LP runs, the forecast ledgers and the run snapshots.

    runs        ``run_case`` (the ``bmo.run`` worker job): ``BmoRunParams`` -> ``services.bmo_case`` (page 9's
                composition and LP step) -> ``BmoRunResult``; ``run_result`` maps an ``LpRun`` with page 9's own
                figures (``services.bmo.blend_table``, ``display_total`` ...), so the plan equals the page's
    forecasts   ``si_forecast`` and ``charged_coke_forecast`` read the worker's ledgers (``services.forecasts``,
                SV-1.5) as page 9 does when it reads; they never issue
    snapshots   ``snapshot_list`` / ``snapshot_detail``: page 9's saved runs (``utils.bmo.snapshot_store``), without
                the frozen plant data

Quantities per day are t/day (wet unless named dry), rates kg/tHM, costs Rs/tHM, prices Rs/t, chemistry mass %,
basicities ratios; times are UTC unless named ``_ist``.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from services.events import ChargedCokeForecastData, SiForecastData
from services.jobs import BmoRunParams

if TYPE_CHECKING:
    from services import bmo_case

# The model modules (services.bmo, services.bmo_case, services.forecasts: the BMO data providers, the fuel and Si
# models) are imported where they are used, so the twin-api starts without them (about 2 s on the laptop).

PLANT_TZ = "Asia/Kolkata"
SNAPSHOT_ID = re.compile(r"^[A-Za-z0-9_-]{1,120}$")


# ============================================================================ views
class BmoOreRow(BaseModel):
    ore_id: str
    ore_name: str
    share_pct: float = Field(..., description="Share of the wet burden, %")
    quantity_t_d: float = Field(..., description="Wet quantity, t/day")
    dry_quantity_t_d: float | None = Field(None, description="Dry quantity, t/day")
    kg_per_charge: float | None = Field(None, description="Wet quantity per charge, kg")
    moisture_pct: float | None = Field(None, description="Moisture, mass %")
    fe_t_d: float | None = Field(None, description="Fe contributed, t/day")
    slag_t_d: float | None = Field(None, description="Slag contributed, t/day")
    price_rs_t: float | None = Field(None, description="Price of the wet ore, Rs/t")
    stock_t: float | None = Field(None, description="Stock, t wet")
    cost_rs_d: float | None = Field(None, description="Ore cost, Rs/day")


class BmoPlanView(BaseModel):
    """A blend as page 9's result cards show it."""

    feasible: bool
    violations: list[str] = Field(default_factory=list)
    total_qty_t_d: float | None = Field(None, description="Wet ore burden, t/day")
    ore_cost_rs_thm: float | None = Field(None, description="Ore cost, Rs/tHM")
    fuel_cost_rs_thm: float | None = Field(None, description="Fuel cost at the run's fuel prices, Rs/tHM")
    flux_cost_rs_thm: float | None = Field(None, description="Flux the LP added, Rs/tHM")
    total_cost_rs_thm: float | None = Field(None, description="Ore + fuel + added flux, Rs/tHM")
    fe_t_pct: float | None = Field(None, description="Fe(T) of the dry burden, mass %")
    fe_production_t_d: float | None = Field(None, description="Fe from the dry burden, t/day")
    slag_t_d: float | None = Field(None, description="Slag, t/day (model basis)")
    slag_rate_kg_thm: float | None = Field(None, description="Slag rate, kg/tHM (model basis)")
    slag_basicity: float | None = Field(None, description="CaO/SiO₂ of the slag, ratio")
    slag_t_basicity: float | None = Field(None, description="(CaO+MgO)/SiO₂ of the slag, ratio")
    slag_al2o3_pct: float | None = Field(None, description="Al₂O₃ in slag, mass %")
    slag_mgo_pct: float | None = Field(None, description="MgO in slag, mass %")
    slag_mgo_al2o3_ratio: float | None = Field(None, description="MgO/Al₂O₃ of the slag, ratio")
    coke_rate_kg_thm: float | None = Field(None, description="Coke rate of the blend, kg/tHM")
    nut_coke_rate_kg_thm: float | None = Field(None, description="Nut-coke rate, kg/tHM")
    pci_rate_kg_thm: float | None = Field(None, description="PCI rate, kg/tHM")
    fuel_rate_kg_thm: float | None = Field(None, description="Coke + nut coke + PCI, kg/tHM")
    coke_correction_kg_thm: float | None = Field(
        None, description="Blend correction of the coke rate against the reference burden, kg/tHM"
    )
    flux_t_d: dict[str, float] = Field(default_factory=dict, description="Flux the LP added, t/day by flux id")
    ores: list[BmoOreRow] = Field(default_factory=list, description="Ores of the run, largest share first")


class BmoRunTargets(BaseModel):
    production_t_d: float = Field(..., description="Hot-metal target, t/day")
    fe_t_d: float = Field(..., description="Fe required in hot metal, t/day")
    hm_fe_pct: float = Field(..., description="Fe in hot metal used for the Fe requirement, mass %")
    slag_rate_cap_kg_thm: float = Field(..., description="Slag cap, kg/tHM (plant basis)")
    slag_cap_t_d: float = Field(..., description="Slag cap the LP enforces, t/day (model basis)")
    model_to_plant_slag_factor: float = Field(..., description="plant slag = model slag × factor, ratio")
    basicity_min: float = Field(..., description="Lowest CaO/SiO₂, ratio")
    basicity_max: float = Field(..., description="Highest CaO/SiO₂, ratio")
    t_basicity_min: float | None = Field(None, description="Lowest (CaO+MgO)/SiO₂, ratio; null = not enforced")
    t_basicity_max: float | None = Field(None, description="Highest (CaO+MgO)/SiO₂, ratio; null = not enforced")
    al2o3_max_pct: float | None = Field(None, description="Highest Al₂O₃ in slag, mass %; null = not enforced")
    mgo_min_pct: float | None = Field(None, description="Lowest MgO in slag, mass %; null = not enforced")
    mgo_al2o3_ratio_min: float | None = Field(None, description="Lowest MgO/Al₂O₃, ratio; null = not enforced")
    max_burden_t_d: float | None = Field(None, description="Ore + flux charging ceiling, t/day; null = not enforced")
    max_charges_per_hour: float = Field(..., description="Charging rate limit, charges/h")
    charge_mass_t: float = Field(..., description="Burden per charge, t")
    nut_coke_rate_kg_thm: float = Field(..., description="Nut-coke rate held in the charges, kg/tHM")
    nut_coke_charge_moisture_pct: float = Field(..., description="Moisture added to the nut-coke charge, mass %")


class BmoFuelRates(BaseModel):
    coke_rate_kg_thm: float | None = Field(None, description="Coke rate read for the run, kg/tHM")
    nut_coke_rate_kg_thm: float | None = Field(None, description="Nut-coke rate of the run, kg/tHM")
    pci_rate_kg_thm: float | None = Field(None, description="PCI rate of the run, kg/tHM")
    coke_source: str | None = None
    nut_coke_source: str | None = None
    pci_source: str | None = None


class BmoAnchor(BaseModel):
    basis: Literal["data_driven", "energy_balance"]
    rate_kg_thm: float | None = Field(None, description="Coke level of the fuel cost, kg/tHM; null = the fuel model's")


class BmoRunResult(BaseModel):
    """The result of a ``bmo.run`` job: the LP plan at a fixed production, or why there is none."""

    status: Literal["ok", "infeasible", "refused"] = Field(
        ..., description="ok: a feasible plan; infeasible: the LP found none or broke a limit; refused: see refusal"
    )
    refusal: str | None = Field(None, description="Why page 9 would not run the optimiser with these inputs")
    errors: list[str] = Field(default_factory=list, description="LP errors")
    warnings: list[str] = Field(default_factory=list, description="Data and input warnings of the run")
    plan: BmoPlanView | None = None
    anchor: BmoAnchor | None = None
    targets: BmoRunTargets
    fuel_rates: BmoFuelRates
    coke_model: Literal["data_driven", "energy_balance"]
    chemistry_mode: str
    chemistry_window_days: int = Field(..., description="Chemistry window, d")
    live_si_pct: float | None = Field(None, description="Hot-metal Si of the Si forecast the run used, mass %")
    pellet_issues: list[str] = Field(default_factory=list)
    computed_at: datetime = Field(..., description="UTC")
    seconds: float = Field(..., description="Run time, s")


class BmoSiForecastView(BaseModel):
    mode: str = Field(..., description="Forecast mode of setting_bmo.yml: forecast, shadow or legacy (none)")
    current: bool = Field(..., description="The forecast is for the current origin (or the one before it)")
    note: str | None = Field(None, description="Why the forecast shown is not the current origin's")
    issuer: str = Field(..., description="Who issues: auto, worker or page (bmo.forecast_issuer)")
    worker_active: bool = Field(..., description="The worker attempted the forecast within 15 min")
    forecast: SiForecastData | None = None


class BmoChargedCokeView(BaseModel):
    current: bool = Field(..., description="The forecast is for the row of the latest build (or the one before)")
    note: str | None = Field(None, description="Why the forecast shown is not the latest build's")
    issuer: str = Field(..., description="Who issues: auto, worker or page (bmo.forecast_issuer)")
    worker_active: bool = Field(..., description="The worker attempted the forecast within 75 min")
    last_issued_row: datetime | None = Field(None, description="Latest row in the ledger when out of date (IST)")
    reasons: list[str] = Field(default_factory=list)
    forecast: ChargedCokeForecastData | None = None


class BmoSnapshotSummary(BaseModel):
    result: str | None = Field(None, description="Balanced Optimizer (LP) or Intensive Optimizer (DE)")
    production_t_d: float | None = Field(None, description="Hot-metal target, t/day")
    blend_pct: dict[str, float] = Field(default_factory=dict, description="Share by ore name, %")
    fuel_rate_kg_thm: float | None = Field(None, description="kg/tHM")
    coke_rate_kg_thm: float | None = Field(None, description="kg/tHM")
    nut_coke_kg_thm: float | None = Field(None, description="kg/tHM")
    pci_kg_thm: float | None = Field(None, description="kg/tHM")
    slag_rate_kg_thm: float | None = Field(None, description="kg/tHM (model basis)")
    slag_t_d: float | None = Field(None, description="t/day (model basis)")
    basicity: float | None = Field(None, description="CaO/SiO₂, ratio")
    t_basicity: float | None = Field(None, description="(CaO+MgO)/SiO₂, ratio")
    fe_pct: float | None = Field(None, description="Fe(T) of the dry burden, mass %")
    total_cost_rs_thm: float | None = Field(None, description="Ore + fuel + added flux, Rs/tHM")
    feasible: bool | None = None
    violations: int | None = Field(None, description="Limits broken, count")
    frozen_context: bool = Field(False, description="The snapshot holds the plant data the run read (replayable)")


class BmoSnapshotHeader(BaseModel):
    id: str
    created_at: datetime | None = Field(None, description="When the snapshot was saved (IST, with offset)")
    source: str = Field("", description="Blend Mix Optimiser or Sandbox")
    label: str = ""
    summary: BmoSnapshotSummary


class BmoSnapshotList(BaseModel):
    snapshots: list[BmoSnapshotHeader] = Field(default_factory=list, description="Newest first")
    unreadable: list[str] = Field(default_factory=list, description="Snapshot files that could not be read")


class BmoSnapshotRun(BaseModel):
    at: str | None = Field(None, description="When the recorded run was made (IST, with offset)")
    pinned: bool = Field(False, description="The inputs and plant data are those of the run, not of the save")
    inputs_changed_after_run: bool = False


class BmoSnapshotDetail(BmoSnapshotHeader):
    schema_version: str = Field("", description="bmo-snapshot/<n>")
    run: BmoSnapshotRun
    provenance: dict[str, Any] = Field(default_factory=dict, description="Code version of the run")
    recommended: str | None = Field(None, description="The result page 9 recommends (DE unless it fell back)")
    lp: BmoPlanView | None = Field(None, description="Balanced Optimizer (LP) result")
    de: BmoPlanView | None = Field(None, description="Intensive Optimizer (DE) result")
    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Widget values and input tables (tables as row lists) by page key"
    )
    not_restorable: list[str] = Field(default_factory=list)
    model_check: dict[str, Any] | None = Field(None, description="Recorded fuel-model check of the run")


# ============================================================================ helpers
def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def plain(value: Any) -> Any:
    """JSON-ready: tables as row lists, times as ISO text, NaN and infinity as null."""
    if isinstance(value, pd.DataFrame):
        return [plain(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def plan_view(blend: Any, *, ores: list[Any] | None = None, charge_mass_mt: float = 26.4,
              names: Mapping[str, str] | None = None) -> BmoPlanView:  # fmt: skip
    """A ``BlendEvaluation`` (or its fields as a mapping) with page 9's cost figures. With ``ores`` the ore rows are
    page 9's result table (``services.bmo.blend_table``); else shares and tonnes by id, named by ``names``."""
    from services import bmo

    if isinstance(blend, Mapping):
        fields = dict(blend)
        fields.setdefault("diagnostics", {})
        fields["diagnostics"] = dict(fields.get("diagnostics") or {})
        fields.setdefault("objective_rs_per_thm", float("nan"))
        fields.setdefault("fuel_cost_per_thm_rs", float("nan"))
        blend = SimpleNamespace(**fields)
    diagnostics = blend.diagnostics or {}
    rates = diagnostics.get("fuel_rate_estimate") or {}
    if ores is not None:
        by_name = {o.display_name: o.ore_id for o in ores}
        table = bmo.blend_table(blend, ores, charge_mass_mt=charge_mass_mt)
        rows = [
            BmoOreRow(
                ore_id=by_name.get(r["ore_name"], r["ore_name"]),
                ore_name=r["ore_name"],
                share_pct=r["share_pct"],
                quantity_t_d=r["quantity_mt"],
                dry_quantity_t_d=_num(r["dry_quantity_mt"]),
                kg_per_charge=_num(r["kg_per_charge"]),
                moisture_pct=_num(r["moisture_pct"]),
                fe_t_d=_num(r["fe_contribution_mt"]),
                slag_t_d=_num(r["slag_contribution_mt"]),
                price_rs_t=_num(r["price_rs_per_mt"]),
                stock_t=_num(r["stock_mt"]),
                cost_rs_d=_num(r["ore_cost_rs"]),
            )
            for r in table.to_dict(orient="records")
        ]
    else:
        quantities = getattr(blend, "quantities_mt", None) or {}
        shares = getattr(blend, "shares_pct", None) or {}
        rows = sorted(
            (
                BmoOreRow(
                    ore_id=str(ore_id),
                    ore_name=(names or {}).get(str(ore_id), str(ore_id)),
                    share_pct=_num(share) or 0.0,
                    quantity_t_d=_num(quantities.get(ore_id)) or 0.0,
                )
                for ore_id, share in shares.items()
            ),
            key=lambda r: -r.share_pct,
        )
    total = _num(bmo.display_total(blend))
    fuel = _num(bmo.display_fuel(blend))
    return BmoPlanView(
        feasible=bool(getattr(blend, "feasible", False)),
        violations=[str(v) for v in getattr(blend, "violations", None) or []],
        total_qty_t_d=_num(getattr(blend, "total_qty_mt", None)),
        ore_cost_rs_thm=_num(getattr(blend, "ore_cost_per_thm_rs", None)),
        fuel_cost_rs_thm=fuel,
        flux_cost_rs_thm=_num(bmo.flux_cost(blend)),
        total_cost_rs_thm=total,
        fe_t_pct=_num(getattr(blend, "fe_t_pct", None)),
        fe_production_t_d=_num(getattr(blend, "fe_production_mt", None)),
        slag_t_d=_num(getattr(blend, "slag_mt", None)),
        slag_rate_kg_thm=_num(getattr(blend, "slag_rate_kg_per_thm", None)),
        slag_basicity=_num(getattr(blend, "slag_basicity", None)),
        slag_t_basicity=_num(getattr(blend, "slag_t_basicity", None)),
        slag_al2o3_pct=_num(getattr(blend, "slag_al2o3_pct", None)),
        slag_mgo_pct=_num(getattr(blend, "slag_mgo_pct", None)),
        slag_mgo_al2o3_ratio=_num(getattr(blend, "slag_mgo_al2o3_ratio", None)),
        coke_rate_kg_thm=_num(rates.get("coke_rate_kg_thm")),
        nut_coke_rate_kg_thm=_num(rates.get("nut_coke_rate_kg_thm")),
        pci_rate_kg_thm=_num(rates.get("pci_rate_kg_thm")),
        fuel_rate_kg_thm=_num(rates.get("total_fuel_rate_kg_thm")),
        coke_correction_kg_thm=_num(diagnostics.get("coke_correction_delta_kg_thm")),
        flux_t_d={
            str(k): float(v) for k, v in (diagnostics.get("lp_flux_quantities_mt") or {}).items() if _num(v) is not None
        },
        ores=rows,
    )


# ============================================================================ runs
def inputs_from_params(params: BmoRunParams) -> bmo_case.CaseInputs:
    """The run's ``CaseInputs``: the request's values, None for page 9's defaults."""
    from services import bmo_case

    scalar = {
        name: getattr(params, name)
        for name in (
            "target_production_mt",
            "target_slag_rate_kg_per_thm",
            "target_slag_basicity_min",
            "target_slag_basicity_max",
            "target_slag_t_basicity_min",
            "target_slag_t_basicity_max",
            "target_slag_al2o3_max_pct",
            "target_slag_mgo_min_pct",
            "target_slag_mgo_al2o3_ratio_min",
            "max_charges_per_hour",
            "charge_mass_mt",
            "burden_capacity_enabled",
            "chemistry_mode",
            "chemistry_window_days",
            "coke_model",
            "blend_response_kg_coke_per_kg_slag",
            "manual_blend_lookback_hours",
            "pci_override_kg_thm",
            "nut_coke_override_kg_thm",
            "pellet_inputs_confirmed",
            "selected_ore_ids",
        )
    }
    chem = params.hm_chemistry.model_dump(exclude_none=True) if params.hm_chemistry else None
    prices = params.fuel_prices.model_dump(exclude_none=True) if params.fuel_prices else None
    overrides = (
        {ore_id: o.model_dump(exclude_none=True) for ore_id, o in params.ore_overrides.items()}
        if params.ore_overrides
        else None
    )
    return bmo_case.CaseInputs(
        **scalar,
        force_data_driven_prediction=params.force_data_driven,
        hm_chemistry=chem or None,
        fuel_prices_rs_per_mt=prices or None,
        ore_overrides=overrides,
    )


def run_result(run: bmo_case.LpRun, *, computed_at: datetime, seconds: float) -> BmoRunResult:
    """The API view of an LP run: page 9's plan, anchor, targets and fuel rates."""
    composed, case = run.composed, run.composed.case
    lp = run.lp_result
    plan = plan_view(lp, ores=case.ores, charge_mass_mt=case.charge_mass_mt) if lp is not None else None
    if run.refusal:
        status = "refused"
    elif plan is not None and plan.feasible:
        status = "ok"
    else:
        status = "infeasible"
    slag, rates = composed.slag, composed.fuel_rates
    return BmoRunResult(
        status=status,
        refusal=run.refusal,
        errors=[str(e) for e in (run.balanced.lp_errors if run.balanced is not None else [])],
        warnings=list(dict.fromkeys([*composed.warnings, *run.warnings])),
        plan=plan,
        anchor=BmoAnchor(basis=run.anchor.basis, rate_kg_thm=_num(run.anchor.rate_kg_thm)) if run.anchor else None,
        targets=BmoRunTargets(
            production_t_d=case.target_production_mt,
            fe_t_d=case.target_fe_mt,
            hm_fe_pct=composed.hm_fe_pct,
            slag_rate_cap_kg_thm=slag.slag_rate_kg_per_thm,
            slag_cap_t_d=slag.slag_qty_mt,
            model_to_plant_slag_factor=slag.model_to_plant_factor,
            basicity_min=slag.basicity_min,
            basicity_max=slag.basicity_max,
            t_basicity_min=slag.t_basicity_min,
            t_basicity_max=slag.t_basicity_max,
            al2o3_max_pct=slag.al2o3_max_pct,
            mgo_min_pct=slag.mgo_min_pct,
            mgo_al2o3_ratio_min=slag.mgo_al2o3_ratio_min,
            max_burden_t_d=_num(case.max_burden_qty_mt),
            max_charges_per_hour=composed.max_charges_per_hour,
            charge_mass_t=case.charge_mass_mt,
            nut_coke_rate_kg_thm=composed.nut_coke.rate_kg_per_thm,
            nut_coke_charge_moisture_pct=composed.nut_coke.charge_moisture_pct,
        ),
        fuel_rates=BmoFuelRates(
            coke_rate_kg_thm=_num(rates.get("coke_rate_kg_thm")),
            nut_coke_rate_kg_thm=_num(rates.get("nut_coke_rate_kg_thm")),
            pci_rate_kg_thm=_num(rates.get("pci_rate_kg_thm")),
            coke_source=rates.get("coke_source"),
            nut_coke_source=rates.get("nut_coke_source"),
            pci_source=rates.get("pci_source"),
        ),
        coke_model=case.fuel_rate_anchor_basis,
        chemistry_mode=composed.chemistry_mode,
        chemistry_window_days=composed.chemistry_window_days,
        live_si_pct=_num(case.live_si_pct),
        pellet_issues=[str(i) for i in composed.pellet_issues],
        computed_at=computed_at,
        seconds=round(seconds, 2),
    )


def run_case(params: Mapping[str, Any], *, sources: bmo_case.CaseSources | None = None) -> dict[str, Any]:
    """One ``bmo.run``: compose page 9's case from the live sources (or ``sources``), run the LP, return the result
    as JSON."""
    from services import bmo_case

    started = time.perf_counter()
    computed_at = datetime.now(UTC)
    request = BmoRunParams.model_validate(dict(params))
    src = sources if sources is not None else bmo_case.live_sources()
    inputs = inputs_from_params(request)
    composed = bmo_case.compose_case(src, inputs)
    run = bmo_case.run_lp(src, composed, inputs)
    return run_result(run, computed_at=computed_at, seconds=time.perf_counter() - started).model_dump(mode="json")


# ============================================================================ forecasts (ledger reads)
def _ist(value: Any) -> pd.Timestamp | None:
    """A ledger time (naive = IST) with its offset."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(PLANT_TZ) if stamp.tzinfo is None else stamp.tz_convert(PLANT_TZ)


def si_forecast(cfg: Mapping[str, Any] | None = None, *, now: Any = None) -> BmoSiForecastView:
    """The hot-metal Si forecast page 9 shows when it reads the ledger (``services.forecasts.si_for_page``)."""
    from services import forecasts as forecast_service
    from utils.bmo.si_forecast.deployments import active_bundle
    from utils.bmo.si_forecast.ledger import SiliconForecastLedger

    cfg = forecast_service.bmo_config() if cfg is None else cfg
    paths = forecast_service.si_paths(cfg)
    who = forecast_service.issuer(cfg)
    active = forecast_service.worker_active(paths.storage_dir, "si", now)
    if paths.mode == "legacy":
        note = "The production HM Si forecast is off (bmo.si_forecast.mode legacy)."
        return BmoSiForecastView(mode=paths.mode, current=False, note=note, issuer=who, worker_active=active)
    bundle = active_bundle(paths.storage_dir, paths.default_bundle)
    view = forecast_service.si_for_page(paths.storage_dir, bundle, now=now)
    forecast = None
    record = view.record
    origin = (record or {}).get("origin_at")
    if record is not None and origin is not None:
        recorded = SiliconForecastLedger(paths.storage_dir).get(origin) is not None
        attempt = {"key": origin, "at": record.get("issued_at") or origin, "written": recorded}
        forecast = forecast_service.si_event(attempt, record)
    return BmoSiForecastView(
        mode=paths.mode, current=view.current, note=view.note, issuer=who, worker_active=active, forecast=forecast
    )


def charged_coke_forecast(cfg: Mapping[str, Any] | None = None, *, now: Any = None) -> BmoChargedCokeView:
    """The charged-coke forecast page 9 uses when it reads the ledger (``services.forecasts.charged_for_page``)."""
    from services import forecasts as forecast_service

    cfg = forecast_service.bmo_config() if cfg is None else cfg
    storage = forecast_service.charged_paths(cfg).storage_dir
    view = forecast_service.charged_for_page(storage, now=now)
    record = view.record or {}
    last = record.get("last_issued_row")
    return BmoChargedCokeView(
        current=view.current,
        note=view.note,
        issuer=forecast_service.issuer(cfg),
        worker_active=forecast_service.worker_active(storage, "charged_coke", now),
        last_issued_row=_ist(last),
        reasons=[str(r) for r in record.get("reasons") or []],
        forecast=forecast_service.charged_event(record) if record.get("data_row") is not None else None,
    )


# ============================================================================ snapshots
def _created(value: Any) -> datetime | None:
    try:
        return pd.Timestamp(value).to_pydatetime() if value else None
    except (TypeError, ValueError):
        return None


def _summary(summary: Mapping[str, Any]) -> BmoSnapshotSummary:
    s = dict(summary or {})
    return BmoSnapshotSummary(
        result=s.get("result") or None,
        production_t_d=_num(s.get("production_mt")),
        blend_pct={str(k): float(v) for k, v in (s.get("blend_pct") or {}).items() if _num(v) is not None},
        fuel_rate_kg_thm=_num(s.get("fuel_rate_kg_thm")),
        coke_rate_kg_thm=_num(s.get("coke_rate_kg_thm")),
        nut_coke_kg_thm=_num(s.get("nut_coke_kg_thm")),
        pci_kg_thm=_num(s.get("pci_kg_thm")),
        slag_rate_kg_thm=_num(s.get("slag_rate_kg_thm")),
        slag_t_d=_num(s.get("slag_mt")),
        basicity=_num(s.get("basicity_b2")),
        t_basicity=_num(s.get("t_basicity")),
        fe_pct=_num(s.get("fe_pct")),
        total_cost_rs_thm=_num(s.get("total_cost_rs_thm")),
        feasible=s.get("feasible"),
        violations=s.get("violations"),
        frozen_context=bool(s.get("frozen_context")),
    )


def snapshot_list(*, limit: int = 50, directory: Path | None = None) -> BmoSnapshotList:
    """Page 9's saved snapshots, newest first (headers only)."""
    from utils.bmo import snapshot_store

    rows, broken = snapshot_store.list_snapshots(directory)
    return BmoSnapshotList(
        snapshots=[
            BmoSnapshotHeader(
                id=str(r["id"]),
                created_at=_created(r.get("created_at")),
                source=str(r.get("source") or ""),
                label=str(r.get("label") or ""),
                summary=_summary(r.get("summary") or {}),
            )
            for r in rows[:limit]
        ],
        unreadable=broken,
    )


def snapshot_detail(snapshot_id: str, *, directory: Path | None = None) -> BmoSnapshotDetail | None:
    """One snapshot without its frozen plant data; None when there is none with this id (or the id is not one)."""
    from utils.bmo import snapshot as snap
    from utils.bmo import snapshot_store

    if not SNAPSHOT_ID.match(snapshot_id or ""):
        return None
    try:
        doc = snapshot_store.load(snapshot_id, directory)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    names = snap.ore_names(doc)
    results = doc.get("results") or {}

    def result(key: str) -> BmoPlanView | None:
        fields = snap.decode(results.get(key), typed=False) if results.get(key) is not None else None
        return plan_view(fields, names=names) if isinstance(fields, Mapping) and fields else None

    label, _fields = snap.recommended_result(doc)
    run = doc.get("run") or {}
    frozen = doc.get("frozen") or {}
    return BmoSnapshotDetail(
        id=str(doc.get("id") or snapshot_id),
        created_at=_created(doc.get("created_at")),
        source=str(doc.get("source") or ""),
        label=str(doc.get("label") or ""),
        summary=_summary(doc.get("summary") or snap.summarise(doc)),
        schema_version=str(doc.get("schema") or ""),
        run=BmoSnapshotRun(
            at=run.get("at"),
            pinned=bool(run.get("pinned")),
            inputs_changed_after_run=bool(run.get("inputs_changed_after_run")),
        ),
        provenance=plain(doc.get("provenance") or {}),
        recommended=label or None,
        lp=result("lp_result"),
        de=result("de_result"),
        inputs={str(k): plain(snap.decode(v, typed=False)) for k, v in (doc.get("inputs") or {}).items()},
        not_restorable=[str(s) for s in doc.get("not_restorable") or []],
        model_check=plain(frozen["model_check"]) if isinstance(frozen.get("model_check"), Mapping) else None,
    )


__all__ = [
    "BmoAnchor",
    "BmoChargedCokeView",
    "BmoFuelRates",
    "BmoOreRow",
    "BmoPlanView",
    "BmoRunResult",
    "BmoRunTargets",
    "BmoSiForecastView",
    "BmoSnapshotDetail",
    "BmoSnapshotHeader",
    "BmoSnapshotList",
    "BmoSnapshotSummary",
    "charged_coke_forecast",
    "inputs_from_params",
    "plain",
    "plan_view",
    "run_case",
    "run_result",
    "si_forecast",
    "snapshot_detail",
    "snapshot_list",
]
