"""Blend Mix Optimiser (page 9) orchestration without Streamlit (SV-3.4, #301).

Page 9 (``custom_pages/9_Blend_Optimizer.py``) used to compose every result inside its script. The composition now
lives here and the page only renders what these functions return, so the twin-api and the worker can build the same
plan, tables and panels later without re-implementing any of it:

    LP plan          ``BlendCase`` (one run's inputs), ``run_refusal``, ``coke_anchor``, ``correction_reference``,
                     ``run_balanced`` (LP solve, fuel re-evaluation, feasibility re-check), ``run_intensive`` (DE) and
                     ``fall_back_to_lp`` (the DE guardrail)
    result tables    ``blend_table`` (per-ore result table), ``manual_seed`` / ``evaluate_manual_blend`` /
                     ``comparison_view`` (manual blend vs optimiser blends), ``transition_rows`` (path from the current
                     blend), ``de_exploration`` (search cloud and its 100 cheapest alternatives),
                     ``control_basis`` / ``recommend_process`` / ``coke_source_rows`` / ``control_rows`` (process
                     parameters for a blend), ``live_si_result_tile`` (the HM Si tile of the result cards)
    charged coke     ``charged_prediction`` (the ledger record or the page's own issue, and the BMO prediction)
    HM Si            ``si_forecast_for_page`` (ledger read or page issue), ``replay_si_history`` (the 7-day trend)
    snapshots        ``take_snapshot`` (capture and save), ``finalize_run``

Every data source is an argument: the provider, the model service, the hourly dataset, the clock (``now``) and the
live process tags are passed by the caller. Page 9 passes its own cached loaders, and the Sandbox (``utils/bmo/
replay.py``) and the golden runs (``tests/bmo_golden.py``) substitute those page names, so a replay never reaches a
live source through this module. Moved from the page unchanged (golden records of five plant days,
``tests/test_bmo_golden.py``).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services import forecasts as forecast_service
from utils.bmo import (
    CokeCorrectionDrivers,
    CokeCorrectionReference,
    CokeCorrectionSettings,
    DustInput,
    FluxInput,
    FuelAshInput,
    FuelUnitCostModelService,
    OreInput,
    SlagBalanceSettings,
    build_reference,
    compute_burden_oxygen_kg_per_thm,
    compute_charging_requirements,
    compute_flux_co2_kg_per_thm,
    run_lp_baseline,
    run_nonlinear_optimizer,
)
from utils.bmo.calculations import evaluate_blend, scale_ore_quantities_to_hot_metal
from utils.bmo.charged_coke.bmo import ChargedCokePrediction, charged_coke_prediction
from utils.bmo.constraints import CHARGING_HOURS_PER_DAY, check_blend_constraints
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.shift_production import production_change_pct
from utils.bmo.si_forecast.service import forecast_horizons
from utils.bmo.types import BlendEvaluation

log = logging.getLogger(__name__)

#: (model service, process context, history, bundle status, warnings): what the fuel model reads for one run.
FuelContext = tuple[FuelUnitCostModelService, dict[str, float], pd.DataFrame, dict[str, Any], list[str]]


# ============================================================================ one run's inputs
@dataclass
class BlendCase:
    """The inputs of one optimiser run, as page 9 derives them from its editors and plant data.

    Quantities are per day: ``target_fe_mt`` and ``target_production_mt`` in t/day of Fe and hot metal,
    ``target_slag_qty_mt`` and ``max_burden_qty_mt`` in t/day; rates in kg/tHM; shares and chemistry in %.
    """

    ores: list[OreInput]
    target_fe_mt: float  # t/day of Fe in hot metal
    target_production_mt: float  # t/day of hot metal
    target_slag_qty_mt: float  # t/day of slag (model basis)
    feo_in_slag_pct: float  # % FeO in slag
    target_slag_basicity_min: float  # CaO/SiO2
    target_slag_basicity_max: float  # CaO/SiO2
    target_slag_t_basicity_min: float | None  # (CaO+MgO)/SiO2; None = not enforced
    target_slag_t_basicity_max: float | None  # (CaO+MgO)/SiO2; None = not enforced
    target_slag_al2o3_max_pct: float  # % Al2O3 in slag
    target_slag_mgo_min_pct: float | None  # % MgO in slag; None = not enforced
    target_slag_mgo_al2o3_ratio_min: float  # MgO/Al2O3
    max_burden_qty_mt: float | None  # t/day charging ceiling; None = not enforced
    fuel_ash_inputs: list[FuelAshInput]
    flux_inputs: list[FluxInput]
    dust_inputs: list[DustInput]
    slag_balance_settings: SlagBalanceSettings
    charge_mass_mt: float  # t per charge
    coke_correction_settings: CokeCorrectionSettings
    observed_slag_rate_kg_per_thm: float  # kg/tHM, DPR
    live_si_pct: float | None  # % Si, the furnace forecast
    fuel_rate_anchor_basis: str  # "data_driven" or "energy_balance"
    lookback_hours: float  # h of the last-shift blend

    @property
    def use_charged_engine(self) -> bool:
        return self.fuel_rate_anchor_basis == "data_driven"

    def limits(self) -> dict[str, Any]:
        """The constraint keywords shared by the LP, DE and the feasibility check."""

        return {
            "target_production_mt": self.target_fe_mt,
            "target_slag_qty_mt": self.target_slag_qty_mt,
            "target_slag_basicity_min": self.target_slag_basicity_min,
            "target_slag_basicity_max": self.target_slag_basicity_max,
            "target_slag_t_basicity_min": self.target_slag_t_basicity_min,
            "target_slag_t_basicity_max": self.target_slag_t_basicity_max,
            "target_slag_al2o3_max_pct": self.target_slag_al2o3_max_pct,
            "target_slag_mgo_min_pct": self.target_slag_mgo_min_pct,
            "target_slag_mgo_al2o3_ratio_min": self.target_slag_mgo_al2o3_ratio_min,
            "max_burden_qty_mt": self.max_burden_qty_mt,
        }

    def lp_kwargs(self) -> dict[str, Any]:
        """The LP keywords without the coke correction (the transition ladder solves with these)."""

        return {
            "target_production_mt": self.target_fe_mt,
            "target_slag_qty_mt": self.target_slag_qty_mt,
            "feo_in_slag_pct": self.feo_in_slag_pct,
            "target_slag_basicity_min": self.target_slag_basicity_min,
            "target_slag_basicity_max": self.target_slag_basicity_max,
            "target_slag_t_basicity_min": self.target_slag_t_basicity_min,
            "target_slag_t_basicity_max": self.target_slag_t_basicity_max,
            "target_slag_al2o3_max_pct": self.target_slag_al2o3_max_pct,
            "target_slag_mgo_min_pct": self.target_slag_mgo_min_pct,
            "target_slag_mgo_al2o3_ratio_min": self.target_slag_mgo_al2o3_ratio_min,
            "max_burden_qty_mt": self.max_burden_qty_mt,
            "fuel_ash_inputs": self.fuel_ash_inputs,
            "flux_inputs": self.flux_inputs,
            "dust_inputs": self.dust_inputs,
            "slag_balance_settings": self.slag_balance_settings,
            "hot_metal_target_mt": self.target_production_mt,
            "charge_mass_mt": self.charge_mass_mt,
        }

    @property
    def slag_rate_cap_kg_per_thm(self) -> float | None:
        """kg/tHM: the slag cap of the run on the hot-metal basis."""

        return self.target_slag_qty_mt / self.target_production_mt * 1000.0 if self.target_production_mt else None


# ============================================================================ building blocks
def load_fuel_prediction_context(provider: Any, model_service: FuelUnitCostModelService) -> FuelContext:
    """
    Load process, history, and model inputs required for fuel inference.

    Fuel prediction needs the latest process context, recent history for lagged
    features, and the cached model service. Keeping this as a small helper lets
    LP post-solve inference and DE candidate optimization share the same loaded
    inputs during one run.

    Args:
         - provider: EvonithBmoContextProvider - Source for process context and history.
         - model_service: FuelUnitCostModelService - The (cached) fuel-cost model service.

    Returns:
         - return FuelContext - Model service, process context, history, bundle status, and warnings.
    """

    max_lag_hours = model_service.get_max_lag_steps()
    history_df, history_warnings = provider.get_history_frame(online_lag_hours=max_lag_hours)
    process_context, process_warnings = provider.get_process_context(history_df=history_df)
    bundle_status = model_service.get_bundle_status()
    warnings = [*process_warnings, *history_warnings]
    return model_service, process_context, history_df, bundle_status, warnings


def live_si_result_tile(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build the compact optimiser tile from the independent furnace forecast."""

    if not record:
        return None
    usable = [row for row in forecast_horizons(record) if row.get("si_pct") is not None]
    if not usable:
        return None
    headline = usable[0]
    horizon_minutes = int(headline.get("horizon_minutes") or 0)
    label = "HM Si if sampled now" if horizon_minutes == 0 else f"HM Si forecast (+{horizon_minutes / 60:g} h)"
    return {
        "si_pct": float(headline["si_pct"]),
        "horizon_minutes": horizon_minutes,
        "metric_label": label,
        "target_at": headline.get("target_start"),
        "issued_at": record.get("issued_at") or record.get("origin_at"),
        "last_cast_at": record.get("last_sample_at"),
        "last_cast_si_pct": record.get("last_sample_si_pct"),
        "model_version": record.get("model_version") or record.get("model_id"),
        "warnings": list(record.get("reasons") or []) if str(record.get("status") or "ok") != "ok" else [],
    }


def effective_pci_kg_thm(snapshot: Mapping[str, Any], override_kg_thm: float | None) -> float:
    """The PCI the WHOLE page must use: the operator's override, else the tag (kg/tHM).

    There is one of these because there used to be three. PCI was read straight
    off the live tag in the process-recommendation panel and in the energy
    anchor, and separately off the Fuel Ash table everywhere else. An override
    that did not reach all three would move the slag and the cost while leaving
    the coke rate solved against the old figure - the worst possible outcome,
    because every number on screen would still look self-consistent.

    ``override_kg_thm`` is the operator's PCI override when it is switched on, else None.
    """

    if override_kg_thm is not None:
        return float(override_kg_thm)
    return float(snapshot.get("coal_rate_actual_value") or 0.0)


def target_quantities_from_shares(
    shares_pct: dict[str, float],
    ores: list[OreInput],
    target_fe_mt: float,
    *,
    target_hot_metal_mt: float | None = None,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    charge_mass_mt: float = 26.4,
) -> tuple[dict[str, float], float, list[str]]:
    """
    Scale operator-edited burden shares into wet quantities for the target HM.

    The manual-vs-optimizer comparison lets the operator type any share split.
    Shares are normalized to 100%, the blend Fe per wet MT is derived from each
    ore's dry Fe, and the total wet quantity is sized so dry Fe equals the same
    target Fe used by the optimizer. This keeps the manual blend on the same Fe
    basis as the LP/DE result so the cost and slag comparison is apples-to-apples.

    Args:
         - shares_pct: dict[str, float] - Operator share percentages keyed by ore id.
         - ores: list[OreInput] - Candidate ores (chemistry supplies Fe and moisture).
         - target_fe_mt: float - Required dry Fe production in MT.

    Returns:
         - return tuple[dict[str, float], float, list[str]] - Quantities by ore id,
           the normalized share total before scaling, and any warnings.
    """

    total_share = sum(max(0.0, float(v)) for v in shares_pct.values())
    if total_share <= 0:
        return (
            {},
            0.0,
            ["Manual shares sum to zero; enter at least one positive share."],
        )
    shares = {oid: max(0.0, float(v)) / total_share for oid, v in shares_pct.items()}
    fe_per_blend_mt = 0.0
    for ore in ores:
        dry_fraction = max(0.0, 1.0 - float(ore.chemistry.moisture_pct) / 100.0)
        fe_fraction = max(0.0, float(ore.chemistry.fe_t_pct) / 100.0)
        fe_per_blend_mt += shares.get(ore.ore_id, 0.0) * dry_fraction * fe_fraction
    if fe_per_blend_mt <= 0:
        return (
            {},
            total_share,
            ["Manual blend Fe% is unavailable; cannot scale to target HM."],
        )
    total_qty = float(target_fe_mt) / fe_per_blend_mt
    quantities = {oid: sh * total_qty for oid, sh in shares.items()}
    if target_hot_metal_mt and slag_balance_settings and slag_balance_settings.enabled:
        quantities = scale_ore_quantities_to_hot_metal(
            ores=ores,
            reference_quantities_mt=quantities,
            target_hot_metal_mt=target_hot_metal_mt,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            charge_mass_mt=charge_mass_mt,
        )
    return quantities, total_share, []


def plant_window_text(start: Any, end: Any) -> str:
    """'29 Sep, 15:30-20:30 IST' for a UTC window; the raw values if unparseable."""

    try:
        first, last = pd.Timestamp(start), pd.Timestamp(end)
    except (TypeError, ValueError):
        return f"{start} to {end}"
    if pd.isna(first) or pd.isna(last):
        return f"{start} to {end}"
    first, last = (
        (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp).tz_convert("Asia/Kolkata")
        for stamp in (first, last)
    )
    if first.date() == last.date():
        return f"{first:%d %b}, {first:%H:%M}-{last:%H:%M} IST"
    return f"{first:%d %b %H:%M} to {last:%d %b %H:%M} IST"


def burden_slag_rate(case: BlendCase, ores: list[OreInput], quantities_mt: Mapping[str, float] | None) -> float | None:
    """Modelled slag rate of a burden on the BMO basis (kg/tHM), or None."""

    if not ores or not quantities_mt:
        return None
    try:
        blend = evaluate_blend(
            ores=ores,
            quantities_mt={str(k): float(v) for k, v in quantities_mt.items()},
            feo_in_slag_pct=case.feo_in_slag_pct,
            fuel_ash_inputs=case.fuel_ash_inputs,
            flux_inputs=case.flux_inputs,
            dust_inputs=case.dust_inputs,
            slag_balance_settings=case.slag_balance_settings,
            hot_metal_target_mt=case.target_production_mt,
            charge_mass_mt=case.charge_mass_mt,
        )
    except Exception:  # noqa: BLE001 - an unknown reference disables the term
        log.exception("Could not evaluate the current burden's slag rate")
        return None
    value = float(blend.slag_rate_kg_per_thm or 0.0)
    return value if value > 0 else None


def build_coke_correction_reference(
    *,
    settings: CokeCorrectionSettings,
    observed_slag_rate_kg_per_thm: float | None,
    flux_inputs: list[FluxInput] | None,
    hot_metal_target_mt: float,
    process_context: Mapping[str, Any] | None = None,
    current_quantities_mt: Mapping[str, float] | None = None,
    ores: list[OreInput] | None = None,
    current_si_pct: float | None = None,
    current_slag_rate_kg_per_thm: float | None = None,
) -> CokeCorrectionReference:
    """
    Resolve the operating point the coke correction is anchored to.

    Built once per run and passed unchanged to the LP, to DE, and to every DE
    candidate, so the corrected objective is one fixed function rather than one
    that shifts as reference data appears and disappears.

    The two terms enabled by default need no solved blend: slag comes from the
    observed DPR rate and flux CO2 from the flux rows as the operator has them
    charged. That avoids a circular dependency where the reference would need
    the LP result the LP needs the reference to produce.

    Args:
         - settings: CokeCorrectionSettings - Parsed correction settings.
         - observed_slag_rate_kg_per_thm: float | None - Observed DPR slag rate.
         - flux_inputs: list[FluxInput] | None - Flux rows as currently charged.
         - hot_metal_target_mt: float - HM basis for the per-THM drivers.
         - process_context: Mapping[str, Any] | None - Recent process values.
         - current_quantities_mt: Mapping[str, float] | None - Current burden, when known.
         - ores: list[OreInput] | None - Ores backing ``current_quantities_mt``.
         - current_si_pct: float | None - Si model output for the current burden.

    Returns:
         - return CokeCorrectionReference - Reference operating point.
    """

    burden_oxygen = None
    if ores and current_quantities_mt:
        burden_oxygen = compute_burden_oxygen_kg_per_thm(
            ores=ores,
            quantities_mt=current_quantities_mt,
            hot_metal_mt=hot_metal_target_mt,
        )

    current_drivers = CokeCorrectionDrivers(
        # Only a model_current slag reference reads this (the charged engine's
        # response); the physics correction measures slag from observed DPR.
        slag_rate_kg_per_thm=current_slag_rate_kg_per_thm,
        flux_co2_kg_per_thm=compute_flux_co2_kg_per_thm(flux_inputs=flux_inputs, hot_metal_mt=hot_metal_target_mt),
        burden_oxygen_kg_per_thm=burden_oxygen,
        hot_metal_si_pct=current_si_pct,
    )
    return build_reference(
        settings=settings,
        observed_slag_rate_kg_per_thm=observed_slag_rate_kg_per_thm,
        current_drivers=current_drivers,
        process_context=process_context,
    )


def current_burden_quantities(
    *,
    provider: Any,
    ores: list[OreInput],
    target_fe_mt: float,
    hot_metal_mt: float,
    fuel_ash_inputs: list[FuelAshInput],
    flux_inputs: list[FluxInput],
    dust_inputs: list[DustInput],
    slag_balance_settings: SlagBalanceSettings,
    charge_mass_mt: float,
    lookback_hours: float,
) -> dict[str, float]:
    """What the plant charged last shift, scaled onto the target HM basis (t/day by ore id).

    Runs the last-shift SHARES through the same Fe/material closure the
    comparison tab uses, rather than scaling the raw shift tonnage. The shift's
    own production is not the target production, so raw tonnes would put the
    burden and the hot-metal basis on different footings and the energy balance
    would then solve for a coke rate that belongs to neither.

    An ore that was not charged stays out. A zero here is the plant's decision,
    not a gap to be filled - see the note in the manual-blend seeding.
    """

    try:
        snapshot = provider.get_recent_manual_blend_snapshot(ores, lookback_hours=lookback_hours)
    except Exception as exc:  # noqa: BLE001 - the anchor is optional
        log.warning("Could not read the last-shift burden: %s", exc)
        return {}

    shares = {
        str(row.get("ore_id")): float(row.get("share_pct", 0.0) or 0.0)
        for row in snapshot.get("rows", [])
        if float(row.get("share_pct", 0.0) or 0.0) > 0.0
    }
    if not shares:
        return {}

    quantities, _total, _warnings = target_quantities_from_shares(
        shares,
        [ore for ore in ores if ore.ore_id in shares],
        target_fe_mt,
        target_hot_metal_mt=hot_metal_mt,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        charge_mass_mt=charge_mass_mt,
    )
    return {k: float(v) for k, v in (quantities or {}).items() if float(v) > 0.0}


def case_burden_quantities(case: BlendCase, provider: Any) -> dict[str, float]:
    """``current_burden_quantities`` of the case's ores on its basis (t/day by ore id)."""

    return current_burden_quantities(
        provider=provider,
        ores=case.ores,
        target_fe_mt=case.target_fe_mt,
        hot_metal_mt=case.target_production_mt,
        fuel_ash_inputs=case.fuel_ash_inputs,
        flux_inputs=case.flux_inputs,
        dust_inputs=case.dust_inputs,
        slag_balance_settings=case.slag_balance_settings,
        charge_mass_mt=case.charge_mass_mt,
        lookback_hours=case.lookback_hours,
    )


def resolve_energy_anchor(
    *,
    basis: str,
    provider: Any,
    ores: list[OreInput],
    fuel_ash_inputs: list[FuelAshInput],
    flux_inputs: list[FluxInput],
    dust_inputs: list[DustInput],
    slag_balance_settings: SlagBalanceSettings,
    hm_chem_values: Mapping[str, float],
    hm_snapshot: Mapping[str, Any],
    hot_metal_mt: float,
    target_fe_mt: float,
    charge_mass_mt: float,
    observed_slag_rate_kg_per_thm: float,
    lookback_hours: float,
    process_snapshot: Callable[[], Mapping[str, Any]],
    pci_override_kg_thm: float | None,
) -> Any | None:
    """Solve the energy-balance coke anchor for the current operating point.

    Returns ``None`` when the page is not configured to use it, so the caller
    can tell "switched off" from "tried and could not" — the second carries
    notes worth showing the operator, the first does not.

    THE BURDEN IS FETCHED HERE, not read out of session state. It used to come
    from ``bmo_manual_quantities_mt``, which the Comparison tab writes - and
    that tab renders AFTER this runs. So on the first optimizer run of a session
    the key was empty and the anchor silently never engaged, leaving the fuel
    cost on the ML model's near-constant; on later runs it used the PREVIOUS
    run's burden. Reading the last shift directly makes the anchor independent
    of which tabs the operator happened to open, and of the order they render in.

    ``process_snapshot`` returns the live 1-h process tags (called only when a burden is found);
    ``pci_override_kg_thm`` is the operator's PCI override when it is switched on, else None.
    """

    if basis != "energy_balance":
        return None

    from utils.bmo.energy_anchor import solve_energy_anchor

    quantities = current_burden_quantities(
        provider=provider,
        ores=ores,
        target_fe_mt=target_fe_mt,
        hot_metal_mt=hot_metal_mt,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        charge_mass_mt=charge_mass_mt,
        lookback_hours=lookback_hours,
    )
    if not quantities:
        return None

    snapshot = process_snapshot()
    # PCI from the LIVE TAG, for the same reason the recommendation panel takes
    # it from there: the balance solves coke around whatever PCI it is handed,
    # so a stale editor value lands on the coke rate at about 0.86 kg per kg.
    editor = {str(fuel.fuel_id): float(getattr(fuel, "rate_kg_per_thm", 0.0) or 0.0) for fuel in fuel_ash_inputs}
    live_pci = effective_pci_kg_thm(snapshot, pci_override_kg_thm)

    return solve_energy_anchor(
        quantities_mt=quantities,
        ores=ores,
        # Slag from the OBSERVED rate rather than from a solved blend: the
        # anchor describes what the furnace is doing now, and the DPR rate is
        # the measurement of exactly that.
        slag_mt=observed_slag_rate_kg_per_thm * hot_metal_mt / 1000.0,
        hot_metal_mt=hot_metal_mt,
        fuel_rates_kg_per_thm={
            "coke_rate_kg_thm": editor.get("coke", 0.0),
            "nut_coke_rate_kg_thm": editor.get("nut_coke", 70.0),
            "pci_rate_kg_thm": live_pci if live_pci > 0.0 else editor.get("pci", 0.0),
        },
        hm_chemistry=_hm_chemistry(hm_chem_values, hm_snapshot),
        process_snapshot=snapshot,
        flux_mt=sum(max(0.0, float(flux.wet_qty_mt or 0.0)) for flux in flux_inputs if flux.enabled),
        fuel_vm_pct={str(fuel.fuel_id): float(getattr(fuel, "vm_pct", 0.0) or 0.0) for fuel in fuel_ash_inputs},
        shell_loss_gj_per_hr=snapshot.get("shell_loss_gj_per_hr"),
    )


def _hm_chemistry(hm_chem_values: Mapping[str, float], hm_snapshot: Mapping[str, Any]) -> dict[str, float]:
    """Hot-metal and slag chemistry (%) of the energy balance: the operator's C / Si, the snapshot's Fe / Mn / FeO."""

    return {
        "carbon_pct": hm_chem_values.get("carbon_pct", 4.3),
        "silicon_pct": hm_chem_values.get("silicon_pct", 0.5),
        "iron_pct": float(hm_snapshot.get("hm_fe_pct_for_target") or 94.5),
        "manganese_pct": float(hm_snapshot.get("chem_pct_mn") or 0.2),
        "slag_feo_pct": float(hm_snapshot.get("slag_pct_feo") or 0.4),
    }


# ============================================================================ the LP plan
DATA_DRIVEN_REFUSAL = (
    "Optimization was not started because the Data-Driven coke-rate model "
    "has no fresh, eligible current-state prediction. Refresh/fix the "
    "burden quantities or switch to Physics-Driven."
)


def run_refusal(
    case: BlendCase,
    *,
    data_driven_prediction: ChargedCokePrediction | None,
    basicity_bounds_valid: bool,
    pellet_blocked: bool,
) -> str | None:
    """Why a run must not start (the message for the operator), or None when it may."""

    if case.fuel_rate_anchor_basis == "data_driven" and (
        data_driven_prediction is None or not data_driven_prediction.usable
    ):
        return DATA_DRIVEN_REFUSAL
    if not basicity_bounds_valid:
        return "Correct the slag basicity bounds before running BMO."
    if pellet_blocked:
        return "Confirm the selected pellet stock and chemistry values before running BMO."
    if len(case.ores) < 2:
        return "Select at least two ores before running optimization."
    return None


@dataclass(frozen=True)
class CokeAnchor:
    """The coke level the run's fuel cost sits on."""

    basis: str  # "data_driven" or "energy_balance"
    rate_kg_thm: float | None  # kg/tHM; None = no usable anchor (the fuel model's level is used)
    prediction_details: dict[str, Any] | None = None  # the Data-Driven prediction, recorded in the diagnostics


def coke_anchor(
    basis: str, data_driven_prediction: ChargedCokePrediction | None, energy_anchor: Any | None
) -> CokeAnchor:
    """The LEVEL the whole fuel cost sits on, resolved once per run.

    One number, identical for LP, DE, and every DE candidate, so the objective is a single fixed function. Blend
    sensitivity is the correction's job, not the anchor's.
    """

    if basis == "data_driven":
        rate = (
            float(data_driven_prediction.value_kg_per_thm)
            if data_driven_prediction is not None
            and data_driven_prediction.usable
            and data_driven_prediction.value_kg_per_thm is not None
            else None
        )
        details = data_driven_prediction.to_dict() if data_driven_prediction is not None else None
        return CokeAnchor(basis, rate, details)
    rate = energy_anchor.coke_rate_kg_thm if energy_anchor is not None and energy_anchor.usable else None
    return CokeAnchor(basis, rate, None)


def correction_reference(
    case: BlendCase,
    *,
    current_quantities_mt: Mapping[str, float] | None,
    previous_manual_quantities_mt: Mapping[str, float] | None,
) -> CokeCorrectionReference:
    """The coke-correction reference of a run: the current burden (else the last manual blend) at the furnace Si."""

    quantities = current_quantities_mt or previous_manual_quantities_mt
    return build_coke_correction_reference(
        settings=case.coke_correction_settings,
        observed_slag_rate_kg_per_thm=case.observed_slag_rate_kg_per_thm,
        flux_inputs=case.flux_inputs,
        hot_metal_target_mt=case.target_production_mt,
        current_quantities_mt=quantities,
        ores=case.ores,
        current_si_pct=case.live_si_pct,
        current_slag_rate_kg_per_thm=(
            burden_slag_rate(case, case.ores, quantities) if case.use_charged_engine else None
        ),
    )


@dataclass
class BalancedRun:
    """The Balanced Optimizer (LP) result of one run."""

    lp_result: BlendEvaluation | None
    lp_errors: list[str]
    fuel_context: FuelContext | None = None  # loaded when the LP solved; reused by DE
    fuel_warnings: list[str] = field(default_factory=list)


def run_balanced(
    case: BlendCase,
    *,
    reference: CokeCorrectionReference,
    anchor: CokeAnchor,
    load_fuel_context: Callable[[], FuelContext],
) -> BalancedRun:
    """Solve the LP, re-evaluate its blend with the fuel prediction and re-check it against the limits."""

    lp_result, lp_errors = run_lp_baseline(
        case.ores,
        **case.lp_kwargs(),
        coke_correction_settings=case.coke_correction_settings,
        coke_correction_reference=reference,
    )
    if lp_result is None:
        return BalancedRun(lp_result, lp_errors)
    lp_physical_result = lp_result
    fuel_context = load_fuel_context()
    model_service, process_context, history_df, _bundle_status, fuel_warnings = fuel_context
    # The LP may have added optimizable flux (e.g. dolomite) to hold
    # slag basicity within bounds. Carry those solved quantities into
    # this display re-evaluation -- otherwise it silently falls back
    # to the raw editor flux rows (wet_qty_mt=0 for LP-auto fluxes),
    # dropping the flux the LP actually added and understating the
    # displayed basicity/slag versus what the LP solved for.
    lp_solved_flux_mt = {
        str(flux_id): float(qty)
        for flux_id, qty in (lp_physical_result.diagnostics.get("lp_flux_quantities_mt", {}) or {}).items()
    }
    flux_inputs_for_display = [
        (replace(flux, wet_qty_mt=lp_solved_flux_mt[flux.flux_id]) if flux.flux_id in lp_solved_flux_mt else flux)
        for flux in case.flux_inputs
    ]
    lp_result = evaluate_blend_with_fuel_prediction(
        ores=case.ores,
        quantities_mt=lp_physical_result.quantities_mt,
        feo_in_slag_pct=case.feo_in_slag_pct,
        model_service=model_service,
        process_context=process_context,
        history_df=history_df,
        fuel_ash_inputs=case.fuel_ash_inputs,
        flux_inputs=flux_inputs_for_display,
        dust_inputs=case.dust_inputs,
        slag_balance_settings=case.slag_balance_settings,
        hot_metal_target_mt=case.target_production_mt,
        coke_correction_settings=case.coke_correction_settings,
        coke_correction_reference=reference,
        hot_metal_si_pct=case.live_si_pct,
        fuel_rate_anchor_basis=anchor.basis,
        anchor_coke_rate_kg_thm=anchor.rate_kg_thm,
        anchor_prediction_details=anchor.prediction_details,
        charge_mass_mt=case.charge_mass_mt,
    )
    lp_result.diagnostics["lp_flux_quantities_mt"] = lp_solved_flux_mt
    lp_result.diagnostics["flux_cost_per_thm_rs"] = float(
        lp_physical_result.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0
    )
    # Re-check feasibility against the blend actually being displayed,
    # rather than copying the pre-recompute blend's flags -- so a
    # re-pricing/re-evaluation drift can never silently show an
    # out-of-bounds result as feasible with no warning.
    lp_result.violations = check_blend_constraints(lp_result, case.ores, **case.limits())
    lp_result.feasible = len(lp_result.violations) == 0
    return BalancedRun(lp_result, lp_errors, fuel_context, list(fuel_warnings))


def run_intensive(
    case: BlendCase,
    *,
    fuel_context: FuelContext,
    reference: CokeCorrectionReference,
    anchor: CokeAnchor,
    de_cfg: Mapping[str, Any],
    progress_callback: Callable[..., bool] | None = None,
) -> tuple[BlendEvaluation | None, list[str]]:
    """Run the Intensive Optimizer (differential evolution) on the same fuel context and anchor as the LP."""

    model_service, process_context, history_df, _bundle_status, _warnings = fuel_context
    return run_nonlinear_optimizer(
        case.ores,
        target_production_mt=case.target_fe_mt,
        target_slag_qty_mt=case.target_slag_qty_mt,
        feo_in_slag_pct=case.feo_in_slag_pct,
        target_slag_basicity_min=case.target_slag_basicity_min,
        target_slag_basicity_max=case.target_slag_basicity_max,
        target_slag_t_basicity_min=case.target_slag_t_basicity_min,
        target_slag_t_basicity_max=case.target_slag_t_basicity_max,
        target_slag_al2o3_max_pct=case.target_slag_al2o3_max_pct,
        target_slag_mgo_min_pct=case.target_slag_mgo_min_pct,
        target_slag_mgo_al2o3_ratio_min=case.target_slag_mgo_al2o3_ratio_min,
        max_burden_qty_mt=case.max_burden_qty_mt,
        model_service=model_service,
        process_context=process_context,
        history_df=history_df,
        de_cfg=dict(de_cfg),
        fuel_ash_inputs=case.fuel_ash_inputs,
        flux_inputs=case.flux_inputs,
        dust_inputs=case.dust_inputs,
        slag_balance_settings=case.slag_balance_settings,
        hot_metal_target_mt=case.target_production_mt,
        coke_correction_settings=case.coke_correction_settings,
        coke_correction_reference=reference,
        # This is the independent live furnace forecast, not a proposed-
        # blend response. It is constant across candidates by design.
        hot_metal_si_pct=case.live_si_pct,
        fuel_rate_anchor_basis=anchor.basis,
        anchor_coke_rate_kg_thm=anchor.rate_kg_thm,
        anchor_prediction_details=anchor.prediction_details,
        progress_callback=progress_callback,
        charge_mass_mt=case.charge_mass_mt,
    )


def baseline_total_with_flux(blend: Any) -> float:
    """Rs/tHM: ore + fuel at baseline prices (the basis DE optimises on) plus the flux the optimiser added."""

    return float(blend.objective_rs_per_thm) + float(blend.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0)


def fall_back_to_lp(lp_result: BlendEvaluation | None, de_result: BlendEvaluation | None) -> BlendEvaluation | None:
    """The DE guardrail: the LP blend when DE did not improve on it.

    DE jointly minimises ore + fuel and is seeded from the LP baseline, so its total cost can never legitimately
    exceed LP's. If it does (DE hit its iteration/time budget, or returned infeasible), the cheaper LP solution is
    reported as the DE result so the operator never sees a "worse optimum" than the baseline. Costs are compared as
    ore + fuel + flux at baseline prices, so a DE blend that saved ore cost by over-dosing expensive flux cannot look
    cheaper than LP.
    """

    if (
        lp_result is not None
        and lp_result.feasible
        and (
            de_result is None
            or not de_result.feasible
            or baseline_total_with_flux(de_result) > baseline_total_with_flux(lp_result) + 1e-6
        )
    ):
        de_result = copy.deepcopy(lp_result)
        de_result.diagnostics = dict(de_result.diagnostics)
        de_result.diagnostics["de_fell_back_to_lp"] = True
    return de_result


# ============================================================================ result tables
def blend_table(
    blend: BlendEvaluation,
    selected_ores: list[OreInput],
    *,
    charge_mass_mt: float = 26.4,
) -> pd.DataFrame:
    """
    Build the result table data for a blend.

    Args:
         - blend: BlendEvaluation - Evaluated blend result to display.
         - selected_ores: list[OreInput] - Ores included in the current run.
         - charge_mass_mt: float - Tonnes carried by one furnace charge.

    Returns:
         - return pd.DataFrame - Per-ore result table sorted by share (quantities in t/day, kg per charge,
           shares in %, prices in Rs/t, cost in Rs and Rs lakh).
    """

    charges_per_hour = compute_charging_requirements(
        blend,
        charge_mass_mt=charge_mass_mt,
        hours_per_day=CHARGING_HOURS_PER_DAY,
    )["required_charges_per_hour"]
    daily_charges = (
        float(charges_per_hour) * CHARGING_HOURS_PER_DAY
        if charges_per_hour is not None and float(charges_per_hour) > 0.0
        else None
    )

    rows = []
    dry_weight_by_ore = blend.diagnostics.get("dry_weight_mt_by_ore", {}) or {}
    fe_contribution_by_ore = blend.diagnostics.get("fe_contribution_mt_by_ore", {}) or {}
    slag_contribution_by_ore = blend.diagnostics.get("slag_contribution_mt_by_ore", {}) or {}
    for ore in selected_ores:
        qty = float(blend.quantities_mt.get(ore.ore_id, 0.0))
        share = float(blend.shares_pct.get(ore.ore_id, 0.0))
        fe_mt = float(fe_contribution_by_ore.get(ore.ore_id, 0.0))
        slag_mt = float(slag_contribution_by_ore.get(ore.ore_id, 0.0))
        rows.append(
            {
                "ore_name": ore.display_name,
                "quantity_mt": qty,
                "kg_per_charge": (qty * 1000.0 / daily_charges if daily_charges else None),
                "dry_quantity_mt": float(dry_weight_by_ore.get(ore.ore_id, 0.0)),
                "moisture_pct": float(ore.chemistry.moisture_pct),
                "fe_contribution_mt": fe_mt,
                "slag_contribution_mt": slag_mt,
                "slag_per_fe": slag_mt / fe_mt if fe_mt > 0 else 0.0,
                "share_pct": share,
                "stock_mt": ore.stock_mt,
                "price_rs_per_mt": ore.price_rs_per_mt,
                "ore_cost_rs": qty * ore.price_rs_per_mt,
                "ore_cost_lakhs": (qty * ore.price_rs_per_mt) / 1.0e5,
            }
        )
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False)


# ---------------------------------------------------------------------------- manual blend vs optimiser
@dataclass
class ManualSeed:
    """How the manual-burden editor is seeded: the last shift's shares, else the optimiser's."""

    rows: list[dict[str, Any]]  # ore_id, ore_name, manual_share_pct (%, normalised to 100)
    note: str  # the caption naming the source of the seed
    use_last_shift: bool  # the seed is the plant's record (any selected ore was charged)
    rows_by_ore: dict[str, dict[str, Any]]  # the snapshot's rows by ore id (share_pct, quantity_mt per hour)


def manual_seed(
    snapshot: Mapping[str, Any],
    compare_ores: list[OreInput],
    primary_blend: BlendEvaluation,
    *,
    lookback_hours: float,
) -> ManualSeed:
    """Seed rows of the manual-burden editor from the last-shift snapshot (``get_recent_manual_blend_snapshot``)."""

    rows_by_ore = {str(row.get("ore_id")): row for row in snapshot.get("rows", [])}
    start_time, end_time = snapshot.get("start_time"), snapshot.get("end_time")
    charged_ids = {ore_id for ore_id, row in rows_by_ore.items() if float(row.get("share_pct", 0.0) or 0.0) > 0.0}
    selected_charged = charged_ids & {ore.ore_id for ore in compare_ores}
    if rows_by_ore and selected_charged and start_time and end_time:
        skipped = [ore.display_name for ore in compare_ores if ore.ore_id not in selected_charged]
        seed_note = (
            f"Seeded from the last {lookback_hours:g} hours of furnace data "
            f"({plant_window_text(start_time, end_time)})."
            + (
                # Named rather than silently zeroed. An operator seeing 0% against
                # an ore they selected needs to know it reflects the shift record,
                # not a data gap.
                f" Not charged in this window, so shown at 0%: {', '.join(skipped)}." if skipped else ""
            )
        )
    elif rows_by_ore and not selected_charged:
        seed_note = (
            "None of the selected ores were charged in this window, so the manual "
            "blend is seeded from the optimizer shares instead."
        )
    else:
        seed_note = "No furnace-dataset manual blend found; seeded from the optimizer shares."

    # A ZERO IN A POPULATED SNAPSHOT IS INFORMATION, NOT A GAP.
    #
    # This used to fall back to the optimizer's share per ORE whenever an ore
    # read zero for the last shift. But an ore reading zero means the plant
    # chose not to charge it - and substituting the optimizer's own
    # recommendation there both invents material the furnace never saw and,
    # because the table is then renormalised to 100%, dilutes every material
    # that WAS charged.
    #
    # Measured on 2026-09-10: true last-shift burden was sinter 60.6%, pellet
    # 15.1%, Lloyds CLO 14.3%, Geomin CLO 10.0%, NMDC ROM 0.0%. NMDC ROM
    # inherited the LP's ~20% and the table renormalised, so sinter displayed as
    # 50.4% - the plant's 60% sinter reading as 50%. Every downstream consumer
    # inherited that: the manual-vs-optimizer cost comparison, the coke
    # correction reference, the energy-balance anchor, and the commentary's
    # idea of "current operation".
    #
    # The optimizer is still a reasonable seed when there is NO last-shift data
    # at all - that case is whole-table, is handled below, and already says so
    # in its own caption.
    use_last_shift = bool(selected_charged)
    seed_rows = []
    for ore in compare_ores:
        seed_share = float(rows_by_ore.get(ore.ore_id, {}).get("share_pct", 0.0) or 0.0)
        if seed_share <= 0 and not use_last_shift:
            seed_share = float(primary_blend.shares_pct.get(ore.ore_id, 0.0))
        seed_rows.append({"ore_id": ore.ore_id, "ore_name": ore.display_name, "manual_share_pct": seed_share})
    # Seed the editor with shares already normalised to 100%. The blend is
    # normalised before evaluation, so the "Suggested blend mix" table shows
    # normalised shares; seeding raw last-shift shares (which sum to <100% when
    # some materials aren't selected) would show a different % for the same ore
    # in the editor vs the mix table.
    _seed_total = sum(row["manual_share_pct"] for row in seed_rows)
    if _seed_total > 0:
        for row in seed_rows:
            row["manual_share_pct"] = round(row["manual_share_pct"] / _seed_total * 100.0, 1)
    return ManualSeed(seed_rows, seed_note, use_last_shift, rows_by_ore)


def evaluate_manual_blend(
    case: BlendCase,
    *,
    compare_ores: list[OreInput],
    quantities_mt: Mapping[str, float],
    fuel_context: FuelContext,
    manual_si: float | None,
) -> BlendEvaluation:
    """The manual blend at the current fuel rates (the realised-cost reference of the comparison)."""

    model_service, process_context, history_df, _bundle_status, _fuel_warnings = fuel_context
    return evaluate_blend_with_fuel_prediction(
        ores=compare_ores,
        quantities_mt=quantities_mt,
        feo_in_slag_pct=case.feo_in_slag_pct,
        model_service=model_service,
        process_context=process_context,
        history_df=history_df,
        fuel_ash_inputs=case.fuel_ash_inputs,
        flux_inputs=case.flux_inputs,
        dust_inputs=case.dust_inputs,
        slag_balance_settings=case.slag_balance_settings,
        hot_metal_target_mt=case.target_production_mt,
        # Manual blend = current operation: show the realised fuel cost
        # from the actual current rates (Fuel Ash table, seeded from the
        # latest static-dataset row) at current prices. Optimized blends
        # instead convert the ML-predicted cost (default "model_cost").
        fuel_rate_basis="inputs",
        # Computed and shown but not applied (apply_to_manual_blend is
        # false): the manual blend is the realised-cost reference, and a
        # large correction here is the clearest signal that the reference
        # operating point itself is mis-set.
        coke_correction_settings=case.coke_correction_settings,
        coke_correction_reference=build_coke_correction_reference(
            settings=case.coke_correction_settings,
            observed_slag_rate_kg_per_thm=case.observed_slag_rate_kg_per_thm,
            flux_inputs=case.flux_inputs,
            hot_metal_target_mt=case.target_production_mt,
            process_context=process_context,
            current_quantities_mt=quantities_mt,
            ores=compare_ores,
            current_si_pct=manual_si,
            current_slag_rate_kg_per_thm=(
                burden_slag_rate(case, compare_ores, quantities_mt) if case.use_charged_engine else None
            ),
        ),
        hot_metal_si_pct=manual_si,
        charge_mass_mt=case.charge_mass_mt,
    )


#: Rows where the smaller number is plainly the better outcome, so the cheapest option can be picked out. Other rows
#: are specs, not scores.
LOWER_IS_BETTER = frozenset({"Total Cost (Rs/THM)", "Ore Cost (Rs/THM)", "Fuel Cost (Rs/THM)", "Flux Cost (Rs/THM)"})

#: Grouped rather than one 25-row wall. The groups are the questions an operator asks in order: what does it cost, what
#: fuel does it need, what slag does it make, and how does it get charged.
COMPARISON_GROUPS: dict[str, list[str]] = {
    "💰 Cost": [
        "Total Cost (Rs/THM)",
        "Ore Cost (Rs/THM)",
        "Fuel Cost (Rs/THM)",
        "Flux Cost (Rs/THM)",
    ],
    "🔥 Fuel": [
        "Fuel Rate (kg/THM)",
        "Coke Rate, uncorrected (kg/THM)",
        "Coke Rate, corrected (kg/THM)",
        "Coke Correction (kg/THM)",
    ],
    "🌋 Slag": [
        "Slag Rate (kg/THM)",
        "Slag Basicity (CaO/SiO2)",
        "Slag T-Basicity (CaO+MgO)/SiO2",
        "IB4 (CaO+MgO)/(SiO2+Al2O3)",
        "Slag Al2O3 (%)",
        "Slag MgO (%)",
        "Slag MgO/Al2O3",
    ],
    "🚚 Charging": [
        "Production (MT/day)",
        "Δ Production vs last shift (MT/day)",
        "Δ Production vs last shift (%)",
        "Flux Rate (kg/THM)",
        "Coke in Charges (MT)",
        "Nut Coke in Charges (MT)",
        "PCI in Charges (MT)",
        "Required Charges (/hr)",
        "Chemical Hotmetal per Charge (MT)",
    ],
}


@dataclass
class OutcomeRows:
    """One group of the outcome table: pre-formatted text per option and the raw numbers behind it."""

    rows: list[dict[str, Any]]  # {"Outcome": label, <option>: text}
    raw: list[dict[str, float]]  # per row: option -> value (finite values only)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def best(self, index: int) -> set[str]:
        """Options holding the lowest value of a lower-is-better row (empty when the row is not a cost or ties)."""

        row = self.rows[index]
        if row["Outcome"] not in LOWER_IS_BETTER:
            return set()
        values = self.raw[index]
        if len(values) < 2 or np.isclose(max(values.values()), min(values.values())):
            return set()
        best = min(values.values())
        return {column for column, value in values.items() if np.isclose(value, best)}


@dataclass
class OptionHeadline:
    """The headline tile of one option: total cost and production against the last shift."""

    label: str
    total_rs_per_thm: float  # Rs/tHM: ore + fuel at current prices + flux the optimiser added
    production_mt: float  # t/day
    production_delta_mt: float | None  # t/day against the last shift (None for the manual blend)
    production_change_pct: float | None  # % against the last shift
    is_manual: bool


@dataclass
class ComparisonView:
    """The manual blend against the optimiser blends: headline, outcome groups and the blend mix."""

    labels: list[str]
    headlines: list[OptionHeadline]
    optimizer_labels: list[str]
    groups: dict[str, OutcomeRows]
    mix: pd.DataFrame  # ore, share (%) and tonnage (t/day) per option
    manual_actual_tonnage: bool  # the manual tonnage is what was charged (else the solution's charge per day)
    baseline: dict[str, Any]  # the last complete shift (``_last_shift_production``)
    saving: tuple[str, float] | None  # (best optimiser, Rs/tHM saved against the manual blend), with a manual blend


def comparison_view(
    *,
    options: list[tuple[str, Any, float | None]],
    optimizer_candidates: list[tuple[str, Any, float | None]],
    manual_blend: BlendEvaluation | None,
    compare_ores: list[OreInput],
    charge_mass_mt: float,
    shift_baseline: Mapping[str, Any] | None,
    rows_by_ore: Mapping[str, Mapping[str, Any]],
    use_last_shift: bool,
) -> ComparisonView:
    """The outcome tables of the Comparison tab, for options in display order (Manual first, then LP / DE)."""

    labels = [label for label, _, _ in options]
    charging_by_blend_id = {
        id(blend): compute_charging_requirements(
            blend,
            charge_mass_mt=charge_mass_mt,
        )
        for _, blend, _ in options
    }

    def _charging_value(blend: Any, key: str) -> float | None:
        value = charging_by_blend_id[id(blend)].get(key)
        return float(value) if value is not None else None

    # Current production is what the furnace made over the last complete
    # shift; each optimizer's production is its target. Every "vs last shift"
    # figure on this tab is measured against the former.
    baseline = dict(shift_baseline or {})
    baseline_mt = baseline.get("daily_production_mt")

    def _production_of(blend: Any) -> float:
        if blend is manual_blend and baseline_mt is not None:
            return float(baseline_mt)
        return float(blend.diagnostics.get("hot_metal_target_mt", 0.0) or 0.0)

    def _production_change(blend: Any) -> float | None:
        if blend is manual_blend:
            return None
        return production_change_pct(_production_of(blend), baseline_mt)

    def _production_delta_mt(blend: Any) -> float | None:
        if blend is manual_blend or _production_change(blend) is None:
            return None
        return _production_of(blend) - float(baseline_mt)

    # ---- Inputs: blend mix as share and tonnage, side by side ----
    # Manual tonnage is what was actually charged (the snapshot's hourly rate x
    # 24) whenever the snapshot came from the plant record; the optimizer
    # tonnage is each solution's wet charge per day at its target production.
    manual_actual_t_per_day = (
        {ore_id: float(row.get("quantity_mt", 0.0) or 0.0) * 24.0 for ore_id, row in rows_by_ore.items()}
        if use_last_shift
        else None
    )

    def _tonnage_column(label: str, blend: Any) -> tuple[str, dict[str, float]]:
        if blend is manual_blend and manual_actual_t_per_day is not None:
            return f"{label} actual (t/day)", manual_actual_t_per_day
        return f"{label} (t/day)", {str(ore_id): float(qty) for ore_id, qty in blend.quantities_mt.items()}

    tonnage_columns = [_tonnage_column(label, blend) for label, blend, _ in options]
    mix_frame = pd.DataFrame(
        [
            {
                "Ore": ore.display_name,
                **{
                    key: value
                    for (label, blend, _), (tonnage_label, tonnage) in zip(options, tonnage_columns, strict=True)
                    for key, value in (
                        (f"{label} (%)", float(blend.shares_pct.get(ore.ore_id, 0.0))),
                        # Whole tonnes: a daily tonnage to the kilogram is noise.
                        (tonnage_label, round(float(tonnage.get(ore.ore_id, 0.0)))),
                    )
                },
            }
            for ore in compare_ores
        ]
    )

    # ---- Outputs: the KPIs an operator decides on ----
    def _basicity(blend: Any, denom_key: str, attr: str) -> float | None:
        if float(blend.diagnostics.get(denom_key, 0.0) or 0.0) <= 0:
            return None
        return float(getattr(blend, attr, 0.0) or 0.0)

    def _rate_field(blend: Any, estimate_key: str, field: str) -> float | None:
        value = (blend.diagnostics.get(estimate_key) or {}).get(field)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _fuel_rate_total(blend: Any) -> float | None:
        return _rate_field(blend, "fuel_rate_estimate", "total_fuel_rate_kg_thm")

    metric_specs = _metric_specs(
        production_of=_production_of,
        production_delta_mt=_production_delta_mt,
        production_change=_production_change,
        charging_value=_charging_value,
        fuel_rate_total=_fuel_rate_total,
        rate_field=_rate_field,
        basicity=_basicity,
    )

    def _rows_for(spec_labels: list[str]) -> OutcomeRows:
        """One group of the outcome table, each row in its own format.

        Cells are pre-formatted text (right-aligned by the column config): a
        missing number must read "n/a", and Streamlit draws an empty numeric
        cell as a grey "None" whatever the Styler says.
        """

        rows, raw_rows = [], []
        for row_label, accessor, fmt in metric_specs:
            if row_label not in spec_labels:
                continue
            row: dict[str, Any] = {"Outcome": row_label}
            raw: dict[str, float] = {}
            for label, blend, si in options:
                value = accessor(blend, si)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    row[label] = fmt.format(value)
                    raw[label] = float(value)
                elif row_label.startswith("Δ Production") and blend is manual_blend:
                    row[label] = "—"  # Manual IS the last shift.
                else:
                    row[label] = "n/a"
            rows.append(row)
            raw_rows.append(raw)
        return OutcomeRows(rows, raw_rows)

    groups = dict(COMPARISON_GROUPS)
    # A row that belongs to no group would simply vanish, and nothing on screen
    # would say a metric had gone missing. Cheap to check, so check.
    _ungrouped = {label for label, _, _ in metric_specs} - {
        label for labels_in_group in groups.values() for label in labels_in_group
    }
    if _ungrouped:
        groups["Other"] = sorted(_ungrouped)

    headlines = [
        OptionHeadline(
            label=label,
            total_rs_per_thm=display_total(blend),
            production_mt=_production_of(blend),
            production_delta_mt=_production_delta_mt(blend),
            production_change_pct=_production_change(blend),
            is_manual=blend is manual_blend,
        )
        for label, blend, _ in options
    ]
    saving = None
    if manual_blend is not None:
        best_label, best_optimizer = min(
            ((label, blend) for label, blend, _ in optimizer_candidates),
            key=lambda item: display_total(item[1]),
        )
        saving = (best_label, display_total(manual_blend) - display_total(best_optimizer))
    return ComparisonView(
        labels=labels,
        headlines=headlines,
        optimizer_labels=[label for label, _, _ in optimizer_candidates],
        groups={name: _rows_for(spec_labels) for name, spec_labels in groups.items()},
        mix=mix_frame,
        manual_actual_tonnage=manual_actual_t_per_day is not None,
        baseline=baseline,
        saving=saving,
    )


# Fuel + total costs are shown re-priced at the operator's current fuel
# prices when available (display-only; the optimizer used baseline prices).
def display_fuel(blend: Any) -> float:
    """Rs/tHM: the fuel cost at the currently applied fuel prices (else at baseline prices)."""

    adjusted = blend.diagnostics.get("adjusted_fuel_cost_per_thm_rs")
    return float(adjusted) if adjusted is not None else float(blend.fuel_cost_per_thm_rs)


def flux_cost(blend: Any) -> float:
    """Rs/tHM: the flux the optimiser added."""

    return float(blend.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0)


def display_total(blend: Any) -> float:
    """Rs/tHM: ore + (re-priced) fuel + optimiser-added flux.

    The cheapest option and the saving vs manual account for the flux spend too.
    """

    adjusted = blend.diagnostics.get("adjusted_objective_rs_per_thm")
    base = float(adjusted) if adjusted is not None else float(blend.objective_rs_per_thm)
    return base + flux_cost(blend)


def _metric_specs(
    *,
    production_of: Callable[[Any], float],
    production_delta_mt: Callable[[Any], float | None],
    production_change: Callable[[Any], float | None],
    charging_value: Callable[[Any, str], float | None],
    fuel_rate_total: Callable[[Any], float | None],
    rate_field: Callable[[Any, str, str], float | None],
    basicity: Callable[[Any, str, str], float | None],
) -> list[tuple[str, Callable[[Any, Any], Any], str]]:
    """(row label, accessor(blend, si) -> value | None, format string) of every outcome row."""

    return [
        ("Production (MT/day)", lambda b, si: production_of(b), "{:,.1f}"),
        (
            "Δ Production vs last shift (MT/day)",
            lambda b, si: production_delta_mt(b),
            "{:+,.0f}",
        ),
        (
            "Δ Production vs last shift (%)",
            lambda b, si: production_change(b),
            "{:+.1f}%",
        ),
        ("Total Cost (Rs/THM)", lambda b, si: display_total(b), "{:,.0f}"),
        ("Ore Cost (Rs/THM)", lambda b, si: b.ore_cost_per_thm_rs, "{:,.0f}"),
        (
            "Flux Rate (kg/THM)",
            lambda b, si: charging_value(b, "flux_rate_kg_per_thm"),
            "{:,.1f}",
        ),
        (
            "Coke in Charges (MT)",
            lambda b, si: charging_value(b, "coke_total_mt"),
            "{:,.1f}",
        ),
        (
            "Nut Coke in Charges (MT)",
            lambda b, si: charging_value(b, "nut_coke_total_mt"),
            "{:,.1f}",
        ),
        (
            "PCI in Charges (MT)",
            lambda b, si: charging_value(b, "pci_total_mt"),
            "{:,.1f}",
        ),
        (
            "Required Charges (/hr)",
            lambda b, si: charging_value(b, "required_charges_per_hour"),
            "{:,.2f}",
        ),
        (
            "Chemical Hotmetal per Charge (MT)",
            lambda b, si: charging_value(b, "chemical_hot_metal_per_charge_mt"),
            "{:,.2f}",
        ),
        # 1-decimal so small but real blend-to-blend differences aren't hidden by
        # rounding (the fuel model is only weakly blend-sensitive -- see the help
        # note on the outcomes table).
        ("Fuel Cost (Rs/THM)", lambda b, si: display_fuel(b), "{:,.1f}"),
        ("Flux Cost (Rs/THM)", lambda b, si: flux_cost(b), "{:,.1f}"),
        ("Fuel Rate (kg/THM)", lambda b, si: fuel_rate_total(b), "{:,.1f}"),
        # Uncorrected and corrected side by side: a correction is only worth
        # trusting if the number it replaced is still visible next to it.
        (
            "Coke Rate, uncorrected (kg/THM)",
            lambda b, si: rate_field(b, "fuel_rate_estimate_anchor", "coke_rate_kg_thm"),
            "{:,.1f}",
        ),
        (
            "Coke Rate, corrected (kg/THM)",
            lambda b, si: rate_field(b, "fuel_rate_estimate", "coke_rate_kg_thm"),
            "{:,.1f}",
        ),
        (
            "Coke Correction (kg/THM)",
            lambda b, si: (
                float(b.diagnostics["coke_correction_delta_kg_thm"])
                if "coke_correction_delta_kg_thm" in b.diagnostics
                else None
            ),
            "{:+,.1f}",
        ),
        (
            "Slag Basicity (CaO/SiO2)",
            lambda b, si: basicity(b, "slag_basicity_denominator_mt", "slag_basicity"),
            "{:,.3f}",
        ),
        (
            # Constrained. Plant runs ~1.31.
            "Slag T-Basicity (CaO+MgO)/SiO2",
            lambda b, si: basicity(b, "slag_t_basicity_denominator_mt", "slag_t_basicity"),
            "{:,.3f}",
        ),
        (
            # Display only. Reads ~0.84 at the plant, not ~1.31, because Al2O3
            # joins the denominator - the two are easily mistaken for each other.
            "IB4 (CaO+MgO)/(SiO2+Al2O3)",
            lambda b, si: basicity(b, "slag_ib4_denominator_mt", "slag_ib4"),
            "{:,.3f}",
        ),
        ("Slag Rate (kg/THM)", lambda b, si: b.slag_rate_kg_per_thm, "{:,.0f}"),
        (
            "Slag Al2O3 (%)",
            lambda b, si: basicity(b, "slag_chemistry_denominator_mt", "slag_al2o3_pct"),
            "{:,.2f}",
        ),
        (
            "Slag MgO (%)",
            lambda b, si: basicity(b, "slag_chemistry_denominator_mt", "slag_mgo_pct"),
            "{:,.2f}",
        ),
        (
            "Slag MgO/Al2O3",
            lambda b, si: basicity(b, "slag_mgo_al2o3_denominator_mt", "slag_mgo_al2o3_ratio"),
            "{:,.3f}",
        ),
    ]


# ---------------------------------------------------------------------------- path from the current blend
def manual_shares(snapshot: Mapping[str, Any]) -> dict[str, float]:
    """% share by ore id of the last-shift snapshot (``get_recent_manual_blend_snapshot``)."""

    return {str(row.get("ore_id")): float(row.get("share_pct", 0.0) or 0.0) for row in snapshot.get("rows", [])}


def transition_ladder(case: BlendCase, *, shares_pct: Mapping[str, float], max_share_move_pct: float) -> Any:
    """The path from the current blend to the LP optimum: one full LP solve per rung under a per-ore move cap."""

    from utils.bmo.transition import build_transition_ladder

    return build_transition_ladder(
        case.ores,
        dict(shares_pct),
        max_share_move_pct=float(max_share_move_pct),
        _slag_rate_cap=case.slag_rate_cap_kg_per_thm,
        **case.lp_kwargs(),
    )


def transition_note(ladder: Any) -> str | None:
    """The warning when the blend being charged is outside the limits, else None."""

    if ladder.start_is_admissible:
        return None
    recovery = ladder.diagnostics.get("recovery_move_pct")
    note = (
        "**The blend currently being charged is outside your limits.** "
        "Step 1 below brings it back inside — make that move first, then "
        "continue down the path.\n- " + "\n- ".join(ladder.start_violations)
    )
    if recovery:
        note += (
            f"\n\n**Step 1 needs a {recovery:.1f}% share move**, more than "
            f"the {ladder.max_share_move_pct:.1f}% step you have set — "
            "recovery is not possible within it."
        )
    return note


def transition_rows(ladder: Any, ores: list[OreInput]) -> list[dict[str, Any]]:
    """One row per rung: shares (%), ore cost (Rs/tHM), slag (kg/tHM), B2 and the binding limits."""

    rows = []
    if ladder.start_blend is not None:
        rows.append(
            {
                "Step": "now",
                **{ore.display_name: ladder.start_shares_pct.get(ore.ore_id, 0.0) for ore in ores},
                "Ore Cost (Rs/THM)": ladder.start_blend.ore_cost_per_thm_rs,
                "Slag (kg/THM)": ladder.start_blend.slag_rate_kg_per_thm,
                "B2": ladder.start_blend.slag_basicity,
                "Binding": "",
            }
        )
    for rung in ladder.rungs:
        if not rung.feasible:
            rows.append({"Step": str(rung.index), "Binding": "infeasible"})
            continue
        rows.append(
            {
                "Step": str(rung.index),
                **{ore.display_name: rung.shares_pct.get(ore.ore_id, 0.0) for ore in ores},
                "Ore Cost (Rs/THM)": rung.blend.ore_cost_per_thm_rs,
                "Slag (kg/THM)": rung.blend.slag_rate_kg_per_thm,
                "B2": rung.blend.slag_basicity,
                "Binding": "; ".join(rung.binding_limits),
            }
        )
    return rows


# ---------------------------------------------------------------------------- DE search cloud
@dataclass
class DeExploration:
    """The DE candidates (one row per evaluated blend) and the 100 cheapest distinct ones as a table."""

    cloud: pd.DataFrame  # total_cost_rs_per_thm (Rs/tHM), slag_basicity, slag_mt (t/day), ...
    table: pd.DataFrame  # Rank, Total Cost (Rs/THM), Slag Basicity, Slag (MT), ore shares (%), flux (MT), Feasible
    has_shares: bool  # the candidates carry their ore shares (older session records do not)


def de_exploration(
    candidates: list[dict[str, Any]] | None, selected_ores: list[OreInput] | None = None
) -> DeExploration | None:
    """The DE search cloud and its top-100 table, or None without usable candidates."""

    rows = [c for c in (candidates or []) if isinstance(c, dict)]
    if not rows:
        return None

    df = pd.DataFrame(rows)
    required = ("total_cost_rs_per_thm", "slag_basicity", "slag_mt")
    if any(col not in df.columns for col in required):
        return None
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    if df.empty:
        return None

    # Top 100 cheapest distinct candidates as alternative solutions.
    ranked = (
        df.assign(
            _b=df["slag_basicity"].round(3),
            _s=df["slag_mt"].round(1),
            _c=df["total_cost_rs_per_thm"].round(0),
        )
        .drop_duplicates(subset=["_b", "_s", "_c"])
        .sort_values("total_cost_rs_per_thm")
        .head(100)
        .reset_index(drop=True)
    )
    feasible_vals = ranked["feasible"].astype(bool).tolist() if "feasible" in ranked.columns else [True] * len(ranked)
    table = pd.DataFrame(
        {
            "Rank": list(range(1, len(ranked) + 1)),
            "Total Cost (Rs/THM)": ranked["total_cost_rs_per_thm"].round(0).tolist(),
            "Slag Basicity": ranked["slag_basicity"].round(3).tolist(),
            "Slag (MT)": ranked["slag_mt"].round(1).tolist(),
        }
    )

    # Blend combination columns: one share column per ore, one MT column per
    # flux the optimizer controls. Older session records may lack these dicts.
    ore_names = {ore.ore_id: ore.display_name for ore in (selected_ores or [])}
    if "shares_pct" in ranked.columns:
        share_dicts = [d if isinstance(d, dict) else {} for d in ranked["shares_pct"]]
        ore_ids: list[str] = []
        for d in share_dicts:
            for ore_id in d:
                if ore_id not in ore_ids:
                    ore_ids.append(ore_id)
        for ore_id in ore_ids:
            label = f"{ore_names.get(ore_id, ore_id.upper())} (%)"
            table[label] = [round(float(d.get(ore_id, 0.0)), 1) for d in share_dicts]
    if "flux_mt" in ranked.columns:
        flux_dicts = [d if isinstance(d, dict) else {} for d in ranked["flux_mt"]]
        flux_ids: list[str] = []
        for d in flux_dicts:
            for flux_id in d:
                if flux_id not in flux_ids:
                    flux_ids.append(flux_id)
        for flux_id in flux_ids:
            table[f"{flux_id.title()} (MT)"] = [round(float(d.get(flux_id, 0.0)), 1) for d in flux_dicts]

    table["Feasible"] = feasible_vals
    return DeExploration(df, table, "shares_pct" in ranked.columns)


# ---------------------------------------------------------------------------- process parameters for a blend
@dataclass
class ControlBasis:
    """What the control recommendation of a blend starts from, or why it cannot run."""

    unavailable: str | None = None  # the operator message when the recommendation cannot run
    fuel_rates: dict[str, Any] = field(default_factory=dict)  # the blend's fuel-rate estimate (kg/tHM)
    pci_now: float = 0.0  # kg/tHM: the PCI the balance uses (live tag or override, else the Fuel Ash box)
    pci_note: str | None = None  # the caption when the live tag and the box disagree by more than 10 kg/tHM


def control_basis(blend: Any, *, snapshot: Mapping[str, Any], live_pci: float) -> ControlBasis:
    """Check the live tags and the blend's fuel rates and resolve the PCI of the control recommendation.

    ``live_pci`` is ``effective_pci_kg_thm`` of the live snapshot (kg/tHM).
    """

    if not snapshot.get("hot_blast_vol_nm3h"):
        return ControlBasis(
            unavailable=(
                "Live blast and top-gas tags are unavailable, so control parameters "
                "cannot be recommended for this blend. The blend itself is unaffected."
            )
        )
    fuel_rates = blend.diagnostics.get("fuel_rate_estimate") or {}
    if not fuel_rates:
        return ControlBasis(unavailable="Fuel-rate estimate unavailable; cannot run the Physics-Driven model.")
    # PCI from the LIVE TAG, never from the Fuel Ash box. The box is
    # operator-editable and exists for slag estimation; the energy balance
    # solves coke around whatever PCI it is handed, so a stale box value lands
    # on the coke rate at roughly 0.86 kg per kg. Resolved here, before the
    # blend inputs are built, so both the balance and the control baseline see
    # the same figure.
    box_pci = float(fuel_rates.get("pci_rate_kg_thm", 0.0) or 0.0)
    pci_now = live_pci if live_pci > 0.0 else box_pci
    note = None
    if live_pci > 0.0 and box_pci > 0.0 and abs(live_pci - box_pci) > 10.0:
        note = (
            f"⚠️ PCI: using the live tag **{live_pci:,.0f} kg/THM** for the energy "
            f"balance, not the Fuel Ash box's {box_pci:,.0f}. The box drives slag only."
        )
    return ControlBasis(fuel_rates=dict(fuel_rates), pci_now=pci_now, pci_note=note)


def control_inputs(
    blend: Any,
    basis: ControlBasis,
    *,
    hot_metal_mt: float,
    ores: list[Any],
    hm_chem_values: Mapping[str, float],
    hm_snapshot: Mapping[str, Any],
    flux_inputs: list[Any],
    fuel_ash_inputs: list[Any],
    snapshot: Mapping[str, Any],
) -> tuple[Any, Any]:
    """The energy-balance inputs of the blend and the current control settings (raises when they cannot be built)."""

    from utils.bmo.process_recommendation import ControlSettings, blend_to_energy_inputs

    flux_mt = sum(max(0.0, float(flux.wet_qty_mt or 0.0)) for flux in flux_inputs if flux.enabled)
    vm_by_fuel = {str(fuel.fuel_id): float(getattr(fuel, "vm_pct", 0.0) or 0.0) for fuel in fuel_ash_inputs}
    energy_inputs = blend_to_energy_inputs(
        blend,
        hot_metal_mt=hot_metal_mt,
        ores=ores,
        fuel_rates_kg_per_thm={**basis.fuel_rates, "pci_rate_kg_thm": basis.pci_now},
        hm_chemistry=_hm_chemistry(hm_chem_values, hm_snapshot),
        process_snapshot=snapshot,
        flux_mt=flux_mt,
        fuel_vm_pct=vm_by_fuel,
        shell_loss_gj_per_hr=snapshot.get("shell_loss_gj_per_hr"),
    )
    current = ControlSettings(
        blast_temperature_c=snapshot.get("hot_blast_temp", 0.0),
        oxygen_enrichment_pct=snapshot.get("oxygen_enrichment_pct", 0.0),
        blast_volume_nm3_per_hr=snapshot.get("hot_blast_vol_nm3h", 0.0),
        pci_kg_per_thm=basis.pci_now,
        hot_blast_pressure_bar=snapshot.get("hot_blast_press", 0.0),
        top_pressure_bar=snapshot.get("top_press_avg", 0.0),
        steam_kg_per_hr=snapshot.get("steam_injection", 0.0),
    )
    return energy_inputs, current


def recommend_process(energy_inputs: Any, current: Any, *, fuel_ash_inputs: list[Any], release_pci: bool) -> Any:
    """The blast settings that supply the blend's energy demand at least fuel cost (``recommend_controls``)."""

    from utils.bmo.process_recommendation import recommend_controls
    from utils.energy_balance.assumptions import apply_overrides as apply_energy_overrides
    from utils.energy_balance.constants import load_config as load_energy_config

    return recommend_controls(
        blend_inputs=energy_inputs,
        current=current,
        prices_rs_per_kg={str(fuel.fuel_id): float(fuel.price_rs_per_mt or 0.0) / 1000.0 for fuel in fuel_ash_inputs},
        optimise_pci=release_pci,
        # Operator-supplied values for the unmeasured constants. Applied
        # here at the app boundary rather than inside the math layer, so
        # the balance stays pure and tests stay hermetic - a saved override
        # file on a developer's machine must not change test results.
        energy_cfg=apply_energy_overrides(load_energy_config()),
    )


def coke_source_rows(
    recommendation: Any, calib: Any, *, plant_coke: float, coke_now: float, coke_shown: float
) -> list[dict[str, Any]]:
    """The three coke rates of the app side by side (kg/tHM), with what each one is."""

    return [
        {
            "Figure": "Plant actual (recent charge reports)",
            "kg/THM": round(plant_coke, 1) if plant_coke else None,
            "What it is": "What the furnace is actually being charged.",
        },
        {
            "Figure": "Physics-Driven, raw at current controls",
            "kg/THM": round(recommendation.current_coke_rate_kg_per_thm, 1),
            "What it is": ("Uncorrected. Runs high — the balance has a known bias."),
        },
        {
            "Figure": f"less bias offset ({calib.offset_kg_per_thm:+.1f})",
            "kg/THM": round(coke_now, 1),
            "What it is": (
                f"Offset fitted on {calib.sample_days} recent days"
                f" ({calib.first_day} → {calib.last_day}), residual sd "
                f"{calib.residual_sd_kg_per_thm:.0f} kg/THM."
                if calib.is_usable
                else "No calibration on file."
            ),
        },
        {
            "Figure": "Corrected, at RECOMMENDED controls",
            "kg/THM": round(coke_shown, 1),
            "What it is": "The headline figure above.",
        },
    ]


def control_rows(recommendation: Any) -> list[dict[str, Any]]:
    """Current against recommended value of every control, with its role (optimised or pass-through)."""

    optimised = set(recommendation.diagnostics["optimised_controls"])
    rows = []
    for key, delta in recommendation.deltas().items():
        rows.append(
            {
                "Control": key.replace("_", " ").title(),
                "Current": getattr(recommendation.current, key),
                "Recommended": getattr(recommendation.settings, key),
                "Change": delta,
                "Role": "optimised" if key in optimised else "pass-through",
            }
        )
    return rows


# ============================================================================ charged-coke panel
def charged_prediction(
    *,
    cfg: Mapping[str, Any],
    source: pd.DataFrame,
    engine: Any,
    storage: Path,
    built_at: Any,
    now: pd.Timestamp,
    pci_override: float | None,
    nut_override: float | None,
    slag_response_coefficient: float,
    force_predict: bool,
) -> tuple[ChargedCokePrediction, dict[str, Any]]:
    """The charged forecast for the BMO level.

    Read from the ledger while the worker issues it (SV-1.5), else issued here once per completed hour, as before
    (``bmo.forecast_issuer``). ``now`` is the naive IST clock; ``source`` the hourly dataset, ``engine`` the active
    charged-coke model and ``built_at`` the dataset's build time.
    """

    note = None
    if forecast_service.page_issues(storage, "charged_coke", cfg):
        record = forecast_service.issue_charged(
            now=now, cfg=cfg, issued_by="page", refresh=False, source=source, engine=engine
        )["record"]
    else:
        view = forecast_service.charged_for_page(storage, now=now)
        record, note = view.record, view.note
    prediction, detail = charged_coke_prediction(
        source,
        engine,
        now=now,
        storage_dir=storage,
        mode="fitted",
        slag_response_coefficient=slag_response_coefficient,
        built_at=built_at,
        pci_override=pci_override,
        nut_override=nut_override,
        force_predict=force_predict,
        record=record,
    )
    detail["note"] = note
    return prediction, detail


# ============================================================================ HM Si panel
@dataclass
class SiForPage:
    """The HM Si forecast page 9 shows: the ledger's record (worker issues) or the page's own issue."""

    mode: str  # forecast mode of setting_bmo.yml (``legacy`` = no production forecast)
    record: dict[str, Any] | None
    note: str | None  # shown when the record is not the current origin's
    issue: dict[str, Any] | None = None  # when the page issues: the keyword arguments of its (cached) issue call


def si_forecast_for_page(
    cfg: Mapping[str, Any],
    *,
    frozen: bool,
    frozen_record: dict[str, Any] | None,
    now: pd.Timestamp,
    source_cache_version: int,
) -> SiForPage:
    """Return the global production forecast without touching blend Si state.

    Read from the ledger while the worker issues it (SV-1.5); when the page issues (``bmo.forecast_issuer``), the
    result carries the arguments of the page's cached issue call (``issue``) and no record yet. A loaded Sandbox
    (``frozen``) shows the archived result and never queries today's production feeds. ``now`` is the IST clock.
    """

    from utils.bmo.si_forecast.deployments import active_bundle as active_si_bundle
    from utils.bmo.si_forecast.service import forecast_origin as si_forecast_origin

    paths = forecast_service.si_paths(cfg)
    mode, default_bundle, storage_dir, static_dataset = (
        paths.mode,
        paths.default_bundle,
        paths.storage_dir,
        paths.static_dataset,
    )
    if mode == "legacy":
        return SiForPage(mode, None, None)
    # TestBMO rewrites this key to testbmo_ui_frozen. A loaded sandbox must show
    # the archived result and must never query today's production feeds.
    if frozen:
        return SiForPage(mode, frozen_record, None)
    active = active_si_bundle(storage_dir, default_bundle)
    if not forecast_service.page_issues(storage_dir, "si", cfg):
        view = forecast_service.si_for_page(storage_dir, active)
        return SiForPage(mode, view.record, view.note)
    try:
        model_mtime = int((active / "manifest.json").stat().st_mtime_ns)
    except OSError:
        model_mtime = 0
    cadence = forecast_service.si_cadence(active)
    origin = si_forecast_origin(now, cadence_minutes=cadence)
    issue = {
        "mode": mode,
        "bundle_path": str(active),
        "storage_path": str(storage_dir),
        "static_dataset_path": str(static_dataset),
        "origin_at": origin.isoformat(),
        "source_cache_version": int(source_cache_version),
        "forest_mtime_ns": model_mtime,
    }
    return SiForPage(mode, None, None, issue)


def issue_si_for_page(origin_at: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Issue the Si forecast at an origin from the page; a failure becomes a failure record (Si never stops BMO)."""

    origin = pd.Timestamp(origin_at)
    try:
        return forecast_service.issue_si(origin=origin, cfg=cfg, issued_by="page")["record"]
    except Exception as exc:  # noqa: BLE001 - Si must never stop BMO/coke
        log.exception("Production HM Si forecast failed")
        return forecast_service.si_failure_record(origin, exc)


def replay_si_history(*, service: Any, make_source: Callable[[], Any], history_end_at: str) -> dict[str, Any]:
    """Causally replay the active model over the 7 days before ``history_end_at`` (the trend); never write live issues.

    ``service`` is the active ``SiliconForecastService``; ``make_source`` builds the ``LiveSiliconForecastSource`` of
    its bundle (a failure there is reported like a failed fetch).
    """

    end = pd.Timestamp(history_end_at)
    start = end - pd.Timedelta(days=7)
    try:
        source = make_source()
        feeds = source.fetch_range(
            origin_start=start,
            origin_end=end,
            lab_end=end,
        )
        replay = service.replay_history(
            feeds,
            origin_start=start,
            origin_end=end,
            available_by=feeds.fetched_at,
        )
        return {
            "forecasts": replay.forecasts,
            "actuals": replay.actuals,
            "scored": replay.scored,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - history cannot block live BMO
        log.exception("Historical HM Si replay failed")
        return {
            "forecasts": pd.DataFrame(),
            "actuals": pd.DataFrame(),
            "scored": pd.DataFrame(),
            "error": str(exc),
        }


# ============================================================================ snapshots
def finalize_run(state: MutableMapping[str, Any], prefix: str, page_vars: Mapping[str, Any]) -> bool:
    """Pin the plant data of a run finished in this rerun (``utils.bmo.replay.finalize_run``); never raises."""

    from utils.bmo.replay import finalize_run as _finalize

    try:
        return _finalize(state, prefix, page_vars)
    except Exception:  # noqa: BLE001 - recording must never take the page down
        log.exception("Could not pin the run context")
        return False


def take_snapshot(
    state: Mapping[str, Any],
    *,
    prefix: str,
    page_vars: Mapping[str, Any],
    label: str = "",
    now: datetime | None = None,
    directory: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Capture the last run (inputs, results, the plant data it read) and save it; returns the snapshot and its file."""

    from utils.bmo.snapshot import capture
    from utils.bmo.snapshot_store import save

    snapshot = capture(state, prefix=prefix, page_vars=page_vars, label=label, now=now)
    return snapshot, save(snapshot, directory)


__all__ = [
    "BalancedRun",
    "BlendCase",
    "CokeAnchor",
    "ComparisonView",
    "ControlBasis",
    "DeExploration",
    "ManualSeed",
    "OptionHeadline",
    "OutcomeRows",
    "SiForPage",
    "baseline_total_with_flux",
    "blend_table",
    "build_coke_correction_reference",
    "burden_slag_rate",
    "case_burden_quantities",
    "charged_prediction",
    "coke_anchor",
    "coke_source_rows",
    "comparison_view",
    "control_basis",
    "control_inputs",
    "control_rows",
    "correction_reference",
    "current_burden_quantities",
    "de_exploration",
    "display_fuel",
    "display_total",
    "effective_pci_kg_thm",
    "evaluate_manual_blend",
    "fall_back_to_lp",
    "finalize_run",
    "flux_cost",
    "issue_si_for_page",
    "live_si_result_tile",
    "load_fuel_prediction_context",
    "manual_seed",
    "manual_shares",
    "plant_window_text",
    "recommend_process",
    "replay_si_history",
    "resolve_energy_anchor",
    "run_balanced",
    "run_intensive",
    "run_refusal",
    "si_forecast_for_page",
    "take_snapshot",
    "target_quantities_from_shares",
    "transition_ladder",
    "transition_note",
    "transition_rows",
]
