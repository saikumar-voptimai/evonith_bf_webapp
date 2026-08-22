"""Layer 2: control parameters for a blend Layer 1 has already chosen.

    Layer 1  BMO LP        -> cheapest blend meeting the slag limits
    Layer 2  THIS MODULE   -> control settings that supply what that blend needs

The blend is an INPUT here. Nothing in this module reconsiders it.

WHICH KNOBS THIS OPTIMISES, AND WHICH IT CANNOT.

Only three of the seven controls appear in an energy balance:

    blast temperature   blast sensible heat
    oxygen enrichment   less N2 ballast to heat per Nm3
    blast volume        both of the above, per tonne of hot metal

plus PCI when the operator releases it. Hot blast pressure and top pressure do
not appear in a heat balance at all - they act through burden permeability and
gas utilisation. They are carried through with a bounds check and reported as
pass-through. Steam is nil at this plant. Optimising any of those three here
would be fabricating a recommendation, so the module refuses to.

HOW THE OBJECTIVE WORKS.

For a candidate set of controls, ``solve_coke_rate_kg_per_thm`` inverts the
closed energy balance to find the coke rate that supplies the blend's demand.
Fuel cost follows from that coke rate plus nut coke and PCI at operator prices.
So the objective is grounded in physics, not in a fitted response surface - a
distinction that matters because every fitted alternative tried on this plant's
record failed to generalise forward.

RAFT IS ADVISORY.

The calibration against RAFTOC gives textbook coefficients but only R2 = 0.11
forward, with an unattributed seasonal bias up to +46 C. It is reported with
that uncertainty and does NOT block a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from utils.energy_balance.constants import load_config as load_energy_config
from utils.energy_balance.solve import solve_coke_rate_kg_per_thm
from utils.energy_balance.types import EnergyBalanceInputs

# The three the balance can see, plus PCI. Order fixes the decision vector.
OPTIMISABLE = ("blast_temperature_c", "oxygen_enrichment_pct", "blast_volume_nm3_per_hr")
PCI_CONTROL = "pci_kg_per_thm"
# Real controls the balance is blind to. Passed through, never optimised.
PASS_THROUGH = ("hot_blast_pressure_bar", "top_pressure_bar", "steam_kg_per_hr")


@dataclass
class ControlSettings:
    """One set of furnace control settings."""

    blast_temperature_c: float
    oxygen_enrichment_pct: float
    blast_volume_nm3_per_hr: float
    pci_kg_per_thm: float
    hot_blast_pressure_bar: float = 0.0
    top_pressure_bar: float = 0.0
    steam_kg_per_hr: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass
class ProcessRecommendation:
    """What Layer 2 returns for one blend."""

    settings: ControlSettings
    current: ControlSettings
    coke_rate_kg_per_thm: float
    current_coke_rate_kg_per_thm: float
    fuel_cost_rs_per_thm: float
    current_fuel_cost_rs_per_thm: float
    raft_c: float | None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def fuel_cost_saving_rs_per_thm(self) -> float:
        return self.current_fuel_cost_rs_per_thm - self.fuel_cost_rs_per_thm

    def deltas(self) -> dict[str, float]:
        return {
            key: getattr(self.settings, key) - getattr(self.current, key)
            for key in self.settings.as_dict()
        }


def raft_from_controls(
    settings: ControlSettings, cfg: dict[str, Any] | None = None
) -> float:
    """
    RAFT from the calibrated correlation. ADVISORY only - see module docstring.

    Args:
         - settings: ControlSettings - Candidate controls.
         - cfg: dict[str, Any] | None - ``process_recommendation`` block.

    Returns:
         - return float - Estimated RAFT in degrees C.
    """

    raft = (cfg or _load_pr_config())["raft"]
    return (
        float(raft["intercept"])
        + float(raft["per_c_blast_temp"]) * settings.blast_temperature_c
        + float(raft["per_pct_o2"]) * settings.oxygen_enrichment_pct
        + float(raft["per_kg_pci"]) * settings.pci_kg_per_thm
    )


def _load_pr_config(path: str | None = None) -> dict[str, Any]:
    import yaml
    from utils.energy_balance.constants import CONFIG_PATH

    target = path or CONFIG_PATH
    try:
        loaded = yaml.safe_load(open(target, encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        loaded = {}
    return loaded.get("process_recommendation", {}) or {}


def _fuel_cost(
    coke_rate: float, nut_rate: float, pci_rate: float, prices: dict[str, float]
) -> float:
    return (
        coke_rate * prices.get("coke", 28.0)
        + nut_rate * prices.get("nut_coke", 24.0)
        + pci_rate * prices.get("pci", 18.0)
    )


def recommend_controls(
    *,
    blend_inputs: EnergyBalanceInputs,
    current: ControlSettings,
    prices_rs_per_kg: dict[str, float] | None = None,
    optimise_pci: bool | None = None,
    pr_cfg: dict[str, Any] | None = None,
    energy_cfg: dict[str, Any] | None = None,
) -> ProcessRecommendation:
    """
    Choose control settings that minimise fuel cost for an already-chosen blend.

    Args:
         - blend_inputs: EnergyBalanceInputs - The blend from Layer 1, carrying
           its slag rate, burden masses, moisture and hot-metal chemistry. Its
           control fields are overwritten by the candidate settings.
         - current: ControlSettings - What the furnace is running now. Sets the
           move-limit origin and the comparison baseline.
         - prices_rs_per_kg: dict | None - Operator fuel prices.
         - optimise_pci: bool | None - Release PCI as a decision variable.
           Defaults to the config value, which is False.
         - pr_cfg / energy_cfg: dict | None - Loaded config blocks.

    Returns:
         - return ProcessRecommendation - Settings, expected coke rate and fuel
           cost, RAFT advisory, and any warnings about bounds or extrapolation.
    """

    cfg = pr_cfg if pr_cfg is not None else _load_pr_config()
    ecfg = energy_cfg or load_energy_config()
    prices = prices_rs_per_kg or {}
    release_pci = (
        cfg.get("optimise_pci_by_default", False)
        if optimise_pci is None
        else optimise_pci
    )

    controls = list(OPTIMISABLE) + ([PCI_CONTROL] if release_pci else [])
    bounds_cfg = cfg.get("bounds", {})
    moves = cfg.get("move_limits", {})
    hm = float(blend_inputs.hot_metal_mt)

    def _apply(vector: np.ndarray) -> ControlSettings:
        values = current.as_dict()
        for name, value in zip(controls, vector):
            values[name] = float(value)
        return ControlSettings(**values)

    def _balance_inputs(settings: ControlSettings) -> EnergyBalanceInputs:
        return replace(
            blend_inputs,
            blast_temperature_c=settings.blast_temperature_c,
            oxygen_enrichment_pct=settings.oxygen_enrichment_pct,
            blast_volume_nm3_per_hr=settings.blast_volume_nm3_per_hr,
            pci_mt=settings.pci_kg_per_thm * hm / 1000.0,
        )

    def _cost_of(settings: ControlSettings) -> tuple[float, float]:
        coke = solve_coke_rate_kg_per_thm(_balance_inputs(settings), ecfg)
        nut = blend_inputs.nut_coke_mt / hm * 1000.0
        return coke, _fuel_cost(coke, nut, settings.pci_kg_per_thm, prices)

    def objective(vector: np.ndarray) -> float:
        try:
            return _cost_of(_apply(vector))[1]
        except ValueError:
            # No coke rate closes the balance here; steer the solver away.
            return 1.0e12

    # Bounds are the intersection of the plant limit and the move limit, so a
    # recommendation is always something the operator can actually do next shift.
    bounds: list[tuple[float, float]] = []
    for name in controls:
        lo_cfg, hi_cfg = bounds_cfg.get(name, (-np.inf, np.inf))
        here = getattr(current, name)
        step = float(moves.get(name, np.inf))
        bounds.append((max(float(lo_cfg), here - step), min(float(hi_cfg), here + step)))

    x0 = np.array([getattr(current, name) for name in controls], dtype=float)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                      options={"maxiter": 60, "ftol": 1e-3})
    best = _apply(result.x if result.success else x0)

    coke, cost = _cost_of(best)
    current_coke, current_cost = _cost_of(current)

    warnings: list[str] = []
    if not result.success:
        warnings.append(
            f"optimiser did not converge ({result.message}); showing current settings"
        )
    envelope = cfg.get("observed_envelope", {})
    for name in controls:
        lo, hi = envelope.get(name, (None, None)) or (None, None)
        value = getattr(best, name)
        if lo is not None and not (float(lo) <= value <= float(hi)):
            warnings.append(
                f"{name} = {value:,.1f} is outside the observed envelope "
                f"[{float(lo):,.1f}, {float(hi):,.1f}] - this is extrapolation"
            )

    raft = raft_from_controls(best, cfg)
    band = cfg.get("raft", {}).get("advisory_band_c")
    if band and not (float(band[0]) <= raft <= float(band[1])):
        warnings.append(
            f"RAFT advisory {raft:,.0f} C is outside {band}. Advisory only: the "
            "RAFT correlation has forward R2 0.11 and an unattributed seasonal "
            f"bias up to {cfg.get('raft', {}).get('stated_uncertainty_c', 46)} C."
        )

    return ProcessRecommendation(
        settings=best,
        current=current,
        coke_rate_kg_per_thm=coke,
        current_coke_rate_kg_per_thm=current_coke,
        fuel_cost_rs_per_thm=cost,
        current_fuel_cost_rs_per_thm=current_cost,
        raft_c=raft,
        warnings=warnings,
        diagnostics={
            "optimised_controls": controls,
            "pass_through_controls": list(PASS_THROUGH),
            "pass_through_note": (
                "hot blast pressure, top pressure and steam do not appear in an "
                "energy balance; they act through permeability and gas "
                "utilisation and are not optimised here"
            ),
            "bounds_used": dict(zip(controls, bounds)),
            "converged": bool(result.success),
            "raft_is_advisory": True,
        },
    )


def _category_for(ore: Any) -> str:
    """Classify a BMO ore row into the burden categories the balance uses."""

    key = (
        f"{ore.metadata.get('material_key', '')} {ore.display_name}"
    ).lower()
    if "sinter" in key:
        return "sinter"
    if "pellet" in key:
        return "pellet"
    return "ore"


def blend_to_energy_inputs(
    blend: Any,
    *,
    hot_metal_mt: float,
    ores: list[Any],
    fuel_rates_kg_per_thm: dict[str, float],
    hm_chemistry: dict[str, float],
    process_snapshot: dict[str, float],
    flux_mt: float = 0.0,
    flux_loi_pct: float = 40.0,
    fuel_vm_pct: dict[str, float] | None = None,
    shell_loss_gj_per_hr: float | None = None,
) -> EnergyBalanceInputs:
    """
    Adapt a Layer 1 blend into the measured-quantity form Layer 2 needs.

    This is the seam between the two layers. Layer 1 works in wet ore tonnes and
    slag chemistry; the energy balance works in per-tHM heat terms. Everything
    here is a restatement of the blend, not a new assumption - the only external
    inputs are the live process snapshot and the hot-metal analysis.

    Args:
         - blend: BlendEvaluation - The blend Layer 1 chose.
         - hot_metal_mt: float - Operator hot-metal target for the day.
         - ores: list[OreInput] - Ores in the blend, for classifying burden
           categories and pulling their moisture.
         - fuel_rates_kg_per_thm: dict - coke / nut_coke / pci rates, normally
           from ``blend.diagnostics['fuel_rate_estimate']``.
         - hm_chemistry: dict - carbon/iron/silicon/manganese percentages and
           slag FeO.
         - process_snapshot: dict - Live blast and top-gas tags.
         - flux_mt / flux_loi_pct: float - Flux charged and its LOI.
         - fuel_vm_pct: dict | None - Volatile matter per fuel, for the hydrogen
           estimate. Only used when the hydrogen term is switched on.
         - shell_loss_gj_per_hr: float | None - Measured stave heat load.

    Returns:
         - return EnergyBalanceInputs - Ready for ``run_energy_balance`` or
           ``recommend_controls``.
    """

    masses = {"sinter": 0.0, "ore": 0.0, "pellet": 0.0}
    water = {"sinter": 0.0, "ore": 0.0, "pellet": 0.0}
    for ore in ores:
        qty = float(blend.quantities_mt.get(ore.ore_id, 0.0) or 0.0)
        if qty <= 0.0:
            continue
        category = _category_for(ore)
        masses[category] += qty
        water[category] += qty * float(getattr(ore.chemistry, "moisture_pct", 0.0) or 0.0)

    # Tonnage-weighted moisture per category, so a category with no material
    # contributes nothing rather than a spurious average.
    moisture = {
        category: (water[category] / masses[category]) if masses[category] > 0 else 0.0
        for category in masses
    }
    moisture["flux"] = 0.0
    moisture["coke"] = 0.0
    moisture["nut_coke"] = 0.0

    rate = lambda key: float(fuel_rates_kg_per_thm.get(key, 0.0) or 0.0)  # noqa: E731
    to_mt = lambda kg_per_thm: kg_per_thm * hot_metal_mt / 1000.0  # noqa: E731

    return EnergyBalanceInputs(
        hot_metal_mt=float(hot_metal_mt),
        slag_mt=float(getattr(blend, "slag_mt", 0.0) or 0.0),
        coke_mt=to_mt(rate("coke_rate_kg_thm")),
        nut_coke_mt=to_mt(rate("nut_coke_rate_kg_thm")),
        pci_mt=to_mt(rate("pci_rate_kg_thm")),
        blast_volume_nm3_per_hr=float(process_snapshot.get("hot_blast_vol_nm3h", 0.0)),
        blast_temperature_c=float(process_snapshot.get("hot_blast_temp", 0.0)),
        oxygen_enrichment_pct=float(process_snapshot.get("oxygen_enrichment_pct", 0.0)),
        top_gas_co_pct=float(process_snapshot.get("co_pct", 0.0)),
        top_gas_co2_pct=float(process_snapshot.get("co2_pct", 0.0)),
        top_gas_h2_pct=float(process_snapshot.get("h2_pct", 0.0)),
        top_gas_temperature_c=float(process_snapshot.get("top_temp_avg", 0.0)),
        hm_carbon_pct=float(hm_chemistry.get("carbon_pct", 4.3)),
        hm_iron_pct=float(hm_chemistry.get("iron_pct", 94.5)),
        hm_silicon_pct=float(hm_chemistry.get("silicon_pct", 0.5)),
        hm_manganese_pct=float(hm_chemistry.get("manganese_pct", 0.2)),
        slag_feo_pct=float(hm_chemistry.get("slag_feo_pct", 0.4)),
        sinter_mt=masses["sinter"],
        ore_mt=masses["ore"],
        pellet_mt=masses["pellet"],
        flux_mt=float(flux_mt),
        flux_loi_pct=float(flux_loi_pct),
        fuel_vm_pct=dict(fuel_vm_pct or {}),
        moisture_pct=moisture,
        shell_loss_gj_per_hr=shell_loss_gj_per_hr,
    )
