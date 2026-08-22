"""Typed contracts for the blast-furnace energy balance.

The inputs are deliberately all MEASURED quantities on a daily basis. Nothing
here is fitted or inferred, so a balance built from this record can be checked
against the plant's own logs line by line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnergyBalanceInputs:
    """One day of measured plant data, ready for the balance.

    All masses are tonnes per day and all percentages are on the basis the
    source table reports. ``compute`` converts to per-tonne-hot-metal itself so
    callers never have to agree a convention.

    Args:
         - hot_metal_mt: float - Hot metal produced, DPR ``total_hot_metal_mt``.
         - slag_mt: float - Slag generated, DPR ``slag_generation_mt``.
         - coke_mt / nut_coke_mt: float - Charge-report tonnes. NOT the DPR or
           static-CSV figures: DPR under-reports coke by ~13% and the CSV's
           COKE_CALC_MT correlates only +0.16 with actual dumps.
         - pci_mt: float - PCI injected, DPR ``pci_mt``. PCI never appears in a
           charge report because it goes in at the tuyeres.
         - flux_mt / sinter_mt / ore_mt / pellet_mt: float - Charged tonnes,
           needed for burden moisture and calcination.
         - blast_volume_nm3_per_hr: float - Hot blast volume.
         - blast_temperature_c: float - Hot blast temperature.
         - oxygen_enrichment_pct: float - O2 enrichment over the 20.8% in air.
         - top_gas_co_pct / co2_pct / h2_pct: float - Top gas analysis.
         - top_gas_temperature_c: float - Top gas temperature.
         - hm_carbon_pct: float - Carbon dissolved in hot metal. Critical: this
           carbon never burns, and crediting it as fuel over-states the input by
           roughly 1,400 MJ/tHM.
         - hm_iron_pct / hm_silicon_pct / hm_manganese_pct: float - HM analysis.
         - slag_feo_pct: float - FeO remaining in slag.
         - fuel_vm_pct: dict[str, float] - Volatile matter per fuel id, used to
           estimate hydrogen while ultimate analysis is unavailable.
         - moisture_pct: dict[str, float] - Free moisture per material id.
         - flux_loi_pct: float - Flux loss on ignition.
         - shell_loss_gj_per_hr: float | None - Measured stave heat load already
           converted to GJ/hr. ``None`` falls back to the configured estimate.

    Returns:
         - return EnergyBalanceInputs - One day of measured inputs.
    """

    hot_metal_mt: float
    slag_mt: float
    coke_mt: float
    nut_coke_mt: float
    pci_mt: float
    blast_volume_nm3_per_hr: float
    blast_temperature_c: float
    oxygen_enrichment_pct: float
    top_gas_co_pct: float
    top_gas_co2_pct: float
    top_gas_h2_pct: float
    top_gas_temperature_c: float
    hm_carbon_pct: float
    hm_iron_pct: float
    hm_silicon_pct: float = 0.0
    hm_manganese_pct: float = 0.0
    slag_feo_pct: float = 0.0
    flux_mt: float = 0.0
    sinter_mt: float = 0.0
    ore_mt: float = 0.0
    pellet_mt: float = 0.0
    flux_loi_pct: float = 0.0
    fuel_vm_pct: dict[str, float] = field(default_factory=dict)
    moisture_pct: dict[str, float] = field(default_factory=dict)
    shell_loss_gj_per_hr: float | None = None


@dataclass
class EnergyBalanceResult:
    """Balance outcome, everything in MJ per tonne of hot metal.

    ``closure`` is the headline: output divided by input, target 1.00. Because
    reductants are credited at full oxidation potential and whatever leaves
    unburnt is booked as an output, a correct balance lands near 1.00 rather
    than at some lumped efficiency.

    Args:
         - demand: dict[str, float] - Heat consumed, term by term.
         - top_gas: dict[str, float] - Heat leaving at the top, sensible and chemical.
         - supply: dict[str, float] - Heat in, term by term.
         - total_demand_mj_per_thm / total_top_gas_mj_per_thm /
           total_output_mj_per_thm / total_input_mj_per_thm: float - Rolled up.
         - closure: float - output / input.
         - implied_shell_loss_mj_per_thm: float - What shell loss WOULD have to be
           for closure to be exactly 1.0. Compare against the measured value: if
           the two disagree badly, a term is missing and calling the residual
           "shell loss" would simply hide the error.
         - diagnostics: dict[str, Any] - Rates, top gas volume, carbon split, and
           the provenance of every estimated quantity.

    Returns:
         - return EnergyBalanceResult - Balance outcome and full trace.
    """

    demand: dict[str, float]
    top_gas: dict[str, float]
    supply: dict[str, float]
    total_demand_mj_per_thm: float
    total_top_gas_mj_per_thm: float
    total_output_mj_per_thm: float
    total_input_mj_per_thm: float
    closure: float
    implied_shell_loss_mj_per_thm: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def closure_band(self) -> str:
        """Green / amber / red, for the closure report tile."""

        green = self.diagnostics.get("closure_green_range", (0.97, 1.03))
        amber = self.diagnostics.get("closure_amber_range", (0.93, 1.07))
        if green[0] <= self.closure <= green[1]:
            return "green"
        if amber[0] <= self.closure <= amber[1]:
            return "amber"
        return "red"
