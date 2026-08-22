"""Blast-furnace energy balance. Pure math, no Streamlit and no database.

CONVENTION - everything below depends on it:

    Every reductant is credited on the INPUT side at its FULL oxidation
    potential (C -> CO2, H -> H2O). Whatever the furnace fails to extract
    leaves at the top as unburnt CO and H2 and is booked as an OUTPUT.

Two consequences worth knowing before reading the code:

1. Closure should land at 1.00, not at a lumped efficiency. Measured on 221
   days of this plant's data: median 1.002, across-quarter spread 3.2%.

2. Endothermic gasification needs NO separate term. For H2O + C -> CO + H2:

       input   C at full potential             = 393.5 kJ/mol
       output  CO unburnt 283 + H2 unburnt 242 = 525.0 kJ/mol
       net                                     = -131.5 kJ/mol = the endotherm

   So moisture that gasifies must not also be charged a 7.3 MJ/kg
   decomposition term - that is double counting. Only moisture that EVAPORATES
   carries its own term.

Two mistakes cost a lot of time getting here, both recorded so they are not
repeated:

  * Iron oxide reduction was omitted at first. It is the largest single term in
    the furnace at ~7,000 MJ/tHM, and without it closure sat at 0.47.
  * Carbon dissolved in the hot metal was credited as burnt. At 4.3% C that is
    43 kg C/tHM worth ~1,400 MJ/tHM of input that is never released.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from utils.energy_balance.constants import (
    FE_IN_FEO_FRACTION,
    H2O_TO_H_FRACTION,
    hydrogen_pct_for_fuel,
    load_config,
)
from utils.energy_balance.types import EnergyBalanceInputs, EnergyBalanceResult

_FUELS = ("coke", "nut_coke", "pci")


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if isfinite(out) else default


def top_gas_volume_nm3_per_thm(
    *,
    blast_nm3_per_thm: float,
    oxygen_enrichment_pct: float,
    co_pct: float,
    co2_pct: float,
    h2_pct: float,
    cfg: dict[str, Any] | None = None,
) -> float:
    """
    Top gas volume from a nitrogen balance.

    N2 is inert, so every Nm3 entering with the blast leaves at the top::

        N2% in blast   = 100 - O2%          = 79.2 - enrichment
        N2% in top gas = 100 - CO - CO2 - H2
        V_top          = V_blast * N2%_blast / N2%_top

    No fitted constant anywhere. Verified against the record at 1,622 Nm3/tHM
    median against a textbook 1,500-1,700.

    Returns:
         - return float - Nm3 of top gas per tonne hot metal, 0.0 if the gas
           analysis is too degraded for the balance to be meaningful.
    """

    settings = (cfg or load_config())["top_gas"]
    n2_blast = _f(settings["n2_in_air_pct"]) - _f(oxygen_enrichment_pct)
    n2_top = 100.0 - _f(co_pct) - _f(co2_pct) - _f(h2_pct)
    if n2_top < _f(settings["min_n2_top_pct"]) or n2_blast <= 0.0:
        return 0.0
    return max(0.0, _f(blast_nm3_per_thm) * n2_blast / n2_top)


def _hydrogen_charged_kg_per_thm(
    inputs: EnergyBalanceInputs, rates: dict[str, float], cfg: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Fuel hydrogen, plus the hydrogen already oxidised in blast moisture.

    Fuel hydrogen is a reductant and earns its full LHV on the input side.
    Blast moisture arrives ALREADY oxidised, so it earns nothing - it shows up
    only through the extra H2 it puts in the top gas.
    """

    provenance: dict[str, Any] = {}
    fuel_h = 0.0
    for fuel in _FUELS:
        vm = _f(inputs.fuel_vm_pct.get(fuel))
        h_pct, source = hydrogen_pct_for_fuel(fuel, vm, cfg)
        contribution = rates[fuel] * h_pct / 100.0
        fuel_h += contribution
        provenance[fuel] = {
            "vm_pct": vm,
            "hydrogen_pct": h_pct,
            "kg_h_per_thm": contribution,
            "source": source,
        }

    blast_water = (
        _f(cfg.get("blast_moisture_g_per_nm3")) * rates["blast_nm3_per_thm"] / 1000.0
    )
    provenance["blast_moisture"] = {
        "kg_h2o_per_thm": blast_water,
        "kg_h_per_thm": blast_water * H2O_TO_H_FRACTION,
        "source": "plant-supplied constant; no humidity tag exists",
        "credited_as_fuel": False,
    }
    return fuel_h, provenance


def run_energy_balance(
    inputs: EnergyBalanceInputs, cfg: dict[str, Any] | None = None
) -> EnergyBalanceResult:
    """
    Close the energy balance for one day.

    Args:
         - inputs: EnergyBalanceInputs - One day of measured plant data.
         - cfg: dict[str, Any] | None - Loaded ``energy_balance`` settings.

    Returns:
         - return EnergyBalanceResult - Term-by-term balance and closure.
    """

    settings = cfg or load_config()
    demand_cfg = settings["demand"]
    supply_cfg = settings["supply"]
    gas_cfg = settings["top_gas"]
    t_ref = _f(supply_cfg["reference_temperature_c"], 25.0)

    hm = _f(inputs.hot_metal_mt)
    if hm <= 0.0:
        raise ValueError("hot_metal_mt must be positive to form per-tHM rates")

    per_thm = lambda mt: _f(mt) / hm * 1000.0  # noqa: E731  tonnes/day -> kg/tHM
    rates = {
        "coke": per_thm(inputs.coke_mt),
        "nut_coke": per_thm(inputs.nut_coke_mt),
        "pci": per_thm(inputs.pci_mt),
        "slag": per_thm(inputs.slag_mt),
        "flux": per_thm(inputs.flux_mt),
        "blast_nm3_per_thm": _f(inputs.blast_volume_nm3_per_hr) * 24.0 / hm,
    }

    # --- carbon actually burnt ------------------------------------------------
    carbon_fraction = settings["fuels"]["carbon_fraction"]
    carbon_charged = sum(
        rates[fuel] * _f(carbon_fraction.get(fuel)) for fuel in _FUELS
    )
    carbon_to_hm = _f(inputs.hm_carbon_pct) / 100.0 * 1000.0
    carbon_burnt = max(0.0, carbon_charged - carbon_to_hm)

    # --- hydrogen -------------------------------------------------------------
    fuel_hydrogen, hydrogen_provenance = _hydrogen_charged_kg_per_thm(
        inputs, rates, settings
    )

    # --- top gas --------------------------------------------------------------
    v_top = top_gas_volume_nm3_per_thm(
        blast_nm3_per_thm=rates["blast_nm3_per_thm"],
        oxygen_enrichment_pct=inputs.oxygen_enrichment_pct,
        co_pct=inputs.top_gas_co_pct,
        co2_pct=inputs.top_gas_co2_pct,
        h2_pct=inputs.top_gas_h2_pct,
        cfg=settings,
    )
    top_gas = {
        "sensible": v_top
        * _f(gas_cfg["cp_kj_per_nm3_k"])
        * (_f(inputs.top_gas_temperature_c) - t_ref)
        / 1000.0,
        "chemical_co": v_top * _f(inputs.top_gas_co_pct) / 100.0
        * _f(gas_cfg["co_lhv_mj_per_nm3"]),
        "chemical_h2": v_top * _f(inputs.top_gas_h2_pct) / 100.0
        * _f(gas_cfg["h2_lhv_mj_per_nm3"]),
    }

    # --- demand ---------------------------------------------------------------
    burden_water_kg = (
        sum(
            _f(getattr(inputs, f"{material}_mt"))
            * _f(inputs.moisture_pct.get(material))
            for material in ("ore", "pellet", "flux", "sinter")
        )
        + _f(inputs.coke_mt) * _f(inputs.moisture_pct.get("coke"))
        + _f(inputs.nut_coke_mt) * _f(inputs.moisture_pct.get("nut_coke"))
    ) / 100.0 / hm * 1000.0

    shell_loss = (
        _f(inputs.shell_loss_gj_per_hr) * 24.0 * 1000.0 / hm
        if inputs.shell_loss_gj_per_hr is not None
        else 0.0
    )

    demand = {
        "iron_reduction": _f(demand_cfg["fe_reduction_mj_per_kg_fe"])
        * _f(inputs.hm_iron_pct) / 100.0 * 1000.0,
        "hot_metal": _f(demand_cfg["hot_metal_mj_per_t"]),
        "slag": _f(demand_cfg["slag_mj_per_kg"]) * rates["slag"],
        "shell_loss": shell_loss,
        "silicon": _f(demand_cfg["silicon_mj_per_kg"])
        * _f(inputs.hm_silicon_pct) / 100.0 * 1000.0,
        "manganese": _f(demand_cfg["manganese_mj_per_kg"])
        * _f(inputs.hm_manganese_pct) / 100.0 * 1000.0,
        "burden_moisture": _f(demand_cfg["burden_moisture_mj_per_kg"]) * burden_water_kg,
        "feo_in_slag": _f(demand_cfg["fe_to_feo_mj_per_kg_fe"])
        * rates["slag"] * _f(inputs.slag_feo_pct) / 100.0 * FE_IN_FEO_FRACTION,
        "calcination": _f(demand_cfg["calcination_mj_per_kg_co2"])
        * rates["flux"] * _f(inputs.flux_loi_pct) / 100.0,
    }

    # --- supply ---------------------------------------------------------------
    # Fuel hydrogen is physically correct to credit, and its H% is now pinned
    # from published rank data (no ultimate analysis exists or is coming). It
    # stays OFF anyway, on measured grounds: the residual it would fill scales
    # with total fuel, not with hydrogen. Over 152-205 kg/tHM of PCI the
    # correlation between PCI rate and the back-calculated residual is -0.05 -
    # the signature fuel hydrogen would leave is simply not there. Enabling it
    # costs 6 points of closure and buys nothing. See energy_balance.yml.
    # The figure is still reported in diagnostics either way.
    include_hydrogen = bool(supply_cfg.get("include_fuel_hydrogen", False))
    hydrogen_supply = fuel_hydrogen * _f(supply_cfg["hydrogen_lhv_mj_per_kg"])
    supply = {
        "carbon": carbon_burnt * _f(supply_cfg["carbon_full_mj_per_kg"]),
        "hydrogen": hydrogen_supply if include_hydrogen else 0.0,
        "blast_sensible": rates["blast_nm3_per_thm"]
        * _f(supply_cfg["blast_cp_kj_per_nm3_k"])
        * (_f(inputs.blast_temperature_c) - t_ref)
        / 1000.0,
    }

    total_demand = sum(demand.values())
    total_top_gas = sum(top_gas.values())
    total_output = total_demand + total_top_gas
    total_input = sum(supply.values())
    closure = total_output / total_input if total_input > 0 else 0.0

    # What shell loss WOULD have to be for closure to be exactly 1.0. Compare
    # against the measured value: a large disagreement means a term is missing,
    # and calling the residual "shell loss" would merely hide the error.
    implied_shell = total_input - (total_output - demand["shell_loss"])

    closure_cfg = settings.get("closure", {})
    return EnergyBalanceResult(
        demand=demand,
        top_gas=top_gas,
        supply=supply,
        total_demand_mj_per_thm=total_demand,
        total_top_gas_mj_per_thm=total_top_gas,
        total_output_mj_per_thm=total_output,
        total_input_mj_per_thm=total_input,
        closure=closure,
        implied_shell_loss_mj_per_thm=implied_shell,
        diagnostics={
            "rates_kg_per_thm": rates,
            "carbon_charged_kg_per_thm": carbon_charged,
            "carbon_to_hot_metal_kg_per_thm": carbon_to_hm,
            "carbon_burnt_kg_per_thm": carbon_burnt,
            "fuel_hydrogen_kg_per_thm": fuel_hydrogen,
            "fuel_hydrogen_mj_per_thm_if_included": hydrogen_supply,
            "fuel_hydrogen_included": include_hydrogen,
            "hydrogen_provenance": hydrogen_provenance,
            "top_gas_nm3_per_thm": v_top,
            "burden_water_kg_per_thm": burden_water_kg,
            "closure_green_range": tuple(closure_cfg.get("green_range", (0.97, 1.03))),
            "closure_amber_range": tuple(closure_cfg.get("amber_range", (0.93, 1.07))),
        },
    )
