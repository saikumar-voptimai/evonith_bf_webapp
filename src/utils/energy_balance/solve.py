"""Solve the closed balance for coke rate, and derive control coefficients.

WHAT THIS DOES, precisely.

``run_energy_balance`` takes a measured day and reports how well it closes.
This module inverts that: it holds the burden and the control settings fixed and
asks **what coke rate would close the balance**. Perturbing one control and
re-solving then gives d(coke)/d(control) - a coefficient derived from physics
rather than guessed or regressed.

WHY A SIMPLE FORMULA IS WRONG HERE.

The tempting shortcut is

    d(coke)/d(T_blast) = cbv x cp / (32.8 x C_frac) = -5.8 kg per 100 C

which holds everything except blast sensible heat fixed. But the top gas is not
fixed while coke moves. Less coke means less carbon gasified, which means:

    fewer Nm3 of CO and CO2 in the top gas
      -> smaller top gas volume
      -> less chemical energy leaving unburnt
      -> less heat needed
      -> a FURTHER coke reduction

That feedback is why the plant's own figure is -8 to -12 rather than -5.8. To
capture it the solve must carry a CARBON balance alongside the nitrogen balance:

    V_N2       = blast volume x (79.2 - O2 enrichment) / 100      (inert, fixed)
    V_C_gas    = carbon burnt / 12.011 x 22.414                   (moves with coke)
    V_top      = V_N2 + V_C_gas + V_H2
    eta_CO     = CO2 / (CO + CO2)                                 (held at observed)
    V_CO       = (1 - eta_CO) x V_C_gas ;  V_CO2 = eta_CO x V_C_gas

Everything else - iron reduction, hot metal, slag, shell loss - is burden and
operation, and does not move with coke rate, so it stays fixed during the solve.

WHAT IS HELD FIXED, and why that is a real limitation.

eta_CO is held at its observed value. In reality raising blast temperature or
oxygen would shift gas utilisation somewhat, and that second-order effect is not
captured. The coefficients below are therefore thermal-displacement
coefficients at constant gas utilisation. Top pressure has no thermal term at
all - it acts through eta_CO - so it CANNOT be derived here and is reported as
such rather than being given a fabricated number.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from scipy.optimize import brentq

from utils.energy_balance.compute import _f, run_energy_balance
from utils.energy_balance.constants import load_config
from utils.energy_balance.types import EnergyBalanceInputs

C_MOLAR_MASS = 12.011
NM3_PER_KMOL = 22.414

# Controls that can be perturbed, and the step used for the finite difference.
# Steps are chosen to be large enough to dominate solver tolerance and small
# enough to stay in the locally linear region.
CONTROL_STEPS: dict[str, float] = {
    "blast_temperature_c": 25.0,
    "oxygen_enrichment_pct": 0.5,
    "blast_volume_nm3_per_hr": 2000.0,
    "pci_mt": 10.0,
}

# How each derivative is best read by an operator, and what to compare against.
REPORTING: dict[str, dict[str, Any]] = {
    "blast_temperature_c": {
        "scale": 100.0, "unit": "kg coke per 100 C",
        "benchmark": "-8 to -12 (config -10.0)",
    },
    "oxygen_enrichment_pct": {
        "scale": 1.0, "unit": "kg coke per 1% O2 enrichment",
        "benchmark": "-1 to -3 typical; this solve holds eta_CO fixed so it "
                     "misses the gas-utilisation gain and reads low",
    },
    "blast_volume_nm3_per_hr": {
        "scale": 1000.0, "unit": "kg coke per 1000 Nm3/hr",
        "benchmark": "near neutral: more N2 to heat, but more O2 to burn",
    },
    "pci_mt": {
        "scale": None, "unit": "kg coke per kg PCI (replacement ratio)",
        "benchmark": "plant uses 0.53; the balance gives the pure "
                     "carbon-equivalence 0.75/0.87 = 0.86 and CANNOT see coke's "
                     "mechanical role in the burden column, so prefer 0.53",
    },
}


def _closure_residual(
    coke_mt: float, inputs: EnergyBalanceInputs, cfg: dict[str, Any]
) -> float:
    """(output - input) in MJ/tHM for a candidate coke tonnage.

    The top gas is recomputed from the carbon balance at this coke rate, not
    taken from the measured analysis, which is what makes the feedback real.
    """

    hm = _f(inputs.hot_metal_mt)
    trial = replace(inputs, coke_mt=coke_mt)

    fuels = cfg["fuels"]["carbon_fraction"]
    carbon_burnt = (
        (_f(coke_mt) / hm * 1000.0) * _f(fuels["coke"])
        + (_f(inputs.nut_coke_mt) / hm * 1000.0) * _f(fuels["nut_coke"])
        + (_f(inputs.pci_mt) / hm * 1000.0) * _f(fuels["pci"])
        - _f(inputs.hm_carbon_pct) / 100.0 * 1000.0
    )
    if carbon_burnt <= 0.0:
        return -1.0e9

    gas_cfg = cfg["top_gas"]
    blast_per_thm = _f(inputs.blast_volume_nm3_per_hr) * 24.0 / hm
    v_n2 = blast_per_thm * (
        _f(gas_cfg["n2_in_air_pct"]) - _f(inputs.oxygen_enrichment_pct)
    ) / 100.0

    # Carbon leaves as CO + CO2, so its gas volume follows the coke rate.
    v_c_gas = carbon_burnt / C_MOLAR_MASS * NM3_PER_KMOL

    # H2 is held at the measured Nm3, derived from the observed analysis against
    # the observed top gas volume. It is ~3% of the gas and only weakly coupled
    # to coke rate, so holding it is a smaller error than modelling it badly.
    measured = run_energy_balance(inputs, cfg)
    v_top_measured = _f(measured.diagnostics["top_gas_nm3_per_thm"])
    v_h2 = v_top_measured * _f(inputs.top_gas_h2_pct) / 100.0

    v_top = v_n2 + v_c_gas + v_h2
    co_plus_co2 = _f(inputs.top_gas_co_pct) + _f(inputs.top_gas_co2_pct)
    eta_co = _f(inputs.top_gas_co2_pct) / co_plus_co2 if co_plus_co2 > 0 else 0.45
    v_co = (1.0 - eta_co) * v_c_gas

    t_ref = _f(cfg["supply"]["reference_temperature_c"], 25.0)
    q_topgas = (
        v_top * _f(gas_cfg["cp_kj_per_nm3_k"])
        * (_f(inputs.top_gas_temperature_c) - t_ref) / 1000.0
        + v_co * _f(gas_cfg["co_lhv_mj_per_nm3"])
        + v_h2 * _f(gas_cfg["h2_lhv_mj_per_nm3"])
    )

    # Demand is burden and operation; it does not move with coke rate. Take it
    # from the full balance so every term stays consistent with `compute`.
    demand = sum(run_energy_balance(trial, cfg).demand.values())

    q_carbon = carbon_burnt * _f(cfg["supply"]["carbon_full_mj_per_kg"])
    q_blast = (
        blast_per_thm * _f(cfg["supply"]["blast_cp_kj_per_nm3_k"])
        * (_f(inputs.blast_temperature_c) - t_ref) / 1000.0
    )
    if bool(cfg["supply"].get("include_fuel_hydrogen", False)):
        q_carbon += _f(
            run_energy_balance(trial, cfg).diagnostics[
                "fuel_hydrogen_mj_per_thm_if_included"
            ]
        )

    return (demand + q_topgas) - (q_carbon + q_blast)


def solve_coke_rate_kg_per_thm(
    inputs: EnergyBalanceInputs, cfg: dict[str, Any] | None = None
) -> float:
    """
    Coke rate that closes the energy balance for this burden and these settings.

    Args:
         - inputs: EnergyBalanceInputs - Burden and control settings. The
           ``coke_mt`` field is the starting point only; it is solved for.
         - cfg: dict[str, Any] | None - Loaded settings.

    Returns:
         - return float - Coke rate in kg/tHM.

    Raises:
         - ValueError - If no coke rate in a physically sane 50-900 kg/tHM band
           closes the balance, which means the inputs are inconsistent.
    """

    settings = cfg or load_config()
    hm = _f(inputs.hot_metal_mt)
    lo_mt, hi_mt = 50.0 * hm / 1000.0, 900.0 * hm / 1000.0

    f_lo = _closure_residual(lo_mt, inputs, settings)
    f_hi = _closure_residual(hi_mt, inputs, settings)
    if f_lo * f_hi > 0:
        raise ValueError(
            "no coke rate between 50 and 900 kg/tHM closes the balance "
            f"(residuals {f_lo:,.0f} and {f_hi:,.0f} MJ/tHM); inputs are inconsistent"
        )
    solved_mt = brentq(_closure_residual, lo_mt, hi_mt, args=(inputs, settings),
                       xtol=1e-4, rtol=1e-8)
    return solved_mt / hm * 1000.0


def derive_control_coefficients(
    inputs: EnergyBalanceInputs, cfg: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """
    d(coke rate) / d(control), by central difference on the closed balance.

    These are thermal-displacement coefficients at constant gas utilisation.
    See the module docstring for what is held fixed and why.

    Args:
         - inputs: EnergyBalanceInputs - The operating point to linearise about.
         - cfg: dict[str, Any] | None - Loaded settings.

    Returns:
         - return dict - Per control: the derivative, a per-100-unit figure for
           readability, and the step used.
    """

    settings = cfg or load_config()
    base = solve_coke_rate_kg_per_thm(inputs, settings)
    out: dict[str, dict[str, Any]] = {
        "_base_coke_rate_kg_per_thm": {"value": base},
    }

    for control, step in CONTROL_STEPS.items():
        current = _f(getattr(inputs, control))
        try:
            up = solve_coke_rate_kg_per_thm(
                replace(inputs, **{control: current + step}), settings
            )
            down = solve_coke_rate_kg_per_thm(
                replace(inputs, **{control: current - step}), settings
            )
        except ValueError:
            out[control] = {"derivative": None, "note": "no solution within band"}
            continue
        derivative = (up - down) / (2.0 * step)
        report = REPORTING.get(control, {})
        entry: dict[str, Any] = {
            "derivative_kg_coke_per_unit": derivative,
            "step": step,
            "at": current,
            "unit": report.get("unit", "kg coke per unit"),
            "benchmark": report.get("benchmark", ""),
        }
        if control == "pci_mt":
            # pci_mt is tonnes/day; express it as the kg-for-kg replacement ratio
            # the plant actually thinks in.
            pci_kg_per_thm_per_mt = 1000.0 / _f(inputs.hot_metal_mt)
            entry["reported_value"] = -derivative / pci_kg_per_thm_per_mt
        else:
            entry["reported_value"] = derivative * _f(report.get("scale", 1.0), 1.0)
        out[control] = entry

    # Top pressure acts through gas utilisation, not through the heat balance,
    # so it has no thermal derivative. Reporting a number here would be
    # fabricating one.
    out["top_pressure_bar"] = {
        "derivative_kg_coke_per_unit": None,
        "note": (
            "not derivable from the energy balance - top pressure acts on eta_CO, "
            "which this solve holds fixed. Needs an empirical term."
        ),
    }
    return out
