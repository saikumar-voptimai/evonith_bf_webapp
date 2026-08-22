"""Physical constants and config loading for the energy balance.

Every constant has a documented basis. Defaults here mirror
``src/config/energy_balance.yml`` so the math layer works standalone in tests
without a config file present.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "energy_balance.yml"

# Molecular / stoichiometric constants. These are physics, not settings.
FE_IN_FEO_FRACTION = 55.845 / 71.844
H2O_TO_H_FRACTION = 2.016 / 18.015
NM3_PER_KMOL = 22.414
C_MOLAR_MASS = 12.011

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "demand": {
        "fe_reduction_mj_per_kg_fe": 7.38,
        "fe_to_feo_mj_per_kg_fe": 2.56,
        "hot_metal_mj_per_t": 1378.0,
        "slag_mj_per_kg": 1.80,
        "silicon_mj_per_kg": 24.6,
        "manganese_mj_per_kg": 4.8,
        "burden_moisture_mj_per_kg": 2.70,
        "calcination_mj_per_kg_co2": 2.46,
    },
    "supply": {
        "carbon_full_mj_per_kg": 32.8,
        "hydrogen_lhv_mj_per_kg": 120.0,
        "include_fuel_hydrogen": False,
        "blast_cp_kj_per_nm3_k": 1.40,
        "reference_temperature_c": 25.0,
    },
    "top_gas": {
        "n2_in_air_pct": 79.2,
        "co_lhv_mj_per_nm3": 12.63,
        "h2_lhv_mj_per_nm3": 10.78,
        "cp_kj_per_nm3_k": 1.38,
        "min_n2_top_pct": 30.0,
    },
    "fuels": {
        "carbon_fraction": {"coke": 0.87, "nut_coke": 0.87, "pci": 0.75},
        "hydrogen_pct": {"coke": None, "nut_coke": None, "pci": None},
        "hydrogen_from_vm_factor": {"coke": 0.25, "nut_coke": 0.25, "pci": 0.25},
    },
    "blast_moisture_g_per_nm3": 15.0,
    "shell_loss": {
        "heatload_mw_to_gj_per_hr": 3.6,
        "valid_range_mw": [2.0, 12.0],
        "scale_to_all_circuits": True,
    },
    "closure": {"green_range": [0.97, 1.03], "amber_range": [0.93, 1.07]},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge an override onto defaults, so a partial yml stays valid."""

    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """
    Load ``energy_balance`` settings, merged onto the defaults above.

    Args:
         - path: str | None - Override config path, mainly for tests.

    Returns:
         - return dict[str, Any] - Fully populated settings.
    """

    target = Path(path) if path else CONFIG_PATH
    if not target.exists():
        return dict(DEFAULTS)
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return dict(DEFAULTS)
    return _merge(DEFAULTS, loaded.get("energy_balance", {}) or {})


def hydrogen_pct_for_fuel(
    fuel_id: str, vm_pct: float, cfg: dict[str, Any] | None = None
) -> tuple[float, str]:
    """
    Hydrogen content of a fuel, measured if available and estimated otherwise.

    This plant has no ultimate analysis and the vendor does not supply one:
    ``offline_feed.fuel_chemistry`` carries proximate analysis only. The
    configured values are therefore fixed from published data for the coal's
    RANK, which proximate analysis is sufficient to establish - the plant's PCI
    is medium-volatile bituminous at 22.4% VM daf. See ``energy_balance.yml``
    for the derivation.

    Nothing here is a measurement, so the provenance string is returned
    alongside the number and no caller may drop it. Note that "configured" now
    means "literature value for the rank", not "from a certificate".

    The VM fallback applies only where a value is null. It is a backstop
    calibrated to this plant's typical VM, not a general correlation.

    Args:
         - fuel_id: str - "coke", "nut_coke" or "pci".
         - vm_pct: float - Volatile matter on dry basis.
         - cfg: dict[str, Any] | None - Loaded config.

    Returns:
         - return tuple[float, str] - (hydrogen %, provenance).
    """

    settings = cfg or load_config()
    fuels = settings.get("fuels", {})
    measured = (fuels.get("hydrogen_pct") or {}).get(fuel_id)
    if measured is not None:
        try:
            # NOT "ultimate analysis" - no such analysis exists for these fuels.
            # The string is surfaced in diagnostics and must not imply a
            # measurement that was never made.
            return float(measured), "configured (literature value for coal rank)"
        except (TypeError, ValueError):
            pass
    factor = float((fuels.get("hydrogen_from_vm_factor") or {}).get(fuel_id, 0.25))
    return max(0.0, float(vm_pct or 0.0)) * factor, f"estimated from VM x {factor}"
