"""Blast-furnace energy balance.

Pure math with no Streamlit and no database, on the same pattern as
``utils.material_balance``. See ``compute`` for the convention, and
``docs/energy_balance_calculation_procedure.md`` for a worked day.
"""

from utils.energy_balance.compute import (
    run_energy_balance,
    top_gas_volume_nm3_per_thm,
)
from utils.energy_balance.constants import (
    hydrogen_pct_for_fuel,
    load_config,
)
from utils.energy_balance.types import EnergyBalanceInputs, EnergyBalanceResult

__all__ = [
    "EnergyBalanceInputs",
    "EnergyBalanceResult",
    "hydrogen_pct_for_fuel",
    "load_config",
    "run_energy_balance",
    "top_gas_volume_nm3_per_thm",
]
