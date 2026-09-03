"""Blast-furnace energy balance.

Pure math with no Streamlit and no database, on the same pattern as
``utils.material_balance``. See ``compute`` for the convention, and
``docs/energy_balance_calculation_procedure.md`` for a worked day.
"""

from utils.energy_balance.assumptions import (
    ASSUMPTIONS,
    Assumption,
    apply_overrides,
    current_values,
    load_overrides,
    save_overrides,
)
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
    "ASSUMPTIONS",
    "Assumption",
    "EnergyBalanceInputs",
    "EnergyBalanceResult",
    "apply_overrides",
    "current_values",
    "hydrogen_pct_for_fuel",
    "load_config",
    "load_overrides",
    "run_energy_balance",
    "save_overrides",
    "top_gas_volume_nm3_per_thm",
]
