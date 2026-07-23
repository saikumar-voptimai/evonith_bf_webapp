"""Pure-Python Material Balance domain package for BF2.

The package is decoupled from Streamlit and FastAPI.  API code should build a
``MaterialBalanceContext`` once, run ``MaterialBalanceEngine``, then serialize
with ``build_material_balance_result``.
"""

from furnace_data.material_balance.compute import BalanceResult, run_full_balance
from furnace_data.material_balance.context import MaterialBalanceContextBuilder
from furnace_data.material_balance.engine import MaterialBalanceEngine
from furnace_data.material_balance.result_builder import build_material_balance_result

__all__ = [
    "BalanceResult",
    "MaterialBalanceContextBuilder",
    "MaterialBalanceEngine",
    "build_material_balance_result",
    "run_full_balance",
]