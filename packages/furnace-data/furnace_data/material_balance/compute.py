"""Compatibility entrypoints for the Material Balance engine.

New API code builds a :class:`MaterialBalanceContext` and calls
``MaterialBalanceEngine`` directly.  ``run_full_balance`` remains for direct
Streamlit rollback and legacy callers.
"""

from __future__ import annotations

from datetime import date

from furnace_data.material_balance.context import MaterialBalanceContextBuilder
from furnace_data.material_balance.engine import (
    GAS_INPUT_BLAST,
    GAS_INPUT_O2,
    GAS_INPUT_STEAM,
    OUT_DUST,
    OUT_HM,
    OUT_SLAG,
    OUT_TOPGAS,
    OUT_UNACCOUNTED,
    MaterialBalanceEngine,
    build_closure_table,
)
from furnace_data.material_balance.types import BalanceResult


def run_full_balance(
    day: date,
    rm_lag_hours: int = 0,
    blast_lag_hours: int = 0,
    dust_catcher_t: float = 0.0,
) -> BalanceResult:
    """Compute the full element balance for one IST calendar day."""

    context = MaterialBalanceContextBuilder().build(
        day=day,
        rm_lag_hours=int(rm_lag_hours),
        blast_lag_hours=int(blast_lag_hours),
        dust_catcher_t=float(dust_catcher_t),
        algorithm_version="legacy_v1",
    )
    return MaterialBalanceEngine().compute(context)


__all__ = [
    "BalanceResult",
    "GAS_INPUT_BLAST",
    "GAS_INPUT_O2",
    "GAS_INPUT_STEAM",
    "OUT_DUST",
    "OUT_HM",
    "OUT_SLAG",
    "OUT_TOPGAS",
    "OUT_UNACCOUNTED",
    "build_closure_table",
    "run_full_balance",
]