"""Material Balance Visualiser — pure-Python element balance for BF2.

Submodules:
    constants        — atomic weights, oxide→element table, MaterialSpec registry
    data_sources     — Streamlit-cached day-window fetchers
    dpr_mapping      — load/save/apply DPR field mapping
    _helpers         — shared utility (_get_pct)
    elements         — material → element conversion (kind-switch engine)
    mass_resolution  — DPR-first mass picking with RM fallback
    gas_phase        — blast / steam / top-gas element math
    outputs          — HM, slag, dust-catcher, unaccounted-solids converters
    compute          — orchestrator + closure table + BalanceResult

The package is intentionally decoupled from Streamlit (only data_sources
imports streamlit, and only for caching). All other modules can be
unit-tested as plain Python.
"""

from utils.material_balance.compute import BalanceResult, run_full_balance

__all__ = ["run_full_balance", "BalanceResult"]
