"""Material Balance Visualiser — pure-Python element balance for BF2.

Submodules:
    constants    — atomic weights, oxide→element table, MaterialSpec registry
    data_sources — Streamlit-cached day-window fetchers
    dpr_mapping  — load/save/apply DPR field mapping
    compute      — element-balance math + run_full_balance(day) entry point

The package is intentionally decoupled from Streamlit (only data_sources
imports streamlit, and only for caching). compute.* and constants.* can be
unit-tested as plain Python.
"""
