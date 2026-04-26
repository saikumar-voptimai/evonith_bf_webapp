"""Persistence + discovery helpers for the DPR field mapping.

The DPR (`dpr_data`) measurement's column names are not documented in
the codebase. To avoid hard-coding guesses, the Material Balance page
exposes a one-time mapping UI: it lists every column found on a sample
DPR row and asks the user to map nine canonical fields to them. The
chosen mapping is persisted to ``src/config/material_balance.yml`` and
read on every page load.

Functions:
    load_dpr_mapping       — read mapping from yml (returns default skeleton if missing)
    save_dpr_mapping       — atomic write back to yml
    discover_dpr_fields    — list of column names from a recent DPR row
    apply_dpr_mapping      — translate raw DPR row → canonical mass dict
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

log = logging.getLogger("root")

# Canonical mass-field keys the compute layer expects.
CANONICAL_MASS_FIELDS: List[str] = [
    "hm_mass_t",
    "slag_mass_t",
    "coke_mass_t",
    "nut_coke_mass_t",
    "pci_mass_t",
    "ore_mass_t",
    "sinter_mass_t",
    "pellet_mass_t",
    "flux_mass_t",
]

#: Ordered ash-analysis fields (yml key, display label, kind).
#: kind ``"oxide"`` → element + O via OXIDE_TO_ELEMENT_MASS_FRAC.
#: kind ``"direct"`` → elemental wt% of ash (e.g. S, P).
ASH_ANALYSIS_FIELDS: List[Tuple[str, str, str]] = [
    ("SiO2",  "SiO\u2082",   "oxide"),
    ("Al2O3", "Al\u2082O\u2083", "oxide"),
    ("Fe2O3", "Fe\u2082O\u2083", "oxide"),
    ("CaO",   "CaO",    "oxide"),
    ("MgO",   "MgO",    "oxide"),
    ("TiO2",  "TiO\u2082",   "oxide"),
    ("Na2O",  "Na\u2082O",   "oxide"),
    ("K2O",   "K\u2082O",    "oxide"),
    ("S",     "S",      "direct"),
    ("P",     "P",      "direct"),
]

#: Maps ``MaterialSpec.ash_assumption_key`` → yml config block name.
ASH_MATERIAL_CONFIG_KEYS: Dict[str, str] = {
    "coke":    "coke_ash_analysis_pct",
    "nutcoke": "nutcoke_ash_analysis_pct",
    "pci":     "pci_ash_analysis_pct",
}

_YML_FILENAME = "material_balance.yml"


def _yml_path() -> Path:
    """Resolve the absolute path of ``src/config/material_balance.yml``."""
    # This file lives at src/utils/material_balance/dpr_mapping.py;
    # config/ is parent.parent.parent / "config".
    return Path(__file__).resolve().parents[2] / "config" / _YML_FILENAME


def _default_skeleton() -> dict:
    """Fallback yml content used when material_balance.yml is missing."""
    return {
        "elements": ["Fe", "C", "Si", "Ca", "Mg", "Al", "Mn", "S", "P", "O", "N", "H"],
        "constants": {
            "rho_air_ntp_kg_per_nm3": 1.293,
            "air_o2_vol_frac": 0.208,
            "air_n2_vol_frac": 0.792,
            "molar_volume_ntp_nm3_per_kmol": 22.414,
            "air_o2_mass_frac": 0.232,
            "air_n2_mass_frac": 0.755,
        },
        "coke_ash_analysis_pct": {
            "SiO2": 55.60, "Al2O3": 27.29, "Fe2O3": 6.85,
            "CaO": 3.26, "MgO": 1.21, "TiO2": 1.47,
            "Na2O": 0.07, "K2O": 0.18, "S": 0.72, "P": 0.042,
        },
        "nutcoke_ash_analysis_pct": {
            "SiO2": 55.64, "Al2O3": 27.25, "Fe2O3": 7.13,
            "CaO": 3.12, "MgO": 1.12, "TiO2": 1.475,
            "Na2O": 0.07, "K2O": 0.17, "S": 0.76, "P": 0.040,
        },
        "pci_ash_analysis_pct": {
            "SiO2": 47.12, "Al2O3": 24.68, "Fe2O3": 7.88,
            "CaO": 7.68, "MgO": 2.09, "TiO2": 0.857,
            "Na2O": 0.663, "K2O": 1.524, "S": 0.40, "P": 0.042,
        },
        "dpr_field_mapping": {k: None for k in CANONICAL_MASS_FIELDS},
        "future_streams": {
            "dust_catcher_t": None,
            "sludge_t": None,
            "slag_granulation_loss": None,
        },
        "closure_thresholds": {"good": [95, 105], "warning": [85, 115]},
    }


def load_full_config() -> dict:
    """Read the entire ``material_balance.yml`` file (or default skeleton).

    Returns:
        dict: Full configuration with missing keys back-filled from the
        built-in default skeleton for forward compatibility.
    """
    p = _yml_path()
    if not p.is_file():
        return _default_skeleton()
    try:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        log.warning("Failed to read %s: %s — using defaults", p, exc)
        return _default_skeleton()

    # Backward-compat: migrate old _assumption_ key names → _analysis_.
    for old, new in (
        ("coke_ash_assumption_pct", "coke_ash_analysis_pct"),
        ("pci_ash_assumption_pct",  "pci_ash_analysis_pct"),
    ):
        if old in cfg and new not in cfg:
            cfg[new] = cfg.pop(old)

    # Merge missing keys from skeleton (forward-compat).
    skel = _default_skeleton()
    for k, v in skel.items():
        cfg.setdefault(k, v)
    return cfg


def load_dpr_mapping() -> Dict[str, str | None]:
    """Return just the ``dpr_field_mapping`` block.

    Returns:
        dict: ``{canonical_field: dpr_column_name | None}`` for every
        field in :data:`CANONICAL_MASS_FIELDS`.
    """
    cfg = load_full_config()
    raw = cfg.get("dpr_field_mapping") or {}
    return {k: raw.get(k) for k in CANONICAL_MASS_FIELDS}


def save_dpr_mapping(mapping: Dict[str, str | None]) -> None:
    """Atomically write a new mapping back to ``material_balance.yml``.

    Reads-modifies-writes the file via a ``.tmp`` sibling so a crash
    mid-write cannot leave the yml truncated.

    Args:
        mapping (dict): ``{canonical_field: dpr_column_name | None}``
            for every field in :data:`CANONICAL_MASS_FIELDS`.
    """
    p = _yml_path()
    cfg = load_full_config()
    cfg["dpr_field_mapping"] = {
        k: (mapping.get(k) or None) for k in CANONICAL_MASS_FIELDS
    }

    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".material_balance_", suffix=".tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        os.replace(tmp_path, p)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def discover_dpr_fields(day: date) -> List[str]:
    """Return the column names of a one-day DPR fetch (sorted, dedup'd).

    Imports inside the function to keep this module Streamlit-free at
    module-load time.

    Args:
        day (date): IST calendar date used to discover available columns.

    Returns:
        list: Sorted unique column names from the DPR DataFrame.
        Returns ``[]`` if no DPR data exists for the given day.
    """
    from utils.material_balance.data_sources import fetch_dpr_for_day

    df = fetch_dpr_for_day(day)
    if df is None or df.empty:
        return []
    return sorted({str(c) for c in df.columns})


def apply_dpr_mapping(
    dpr_df: pd.DataFrame,
    mapping: Dict[str, str | None],
) -> Dict[str, float]:
    """Translate a raw DPR DataFrame into ``{canonical_field: mass_t}``.

    If multiple rows are returned for the day (rare — DPR is daily) the
    sum is taken: DPR fields are typically already-aggregated daily
    totals, so summing the (usually single) row equals the row value.
    Missing or unmapped fields silently produce 0.0.

    Args:
        dpr_df (pd.DataFrame): Raw DPR row(s) from
            :func:`~utils.material_balance.data_sources.fetch_dpr_for_day`.
        mapping (dict): ``{canonical_field: dpr_column_name | None}``
            from :func:`load_dpr_mapping`.

    Returns:
        dict: ``{canonical_field: mass_t}`` for every field in
        :data:`CANONICAL_MASS_FIELDS`.
    """
    if dpr_df is None or dpr_df.empty:
        return {k: 0.0 for k in CANONICAL_MASS_FIELDS}

    out: Dict[str, float] = {}
    for canonical in CANONICAL_MASS_FIELDS:
        col = mapping.get(canonical)
        if not col or col not in dpr_df.columns:
            out[canonical] = 0.0
            continue
        try:
            v = pd.to_numeric(dpr_df[col], errors="coerce").sum()
            out[canonical] = float(v) if pd.notna(v) else 0.0
        except Exception:  # noqa: BLE001
            out[canonical] = 0.0
    return out


def save_ash_analyses(analyses: Dict[str, Dict[str, float]]) -> None:
    """Atomically persist ash analysis compositions to ``material_balance.yml``.

    Args:
        analyses (dict): ``{material_key: {species: pct}}`` where
            *material_key* is one of ``"coke"``, ``"nutcoke"``, ``"pci"``
            and *species* matches entries in :data:`ASH_ANALYSIS_FIELDS`.
            Values are wt% of ash.
    """
    p = _yml_path()
    cfg = load_full_config()
    for material_key, analysis in analyses.items():
        yml_key = ASH_MATERIAL_CONFIG_KEYS.get(material_key)
        if yml_key:
            cfg[yml_key] = {
                k: float(v) for k, v in analysis.items() if v is not None
            }
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".material_balance_", suffix=".tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        os.replace(tmp_path, p)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def mapping_is_complete(mapping: Dict[str, str | None]) -> bool:
    """Return True iff every canonical field has a non-empty value.

    Args:
        mapping (dict): ``{canonical_field: dpr_column_name | None}``.

    Returns:
        bool: ``True`` when all 9 canonical fields are mapped.
    """
    return all(bool(mapping.get(k)) for k in CANONICAL_MASS_FIELDS)
