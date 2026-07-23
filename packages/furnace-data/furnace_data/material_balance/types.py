"""Typed Material Balance domain objects shared by API and direct mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List

import pandas as pd

from furnace_data.material_balance.data_sources import MaterialBalanceWindow, StaticDatasetSnapshot


@dataclass(frozen=True)
class MaterialBalanceContext:
    """All acquired data needed by the pure Material Balance engine."""

    day: date
    config: dict[str, Any]
    config_version: str
    dataset_snapshot: StaticDatasetSnapshot
    output_window: MaterialBalanceWindow
    raw_material_window: MaterialBalanceWindow
    blast_window: MaterialBalanceWindow
    rm_df: pd.DataFrame
    hm_slag_df: pd.DataFrame
    online: Dict[str, float]
    dpr_df: pd.DataFrame
    dpr_mapping: Dict[str, str | None]
    rm_lag_hours: int = 0
    blast_lag_hours: int = 0
    dust_catcher_t: float = 0.0
    algorithm_version: str = "legacy_v1"
    window_policy_version: str = "hourly_shift_v1"
    data_quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class BalanceResult:
    """Bundle consumed by existing plotters plus richer API adapters."""

    day: date
    inputs: Dict[str, Dict[str, float]]
    outputs: Dict[str, Dict[str, float]]
    closure_table: pd.DataFrame
    material_masses: Dict[str, float]
    gas_phase: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    used_dpr: bool = False
    n_rm_rows: int = 0
    rm_lag_hours: int = 0
    blast_lag_hours: int = 0
    dust_catcher_t: float = 0.0
    algorithm_version: str = "legacy_v1"
    window_policy_version: str = "hourly_shift_v1"
    versions: dict[str, str] = field(default_factory=dict)
    windows: dict[str, MaterialBalanceWindow] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    material_sources: Dict[str, dict[str, Any]] = field(default_factory=dict)
    assumptions: list[dict[str, Any]] = field(default_factory=list)