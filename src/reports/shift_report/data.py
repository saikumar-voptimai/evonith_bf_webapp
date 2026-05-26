"""Data models for the shift report pipeline."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

import pandas as pd


@dataclass
class ShiftRawData:
    """Raw DataFrames straight from InfluxDB — no computation done yet."""

    shift_date: date
    shift_label: Literal["A", "B", "C"]
    shift_start_ist: datetime
    shift_end_ist: datetime
    online_df: pd.DataFrame  # 15-min windowed averages, IST-indexed
    hm_slag_df: pd.DataFrame  # one row per tap
    charge_df: pd.DataFrame  # one row per charge
    ore_chemistry_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fuel_chemistry_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    flux_chemistry_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    material_fines_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    materials_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ParamStats:
    """Mean and std-dev pair for a time-series parameter."""

    mean: Optional[float]
    std: Optional[float]


@dataclass
class TempRow:
    """Q1–Q4 readings for one temperature zone; spread std computed on demand."""

    q1: Optional[float]
    q2: Optional[float]
    q3: Optional[float]
    q4: Optional[float]

    @property
    def spread_std(self) -> Optional[float]:
        vals = [v for v in (self.q1, self.q2, self.q3, self.q4) if v is not None]
        return round(statistics.stdev(vals), 2) if len(vals) >= 2 else None


@dataclass
class ShiftReportData:
    """All computed metrics for one 8-hour shift — no DataFrames, no I/O."""

    # Identity
    shift_date: date
    shift_label: Literal["A", "B", "C"]
    shift_start_ist: datetime
    shift_end_ist: datetime

    # Status (rule-based)
    status: Literal["STABLE", "ATTENTION REQUIRED", "UNSTABLE"]
    status_flags: list[str]

    # Shift Report table
    production_rate: Optional[float]  # t/hr
    theoretical_production: Optional[float]  # tons (rate × 8)
    total_charges: Optional[int]
    coke_t: Optional[float]
    nut_coke_t: Optional[float]
    sinter_t: Optional[float]
    ore_t: Optional[float]
    pellet_t: Optional[float]
    flux_t: Optional[float]
    fuel_rate: Optional[float]  # kg/thm
    coke_rate: Optional[float]  # kg/thm
    nut_coke_rate: Optional[float]  # kg/thm
    pci_rate: Optional[float]  # kg/thm
    hm_si: Optional[float]
    hm_s: Optional[float]
    hm_temp: Optional[float]  # degC
    slag_basicity: Optional[float]
    total_taps: Optional[int]

    # Parameters table
    blast_volume: ParamStats
    blast_temp: ParamStats
    blast_pressure: ParamStats
    furnace_top_dp: ParamStats
    furnace_bottom_dp: ParamStats
    furnace_total_dp: ParamStats
    o2_flow: ParamStats
    o2_enrichment: ParamStats
    permeability: ParamStats
    etaco: ParamStats
    raft: ParamStats

    # Temperatures
    uptake: TempRow
    lower_stack: TempRow
    belly: TempRow
    bosh: TempRow

    # Hearth Pad
    hearth_4_3_a: Optional[float]
    hearth_5_4_c: Optional[float]
    hearth_5_7_c: Optional[float]
    hearth_6_1_b: Optional[float]

    # Material quality derived from charge tonnage and latest analysis
    burden_moisture_input: Optional[float] = None  # kg/thm
    fines_input: Optional[float] = None  # kg/thm
    used_materials: dict[str, str] = field(default_factory=dict)
