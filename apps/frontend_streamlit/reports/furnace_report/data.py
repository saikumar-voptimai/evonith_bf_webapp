"""Data models for the furnace report pipeline."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

import pandas as pd

ReportType = Literal["Shift", "Day"]
ShiftLabel = Literal["A", "B", "C"]


def report_window_id(
    report_type: ReportType,
    selected_date: date,
    shift_label: Optional[ShiftLabel] = None,
) -> str:
    if report_type == "Shift":
        if shift_label is None:
            raise ValueError("shift_label is required for Shift reports.")
        return f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"
    return selected_date.isoformat()


def report_display_name(
    report_type: ReportType,
    selected_date: date,
    shift_label: Optional[ShiftLabel] = None,
) -> str:
    if report_type == "Shift":
        if shift_label is None:
            raise ValueError("shift_label is required for Shift reports.")
        return f"{selected_date} Shift {shift_label}"
    return f"{selected_date} Day"


def report_duration_hours(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600.0


@dataclass(frozen=True)
class ReportTimeframe:
    """Resolved local-time report window used by fetch, build, and rendering."""

    report_type: ReportType
    selected_date: date
    start_ist: datetime
    end_ist: datetime
    shift_label: Optional[ShiftLabel] = None

    @property
    def duration_hours(self) -> float:
        return report_duration_hours(self.start_ist, self.end_ist)

    @property
    def window_id(self) -> str:
        return report_window_id(self.report_type, self.selected_date, self.shift_label)

    @property
    def display_name(self) -> str:
        return report_display_name(
            self.report_type,
            self.selected_date,
            self.shift_label,
        )


@dataclass
class ShiftRawData:
    """Raw DataFrames straight from source systems; no computation done yet."""

    shift_date: date
    shift_label: Optional[ShiftLabel]
    shift_start_ist: datetime
    shift_end_ist: datetime
    online_df: pd.DataFrame  # 15-min windowed averages, IST-indexed
    hm_slag_df: pd.DataFrame  # one row per tap
    charge_df: pd.DataFrame  # one row per charge
    total_o2_flow_max: Optional[float] = None
    ore_chemistry_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fuel_chemistry_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    flux_chemistry_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    material_fines_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    materials_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    report_type: ReportType = "Shift"

    @property
    def duration_hours(self) -> float:
        return report_duration_hours(self.shift_start_ist, self.shift_end_ist)


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
    """All computed metrics for one report window; no DataFrames, no I/O."""

    # Identity
    shift_date: date
    shift_label: Optional[ShiftLabel]
    shift_start_ist: datetime
    shift_end_ist: datetime

    # Status (rule-based)
    status: Literal["STABLE", "ATTENTION REQUIRED", "UNSTABLE"]
    status_flags: list[str]

    # Core report table
    production_rate: Optional[float]  # t/hr
    theoretical_production: Optional[float]  # tons (rate x window hours)
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
    total_o2_flow: Optional[float]
    o2_enrichment: ParamStats
    permeability: ParamStats
    etaco: ParamStats
    raft: ParamStats

    # Temperatures
    uptake: TempRow
    lower_stack: TempRow
    belly: TempRow
    skin_flow: TempRow
    bosh: TempRow

    # Hearth Pad
    hearth_4_3_a: Optional[float]
    hearth_5_4_c: Optional[float]
    hearth_5_7_c: Optional[float]
    hearth_6_1_b: Optional[float]

    # Material quality derived from charge tonnage and latest analysis
    burden_moisture_input: Optional[float] = None  # kg/thm
    fines_input: Optional[float] = None  # kg/thm
    burden_ratio: Optional[str] = None
    ibrm: Optional[float] = None
    used_materials: dict[str, str] = field(default_factory=dict)
    report_type: ReportType = "Shift"

    @property
    def duration_hours(self) -> float:
        return report_duration_hours(self.shift_start_ist, self.shift_end_ist)

    @property
    def window_id(self) -> str:
        return report_window_id(self.report_type, self.shift_date, self.shift_label)

    @property
    def display_name(self) -> str:
        return report_display_name(self.report_type, self.shift_date, self.shift_label)
