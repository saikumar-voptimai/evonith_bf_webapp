"""Compute all shift metrics from raw DataFrames — pure Python, no I/O."""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

from reports.base import ReportBuilder
from reports.shift_report.data import (
    ParamStats,
    ShiftRawData,
    ShiftReportData,
    TempRow,
)

# ── Column name registry ────────────────────────────────────────────────────
# Keys are short internal names; values are the exact DataFrame column names
# produced by fetch_online_df() and fetch_offline_data().

_ONLINE: dict[str, str] = {
    # process_params
    "prod_rate":     "Process Params - BF2_PRODUCTION TONNES PER HR",
    "charges_hr":    "Process Params - BF2_CHARGES PER HR",
    "hb_vol":        "Process Params - BF2_PROC Hot Blast Volume",
    "hb_temp":       "Process Params - BF2_PROC Hot Blast Temp",
    "hb_press":      "Process Params - BF2_PROC Hot Blast Pressure",
    "perm":          "Process Params - BF2_BODY_PERMEABILITY",
    "etaco":         "Process Params - BF2_BODY_ETACO",
    "raft":          "Process Params - BF2_BODY_RAFT",
    "o2_enr":        "Process Params - BF2_OXYGEN ENRICHMENT PCT",
    "fuel_rate":     "Process Params - BF2_FUEL RATE PER THM",
    "coke_rate":     "Process Params - BF2_COKE RATE PER THM",
    "nutcoke_rate":  "Process Params - BF2_NUT COKE RATE PER THM",
    "pci_rate":      "Process Params - BF2_COAL RATE PER THM",
    "runner_temp":   "Process Params - TE_40532A Runner Temp PCI side near to Taphole",
    # temperature_profile
    "hearth_4_3_a":  "Temperature Profile - BF2_BFBD Furnace Body 4373mm Temp A",
    "hearth_5_4_c":  "Temperature Profile - BF2_BFBD Furnace Body 5411mm Temp C",
    "hearth_5_7_c":  "Temperature Profile - BF2_BFBD Furnace Body 5757mm Temp C",
    "hearth_6_1_b":  "Temperature Profile - BF2_BFBD Furnace Body 6103mm Temp B",
    "ls_q1":         "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp A",
    "ls_q2":         "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp B",
    "ls_q3":         "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp C",
    "ls_q4":         "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp D",
    "belly_q1":      "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp A",
    "belly_q2":      "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp B",
    "belly_q3":      "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp C",
    "belly_q4":      "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp D",
    "uptake_q1":     "Process Params - BF2_PROC Top Temp 1",
    "uptake_q2":     "Process Params - BF2_PROC Top Temp 2",
    "uptake_q3":     "Process Params - BF2_PROC Top Temp 3",
    "uptake_q4":     "Process Params - BF2_PROC Top Temp 4",
    # delta_t
    "bosh_q1":       "Delta T - DELTA T avg Row6-10 Q1(Stave 1-8)",
    "bosh_q2":       "Delta T - DELTA T avg Row6-10 Q2(Stave 9-16)",
    "bosh_q3":       "Delta T - DELTA T avg Row6-10 Q3(Stave 17-24)",
    "bosh_q4":       "Delta T - DELTA T avg Row6-10 Q4(Stave 25-32)",
}

_HM: dict[str, str] = {
    "si":        "chem_pct_si",
    "s":         "chem_pct_s",
    "hm_temp":   "hm_temp",
    "slag_cao":  "slag_pct_cao",
    "slag_sio2": "slag_pct_sio2",
}

# CHARGE column names — list = try in order (fallback aliases)
_CH: dict[str, str | list[str]] = {
    "coke":     "coke_total_mt",
    "nut_coke": ["total_nutcoke_mt", "nut_coke_mt"],
    "sinter":   "sinter_mt",
    "ore":      "ore_mt",
    "pellet":   ["ll_pellet_mt", "pellet_mt"],
    "flux":     "flux_mt",
    "pci":      "pci_mt",
}

# ── Status thresholds ────────────────────────────────────────────────────────
_THRESH = {
    "etaco":     {"warn": 42.0, "crit": 40.0},
    "fuel_rate": {"warn": 530.0, "crit": 570.0},
    "raft_lo":   2100.0,
    "raft_hi":   2350.0,
    "perm_lo":   1000.0,
    "perm_hi":   1600.0,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mean(df: pd.DataFrame, key: str) -> Optional[float]:
    col = _ONLINE.get(key, key)
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.mean()), 2) if len(s) else None


def _std(df: pd.DataFrame, key: str) -> Optional[float]:
    col = _ONLINE.get(key, key)
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.std()), 2) if len(s) >= 2 else None


def _ps(df: pd.DataFrame, key: str) -> ParamStats:
    return ParamStats(mean=_mean(df, key), std=_std(df, key))


def _hm_mean(df: pd.DataFrame, key: str) -> Optional[float]:
    col = _HM.get(key, key)
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.mean()), 2) if len(s) else None


def _charge_sum(df: pd.DataFrame, key: str) -> Optional[float]:
    cols = _CH[key]
    if isinstance(cols, str):
        cols = [cols]
    if df.empty:
        return None
    for col in cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                v = float(s.sum())
                return round(v, 2) if v > 0 else None
    return None


def _temp_row(df: pd.DataFrame, q1: str, q2: str, q3: str, q4: str) -> TempRow:
    return TempRow(
        q1=_mean(df, q1),
        q2=_mean(df, q2),
        q3=_mean(df, q3),
        q4=_mean(df, q4),
    )


def _status(
    etaco: Optional[float],
    fuel_rate: Optional[float],
    raft: Optional[float],
    perm: Optional[float],
) -> tuple[Literal["STABLE", "ATTENTION REQUIRED", "UNSTABLE"], list[str]]:
    flags: list[str] = []

    if etaco is not None:
        if etaco < _THRESH["etaco"]["crit"]:
            flags.append(f"ETA CO {etaco:.2f}% (critical < 40%)")
        elif etaco < _THRESH["etaco"]["warn"]:
            flags.append(f"ETA CO {etaco:.2f}% (< 42% normal)")

    if fuel_rate is not None:
        if fuel_rate > _THRESH["fuel_rate"]["crit"]:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (critical > 570)")
        elif fuel_rate > _THRESH["fuel_rate"]["warn"]:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (> 530 attention)")

    if raft is not None and (raft < _THRESH["raft_lo"] or raft > _THRESH["raft_hi"]):
        flags.append(f"RAFT {raft:.2f}°C (normal 2100–2350)")

    if perm is not None and (perm < _THRESH["perm_lo"] or perm > _THRESH["perm_hi"]):
        flags.append(f"Permeability {perm:.2f} (normal 1000–1600)")

    n = len(flags)
    if n == 0:
        return "STABLE", flags
    if n <= 2:
        return "ATTENTION REQUIRED", flags
    return "UNSTABLE", flags


# ── Builder ──────────────────────────────────────────────────────────────────

class ShiftBuilder(ReportBuilder[ShiftRawData, ShiftReportData]):
    def build(self, raw: ShiftRawData) -> ShiftReportData:  # type: ignore[override]
        df = raw.online_df
        hm = raw.hm_slag_df
        ch = raw.charge_df

        # Total charges: Σ(charges_per_hr × 0.25 h per 15-min row)
        charges_col = _ONLINE["charges_hr"]
        if not df.empty and charges_col in df.columns:
            s = df[charges_col].dropna()
            total_charges = round(float(s.sum()) * 0.25) if len(s) else None
        else:
            total_charges = None

        # O2 flow (derived)
        hb_vol = _mean(df, "hb_vol")
        o2_enr = _mean(df, "o2_enr")
        o2_flow_mean = round(hb_vol * o2_enr / 100, 2) if hb_vol and o2_enr else None

        # Slag basicity
        cao = _hm_mean(hm, "slag_cao")
        sio2 = _hm_mean(hm, "slag_sio2")
        slag_basicity = round(cao / sio2, 2) if cao and sio2 else None

        # HM temp: use from hot_metal_slag_data reports 
        hm_temp = _hm_mean(hm, "hmt_gt_1480c")

        prod_rate = _mean(df, "prod_rate")
        fuel_rate_val = _mean(df, "fuel_rate")
        raft_val = _mean(df, "raft")
        perm_val = _mean(df, "perm")
        etaco_val = _mean(df, "etaco")

        status, flags = _status(etaco_val, fuel_rate_val, raft_val, perm_val)

        return ShiftReportData(
            shift_date=raw.shift_date,
            shift_label=raw.shift_label,
            shift_start_ist=raw.shift_start_ist,
            shift_end_ist=raw.shift_end_ist,
            status=status,
            status_flags=flags,
            production_rate=prod_rate,
            theoretical_production=round(prod_rate * 8, 2) if prod_rate else None,
            total_charges=total_charges,
            coke_t=_charge_sum(ch, "coke"),
            nut_coke_t=_charge_sum(ch, "nut_coke"),
            sinter_t=_charge_sum(ch, "sinter"),
            ore_t=_charge_sum(ch, "ore"),
            pellet_t=_charge_sum(ch, "pellet"),
            flux_t=_charge_sum(ch, "flux"),
            fuel_rate=fuel_rate_val,
            coke_rate=_mean(df, "coke_rate"),
            nut_coke_rate=_mean(df, "nutcoke_rate"),
            pci_rate=_mean(df, "pci_rate"),
            hm_si=_hm_mean(hm, "si"),
            hm_s=_hm_mean(hm, "s"),
            hm_temp=hm_temp,
            slag_basicity=slag_basicity,
            total_taps=int(hm.dropna(how="all").shape[0]) if not hm.empty else None,
            blast_volume=_ps(df, "hb_vol"),
            blast_temp=_ps(df, "hb_temp"),
            blast_pressure=_ps(df, "hb_press"),
            o2_flow=ParamStats(mean=o2_flow_mean, std=None),
            o2_enrichment=_ps(df, "o2_enr"),
            permeability=_ps(df, "perm"),
            etaco=_ps(df, "etaco"),
            raft=_ps(df, "raft"),
            uptake=_temp_row(df, "uptake_q1", "uptake_q2", "uptake_q3", "uptake_q4"),
            lower_stack=_temp_row(df, "ls_q1", "ls_q2", "ls_q3", "ls_q4"),
            belly=_temp_row(df, "belly_q1", "belly_q2", "belly_q3", "belly_q4"),
            bosh=_temp_row(df, "bosh_q1", "bosh_q2", "bosh_q3", "bosh_q4"),
            hearth_4_3_a=_mean(df, "hearth_4_3_a"),
            hearth_5_4_c=_mean(df, "hearth_5_4_c"),
            hearth_5_7_c=_mean(df, "hearth_5_7_c"),
            hearth_6_1_b=_mean(df, "hearth_6_1_b"),
        )
