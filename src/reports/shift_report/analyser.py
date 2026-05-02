"""LLM analysis: compare current shift to previous shift.

The only place in the shift report pipeline that calls an LLM.
Sends compact dicts (no DataFrames) — token-efficient by design.
"""
from __future__ import annotations

import json
from typing import Optional

from reports.base import ReportAnalyser
from reports.shift_report.data import ShiftReportData

_SYSTEM = (
    "You are a blast furnace process expert at BF2. "
    "You receive key metrics for the current shift and (optionally) the previous shift. "
    "Write a concise handover analysis in 3 short paragraphs:\n"
    "1. Current shift summary — status and the 2-3 most significant parameters.\n"
    "2. Change vs previous shift — what improved, what deteriorated, and why.\n"
    "3. Watchlist for incoming crew — specific parameters to monitor and suggested actions.\n"
    "Be direct and numeric. No preamble. Max 200 words total."
)


def _compact(r: ShiftReportData) -> dict:
    """Strip a ShiftReportData down to the key numbers the LLM needs."""
    return {
        "shift": f"{r.shift_date} Shift {r.shift_label}",
        "status": r.status,
        "status_flags": r.status_flags,
        "production_rate_t_hr": r.production_rate,
        "total_charges": r.total_charges,
        "total_taps": r.total_taps,
        "fuel_rate_kg_thm": r.fuel_rate,
        "coke_rate_kg_thm": r.coke_rate,
        "pci_rate_kg_thm": r.pci_rate,
        "etaco_pct": r.etaco.mean,
        "etaco_std": r.etaco.std,
        "permeability": r.permeability.mean,
        "raft_degc": r.raft.mean,
        "blast_vol_nm3hr": r.blast_volume.mean,
        "blast_temp_degc": r.blast_temp.mean,
        "o2_enr_pct": r.o2_enrichment.mean,
        "hm_si_pct": r.hm_si,
        "hm_s_pct": r.hm_s,
        "slag_basicity": r.slag_basicity,
        "uptake_spread_std_degc": r.uptake.spread_std,
        "lower_stack_spread_std_degc": r.lower_stack.spread_std,
        "belly_spread_std_degc": r.belly.spread_std,
        "hearth_5_7_c_degc": r.hearth_5_7_c,
        "hearth_6_1_b_degc": r.hearth_6_1_b,
    }


class ShiftAnalyser(ReportAnalyser[ShiftReportData]):
    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def analyse(
        self,
        current: ShiftReportData,
        previous: Optional[ShiftReportData] = None,
    ) -> str:
        payload: dict = {"current_shift": _compact(current)}
        if previous is not None:
            payload["previous_shift"] = _compact(previous)

        user_msg = json.dumps(payload, indent=2, default=str)
        return self._llm.generate(system_prompt=_SYSTEM, user_prompt=user_msg)
