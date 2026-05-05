"""Shared measurement presets for webapp data-fetch utilities."""

from __future__ import annotations

ONLINE_MEASUREMENT_LABELS: dict[str, str] = {
    "heatload_delta_t": "Heatload Delta T",
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile",
}

WINDOW_FREQUENCY_MAP: dict[str, str | None] = {
    "None": None,
    "1 minute": "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour": "1h",
    "6 hours": "6h",
    "8 hours": "8h",
    "12 hours": "12h",
    "1 day": "1d",
}

OFFLINE_REPORT_LABEL_MAP: dict[str, str] = {
    "HM_SLAG": "HM & Slag",
    "CHARGE": "Charge",
    "RM_CHARGE": "RM Charge",
    "RAW_MATERIAL_COMPOSITION": "Bunker Report",
    "RM_COMPOSITION": "Raw Material Composition",
    "DPR": "DPR",
    "RM_DPR": "RM DPR",
    "BURDEN_DISTRIBUTION": "Burden Distribution",
    "HOPPER_MANAGEMENT": "Hopper Management",
}
