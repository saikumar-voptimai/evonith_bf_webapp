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
    "DPR": "DPR",
    "RAW_MATERIAL_COMPOSITION": "Bunker Report",
    "RM_COMPOSITION": "Raw Material Composition",
    "BURDEN_DISTRIBUTION": "Burden Distribution",
    "HOPPER_MANAGEMENT": "Hopper Management",
}

OFFLINE_TABLE_LABEL_MAP: dict[str, str] = {
    "offline_feed.charge_data": "Charge",
    "offline_feed.dpr_data": "DPR",
    "offline_feed.hot_metal_slag_analysis": "HM & Slag",
    "offline_feed.ore_chemistry": "Ore Chemistry",
    "offline_feed.sinter_chemistry": "Sinter Chemistry",
    "offline_feed.fuel_chemistry": "Fuel Chemistry",
    "offline_feed.flux_chemistry": "Flux Chemistry",
    "offline_feed.raw_material_strength_analysis": "Raw Material Strength",
    "offline_feed.raw_material_stock": "Raw Material Stock",
    "offline_feed.v_charge_material_quantities": "Charge Material Quantities",
    "offline_feed.v_dpr_material_quantities": "DPR Material Quantities",
    "offline_feed.feed_material_columns": "Feed Material Column Map",
    "offline_feed.historical_static_ml_dataset": "Static ML Dataset",
    "plant_master.material_property_mapping": "Material Property Mapping",
    "ops_config.burden_history": "Burden History",
    "ops_config.hopper_raw_material_history": "Hopper Raw Material History",
    "plant_master.materials": "Materials",
    "plant_master.hoppers": "Hoppers",
    "plant_master.units": "Units",
    "plant_master.material_categories": "Material Categories",
}


def offline_table_label(table_name: str) -> str:
    """Return the user-facing label for a configured database table."""
    if table_name in OFFLINE_TABLE_LABEL_MAP:
        return OFFLINE_TABLE_LABEL_MAP[table_name]
    short_name = table_name.rsplit(".", 1)[-1]
    return short_name.replace("_", " ").title()
