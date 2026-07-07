"""Control parameter bound management for the V-OptimAIse optimiser."""

import json
from pathlib import Path
from typing import List

import pandas as pd

from furnace_data.runtime_paths import runtime_path


def get_control_bounds_file_path(*, create_parent: bool = False) -> Path:
    """Return the runtime path for mutable control-bound overrides."""
    return runtime_path("cache", "control_bounds.json", create_parent=create_parent)


def load_control_bounds() -> dict:
    """Load persisted control bounds from the runtime cache."""
    bounds_file = get_control_bounds_file_path()
    if not bounds_file.exists():
        return {}
    with open(bounds_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def save_control_bounds(bounds: dict) -> Path:
    """Persist control bounds to the runtime cache directory."""
    bounds_file = get_control_bounds_file_path(create_parent=True)
    with open(bounds_file, "w", encoding="utf-8") as file:
        json.dump(bounds, file, indent=4)
    return bounds_file


def get_control_bounds(
    df: pd.DataFrame, cp_list: List[str], impute_lags: bool = True
) -> List[tuple]:
    """
    TODO: Edit the function to send cp bounds from control_bounds.json file
    Get quantile-based bounds for control params.
    Args:
        df (pd.DataFrame): DataFrame containing control parameters.
        control_params (List[str]): List of control parameter names.
        q_low (float): Lower quantile for bounds.
        q_high (float): Upper quantile for bounds.
    Returns:
        List[tuple]: List of tuples with lower and upper bounds for each control parameter.
    """
    persisted_bounds = load_control_bounds()
    if persisted_bounds:
        for cp in cp_list:
            if "_lag" in cp and impute_lags:
                cp_sfx = cp.split("_lag")[0]
            else:
                cp_sfx = cp
            col_min = float(df[cp_sfx].min())
            col_max = float(df[cp_sfx].max())
            cp_min = persisted_bounds.get(cp_sfx, {}).get("min", col_min)
            cp_max = persisted_bounds.get(cp_sfx, {}).get("max", col_max)
            val = persisted_bounds.get(cp_sfx, {})["value"]
            overide = persisted_bounds.get(cp_sfx, {}).get("override", False)
            if overide:
                persisted_bounds[cp]["min"] = val * 0.99
                persisted_bounds[cp]["max"] = val * 1.01
        if impute_lags:
            return [
                (
                    persisted_bounds[cp.split("_lag")[0] if "_lag" in cp else cp][
                        "min"
                    ],
                    persisted_bounds[cp.split("_lag")[0] if "_lag" in cp else cp][
                        "max"
                    ],
                )
                for cp in cp_list
            ]
        else:
            return [
                (persisted_bounds[cp]["min"], persisted_bounds[cp]["max"])
                for cp in cp_list
            ]
    fallback_bounds = []
    for cp in cp_list:
        column = cp.split("_lag")[0] if "_lag" in cp and impute_lags else cp
        fallback_bounds.append((float(df[column].min()), float(df[column].max())))
    return fallback_bounds