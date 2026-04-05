"""Control parameter bound management for the V-OptimAIse optimiser.

Bounds are persisted in ``src/assets/data/control_bounds.json`` so operators
can adjust operating ranges from the UI without changing source code.
:func:`get_control_bounds` reads the file and returns a list of
``(min, max)`` tuples in the same order as *cp_list*, suitable for
passing directly to :func:`scipy.optimize.differential_evolution`.
"""

import json
from pathlib import Path
from typing import List

import pandas as pd


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
    bounds_file = Path("src/assets/data/control_bounds.json")
    bounds_file.parent.mkdir(parents=True, exist_ok=True)

    if bounds_file.exists():
        with open(bounds_file, "r") as f:
            persisted_bounds = json.load(f)
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
