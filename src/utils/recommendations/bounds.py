import pandas as pd
import json 
from pathlib import Path
from typing import List

def get_control_bounds(df: pd.DataFrame, 
                       cp_list: List[str], 
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
    bounds_file = Path("src/data/control_bounds.json")
    bounds_file.parent.mkdir(parents=True, exist_ok=True)

    if bounds_file.exists():
        with open(bounds_file, "r") as f:
            persisted_bounds = json.load(f)
            for cp in cp_list:
                col_min = float(df[cp].min())
                col_max = float(df[cp].max())
                cp_min = persisted_bounds.get(cp, {}).get("min", col_min)
                cp_max = persisted_bounds.get(cp, {}).get("max", col_max)
                val = persisted_bounds.get(cp, {})["value"]
                overide = persisted_bounds.get(cp, {}).get("override", False)
                if overide:
                    persisted_bounds[cp]["min"] = val * 0.99
                    persisted_bounds[cp]["max"] = val * 1.01
    return [(persisted_bounds[cp]["min"], persisted_bounds[cp]["max"]) for cp in cp_list]
