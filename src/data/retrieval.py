"""Data retrieval utilities.

Re-exports functions from furnace_data for backward compatibility.
"""

import sys
from pathlib import Path

# Add furnace_data to path for imports
furnace_data_src = Path(__file__).resolve().parents[2] / "furnace_data"
if furnace_data_src.is_dir() and str(furnace_data_src) not in sys.path:
    sys.path.insert(0, str(furnace_data_src))

from furnace_data.influx.offline import clean_rm_data, fetch_offline_data
from furnace_data.influx.online import fetch_online_df

__all__ = ["clean_rm_data", "fetch_offline_data", "fetch_online_df"]