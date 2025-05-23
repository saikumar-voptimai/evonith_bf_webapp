from .base_data_fetcher import BaseDataFetcher
from typing import List
import numpy as np

class AverageHeatLoadDataFetcher(BaseDataFetcher):
    """
    Fetcher for average heat load data.
    """

    def __init__(self, debug: bool = False, source: str = "live"):
        super().__init__("heatload_variables", debug, source)

    def fetch_averaged_data(self, average_by: str, start_time=None, end_time=None, row: str = None) -> dict:
        """
        Fetch and process average heatload data for a specific row (R6-R10).

        Parameters:
            average_by (str): Averaging interval or range selection.
            start_time (datetime, optional): Start time for the range.
            end_time (datetime, optional): End time for the range.
            row (str, optional): The row to filter (e.g., 'R6').

        Returns:
            dict: {Q1: value, Q2: value, Q3: value, Q4: value} for the selected row.
        """
        flat_data = super().fetch_averaged_data(average_by, start_time, end_time)
        if row is None:
            raise ValueError("Row must be specified (e.g., 'R6').")
        result = {}
        for q in range(1, 5):
            # Find the variable name for this row and quadrant
            for key in flat_data:
                if key.startswith(f"Heat load {row} Q{q}"):
                    result[f"Q{q}"] = flat_data[key] * 100
                    break
            else:
                result[f"Q{q}"] = None  # or np.nan if you prefer
        return result