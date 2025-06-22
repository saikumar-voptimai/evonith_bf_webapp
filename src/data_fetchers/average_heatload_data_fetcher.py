from .base_data_fetcher import BaseDataFetcher
from typing import List
import numpy as np
import pandas as pd

class AverageHeatLoadDataFetcher(BaseDataFetcher):
    """
    Fetcher for average heat load data.
    """

    def __init__(self, debug: bool = False, source: str = "live", request_type: str = "average"):
        super().__init__("heatload_delta_t", debug, source)
        self.request_type = request_type

    def fetch_averaged_data(self, average_by: str, start_time=None, end_time=None, row: str = None) -> List[List[float]]:
        """
        Fetch and process average heatload data for a specific row (R6-R10).

        Parameters:
            average_by (str): Averaging interval or range selection.
            start_time (datetime, optional): Start time for the range.
            end_time (datetime, optional): End time for the range.
            row (str, optional): The row to filter (e.g., 'r6').

        Returns:
            pd.DataFrame: DataFrame with averaged heat load data for the specified row.
        """
        df_flatdata = super().fetch_averaged_data(average_by, start_time, end_time)
        df_flatdata.set_index("time", inplace=True, drop=True)
        if row is None:
            raise ValueError("Row must be specified (e.g., 'r6').")
        df_result = pd.DataFrame(columns=["Q1", "Q2", "Q3", "Q4"], index=df_flatdata.index)
        for q in range(1, 5):
            for col in df_flatdata.columns:
                if col.startswith(f"heat_load_{row.lower()}_q{q}"):
                    df_result[f"Q{q}"] = df_flatdata[col]
        df_result[df_result < 0] = np.nan
        df_result[df_result > 1] = 1                    
        df_result.dropna(axis=0, how='all', inplace=True)  # Drop rows where all sensors are NaN
        df_result.interpolate(method='linear', axis=1, inplace=True)
        if self.request_type == "ts":
            df_result.reset_index(inplace=True)
            df_result.rename(columns={"index": "time"}, inplace=True)
            df_result.set_index("time", inplace=True)
        else:
            return self.post_process(df_result)
        return df_result
    
    def post_process(self, df_result: pd.DataFrame) -> List[List[float]]:
        """
        Post-processes temperature data by grouping values by level, computing max and min for each level and quadrant.
        Args:
            df_result (pd.DataFrame): DataFrame with averaged heat load data for the specified row.
        """
        # Send data as list of 13 values for each quadrant - for avg, max, min
        mean_list = df_result.mean().tolist()
        max_list = df_result.max().tolist()
        min_list = df_result.min().tolist()
        return [mean_list, max_list, min_list]