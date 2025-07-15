from .ts_data_fetcher import TimeSeriesDataFetcher
from datetime import datetime
import numpy as np
import pandas as pd

class TimeSeriesHeatLoadDataFetcher(TimeSeriesDataFetcher):
    """
    Processes raw data for time-series plots.
    """
    def __init__(self, debug: bool = False, source: str = "live", request_type: str = "ts"):
        super().__init__("heatload_delta_t", debug, source)
        self.request_type = request_type
    
    def fetch_data(self, time_interval: str, start_time: datetime, end_time: datetime, row: str, quadrant: str=None) -> pd.DataFrame:
        """
        Fetch raw time-series data for plotting.

        Parameters:
            start_time (datetime): Start of the time range.
            end_time (datetime): End of the time range.
            row (str): The row to filter variables by (e.g., 'R6').
            quadrant (str, optional): If provided, filter for a specific quadrant.

        Returns:
            dict: Time-series data for each heatload variable in the selected row (using full variable names).
        """
        df_ts = super().fetch_data(time_interval, start_time, end_time)
        if len(df_ts) == 0:
            raise ValueError(f"No data available in the Database for the specified time range \
                             - {start_time.isoformat()} and {end_time.isoformat()}.")
        df_ts['time'] = pd.to_datetime(df_ts['time'], utc=True, errors='coerce')
        df_ts.set_index("time", inplace=True)

        # Replace values greater than 5 with 1
        df_ts[df_ts > 5] = 1
        df_ts[df_ts < 0.1] = 0
        cols_to_check = [f"heat_load_{row.lower()}_q{q}" for q in range(1, 5)]
        df_ts = df_ts[cols_to_check]
        df_ts.rename(columns={
            f"heat_load_{row.lower()}_q{q}": f"Heat load {row} Q{q}" for q in range(1, 5)
        }, inplace=True)
        for col in df_ts.columns:
            # Moving average of columns
            df_ts[col] = df_ts[col].rolling(window=60, min_periods=1).mean()
        return df_ts
