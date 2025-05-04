from .ts_data_fetcher import TimeSeriesDataFetcher
from datetime import datetime, timedelta
import numpy as np

class TimeSeriesHeatLoadDataFetcher(TimeSeriesDataFetcher):
    """
    Processes raw data for time-series plots.
    """
    def __init__(self, debug: bool = False, source: str = "live"):
        super().__init__("heatload_variables", debug, source)
    
    def fetch_data(self, start_time: datetime, end_time: datetime, row: str, quadrant: str=None) -> dict:
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
        ts_data = super().fetch_data(start_time, end_time)
        ts_row_data = {}
        for key, value in ts_data.items():
            if row in key and (quadrant is None or quadrant in key):
                ts_row_data[key] = value
        return ts_row_data
