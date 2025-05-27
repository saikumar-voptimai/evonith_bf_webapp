from .base_data_fetcher import BaseDataFetcher
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class TimeSeriesDataFetcher(BaseDataFetcher):
    """
    Processes raw data for time-series plots.
    """
    def fetch_data(self, time_interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Fetch raw time-series data for plotting.

        Parameters:
            start_time (datetime): Start of the time range.
            end_time (datetime): End of the time range.

        Returns:
            dict: Time-series data for each variable (timestamps and values).
        """
        if self.debug:
            return self._get_dummy_data()

        raw_df = self.fetch_averaged_data(time_interval, start_time, end_time)
        return raw_df
    def _get_dummy_data(self) -> dict:
        """
        Return dummy data for debugging purposes.

        Returns:
            dict: A dictionary of dummy timestamps and values.
        """
        dummy_data = {}
        now = datetime.now(self.timezone)
        for variable in self.variables:
            dummy_data[variable] = {
                "timestamps": [(now - timedelta(minutes=i)).isoformat() for i in range(100)],
                "values": [np.random.random() * 100 for _ in range(100)]
            }
        return dummy_data
