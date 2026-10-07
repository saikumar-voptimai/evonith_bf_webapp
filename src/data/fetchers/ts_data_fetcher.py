from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from furnace_data.influx.base import BaseDataFetcher


class TimeSeriesDataFetcher(BaseDataFetcher):
    """
    Processes raw data for time-series plots.
    """

    def fetch_data(
        self,
        time_interval: str,
        start_time: datetime | None,
        end_time: datetime | None,
        request_type: str = "ts",
        window_by: str = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame | dict:
        """
        Fetch raw time-series data for plotting.

        Args:
            start_time (datetime): Start of the time range.
            end_time (datetime): End of the time range.
            fields: Optional configured canonical fields to request.

        Returns:
            DataFrame of fetched values, or dummy-data mapping in debug mode.
        """
        if self.debug:
            return self._get_dummy_data()

        raw_df = self.fetch_averaged_data(
            time_interval,
            start_time,
            end_time,
            request_type=request_type,
            window_by=window_by,
            fields=fields,
        )
        raw_df = raw_df.select_dtypes(exclude=["object"])
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
                "timestamps": [
                    (now - timedelta(minutes=i)).isoformat() for i in range(100)
                ],
                "values": [np.random.random() * 100 for _ in range(100)],
            }
        return dummy_data
