"""Longitudinal temperature contour data fetcher.

Groups the 110-sensor readings by circumferential quadrant (Q1–Q4) across all
elevations and returns data ready for :class:`~plotters.longitudinal_temp_contour.LongitudinalTemperaturePlotter`:
per-quadrant average temperature arrays along the furnace height.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from furnace_data.domain.temperature import TemperatureDataFetcher

log = logging.getLogger("root")


class LongitudinalTemperatureDataFetcher(TemperatureDataFetcher):
    """
    Fetcher for longitudinal temperature data, returning values grouped by quadrant (Q1-Q4) for each level.
    """

    def __init__(self, debug: bool = False, source: str = "live"):
        super().__init__(debug, source)

    def fetch_averaged_data(
        self,
        recent_data_of: str,
        start_time=None,
        end_time=None,
        request_type=None,
        window_by=None,
    ) -> dict:
        """
        Fetch and process temperature data grouped by longitudinal location and return as dict with Q1-Q4 keys.

        Returns:
            For request_type=="avg-min-max":
                [[mean_Q1, mean_Q2, mean_Q3, mean_Q4],
                 [max_Q1,  max_Q2,  max_Q3,  max_Q4],
                 [min_Q1,  min_Q2,  min_Q3,  min_Q4]]
            Otherwise:
                {level_str: DataFrame with Q_1..Q_4 columns}
        """
        levelwise_dict = super().fetch_averaged_data(
            recent_data_of,
            start_time,
            end_time,
            request_type=request_type,
            window_by=window_by,
        )
        if request_type == "avg-min-max":
            return self.post_process_by_level(levelwise_dict)
        return levelwise_dict

    def post_process_by_level(self, levelwise_dict: Dict[str, pd.DataFrame]) -> list:
        """
        Collect per-level quadrant stats into three lists (mean, max, min), each with 4 quadrant sub-lists.

        Returns:
            [[mean_Q1..Q4 per level], [max_Q1..Q4 per level], [min_Q1..Q4 per level]]
        """
        mean_lists = [[], [], [], []]
        max_lists = [[], [], [], []]
        min_lists = [[], [], [], []]

        for level, df in levelwise_dict.items():
            for qi in range(1, 5):
                mean_col = f"Q_{qi}_mean"
                max_col = f"Q_{qi}_max"
                min_col = f"Q_{qi}_min"

                mean_lists[qi - 1].append(
                    float(df[mean_col].iloc[-1])
                    if mean_col in df.columns and not df.empty
                    else np.nan
                )
                max_lists[qi - 1].append(
                    float(df[max_col].iloc[-1])
                    if max_col in df.columns and not df.empty
                    else np.nan
                )
                min_lists[qi - 1].append(
                    float(df[min_col].iloc[-1])
                    if min_col in df.columns and not df.empty
                    else np.nan
                )

        return [mean_lists, max_lists, min_lists]
