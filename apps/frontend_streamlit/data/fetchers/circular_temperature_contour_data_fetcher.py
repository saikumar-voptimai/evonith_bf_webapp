"""Circumferential temperature contour data fetcher.

Groups the 110-sensor temperature readings by elevation level and returns
data in a format ready for :class:`~plotters.circumferential_contour.CircumferentialPlotter`:
per-quadrant mean/min/max at each of the 11 sensor elevations.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from furnace_data.domain.temperature import TemperatureDataFetcher

log = logging.getLogger("root")


class CircumferentialTemperatureDataFetcher(TemperatureDataFetcher):
    """
    Fetcher for circumferential temperature data grouped by elevation level.
    """

    def __init__(self, debug: bool = False, source: str = "live"):
        super().__init__(debug, source)

    def fetch_averaged_data(
        self,
        average_by: str,
        start_time=None,
        end_time=None,
        request_type=None,
        window_by=None,
    ) -> dict:
        """
        Fetch and process temperature data grouped by circumferential location (elevation).

        Returns:
            For request_type=="avg-min-max":
                {level_str: [mean_list, max_list, min_list]}
                where each list has 4 values (Q1..Q4).
            Otherwise:
                {level_str: DataFrame with Q_1..Q_4 columns}
        """
        levelwise_dict = super().fetch_averaged_data(
            average_by,
            start_time,
            end_time,
            request_type=request_type,
            window_by=window_by,
        )
        if request_type == "avg-min-max":
            return self.post_process_by_level(levelwise_dict)
        return levelwise_dict

    def post_process_by_level(self, levelwise_dict: Dict[str, pd.DataFrame]) -> dict:
        """
        Collect quadrant stats for each elevation level.

        Returns:
            {level_str: [mean_list, max_list, min_list]}
            where each list has 4 values (Q1..Q4).
        """
        levelwise_stats = {}
        for level, df in levelwise_dict.items():
            mean_cols = sorted([c for c in df.columns if c.endswith("_mean")])
            max_cols = sorted([c for c in df.columns if c.endswith("_max")])
            min_cols = sorted([c for c in df.columns if c.endswith("_min")])

            if df.empty or not mean_cols:
                levelwise_stats[level] = [[np.nan] * 4, [np.nan] * 4, [np.nan] * 4]
                continue

            mean = df[mean_cols].iloc[-1].tolist()
            maxi = df[max_cols].iloc[-1].tolist()
            mini = df[min_cols].iloc[-1].tolist()

            # Pad to 4 quadrants if fewer were computed
            mean = (mean + [np.nan] * 4)[:4]
            maxi = (maxi + [np.nan] * 4)[:4]
            mini = (mini + [np.nan] * 4)[:4]

            levelwise_stats[level] = [mean, maxi, mini]

        return levelwise_stats
