from .temp_data_fetcher import TemperatureDataFetcher
from pathlib import Path
from typing import List
import numpy as np
from config.loader import load_config

config = load_config()

SENSORS_AT_Y = config["plot"]["geometry"]["sensors_at_y"][0]

class CircumferentialTemperatureDataFetcher(TemperatureDataFetcher):
    """
    Fetcher for longitudinal temperature data.
    """

    def __init__(self, debug: bool = False, source: str = "live"):
        super().__init__(debug, source)

    def fetch_averaged_data(self, average_by: str, start_time=None, end_time=None) -> dict:
        """
        Fetch and process temperature data grouped by circumferential location (elevation).

        Parameters:
            average_by (str): Averaging interval or range selection.
            start_time (datetime, optional): Start time for the range.
            end_time (datetime, optional): End time for the range.

        Returns:
            dict: {elevation_label: [temps_at_level], ...} for each elevation.
        """
        temp_data = super().fetch_averaged_data(average_by, start_time, end_time)
        level_dict = self.post_process_by_level(temp_data)
        # Elevation labels from settings
        heights = config["plot"]["geometry"]["heights"][0]
        # Map special names for Bosh, Belly, Stack
        special_labels = {
            8: "12.975m - Bosh",
            9: "15.162m - Belly",
            10: "18.660m - Stack"
        }
        result = {}
        for i, (level, temp_list) in enumerate(level_dict.items()):
            if i in special_labels:
                label = special_labels[i]
            else:
                label = f"{heights[i]:.3f}m"
            result[label] = temp_list
        return result