import numpy as np
from .base_data_fetcher import BaseDataFetcher
from typing import List
from config.loader import load_config

config = load_config()

SENSORS_AT_Y = config["plot"]["geometry"]["sensors_at_y"][0]

class TemperatureDataFetcher(BaseDataFetcher):
    """
    Fetcher for longitudinal temperature data.
    """

    def __init__(self, debug: bool, source: str):
        super().__init__("temperature_variables", debug, source)

    def _get_variable_names(self, n_sensors: int) -> List[str]:
        """
        Retrieve variable names by appending quadrant suffixes (A, B, C, D) to each base name.

        Returns:
            List[str]: List of variable names for all quadrants.
        """
        all_quadrants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        variable_names = []
        for i, num_sensors in enumerate(SENSORS_AT_Y):
            quadrants = all_quadrants[:num_sensors]
            variable_names.extend([f"{self.variables[i]}{quadrant}" for quadrant in quadrants])
        return variable_names
    
    def fetch_averaged_data(self, average_by: str, start_time=None, end_time=None) -> dict:
        """
        Fetch and process temperature data as a dictionary for each temperature variable.

        Parameters:
            average_by (str): Averaging interval or range selection.
            start_time (datetime, optional): Start time for the range.
            end_time (datetime, optional): End time for the range.

        Returns:
            dict: {variable_name: value, ...} for all temperature_variables.
        """
        source_clean = self.source.strip().lower()
        if not self.debug and source_clean == "historical":
            temp_data = super().fetch_averaged_data(average_by, start_time, end_time)
        elif not self.debug and source_clean == "live":
            temp_data = self.fetch_live_data()
        else:
            temp_data = self._get_dummy_data()
        # temp_data is now a dict {variable_name: value, ...}
        return temp_data

    def fetch_live_data(self) -> dict:
        """
        Fetch and process live temperature data as a dictionary for each temperature variable.

        Returns:
            dict: {variable_name: value, ...} for all temperature_variables.
        """
        temp_data = super().fetch_live_data()
        return temp_data

    def _get_dummy_data(self) -> dict:
        """
        Return dummy temperature data for all temperature_variables.

        Returns:
            dict: {variable_name: value, ...} for all temperature_variables.
        """
        dummy_data = {}
        for variable in self.variables:
            dummy_data[variable] = float(np.random.randint(100, 500))
        return dummy_data

    def post_process_by_level(self, temp_data: dict) -> dict:
        """
        Groups temperature values by level (e.g., 12975mm) and returns a dict:
        {level_name: [temps_at_level]}
        """
        import re
        level_dict = {}
        for var, value in temp_data.items():
            match = re.search(r"(\d{4,5})mm", var)
            if match:
                level = match.group(1)
                level_name = var.split("mm")[0] + "mm"
                if level_name not in level_dict:
                    level_dict[level_name] = []
                level_dict[level_name].append(value)
        return level_dict