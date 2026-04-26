"""InfluxDB fetcher for longitudinal furnace wall temperature profiles.

Fetches the 110-sensor temperature profile measurement and aggregates readings
by elevation and quadrant (A–N per level) for both time-series and
averaged-min-max contour views.
"""

from typing import List

import numpy as np
import pandas as pd

from config.config_loader import load_config

from .base_data_fetcher import BaseDataFetcher

config = load_config()

SENSORS_AT_Y = config["plot"]["geometry"]["heights_dict"]


class TemperatureDataFetcher(BaseDataFetcher):
    """
    Fetcher for longitudinal temperature data.
    """

    def __init__(self, debug: bool, source: str):
        super().__init__("temperature_profile", debug, source)

    def _get_variable_names(self, n_sensors: int) -> List[str]:
        """
        Retrieve variable names by appending quadrant suffixes (A, B, C, D) to each base name.

        Returns:
            List[str]: List of variable names for all quadrants.
        """
        all_quadrants = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
        ]
        variable_names = []
        for i, num_sensors in enumerate(SENSORS_AT_Y):
            quadrants = all_quadrants[:num_sensors]
            variable_names.extend(
                [f"{self.variables[i]}{quadrant}" for quadrant in quadrants]
            )
        return variable_names

    def fetch_averaged_data(
        self,
        recent_data_of: str,
        start_time=None,
        end_time=None,
        request_type=None,
        window_by=None,
    ) -> dict:
        """
        Fetch and process temperature data as a dictionary for each temperature variable.

        Args:
            recent_data_of (str): Averaging interval or range selection.
            start_time (datetime, optional): Start time for the range.
            end_time (datetime, optional): End time for the range.
            request_type (str, optional): Type of request for data processing.
            window_by (str, optional): Windowing parameter for data aggregation.

        Returns:
            dict: {variable_name: value, ...} for all temperature_variables.
        """
        source_clean = self.source.strip().lower()
        if not self.debug and source_clean == "historical":
            temp_data = super().fetch_averaged_data(
                recent_data_of, start_time, end_time, request_type, window_by
            )
            if not isinstance(temp_data, pd.DataFrame):
                # temp_data['time'] = pd.to_datetime(start_time)
                temp_data = pd.DataFrame(
                    temp_data, index=[start_time], columns=temp_data.keys()
                )
            else:
                temp_data.set_index("time", inplace=True, drop=True)
            level_cols = {}
            for col in temp_data.columns:
                if not col.startswith("temp_"):
                    continue
                parts = col.split("_")
                if len(parts) < 3:
                    continue
                level = parts[1]
                level_cols.setdefault(level, []).append(col)
            val_dict = {f"Q{i+1}": {} for i in range(4)}
            # Add a 'time' key to store timestamps
            val_dict["time"] = (
                temp_data.index.tolist()
                if hasattr(temp_data.index, "tolist")
                else list(temp_data.index)
            )
            levelwise_dict = {}
            for level, cols in level_cols.items():
                n_sensors = SENSORS_AT_Y[level]["n_sensors"]
                df_temp_data = temp_data[cols].copy()
                df_temp_data[df_temp_data <= 25] = np.nan
                df_temp_data.dropna(
                    axis=0, how="all", inplace=True
                )  # Drop rows where all sensors are NaN
                df_temp_data.interpolate(method="linear", axis=1, inplace=True)
                temp_matrix = df_temp_data.to_numpy()
                # If temp_matrix is empty (no sensors), fill with zeros
                if temp_matrix.shape[1] == 0:
                    temp_matrix = np.zeros((temp_matrix.shape[0], 1))
                df_new = pd.DataFrame(
                    index=df_temp_data.index, columns=[f"Q_{i}" for i in range(1, 5)]
                )
                angles = [45, 135, 225, 315]
                weights = [
                    [
                        max(
                            1
                            - abs((angle - i * 360 / n_sensors + 180) % 360 - 180)
                            / (360 / n_sensors),
                            0,
                        )
                        for i in range(n_sensors)
                    ]
                    for angle in angles
                ]
                max_cols = [col for col in df_temp_data.columns if col.endswith("_max")]
                min_cols = [col for col in df_temp_data.columns if col.endswith("_min")]
                mean_cols = [
                    col for col in df_temp_data.columns if col.endswith("_mean")
                ]
                for i, angle in enumerate(angles):
                    indices = np.where(np.array(weights[i]) > 0)[0]
                    if len(indices) > 0:
                        if request_type == "avg-min-max":
                            df_new[f"Q_{i+1}_mean"] = sum(
                                df_temp_data[mean_cols].iloc[:, idx] * weights[i][idx]
                                for idx in indices
                                if weights[i][idx] > 0
                            ) / sum(
                                weights[i][idx]
                                for idx in indices
                                if weights[i][idx] > 0
                            )
                            df_new[f"Q_{i+1}_max"] = sum(
                                df_temp_data[max_cols].iloc[:, idx] * weights[i][idx]
                                for idx in indices
                                if weights[i][idx] > 0
                            ) / sum(
                                weights[i][idx]
                                for idx in indices
                                if weights[i][idx] > 0
                            )
                            df_new[f"Q_{i+1}_min"] = sum(
                                df_temp_data[min_cols].iloc[:, idx] * weights[i][idx]
                                for idx in indices
                                if weights[i][idx] > 0
                            ) / sum(
                                weights[i][idx]
                                for idx in indices
                                if weights[i][idx] > 0
                            )
                        df_new[f"Q_{i+1}"] = sum(
                            df_temp_data.iloc[:, idx] * weights[i][idx]
                            for idx in indices
                            if weights[i][idx] > 0
                        )
                levelwise_dict[level] = df_new
            # Sort dictionary with float(keys) in levelwise dict so that they arranged from bottom to top
            levelwise_dict = {int(k): v for k, v in levelwise_dict.items()}
            levelwise_dict = dict(sorted(levelwise_dict.items()))
            levelwise_dict = {str(k): v for k, v in levelwise_dict.items()}
        elif not self.debug and source_clean == "live":
            temp_data = self.fetch_live_data()
        else:
            temp_data = self._get_dummy_data()
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
