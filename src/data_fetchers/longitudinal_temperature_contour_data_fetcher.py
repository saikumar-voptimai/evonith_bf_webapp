import numpy as np
import pandas as pd
from .temp_data_fetcher import TemperatureDataFetcher
from typing import List
from config.loader import load_config
import logging

config = load_config()
SENSORS_AT_Y = config["plot"]["geometry"]["sensors_at_y"][0]

# Initialize logging
log = logging.getLogger("root")

class LongitudinalTemperatureDataFetcher(TemperatureDataFetcher):
    """
    Fetcher for longitudinal temperature data, returning values grouped by quadrant (Q1-Q4) for each level.
    """

    def __init__(self, debug: bool = False, source: str = "live", request_type:str = "average"):
        super().__init__(debug, source, request_type)

    def fetch_averaged_data(self, average_by: str, start_time=None, end_time=None) -> dict:
        """
        Fetch and process temperature data grouped by longitudinal location and return as dict with Q1-Q4 keys.

        Returns:
            dict: {Q1: [level1, level2, ...], Q2: [...], Q3: [...], Q4: [...]} for each level.
        """
        temp_data = super().fetch_averaged_data(average_by, start_time, end_time)
        level_dict = self.post_process_by_level(temp_data)
        q_data = {f"Q{i+1}": [] for i in range(4)}
        
        # If more than 20% of values in float_dict are NaN, raise an error
        nan_count = sum(pd.isna(v) for v in temp_data.values())
        if (nan_count / (len(temp_data)+0.001) > 0.2) or (len(temp_data) == 0):
            log.error(f"More than 20% of values are NaN: {nan_count} out of {len(temp_data)}")
            raise ValueError(f"More than 20% of values are Not Available/Logged in the DB for {self.measurement_type} for {start_time} and {end_time}.")
        for i, (level, temp_list) in enumerate(level_dict.items()):
            n_sensors = SENSORS_AT_Y[i]
            if n_sensors == 4:
                vals = temp_list
            else:
                vals = self.averager_to_four(temp_list, n_sensors)
            for q in range(4):
                q_data[f"Q{q+1}"].append(vals[q])
        return q_data

    @staticmethod
    def averager_to_four(temp_list: List[float], n_sensors: int) -> List[float]:
        """
        Interpolates temperature values to four quadrants (45°, 135°, 225°, 315°).

        Args:
            temp_list (List[float]): Sensor temperature values.
            n_sensors (int): Number of sensors at the level.

        Returns:
            List[float]: Interpolated values at four quadrants.
        """
        # if any number in temp_list is 0 or Nan, fill replace with the average of the numerical values (but not zeros)
        temp_list = np.array(temp_list)
        temp_list[temp_list == 0] = np.nan
        temp_list = np.nan_to_num(temp_list, nan=np.nanmean(temp_list[np.isfinite(temp_list)]))

        theta_per_sensor = 360 / n_sensors
        sensor_angles = np.radians([theta_per_sensor/2 + theta_per_sensor * i for i in range(len(temp_list))])
        theta_query_locs = np.radians([45, 135, 225, 315])
        temp_array = np.array(temp_list)
        extended_angles = np.append(sensor_angles, sensor_angles[0] + 2 * np.pi)
        extended_temps = np.append(temp_array, temp_array[0])
        temp_query_locs = np.interp(theta_query_locs, extended_angles, extended_temps)
        return temp_query_locs.tolist()