import numpy as np
import pandas as pd
from .temp_data_fetcher import TemperatureDataFetcher
from typing import List, Optional, Dict
from config.loader import load_config
import logging

config = load_config()
SENSORS_AT_Y = config["plot"]["geometry"]["sensors_at_y"][0]
SENSORS_AT_Y_Dict = config["plot"]["geometry"]["heights_dict"]

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
            dict: {Q1: {level: [temps]}, Q2: {...}, Q3: {...}, Q4: {...}} for each level.
        """
        temp_data = super().fetch_averaged_data(average_by, start_time, end_time)
        if not isinstance(temp_data, pd.DataFrame):
            # temp_data['time'] = pd.to_datetime(start_time)
            temp_data = pd.DataFrame(temp_data, index=[start_time], columns=temp_data.keys())
        else:
            temp_data.set_index("time", inplace=True, drop=True)
        # Vectorized, column-wise processing for all timestamps and levels
        # temp_data columns: temp_L1_S1, temp_L1_S2, ..., temp_L2_S1, ...
        # Group columns by level
        level_cols = {}
        for col in temp_data.columns:
            if not col.startswith('temp_'):
                continue
            parts = col.split('_')
            if len(parts) < 3:
                continue
            level = parts[1]
            level_cols.setdefault(level, []).append(col)
        val_dict = {f"Q{i+1}": {} for i in range(4)}
        # Add a 'time' key to store timestamps
        val_dict['time'] = temp_data.index.tolist() if hasattr(temp_data.index, 'tolist') else list(temp_data.index)
        for level, cols in level_cols.items():
            n_sensors = SENSORS_AT_Y_Dict[level]['n_sensors']
            temp_matrix = temp_data[cols].to_numpy()
            temp_matrix[temp_matrix <= 25] = np.nan
            try:
                row_means = np.nanmean(temp_matrix, axis=1, keepdims=True)
            except ValueError as e:
                log.error(f"Possible Null Data {level}: {e}")
                raise
            inds = np.where(np.isnan(temp_matrix))
            temp_matrix[inds] = np.take(row_means, inds[0])
            for row_idx in range(temp_matrix.shape[0]):
                temp_list = temp_matrix[row_idx, :]
                theta_per_sensor = 360 / n_sensors
                sensor_angles = np.radians([theta_per_sensor/2 + theta_per_sensor * i for i in range(n_sensors)])
                theta_query_locs = np.radians([45, 135, 225, 315])
                extended_angles = np.append(sensor_angles, sensor_angles[0] + 2 * np.pi)
                extended_temps = np.append(temp_list, temp_list[0])
                temp_query_locs = np.interp(theta_query_locs, extended_angles, extended_temps)
                for i, temp in enumerate(temp_query_locs):
                    val_dict[f"Q{i+1}"].setdefault(level, []).append(temp)
        # Sort dictionary with float(keys) in val_dict except for 'time'
        for key in val_dict.keys():
            if key != 'time':
                val_dict[key] = {k: v for k, v in sorted(val_dict[key].items(), key=lambda item: float(item[0]))}
        return val_dict

    def post_process_by_level(self, temp_dict: Dict) -> dict:
        # If more than 20% of values in float_dict are NaN, raise an error
        nan_count = sum(pd.isna(v) for _, v in temp_dict.items())
        if (nan_count / (len(temp_dict)+0.001) > 0.2) or (len(temp_dict) == 0):
            log.error(f"More than 20% of values are NaN: {nan_count} out of {len(temp_dict)}")
            raise ValueError(f"More than 20% of values are Not Available/Logged in the DB for {self.measurement_type} for {self.start_time} and {self.end_time}.")
        # This function is now only used for legacy row-wise fallback
        level_dict = {}
        for k in temp_dict.keys():
            if not 'temp' in k:
                continue
            level_val = k.split("_")[1]
            level_dict.setdefault(level_val, {})[k] = temp_dict[k]
        return level_dict


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