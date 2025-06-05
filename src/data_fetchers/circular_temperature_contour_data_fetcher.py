from .temp_data_fetcher import TemperatureDataFetcher
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from config.loader import load_config

config = load_config()

SENSORS_AT_Y = config["plot"]["geometry"]["sensors_at_y"][0]
SENSORS_AT_Y_Dict = config["plot"]["geometry"]["heights_dict"]

class CircumferentialTemperatureDataFetcher(TemperatureDataFetcher):
    """
    Fetcher for longitudinal temperature data.
    """

    def __init__(self, debug: bool = False, source: str = "live", request_type: str = "average"):
        super().__init__(debug, source, request_type)

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
        # temp_data = super().fetch_averaged_data(average_by, start_time, end_time)
        # level_dict = self.post_process_by_level(temp_data)
        
        # for i, (level, temp_list) in enumerate(level_dict.items()):
        #     # if any number in temp_list is 0 or Nan, fill replace with the average of the numerical values (but not zeros)
        #     temp_list = np.array(temp_list)
        #     temp_list[temp_list <= 25] = np.nan
        #     temp_list = np.nan_to_num(temp_list, nan=np.nanmean(temp_list[np.isfinite(temp_list)]))
        #     level_dict[level] = temp_list
        
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
        level_dict = {}
        for level, cols in level_cols.items():
            temp_matrix = temp_data[cols].to_numpy()
            temp_matrix[temp_matrix <= 25] = np.nan
            # replace nans with average values
            temp_matrix[temp_matrix == 0] = np.nan
            try:
                row_means = np.nanmean(temp_matrix, axis=1, keepdims=True)
            except ValueError as e:
                log.error(f"Possible Null Data {level}: {e}")
                raise
            inds = np.where(np.isnan(temp_matrix))
            temp_matrix[inds] = np.take(row_means, inds[0])    
            level_dict[level] = temp_matrix.tolist()

        # Sort dictionary with float(keys) in val_dict except for 'time'
        for key in level_dict.keys():
            level_dict = {k: v for k, v in sorted(level_dict.items(), key=lambda item: float(item[0]))}
        
        # Map special names for Bosh, Belly, Stack
        special_labels = {
            "12.975m": "12.975m - Bosh",
            "15.162m": "15.162m - Belly",
            "18.660m": "18.660m - Stack"
        }
        result = {}
        for level, temp_list in level_dict.items():
            mapped_level = f"{float(level)/1000:.3f}m"
            if mapped_level in list(special_labels.keys()):
                label = special_labels[mapped_level]
            else:
                label = mapped_level
            result[label] = temp_list
        # Add a 'time' key to store timestamps
        result['time'] = temp_data.index.tolist() if hasattr(temp_data.index, 'tolist') else list(temp_data.index)  
        return result