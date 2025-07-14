from .temp_data_fetcher import TemperatureDataFetcher
from typing import List, Dict
import numpy as np
import pandas as pd
from src.config.config_loader import load_config
import logging

# Initialize logging
log = logging.getLogger("root")

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
        temp_data = super().fetch_averaged_data(average_by, start_time, end_time)
        if not isinstance(temp_data, pd.DataFrame):
            # temp_data['time'] = pd.to_datetime(start_time)
            temp_data = pd.DataFrame(temp_data, index=[start_time], columns=temp_data.keys())
        else:
            temp_data.set_index("time", inplace=True, drop=True)
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
        levelwise_dict = {}
        for level, cols in level_cols.items():
            n_sensors = SENSORS_AT_Y_Dict[level]['n_sensors']
            df_temp_data = temp_data[cols]
            df_temp_data[df_temp_data <= 25] = np.nan 
            df_temp_data.dropna(axis=0, how='all', inplace=True)  # Drop rows where all sensors are NaN
            df_temp_data.interpolate(method='linear', axis=1, inplace=True)
            temp_matrix = df_temp_data.to_numpy()
            # If temp_matrix is empty (no sensors), fill with zeros
            if temp_matrix.shape[1] == 0:
                temp_matrix = np.zeros((temp_matrix.shape[0], 1))
            df_new = pd.DataFrame(index=df_temp_data.index, columns=[f"Q_{i}" for i in range(1, 5)])
            angles = [45, 135, 225, 315]
            weights = [
                [
                max(1-abs((angle-i*360/n_sensors +180) %360 -180)/ (360/n_sensors), 0) for i in range(n_sensors)
                ] for angle in angles
            ]
            for i, angle in enumerate(angles):
                indices = np.where(np.array(weights[i]) > 0)[0]
                if len(indices) > 0:
                    df_new[f"Q_{i+1}"] = sum(df_temp_data.iloc[:, idx] * weights[i][idx] for idx in indices if weights[i][idx] > 0) 
            levelwise_dict[level] = df_new
        # Sort dictionary with float(keys) in levelwise dict
        levelwise_dict = {int(k): v for k, v in levelwise_dict.items()}
        levelwise_dict = dict(sorted(levelwise_dict.items()))
        levelwise_dict = {str(k): v for k, v in levelwise_dict.items()}

        if self.request_type == "average":
            return self.post_process_by_level(levelwise_dict)
        else:
            return levelwise_dict
        
    def post_process_by_level(self, levelwise_dict: Dict[str, pd.DataFrame]) -> dict:
        """
        Post-processes temperature data by grouping values by level, computing max and min for each level and quadrant.
        Args:
            levelwise_dict (dict): Dictionary with temperature data for each level and quadrant.
            Returns:
            dict: Processed dictionary with levels as keys and temperature values as lists.
        """
        # Send data as list of 13 values for each quadrant - for avg, max, min
        levelwise_stats = {}
        for level, df in levelwise_dict.items():
            max = df.max().tolist()
            min = df.min().tolist()
            mean = df.mean().tolist()
            
            levelwise_stats[level] = [mean, max, min]
        return levelwise_stats
