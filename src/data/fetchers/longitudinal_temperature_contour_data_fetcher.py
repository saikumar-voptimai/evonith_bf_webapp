"""Longitudinal temperature contour data fetcher.

Groups the 110-sensor readings by circumferential quadrant (Q1–Q4) across all
elevations and returns data ready for :class:`~plotters.longitudinal_temp_contour.LongitudinalTemperaturePlotter`:
per-quadrant average temperature arrays along the furnace height.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config.config_loader import load_config

from .temp_data_fetcher import TemperatureDataFetcher

config = load_config()
SENSORS_AT_Y_Dict = config["plot"]["geometry"]["heights_dict"]

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
        temp_data = super().fetch_averaged_data(
            recent_data_of, start_time, end_time, request_type, window_by
        )

        if not isinstance(temp_data, pd.DataFrame):
            temp_data = pd.DataFrame(
                temp_data, index=[start_time], columns=temp_data.keys()
            )
        else:
            if "time" in temp_data.columns:
                temp_data.set_index("time", inplace=True, drop=True)

        # Group columns by elevation level
        level_cols: Dict[str, list] = {}
        for col in temp_data.columns:
            if not col.startswith("temp_"):
                continue
            parts = col.split("_")
            if len(parts) < 3:
                continue
            level = parts[1]
            level_cols.setdefault(level, []).append(col)

        angles = [45, 135, 225, 315]
        levelwise_dict = {}

        for level, cols in level_cols.items():
            cols_sorted = sorted(cols)
            n_sensors = SENSORS_AT_Y_Dict.get(level, {}).get(
                "n_sensors", len(cols_sorted)
            )

            df_temp_data = temp_data[cols_sorted].copy()
            df_temp_data[df_temp_data <= 25] = np.nan
            df_temp_data.dropna(axis=0, how="all", inplace=True)

            if df_temp_data.empty:
                df_new = pd.DataFrame(columns=[f"Q_{i}" for i in range(1, 5)])
                levelwise_dict[level] = df_new
                continue

            df_temp_data.interpolate(method="linear", axis=1, inplace=True)

            if request_type == "avg-min-max":
                mean_cols = sorted(
                    [c for c in df_temp_data.columns if c.endswith("_mean")]
                )
                max_cols = sorted(
                    [c for c in df_temp_data.columns if c.endswith("_max")]
                )
                min_cols = sorted(
                    [c for c in df_temp_data.columns if c.endswith("_min")]
                )
                n_actual = len(mean_cols)
            else:
                n_actual = len(cols_sorted)

            weights = [
                [
                    max(
                        1
                        - abs((angle - i * 360 / n_actual + 180) % 360 - 180)
                        / (360 / n_actual),
                        0,
                    )
                    for i in range(n_actual)
                ]
                for angle in angles
            ]

            df_new = pd.DataFrame(index=df_temp_data.index)

            for qi, angle in enumerate(angles):
                indices = np.where(np.array(weights[qi]) > 0)[0]
                if len(indices) == 0:
                    continue
                weight_sum = sum(
                    weights[qi][idx] for idx in indices if weights[qi][idx] > 0
                )
                if request_type == "avg-min-max":
                    df_new[f"Q_{qi+1}_mean"] = (
                        sum(
                            df_temp_data[mean_cols].iloc[:, idx] * weights[qi][idx]
                            for idx in indices
                            if weights[qi][idx] > 0
                        )
                        / weight_sum
                    )
                    df_new[f"Q_{qi+1}_max"] = (
                        sum(
                            df_temp_data[max_cols].iloc[:, idx] * weights[qi][idx]
                            for idx in indices
                            if weights[qi][idx] > 0
                        )
                        / weight_sum
                    )
                    df_new[f"Q_{qi+1}_min"] = (
                        sum(
                            df_temp_data[min_cols].iloc[:, idx] * weights[qi][idx]
                            for idx in indices
                            if weights[qi][idx] > 0
                        )
                        / weight_sum
                    )
                else:
                    df_new[f"Q_{qi+1}"] = (
                        sum(
                            df_temp_data.iloc[:, idx] * weights[qi][idx]
                            for idx in indices
                            if weights[qi][idx] > 0
                        )
                        / weight_sum
                    )

            levelwise_dict[level] = df_new

        # Sort levels bottom-to-top
        levelwise_dict = {int(k): v for k, v in levelwise_dict.items()}
        levelwise_dict = dict(sorted(levelwise_dict.items()))
        levelwise_dict = {str(k): v for k, v in levelwise_dict.items()}

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
