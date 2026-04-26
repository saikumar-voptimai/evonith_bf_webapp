"""InfluxDB fetcher for the 110-sensor furnace wall temperature profile.

Fetches the ``temperature_profile`` measurement and aggregates readings by
elevation and circumferential quadrant (A–N per level).

Visualisation-specific reshaping (contour matrices, colour maps) lives in
``src/plotters/`` — not here.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from furnace_data.config import load_config
from furnace_data.influx.base import BaseDataFetcher

_config = load_config("setting_ds_dv.yml")
SENSORS_AT_Y: dict = _config["plot"]["geometry"]["heights_dict"]

_ALL_QUADRANTS = list("ABCDEFGHIJKLMN")


class TemperatureDataFetcher(BaseDataFetcher):
    """Fetcher for the circumferential furnace wall temperature profile.

    Args:
        debug:  If ``True``, returns dummy data without hitting InfluxDB.
        source: ``"historical"`` (default) or ``"live"``.
    """

    def __init__(self, debug: bool = False, source: str = "historical") -> None:
        super().__init__("temperature_profile", debug=debug, source=source)

    def _get_variable_names(self, n_sensors: int) -> List[str]:  # type: ignore[override]
        variable_names: List[str] = []
        for i, num_sensors in enumerate(SENSORS_AT_Y):
            quadrants = _ALL_QUADRANTS[:num_sensors]
            variable_names.extend(
                [f"{self.variables[i]}{q}" for q in quadrants]
            )
        return variable_names

    def fetch_averaged_data(  # type: ignore[override]
        self,
        recent_data_of: str,
        start_time=None,
        end_time=None,
        request_type=None,
        window_by=None,
    ):
        """Fetch and aggregate temperature data by elevation and quadrant.

        For ``source == "historical"`` the raw DataFrame is fetched via the
        parent class and then grouped into 4 circumferential quadrants (Q1–Q4)
        using angular interpolation weights.

        Returns:
            For historical mode: a ``dict`` keyed by ``"Q1"``–``"Q4"`` (each a
            DataFrame of elevation → weighted temperature) plus ``"time"``.

            For dummy/debug mode: ``{variable_name: float}`` dict.
        """
        source_clean = self.source.strip().lower()

        if not self.debug and source_clean == "historical":
            raw = super().fetch_averaged_data(
                recent_data_of, start_time, end_time, request_type, window_by
            )

            if not isinstance(raw, pd.DataFrame):
                raw = pd.DataFrame(raw, index=[start_time], columns=raw.keys())
            else:
                if "time" in raw.columns:
                    raw = raw.set_index("time")

            level_cols: dict[str, list[str]] = {}
            for col in raw.columns:
                if not col.startswith("temp_"):
                    continue
                parts = col.split("_")
                if len(parts) < 3:
                    continue
                level = parts[1]
                level_cols.setdefault(level, []).append(col)

            val_dict: dict = {f"Q{i+1}": {} for i in range(4)}
            val_dict["time"] = (
                raw.index.tolist() if hasattr(raw.index, "tolist") else list(raw.index)
            )

            levelwise_dict: dict[str, pd.DataFrame] = {}
            for level, cols in level_cols.items():
                n_sensors = SENSORS_AT_Y[level]["n_sensors"]
                df_level = raw[cols].copy()
                df_level[df_level <= 25] = np.nan
                df_level.dropna(axis=0, how="all", inplace=True)
                df_level.interpolate(method="linear", axis=1, inplace=True)
                temp_matrix = df_level.to_numpy()
                if temp_matrix.shape[1] == 0:
                    temp_matrix = np.zeros((temp_matrix.shape[0], 1))

                df_new = pd.DataFrame(
                    index=df_level.index, columns=[f"Q_{i}" for i in range(1, 5)]
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

                mean_cols = [c for c in df_level.columns if c.endswith("_mean")]
                max_cols  = [c for c in df_level.columns if c.endswith("_max")]
                min_cols  = [c for c in df_level.columns if c.endswith("_min")]

                for i, angle in enumerate(angles):
                    indices = np.where(np.array(weights[i]) > 0)[0]
                    if len(indices) == 0:
                        continue
                    if request_type == "avg-min-max":
                        for suffix, src_cols in (
                            ("mean", mean_cols), ("max", max_cols), ("min", min_cols)
                        ):
                            if src_cols:
                                df_new[f"Q_{i+1}_{suffix}"] = sum(
                                    df_level[src_cols].iloc[:, idx] * weights[i][idx]
                                    for idx in indices if weights[i][idx] > 0
                                ) / sum(weights[i][idx] for idx in indices if weights[i][idx] > 0)

                    df_new[f"Q_{i+1}"] = sum(
                        df_level.iloc[:, idx] * weights[i][idx]
                        for idx in indices if weights[i][idx] > 0
                    ) / sum(weights[i][idx] for idx in indices if weights[i][idx] > 0)

                levelwise_dict[level] = df_new

            # Sort by elevation (bottom → top)
            levelwise_dict = {
                str(k): v
                for k, v in sorted(
                    {int(k): v for k, v in levelwise_dict.items()}.items()
                )
            }
            return levelwise_dict

        elif not self.debug and source_clean == "live":
            return self.fetch_live_data()

        # debug / dummy mode
        return self._get_dummy_data()

    def _get_dummy_data(self) -> dict:
        return {var: float(np.random.randint(100, 500)) for var in self.variables}

    def post_process_by_level(self, temp_data: dict) -> dict:
        """Group temperature values by elevation level (e.g. ``"12975mm"``)."""
        import re

        level_dict: dict[str, list] = {}
        for var, value in temp_data.items():
            match = re.search(r"(\d{4,5})mm", var)
            if match:
                level_name = var.split("mm")[0] + "mm"
                level_dict.setdefault(level_name, []).append(value)
        return level_dict
