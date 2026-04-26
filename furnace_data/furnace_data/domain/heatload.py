"""Domain fetchers for cooling-stave heat-load data (V-Board).

Provides
--------
AverageHeatLoadDataFetcher  Per-row, per-quadrant min/mean/max fetcher for
                             the circumferential contour view.
TsHeatloadDataFetcher       Time-series heat-load fetcher.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from furnace_data.influx.base import BaseDataFetcher


class AverageHeatLoadDataFetcher(BaseDataFetcher):
    """Fetch per-row, per-quadrant average heat-load for the V-Board contour view.

    Args:
        debug:  If ``True``, returns dummy data.
        source: ``"live"`` or ``"historical"``.
    """

    def __init__(self, debug: bool = False, source: str = "live") -> None:
        super().__init__("heatload_delta_t", debug=debug, source=source)

    def fetch_averaged_data(  # type: ignore[override]
        self,
        recent_data_of: str,
        start_time=None,
        end_time=None,
        row: Optional[str] = None,
        request_type: str = "avg-min-max",
        window_by: str = "1h",
    ):
        """Fetch and aggregate heat-load data for a specific stave row.

        Args:
            recent_data_of: Preset string or ``"over selected range"``.
            start_time:     UTC-aware start (required for explicit range).
            end_time:       UTC-aware end (required for explicit range).
            row:            Stave row identifier, e.g. ``"r6"`` through ``"r10"``.
            request_type:   Query type; ``"avg-min-max"`` returns
                            ``[mean_list, max_list, min_list]``.
            window_by:      Aggregation window (not valid for non-ts types).

        Returns:
            For ``request_type != "ts"``: ``[mean_list, max_list, min_list]``
            where each list has 4 values (Q1–Q4).

            For ``request_type == "ts"``: time-indexed :class:`pandas.DataFrame`
            with columns Q1–Q4.

        Raises:
            ValueError: If *row* is not specified.
        """
        if row is None:
            raise ValueError("row must be specified (e.g., 'r6').")

        df_flat = super().fetch_averaged_data(
            recent_data_of, start_time, end_time, request_type, window_by
        )
        if "time" in df_flat.columns:
            df_flat = df_flat.set_index("time")

        df_result = pd.DataFrame(
            columns=["Q1", "Q2", "Q3", "Q4"], index=df_flat.index
        )
        for q in range(1, 5):
            for col in df_flat.columns:
                if col.startswith(f"heat_load_{row.lower()}_q{q}"):
                    df_result[f"Q{q}"] = df_flat[col]

        df_result = df_result.apply(pd.to_numeric, errors="coerce")
        df_result[df_result < 0] = np.nan
        df_result[df_result > 1] = 1.0
        df_result.dropna(axis=0, how="all", inplace=True)
        df_result.interpolate(method="linear", axis=1, inplace=True)

        if request_type == "ts":
            df_result.index.name = "time"
            return df_result

        return self._post_process(df_result)

    @staticmethod
    def _post_process(df_result: pd.DataFrame) -> List[List[float]]:
        """Return ``[mean_list, max_list, min_list]`` for Q1–Q4."""
        return [
            df_result.mean().tolist(),
            df_result.max().tolist(),
            df_result.min().tolist(),
        ]

    # Keep old name as alias for backward compatibility.
    post_process = _post_process


class TsHeatloadDataFetcher(BaseDataFetcher):
    """Time-series fetcher for heat-load / delta-T data.

    Args:
        debug:  If ``True``, returns dummy data.
        source: ``"live"`` or ``"historical"``.
    """

    def __init__(self, debug: bool = False, source: str = "live") -> None:
        super().__init__("heatload_delta_t", debug=debug, source=source)

    def fetch_ts_data(
        self,
        recent_data_of: str,
        start_time=None,
        end_time=None,
        window_by: str = "15 minutes",
    ) -> pd.DataFrame:
        """Fetch windowed-average heat-load time series.

        Returns:
            Time-indexed :class:`pandas.DataFrame` with heat-load columns.
        """
        df = self.fetch_averaged_data(
            recent_data_of=recent_data_of,
            start_time=start_time,
            end_time=end_time,
            request_type="windowed-average",
            window_by=window_by,
        )
        if "time" in df.columns:
            df = df.set_index("time")
        return df
