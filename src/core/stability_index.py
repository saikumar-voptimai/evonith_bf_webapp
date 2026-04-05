"""Furnace Stability Index (FSI) computation for BF2 shift analysis.

The FSI is a 0–100 composite score that penalises parameter variability,
anomaly counts, and adverse KPI trends.  A higher score means a more stable
furnace operating period.
"""

# core/stability_index.py

import numpy as np
import pandas as pd


class FurnaceStabilityIndex:
    """Compute a 0–100 Furnace Stability Index (FSI) for a shift window.

    The index deducts up to 40 points each for parameter variability (CV)
    and anomaly count, and up to 20 points for an adverse linear trend in the
    primary KPI.

    Attributes:
        critical_parameters: List of column names treated as key stability
            indicators when computing the variability penalty.
        primary_kpi:         Column name used for the linear trend penalty.
    """

    def __init__(
        self,
        critical_parameters: list[str],
        primary_kpi: str = "ETA_CO",
    ) -> None:
        """Initialise the index with the parameters to monitor.

        Args:
            critical_parameters: Column names to include in the variability
                penalty calculation.  Missing columns are skipped silently.
            primary_kpi:         Column name used to evaluate the linear trend
                penalty.  Defaults to ``"ETA_CO"``.
        """
        self.critical_parameters = critical_parameters
        self.primary_kpi = primary_kpi

    def compute_variability_penalty(self, df: pd.DataFrame) -> float:
        """Compute the variability penalty (0–40) from coefficient of variation.

        Calculates the mean coefficient of variation (CV = std / |mean|) across
        all ``critical_parameters`` present in *df*.  The raw CV is multiplied
        by 100 and capped at 40 to produce the penalty.

        Args:
            df: DataFrame containing the shift window data.

        Returns:
            Penalty score in ``[0.0, 40.0]``.
        """
        variabilities = []

        for param in self.critical_parameters:
            if param not in df.columns:
                continue

            series = df[param].dropna()
            if series.empty or series.mean() == 0:
                continue

            variabilities.append(series.std() / abs(series.mean()))

        if not variabilities:
            return 0.0

        variability_score = np.mean(variabilities)
        return min(40.0, variability_score * 100)

    def compute_anomaly_penalty(self, anomaly_count: int) -> float:
        """Compute the anomaly count penalty (0–40).

        Each anomalous parameter contributes 5 penalty points, capped at 40.

        Args:
            anomaly_count: Number of parameters flagged as anomalous during
                the shift.

        Returns:
            Penalty score in ``[0.0, 40.0]``.
        """
        return min(40.0, anomaly_count * 5)

    def compute_trend_penalty(self, df: pd.DataFrame) -> float:
        """Compute the KPI trend penalty (0–20) using linear regression.

        Fits a first-degree polynomial to the ``primary_kpi`` time series.
        The magnitude of the slope (scaled by 10) is capped at 20 to produce
        the penalty.  A large positive or negative slope suggests the KPI is
        drifting and penalises the stability score accordingly.

        Args:
            df: DataFrame containing the shift window data.

        Returns:
            Penalty score in ``[0.0, 20.0]``.
        """
        if self.primary_kpi not in df.columns:
            return 0.0

        series = df[self.primary_kpi].dropna()
        if len(series) < 3:
            return 0.0

        # simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]

        return min(20.0, abs(slope) * 10)

    def compute(
        self,
        df: pd.DataFrame,
        anomaly_count: int,
    ) -> dict:
        """Compute the composite Furnace Stability Index for a shift window.

        Combines the three sub-penalties (variability, anomaly count, and KPI
        trend) and subtracts them from a perfect score of 100.  The result is
        clamped to ``[0.0, 100.0]`` and labelled as:

        * **STABLE**   – FSI ≥ 80
        * **WATCH**    – 60 ≤ FSI < 80
        * **UNSTABLE** – FSI < 60

        Args:
            df:            DataFrame for the shift window (used for variability
                           and trend sub-penalties).
            anomaly_count: Count of anomalous parameters detected in this shift.

        Returns:
            A dict with the following keys:

            * ``stability_index`` – FSI score rounded to 1 decimal place.
            * ``stability_status`` – One of ``"STABLE"``, ``"WATCH"``, or
              ``"UNSTABLE"``.
            * ``penalties`` – Sub-penalty breakdown with keys ``"variability"``,
              ``"anomaly"``, and ``"trend"``.
        """
        v_penalty = self.compute_variability_penalty(df)
        a_penalty = self.compute_anomaly_penalty(anomaly_count)
        t_penalty = self.compute_trend_penalty(df)

        raw_score = 100.0 - (v_penalty + a_penalty + t_penalty)
        fsi = max(0.0, min(100.0, raw_score))

        if fsi >= 80:
            status = "STABLE"
        elif fsi >= 60:
            status = "WATCH"
        else:
            status = "UNSTABLE"

        return {
            "stability_index": round(fsi, 1),
            "stability_status": status,
            "penalties": {
                "variability": round(v_penalty, 1),
                "anomaly": round(a_penalty, 1),
                "trend": round(t_penalty, 1),
            },
        }
