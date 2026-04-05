# core/stability_index.py

import numpy as np
import pandas as pd


class FurnaceStabilityIndex:
    def __init__(
        self,
        critical_parameters: list[str],
        primary_kpi: str = "ETA_CO",
    ):
        self.critical_parameters = critical_parameters
        self.primary_kpi = primary_kpi

    def compute_variability_penalty(self, df: pd.DataFrame) -> float:
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
        return min(40.0, anomaly_count * 5)

    def compute_trend_penalty(self, df: pd.DataFrame) -> float:
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