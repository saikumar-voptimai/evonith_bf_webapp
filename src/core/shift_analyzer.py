"""Shift-level anomaly analysis and LLM reporting for BF2 operations.

This module implements ``ShiftAnalyzer``, which consumes a single
:class:`~core.shift_builder.ShiftData` window, computes z-score-based
anomaly signals, and delegates structured reporting to the configured LLM.
"""

# core/shift_analyzer.py

from typing import Dict

import numpy as np
import pandas as pd

from utils.prompts import PromptTemplates


class ShiftAnalyzer:
    """
    Analyzes a single 8-hour shift using simple z-score anomalies
    and existing prompt templates.
    """

    def __init__(self, anomaly_config) -> None:
        """Initialise the analyzer with anomaly detection thresholds.

        Args:
            anomaly_config: An object (typically
                :class:`~utils.settings.AnomalyConfig`) exposing
                ``z_warn`` and ``z_critical`` float attributes.
        """
        self.z_warn = anomaly_config.z_warn
        self.z_critical = anomaly_config.z_critical

    def analyze(
        self,
        *,
        shift_data,
        prev_shift_data,
        llm,
    ):
        """Analyse a single shift window and produce LLM + structured output.

        For every numeric column with at least 20 observations:

        * Computes per-shift mean, std, z_max, and z_min.
        * Flags parameters whose absolute z-score exceeds ``z_warn`` or
          ``z_critical`` as anomalous.
        * Derives an overall stability label: *stable* (0 anomalies),
          *warning* (1–3), or *unstable* (> 3).

        The structured summary is then formatted into the
        :data:`~utils.prompts.PromptTemplates.SHIFT_ANALYSIS_TASK` template and
        sent to *llm* for a human-readable narrative.

        Args:
            shift_data:      :class:`~core.shift_builder.ShiftData` for the
                             current window.  Must expose ``.data`` (DataFrame)
                             and shift metadata attributes.
            prev_shift_data: Previous shift's :class:`~core.shift_builder.ShiftData`
                             (reserved for future delta analysis; currently unused).
            llm:             LLM client implementing ``.generate(system_prompt, user_prompt)``.

        Returns:
            A two-tuple ``(llm_summary, structured_summary)`` where
            *llm_summary* is the raw LLM text and *structured_summary* is a
            JSON-serialisable dict suitable for Qdrant payload storage.
        """
        df = shift_data.data

        # numeric-only (safe)
        df_num = df.apply(pd.to_numeric, errors="coerce")

        stats: Dict[str, Dict] = {}
        signals: Dict[str, Dict] = {}

        for col in df_num.columns:
            series = df_num[col].dropna()
            if len(series) < 20:
                continue

            mean = series.mean()
            std = series.std(ddof=0)

            if std <= 0 or not np.isfinite(std):
                continue

            z = (series - mean) / std
            z_max = z.max()
            z_min = z.min()

            # stats summary (simple + stable)
            stats[col] = {
                "mean": float(mean),
                "std": float(std),
                "z_max": float(z_max),
                "z_min": float(z_min),
            }

            # anomaly logic
            severity = None
            if abs(z_max) >= self.z_critical or abs(z_min) >= self.z_critical:
                severity = "critical"
            elif abs(z_max) >= self.z_warn or abs(z_min) >= self.z_warn:
                severity = "warning"

            if severity:
                signals[col] = {
                    "severity": severity,
                    "z_max": float(z_max),
                    "z_min": float(z_min),
                }

        # overall stability, anomaly count, num_parameters
        anomaly_count = len(signals)
        num_parameters = len(stats)

        if anomaly_count == 0:
            overall_stability = "stable"
        elif anomaly_count <= 3:
            overall_stability = "warning"
        else:
            overall_stability = "unstable"

        # structured summary (vector DB safe)
        structured_summary = {
            "shift_id": shift_data.shift_id,
            "shift_start": shift_data.shift_start,
            "shift_end": shift_data.shift_end,
            "overall_stability": overall_stability,
            "anomaly_count": anomaly_count,
            "num_parameters": num_parameters,
            "stats": stats,
            "signals": signals,
        }

        # EXISTING prompt (unchanged)
        user_prompt = PromptTemplates.SHIFT_ANALYSIS_TASK.format(
            shift_id=shift_data.shift_id,
            shift_start=shift_data.shift_start,
            shift_end=shift_data.shift_end,
            stats_summary=str(stats),
            anomaly_signals=str(signals),
        )

        # LLM call (unchanged)
        llm_summary = llm.generate(
            system_prompt=PromptTemplates.SHIFT_ANALYZER_SYSTEM,
            user_prompt=user_prompt,
        )

        return llm_summary, structured_summary
