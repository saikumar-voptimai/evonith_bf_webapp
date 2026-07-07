"""Orchestrates the report pipeline: fetch -> build -> analyse -> render."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from datetime import datetime
import logging
from typing import Optional

from apps.frontend_streamlit.reports.rendering import ReportDocument, document_to_markdown
from apps.frontend_streamlit.reports.furnace_report.analyser import ShiftAnalyser
from apps.frontend_streamlit.reports.furnace_report.builder import ShiftBuilder
from apps.frontend_streamlit.reports.furnace_report.data import (
    ReportTimeframe,
    ReportType,
    ShiftLabel,
    ShiftReportData,
)
from apps.frontend_streamlit.reports.furnace_report.fetcher import ShiftFetcher
from apps.frontend_streamlit.reports.furnace_report.renderer import as_document
from apps.frontend_streamlit.reports.furnace_report.timeframe import (
    get_report_timeframe,
    previous_report_timeframe,
)
from apps.frontend_streamlit.utils.shift_windows import LOCAL_TIMEZONE

logger = logging.getLogger(__name__)


class ShiftReportService:
    """Single entry point for generating live shift and day reports.

    Usage::

        from apps.frontend_streamlit.reports.furnace_report import ShiftReportService
        from apps.frontend_streamlit.agents.llm.llm_client import OpenRouterClient

        service = ShiftReportService(llm_client=OpenRouterClient())
        report_data, markdown = service.generate(date.today(), "A")
    """

    def __init__(self, llm_client=None) -> None:
        """
        Initialize the furnace report service dependencies.

        Args:
             - llm_client: object | None - Optional LLM client for analysis.

        Returns:
             - return: None - This function does not return a value.
        """
        self._fetcher = ShiftFetcher()
        self._builder = ShiftBuilder()
        self._analyser = ShiftAnalyser(llm_client) if llm_client else None

    def generate(
        self,
        shift_date: date,
        shift_label: ShiftLabel | str | None = None,
        *,
        report_type: ReportType | str = "Shift",
        include_analysis: bool = True,
    ) -> tuple[ShiftReportData, str]:
        """
        Fetch data, build metrics, optionally run LLM analysis.

        Args:
             - shift_date: date - Selected report date.
             - shift_label: ShiftLabel | str | None - Shift label for Shift reports.
             - report_type: ReportType | str - Report type to generate.
             - include_analysis: bool - Whether to run LLM report analysis.

        Returns:
             - return: tuple[ShiftReportData, str] - Report data and markdown.
        """
        report_data, document = self.generate_document(
            shift_date,
            shift_label,
            report_type=report_type,
            include_analysis=include_analysis,
        )
        return report_data, document_to_markdown(document)

    def generate_document(
        self,
        shift_date: date,
        shift_label: ShiftLabel | str | None = None,
        *,
        report_type: ReportType | str = "Shift",
        include_analysis: bool = True,
    ) -> tuple[ShiftReportData, ReportDocument]:
        """Fetch data, build metrics, and return a structured report document."""
        timeframe = get_report_timeframe(report_type, shift_date, shift_label)
        raw = self._fetcher.fetch(timeframe=timeframe)
        current = self._builder.build(raw)

        previous: Optional[ShiftReportData] = None
        if include_analysis and self._analyser:
            try:
                prev_raw = self._fetcher.fetch(
                    timeframe=previous_report_timeframe(timeframe)
                )
                previous = self._builder.build(prev_raw)
            except Exception:
                logger.warning(
                    "Unable to build previous report window for analysis",
                    exc_info=True,
                )
                previous = None

        analysis = ""
        if include_analysis and self._analyser:
            try:
                analysis = self._analyser.analyse(current, previous)
            except Exception:
                logger.warning("Furnace report analysis failed", exc_info=True)
                analysis = ""

        document = as_document(current, analysis, title=_live_report_title(timeframe))
        return current, replace(document, generated_at_ist=datetime.now(LOCAL_TIMEZONE))


def _live_report_title(timeframe: ReportTimeframe) -> str:
    if timeframe.report_type == "Shift":
        return f"Live Report ({timeframe.window_id})"
    return f"Live {timeframe.report_type} Report ({timeframe.window_id})"
