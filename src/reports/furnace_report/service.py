"""Orchestrates the report pipeline: fetch -> build -> analyse -> render."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from datetime import datetime
import logging
from typing import Optional

from reports.rendering import ReportDocument, document_to_markdown
from reports.furnace_report.analyser import ShiftAnalyser
from reports.furnace_report.builder import ShiftBuilder
from reports.furnace_report.data import (
    ReportTimeframe,
    ReportType,
    ShiftLabel,
    ShiftReportData,
)
from reports.furnace_report.fetcher import ShiftFetcher
from reports.furnace_report.renderer import as_document
from reports.furnace_report.timeframe import (
    get_report_timeframe,
    previous_report_timeframe,
)
from utils.shift_windows import LOCAL_TIMEZONE

logger = logging.getLogger(__name__)


class ShiftReportService:
    """Single entry point for generating live shift and day reports.

    Usage::

        from reports.furnace_report import ShiftReportService
        from agents.llm.llm_client import OpenRouterClient

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
        current = self.build(timeframe)
        analyse = bool(include_analysis and self._analyser)
        previous = self.build_previous(timeframe) if analyse else None

        analysis = ""
        if analyse:
            try:
                analysis = self._analyser.analyse(current, previous)
            except Exception:
                logger.warning("Furnace report analysis failed", exc_info=True)
                analysis = ""

        return current, self.document(current, analysis, timeframe)

    # Building blocks of ``generate_document``, shared with ``services.reports`` (twin-worker, SV-2.3).

    def build(self, timeframe: ReportTimeframe) -> ShiftReportData:
        """Fetch the raw data of one report window and compute its metrics."""
        return self._builder.build(self._fetcher.fetch(timeframe=timeframe))

    def build_previous(self, timeframe: ReportTimeframe) -> Optional[ShiftReportData]:
        """Metrics of the window before ``timeframe`` for the analysis; None when they cannot be built."""
        try:
            return self.build(previous_report_timeframe(timeframe))
        except Exception:
            logger.warning(
                "Unable to build previous report window for analysis",
                exc_info=True,
            )
            return None

    def document(
        self,
        current: ShiftReportData,
        analysis: str,
        timeframe: ReportTimeframe,
    ) -> ReportDocument:
        """The structured report document of ``current`` with the analysis text, stamped now (IST)."""
        document = as_document(current, analysis, title=_live_report_title(timeframe))
        return replace(document, generated_at_ist=datetime.now(LOCAL_TIMEZONE))


def _live_report_title(timeframe: ReportTimeframe) -> str:
    if timeframe.report_type == "Shift":
        return f"Live Report ({timeframe.window_id})"
    return f"Live {timeframe.report_type} Report ({timeframe.window_id})"
