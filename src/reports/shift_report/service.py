"""Orchestrates the full shift report pipeline: fetch → build → analyse → render."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from reports.shift_report.analyser import ShiftAnalyser
from reports.shift_report.builder import ShiftBuilder
from reports.shift_report.data import ShiftReportData
from reports.shift_report.fetcher import ShiftFetcher
from reports.shift_report.renderer import as_markdown
from utils.shift_windows import previous_shift


class ShiftReportService:
    """Single entry point for generating a live shift handover report.

    Usage::

        from reports.shift_report import ShiftReportService
        from agents.llm.llm_client import OpenRouterClient

        service = ShiftReportService(llm_client=OpenRouterClient())
        report_data, markdown = service.generate(date.today(), "A")
    """

    def __init__(self, llm_client=None) -> None:
        """
        Initialize the shift report service dependencies.

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
        shift_label: Literal["A", "B", "C"],
        *,
        include_analysis: bool = True,
    ) -> tuple[ShiftReportData, str]:
        """
        Fetch data, build metrics, optionally run LLM analysis.

        Args:
             - shift_date: date - Calendar date assigned to the shift.
             - shift_label: Literal["A", "B", "C"] - Shift label to generate.
             - include_analysis: bool - Whether to run LLM shift analysis.

        Returns:
             - return: tuple[ShiftReportData, str] - Report data and markdown.
        """
        raw = self._fetcher.fetch(shift_date=shift_date, shift_label=shift_label)
        current = self._builder.build(raw)

        previous: Optional[ShiftReportData] = None
        if include_analysis and self._analyser:
            try:
                prev_date, prev_label = previous_shift(shift_date, shift_label)
                prev_raw = self._fetcher.fetch(
                    shift_date=prev_date, shift_label=prev_label
                )
                previous = self._builder.build(prev_raw)
            except Exception:
                previous = None

        analysis = ""
        if include_analysis and self._analyser:
            try:
                analysis = self._analyser.analyse(current, previous)
            except Exception:
                analysis = ""

        return current, as_markdown(current, analysis)
