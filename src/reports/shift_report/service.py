"""Orchestrates the full shift report pipeline: fetch → build → analyse → render."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

from reports.shift_report.analyser import ShiftAnalyser
from reports.shift_report.builder import ShiftBuilder
from reports.shift_report.data import ShiftReportData
from reports.shift_report.fetcher import ShiftFetcher
from reports.shift_report.renderer import as_markdown

_PREV: dict[str, tuple[int, str]] = {
    # (day_offset, prev_label)
    "A": (0, "C"),   # Shift A is preceded by Shift C on the same calendar day
    "B": (0, "A"),
    "C": (0, "B"),
}

# Special case: Shift A is preceded by Shift C of the *previous* day
_PREV_DAY_OFFSET = {"A": -1, "B": 0, "C": 0}


def _previous_shift(d: date, label: str) -> tuple[date, str]:
    prev_labels = {"A": "C", "B": "A", "C": "B"}
    day_offset = _PREV_DAY_OFFSET[label]
    return d + timedelta(days=day_offset), prev_labels[label]


class ShiftReportService:
    """Single entry point for generating a live shift handover report.

    Usage::

        from reports.shift_report import ShiftReportService
        from agents.llm.llm_client import OpenRouterClient

        service = ShiftReportService(llm_client=OpenRouterClient())
        report_data, markdown = service.generate(date.today(), "A")
    """

    def __init__(self, llm_client=None) -> None:
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
        """Fetch data, build metrics, optionally run LLM analysis.

        Returns ``(ShiftReportData, markdown_string)``.
        The markdown is ready to pass to ``ui.components.show_report()``.
        """
        raw = self._fetcher.fetch(shift_date=shift_date, shift_label=shift_label)
        current = self._builder.build(raw)

        previous: Optional[ShiftReportData] = None
        if include_analysis and self._analyser:
            try:
                prev_date, prev_label = _previous_shift(shift_date, shift_label)
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
