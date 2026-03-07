# FurnaceMind/memory/retriever.py
# Purpose: Retrieve relevant historical context for LLM reasoning
# Fixed: Type annotations corrected, operator_notes extracted from
#        raw search results (not formatted strings), proper error handling

import logging
from typing import List, Dict, Optional

from FurnaceMind.memory.structured_store import StructuredStore
from FurnaceMind.memory.vector_store import QdrantVectorStore
from FurnaceMind.memory.schemas import ShiftSummary

logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Retrieves relevant historical context for a given shift.
    """

    def __init__(
        self,
        structured_store: StructuredStore,
        vector_store: QdrantVectorStore,
    ):
        self.structured_store = structured_store
        self.vector_store = vector_store

    def retrieve_context(
        self,
        current_shift_id: str,
        current_shift_text: str,
        top_k_similar: int = 3,
    ) -> dict:
        """
        Retrieve previous shift summary and similar historical shifts.
        Returns structured context dict with text summaries and operator notes.
        """
        previous_summary = self._get_previous_shift_summary(current_shift_id)

        # Get raw search results first (for operator_notes extraction)
        raw_results = self.vector_store.search_similar_windows(
            query_text=current_shift_text,
            top_k=top_k_similar,
            window_type="shift",
        )

        # Extract operator notes from raw results (before formatting)
        operator_notes = []
        for hit in raw_results:
            payload = hit.get("payload", {})
            ctx = payload.get("operator_context")
            if isinstance(ctx, dict) and ctx.get("notes"):
                operator_notes.append(ctx["notes"])

        # Format similar shifts into text summaries
        similar_summaries = self._format_similar_shifts(raw_results)

        return {
            "previous_shift": previous_summary,
            "historical_similar": similar_summaries,
            "operator_notes": operator_notes,
        }

    def _get_previous_shift_summary(
        self, current_shift_id: str
    ) -> Optional[str]:
        """Fetch the immediate previous shift summary."""
        all_shifts = self.structured_store.load_all_shift_summaries()
        if not all_shifts:
            return None

        all_shifts_sorted = sorted(
            all_shifts, key=lambda x: x.shift_start
        )

        for idx, shift in enumerate(all_shifts_sorted):
            if shift.shift_id == current_shift_id and idx > 0:
                return self._format_shift_summary(
                    all_shifts_sorted[idx - 1]
                )

        return None

    def _format_similar_shifts(
        self, raw_results: List[Dict]
    ) -> List[str]:
        """
        Convert raw vector search results into formatted text summaries.
        """
        summaries = []

        for r in raw_results:
            payload = r.get("payload", {})

            if payload.get("window_type") != "shift":
                continue

            shift_id = payload.get("window_id")
            if not shift_id:
                continue

            shift_summary = self.structured_store.get_shift_by_id(shift_id)
            if shift_summary:
                summaries.append(
                    self._format_shift_summary(shift_summary)
                )

        return summaries

    @staticmethod
    def _format_shift_summary(shift: ShiftSummary) -> str:
        """Convert structured shift summary into readable text for LLM."""
        if shift.stability_index is not None:
            stability_line = (
                f"Overall Stability: "
                f"{shift.stability_status} "
                f"(Index: {shift.stability_index:.1f})"
            )
        else:
            stability_line = "Overall Stability: NOT AVAILABLE"

        lines = [
            f"Shift ID: {shift.shift_id}",
            f"Period: {shift.shift_start} to {shift.shift_end}",
            stability_line,
            f"Number of Anomalies: {shift.num_anomalies}",
        ]

        if shift.anomalous_parameters:
            lines.append(
                "Anomalous Parameters: "
                + ", ".join(shift.anomalous_parameters)
            )

        return "\n".join(lines)