# memory/retriever.py
# Purpose: Retrieve relevant historical context for LLM reasoning

from typing import List, Optional

from agents.memory.schemas import ShiftSummary
from agents.memory.structured_store import StructuredStore
from agents.memory.vector_store import QdrantVectorStore


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

    # Context retrieval
    def retrieve_context(
        self,
        current_shift_id: str,
        current_shift_text: str,
        top_k_similar: int = 3,
    ) -> dict:
        """
        Retrieve previous shift summary and similar historical shifts.
        """

        previous_summary = self._get_previous_shift_summary(current_shift_id)
        similar_summaries = self._get_similar_shifts(current_shift_text, top_k_similar)

        operator_notes = []
        for hit in similar_summaries:
            ctx = hit.payload.get("operator_context")
            if ctx and ctx.get("notes"):
                operator_notes.append(ctx["notes"])

        return {
            "previous_shift": previous_summary,
            "historical_similar": similar_summaries,
            "operator_notes": operator_notes,
        }

    # Internal helpers
    def _get_previous_shift_summary(self, current_shift_id: str) -> Optional[str]:
        """
        Fetch the immediate previous shift summary.
        """
        all_shifts = self.structured_store.load_all_shift_summaries()
        if not all_shifts:
            return None

        all_shifts_sorted = sorted(all_shifts, key=lambda x: x.shift_start)

        for idx, shift in enumerate(all_shifts_sorted):
            if shift.shift_id == current_shift_id and idx > 0:
                return self._format_shift_summary(all_shifts_sorted[idx - 1])

        return None

    def _get_similar_shifts(self, query_text: str, top_k: int) -> List[str]:
        """
        Retrieve semantically similar shifts from vector DB.
        """

        results = self.vector_store.search_similar_windows(
            query_text=query_text,
            top_k=top_k,
            window_type="shift",  # 👈 IMPORTANT
        )

        summaries = []

        for r in results:
            payload = r.get("payload", {})

            # Ensure this is a shift-level payload
            if payload.get("window_type") != "shift":
                continue

            shift_id = payload.get("window_id")
            if not shift_id:
                continue

            shift_summary = self.structured_store.get_shift_by_id(shift_id)
            if shift_summary:
                summaries.append(self._format_shift_summary(shift_summary))

        return summaries

    @staticmethod
    def _format_shift_summary(shift: ShiftSummary) -> str:
        """
        Convert structured shift summary into readable text for LLM.
        """
        # Stability line (robust to old data)
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
                "Anomalous Parameters: " + ", ".join(shift.anomalous_parameters)
            )

        return "\n".join(lines)
