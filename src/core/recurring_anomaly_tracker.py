from collections import defaultdict
from typing import List, Dict
from datetime import datetime


class RecurringAnomalyTracker:
    """
    Detects recurring anomaly patterns from historical ShiftSummary objects.
    """

    def __init__(self, min_occurrences: int = 3):
        """
        :param min_occurrences: Minimum number of shifts an anomaly must appear
                                in to be considered recurring.
        """
        self.min_occurrences = min_occurrences

    def detect(self, shift_summaries: List) -> Dict[str, dict]:
        """
        Detect recurring anomalies from shift summaries.

        :param shift_summaries: List of ShiftSummary objects
        :return: Dict keyed by anomaly parameter
        """
        anomaly_map = defaultdict(list)

        # Step 1 & 2: Extract anomaly events
        for summary in shift_summaries:
            if not summary.anomalous_parameters:
                continue

            for param in summary.anomalous_parameters:
                anomaly_map[param].append({
                    "shift_id": summary.shift_id,
                    "shift_end": summary.shift_end,
                })

        # Step 3 & 4: Apply recurrence rule
        recurring = {}

        for param, events in anomaly_map.items():
            if len(events) >= self.min_occurrences:
                recurring[param] = {
                    "count": len(events),
                    "shifts": [e["shift_id"] for e in events],
                    "last_seen": self._last_seen(events),
                    "pattern": self._infer_pattern(events),
                }

        return recurring

    def _last_seen(self, events: list) -> datetime | None:
        """
        Returns the most recent shift_end timestamp.
        """
        valid_times = [e["shift_end"] for e in events if e["shift_end"] is not None]
        return max(valid_times) if valid_times else None

    def _infer_pattern(self, events: list) -> str:
        """
        Simple, explainable v1 pattern logic.
        """
        if len(events) >= 3:
            return "Repeated across multiple shifts"
        return "Occasional"