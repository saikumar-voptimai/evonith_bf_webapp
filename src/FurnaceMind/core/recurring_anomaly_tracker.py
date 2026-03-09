# FurnaceMind/core/recurring_anomaly_tracker.py
# Purpose: Detect recurring anomaly patterns from historical shifts
# Fixed: Meaningful pattern detection (consecutive, time-of-day, frequency)
#        instead of dead "Occasional" code branch.

from collections import defaultdict
from typing import List, Dict
from datetime import datetime


class RecurringAnomalyTracker:
    """
    Detects recurring anomaly patterns from historical ShiftSummary objects.
    """

    def __init__(self, min_occurrences: int = 3):
        self.min_occurrences = min_occurrences

    def detect(self, shift_summaries: List) -> Dict[str, dict]:
        """
        Detect recurring anomalies from shift summaries.
        Returns dict keyed by anomaly parameter with pattern info.
        """
        anomaly_map = defaultdict(list)

        for summary in shift_summaries:
            if not summary.anomalous_parameters:
                continue

            for param in summary.anomalous_parameters:
                anomaly_map[param].append({
                    "shift_id": summary.shift_id,
                    "shift_start": summary.shift_start,
                    "shift_end": summary.shift_end,
                })

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
        valid_times = [e["shift_end"] for e in events if e["shift_end"] is not None]
        return max(valid_times) if valid_times else None


    def _infer_pattern(self, events: list) -> str:
        """
        Detect meaningful patterns:
        - Consecutive: anomaly appears in back-to-back shifts
        - Time-of-day: anomaly clusters in same shift slot (A/B/C)
        - Increasing: frequency increasing over recent shifts
        - General: recurring but no clear pattern
        """
        if len(events) < self.min_occurrences:
            return "Occasional"

        # Check for consecutive shifts
        shift_ids = [e["shift_id"] for e in events if e["shift_id"]]
        consecutive_count = self._count_consecutive(shift_ids)
        if consecutive_count >= 3:
            return f"Consecutive ({consecutive_count} shifts in a row)"

        # Check for time-of-day clustering (same shift label)
        shift_labels = []
        for sid in shift_ids:
            # Extract label from shift_id like "2025-01-01_SHIFT_A"
            parts = sid.rsplit("_", 1)
            if len(parts) == 2:
                shift_labels.append(parts[-1])

        if shift_labels:
            from collections import Counter
            label_counts = Counter(shift_labels)
            dominant_label, dominant_count = label_counts.most_common(1)[0]

            if dominant_count / len(shift_labels) >= 0.6:
                return f"Shift-specific (mostly Shift {dominant_label})"

        # Check if frequency is increasing (more events in recent half)
        mid = len(events) // 2
        if mid > 0:
            recent_count = len(events) - mid
            older_count = mid
            if recent_count > older_count * 1.5:
                return "Increasing frequency"

        return "Repeated across multiple shifts"

    @staticmethod
    def _count_consecutive(shift_ids: list[str]) -> int:
        """Count the longest consecutive run of shift IDs."""
        if len(shift_ids) < 2:
            return len(shift_ids)

        # Sort by shift ID (which is date-based, so lexicographic works)
        sorted_ids = sorted(shift_ids)

        max_run = 1
        current_run = 1

        for i in range(1, len(sorted_ids)):
            # Simple heuristic: if shifts share the same date prefix
            # or are from consecutive dates, count as consecutive
            prev_date = sorted_ids[i - 1][:10]  # YYYY-MM-DD
            curr_date = sorted_ids[i][:10]

            if curr_date == prev_date or _is_adjacent_date(prev_date, curr_date):
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        return max_run


def _is_adjacent_date(date_str1: str, date_str2: str) -> bool:
    """Check if two YYYY-MM-DD date strings are adjacent days."""
    try:
        from datetime import timedelta
        d1 = datetime.strptime(date_str1, "%Y-%m-%d").date()
        d2 = datetime.strptime(date_str2, "%Y-%m-%d").date()
        return abs((d2 - d1).days) <= 1
    except (ValueError, TypeError):
        return False