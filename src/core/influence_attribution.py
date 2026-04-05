"""Parameter influence attribution for BF2 furnace instability analysis.

This module identifies which process parameters contributed most to observed
instability during a shift by aggregating anomaly signals, stability penalty
shares, and recurrence weights into a ranked influence index.
"""

from collections import defaultdict
from typing import Dict, List


class InfluenceAttribution:
    """
    Computes parameter influence scores for furnace instability
    using structured shift summaries (v1).
    """

    def __init__(self) -> None:
        """Initialise the attribution engine (stateless)."""
        pass

    def compute(
        self,
        shift_summary,
        recurring_anomalies: Dict[str, dict] | None = None,
    ) -> List[dict]:
        """Compute a ranked list of parameter influence scores.

        Three signals are combined:

        1. **Anomaly occurrence** – each anomalous parameter receives +10 points.
        2. **Stability penalty share** – the anomaly component of the FSI
           penalty is split equally among anomalous parameters.
        3. **Recurrence multiplier** – parameters classified as recurring
           (``count ≥ 3``) have their score scaled by ``×1.5``.

        Scores are then normalised to``[0, 1]`` and ranked in descending order.

        Args:
            shift_summary:      A :class:`~memory.schemas.ShiftSummary` object
                                for the current shift.
            recurring_anomalies: Optional output of
                                 :meth:`core.recurring_anomaly_tracker\
.RecurringAnomalyTracker.detect` mapping parameter names to
                                 recurrence info dicts.

        Returns:
            A list of influence dicts sorted by descending ``influence_index``.
            Each dict contains:

            * ``parameter``      – column name.
            * ``influence_index`` – normalised score in ``[0, 1]``.
            * ``contributors``   – list of human-readable reason strings.
            * ``rank``           – 1-based rank.
        """
        influence = defaultdict(float)
        reasons = defaultdict(list)

        anomalous_params = shift_summary.anomalous_parameters or []

        # 1. Anomaly impact (uniform v1 signal)
        for param in anomalous_params:
            influence[param] += 10.0
            reasons[param].append("Anomaly occurrence")

        # 2. Stability penalty contribution
        penalties = shift_summary.stability_penalties or {}
        anomaly_penalty = penalties.get("anomaly", 0.0)

        if anomalous_params and anomaly_penalty > 0:
            share = anomaly_penalty / len(anomalous_params)
            for param in anomalous_params:
                influence[param] += share
                reasons[param].append("Anomaly penalty contribution")

        # 3. Recurrence weight
        if recurring_anomalies:
            for param, info in recurring_anomalies.items():
                if param in influence and info.get("count", 0) >= 3:
                    influence[param] *= 1.5
                    reasons[param].append("Recurring anomaly pattern")

        # Normalize & rank
        total = sum(influence.values()) or 1.0

        ranked = sorted(
            influence.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for rank, (param, score) in enumerate(ranked, start=1):
            result.append(
                {
                    "parameter": param,
                    "influence_index": round(score / total, 3),
                    "contributors": reasons[param],
                    "rank": rank,
                }
            )

        return result
