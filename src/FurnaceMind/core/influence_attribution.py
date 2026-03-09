from collections import defaultdict
from typing import Dict, List


class InfluenceAttribution:
    """
    Computes parameter influence scores for furnace instability
    using structured shift summaries (v1).
    """

    def __init__(self):
        pass

    def compute(
        self,
        shift_summary,
        recurring_anomalies: Dict[str, dict] | None = None,
    ) -> List[dict]:

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
            result.append({
                "parameter": param,
                "influence_index": round(score / total, 3), 
                "contributors": reasons[param],
                "rank": rank,
            })

        return result