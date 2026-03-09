# memory/schemas.py
# Purpose: Structured schemas for storing furnace operational intelligence

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field



# Shift-Level Schema
class ShiftAnomalyDetail(BaseModel):
    """
    Detailed anomaly information for a single parameter.
    """
    severity: str
    trend: str
    z_score: float
    delta_vs_prev: Optional[float]
    reasons: List[str]


class OperatorFeedback(BaseModel):
    rating: Optional[int] = None  # 1–5
    comment: Optional[str] = None


class OperatorContext(BaseModel):
    notes: Optional[str] = None
    feedback: Optional[OperatorFeedback] = None


class ShiftSummary(BaseModel):
    """
    Canonical structured summary for one 8-hour shift.
    """
    shift_id: str

    # Timing (older records may not have these)
    shift_start: Optional[datetime] = None
    shift_end: Optional[datetime] = None
    generated_at: Optional[datetime] = None

    # Furnace Stability Index
    stability_index: Optional[float] = None                  # 0–100 score
    stability_status: Optional[str] = "UNKNOWN"
    stability_penalties: Dict[str, float] = Field(default_factory=dict)

    # High-level indicators
    num_parameters: Optional[int] = 0
    num_anomalies: Optional[int] = 0

    # Drift & anomaly intelligence
    early_drift: Dict[str, dict] = Field(default_factory=dict)

    anomalous_parameters: List[str] = Field(default_factory=list)
    anomaly_details: Dict[str, ShiftAnomalyDetail] = Field(default_factory=dict)

    # Operator human context (NEW)
    operator_context: Optional[OperatorContext] = None

    # Optional future extensions
    control_actions: Optional[Dict[str, float]] = None
    fuel_efficiency_indicator: Optional[float] = None
    thermal_balance_indicator: Optional[float] = None
    material_condition_notes: Optional[str] = None


# Aggregated Schemas (Future Use)

class DailySummary(BaseModel):
    """
    Aggregated operational summary for one day.
    """
    date: str
    generated_at: datetime
    shift_ids: List[str]
    overall_stability: str
    key_issues: List[str]
    summary_text: str


class WeeklySummary(BaseModel):
    """
    Aggregated operational summary for one week.
    """
    week_id: str
    generated_at: datetime
    shift_ids: List[str]
    recurring_issues: List[str]
    learning_points: List[str]
    summary_text: str


class BiWeeklySummary(BaseModel):
    """
    Aggregated operational summary for two weeks.
    """
    period_id: str
    generated_at: datetime
    shift_ids: List[str]
    major_trends: List[str]
    recommendations: List[str]
    summary_text: str
    