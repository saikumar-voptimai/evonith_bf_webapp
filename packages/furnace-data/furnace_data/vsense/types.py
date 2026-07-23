"""Lightweight V-Sense domain dataclasses used outside web frameworks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


ControlMode = Literal["optimize", "fixed"]
RunStatus = Literal["pending", "running", "completed", "failed", "cancelled", "expired"]


@dataclass(frozen=True)
class ControlPlanItem:
    parameter_id: str
    mode: ControlMode
    lower_bound: float
    upper_bound: float
    fixed_value: float | None = None


@dataclass(frozen=True)
class InputOverride:
    parameter_id: str
    value: float


@dataclass(frozen=True)
class ContextReference:
    context_id: str
    owner_user_id: str
    optimization_type_id: str
    catalog_version: str
    algorithm_version: str
    created_at: datetime
    expires_at: datetime
    snapshot: dict[str, Any] = field(default_factory=dict)
