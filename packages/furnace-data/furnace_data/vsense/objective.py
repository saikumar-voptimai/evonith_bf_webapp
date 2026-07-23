"""Small objective helpers for V-Sense legacy_v1 diagnostics."""

from __future__ import annotations

from typing import Any


def objective_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the public objective summary from a completed V-Sense result."""

    target = result.get("target") or {}
    diagnostics = result.get("diagnostics") or {}
    return {
        "target_parameter_id": target.get("parameter_id"),
        "direction": target.get("direction"),
        "baseline": target.get("baseline"),
        "recommended": target.get("recommended"),
        "algorithm_version": diagnostics.get("algorithm_version"),
    }
