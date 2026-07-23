"""Control-profile and run-control validation for V-Sense."""

from __future__ import annotations

import math
from typing import Any

from furnace_data.vsense.catalog import (
    default_control_profile_parameters,
    load_vsense_catalog,
    optimization_by_id,
    parameter_by_id,
)


class VSenseValidationError(ValueError):
    """Domain validation error with an API-stable code."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def validate_control_profile(
    optimization_type_id: str,
    parameters: list[dict[str, Any]],
    *,
    require_approved_bounds: bool = True,
    catalog: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate and normalize a complete control-profile snapshot."""

    catalog_data = catalog or load_vsense_catalog()
    optimizations = optimization_by_id(catalog_data)
    if optimization_type_id not in optimizations:
        raise VSenseValidationError(
            "VSENSE_INVALID_OPTIMIZATION_TYPE",
            "Unknown V-Sense optimization type.",
        )

    expected_ids = [str(item) for item in optimizations[optimization_type_id]["control_parameter_ids"]]
    received_ids = [str(item.get("parameter_id") or "") for item in parameters]
    if len(received_ids) != len(set(received_ids)):
        raise VSenseValidationError(
            "VSENSE_INVALID_CONTROL_PLAN",
            "Control plan contains duplicate parameter IDs.",
        )
    if set(received_ids) != set(expected_ids):
        raise VSenseValidationError(
            "VSENSE_INVALID_CONTROL_PLAN",
            "Control plan must submit one complete snapshot for the selected optimization type.",
        )

    params = parameter_by_id(catalog_data)
    normalized_by_id = {
        item["parameter_id"]: _validate_item(
            item,
            params=params,
            require_approved_bounds=require_approved_bounds,
        )
        for item in parameters
    }
    return [normalized_by_id[parameter_id] for parameter_id in expected_ids]


def default_control_profile(
    optimization_type_id: str,
    *,
    require_approved_bounds: bool = True,
    catalog: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the validated plant-default profile for an optimization type."""

    return validate_control_profile(
        optimization_type_id,
        default_control_profile_parameters(optimization_type_id),
        require_approved_bounds=require_approved_bounds,
        catalog=catalog,
    )


def _validate_item(
    item: dict[str, Any],
    *,
    params: dict[str, dict[str, Any]],
    require_approved_bounds: bool,
) -> dict[str, Any]:
    parameter_id = str(item.get("parameter_id") or "")
    definition = params.get(parameter_id)
    if definition is None or definition.get("role") != "control":
        raise VSenseValidationError(
            "VSENSE_INVALID_PARAMETER",
            "Unknown V-Sense control parameter.",
        )
    mode = str(item.get("mode") or "").strip().lower()
    if mode not in {"optimize", "fixed"}:
        raise VSenseValidationError(
            "VSENSE_INVALID_CONTROL_PLAN",
            "Control mode must be optimize or fixed.",
        )
    lower = _finite(item.get("lower_bound"), "lower_bound")
    upper = _finite(item.get("upper_bound"), "upper_bound")
    if lower > upper:
        raise VSenseValidationError(
            "VSENSE_INVALID_CONTROL_PLAN",
            "Control lower_bound must be less than or equal to upper_bound.",
        )
    fixed_value = item.get("fixed_value")
    fixed = _finite(fixed_value, "fixed_value") if fixed_value is not None else None
    if mode == "fixed":
        if fixed is None:
            raise VSenseValidationError(
                "VSENSE_INVALID_CONTROL_PLAN",
                "fixed_value is required when mode is fixed.",
            )
        if fixed < lower or fixed > upper:
            raise VSenseValidationError(
                "VSENSE_INVALID_CONTROL_PLAN",
                "fixed_value must be inside the submitted bounds.",
            )
    if require_approved_bounds:
        approved_min = definition.get("approved_min")
        approved_max = definition.get("approved_max")
        if approved_min is not None and lower < float(approved_min):
            raise VSenseValidationError(
                "VSENSE_BOUND_OUTSIDE_APPROVED_ENVELOPE",
                "Submitted lower_bound is outside the approved operating envelope.",
            )
        if approved_max is not None and upper > float(approved_max):
            raise VSenseValidationError(
                "VSENSE_BOUND_OUTSIDE_APPROVED_ENVELOPE",
                "Submitted upper_bound is outside the approved operating envelope.",
            )
        if fixed is not None and (
            (approved_min is not None and fixed < float(approved_min))
            or (approved_max is not None and fixed > float(approved_max))
        ):
            raise VSenseValidationError(
                "VSENSE_BOUND_OUTSIDE_APPROVED_ENVELOPE",
                "Submitted fixed_value is outside the approved operating envelope.",
            )
    return {
        "parameter_id": parameter_id,
        "mode": mode,
        "lower_bound": lower,
        "upper_bound": upper,
        "fixed_value": fixed,
    }


def _finite(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise VSenseValidationError(
            "VSENSE_INVALID_CONTROL_PLAN",
            f"{field_name} must be a finite number.",
        ) from exc
    if not math.isfinite(numeric):
        raise VSenseValidationError(
            "VSENSE_INVALID_CONTROL_PLAN",
            f"{field_name} must be a finite number.",
        )
    return numeric
