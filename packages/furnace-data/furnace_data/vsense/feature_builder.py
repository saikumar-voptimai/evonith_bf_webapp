"""V-Sense feature mapping helpers.

The API never accepts browser-supplied feature vectors. This module exposes
trusted public-ID to internal-feature mapping for backend/shared code only.
"""

from __future__ import annotations

from typing import Any

from furnace_data.vsense.catalog import feature_for_parameter_id


def apply_input_overrides(
    base_sample: dict[str, Any],
    overrides: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply validated public input overrides to trusted internal feature names."""

    sample = dict(base_sample)
    for item in overrides:
        feature_name = feature_for_parameter_id(str(item["parameter_id"]))
        if feature_name:
            sample[feature_name] = item["value"]
    return sample
