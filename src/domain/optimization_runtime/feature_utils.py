"""Shared feature-name helpers for optimization runtime models.

This module centralizes feature-name normalization and lag suffix parsing so
BMO services can resolve model columns consistently across payloads, history
frames, scaler metadata, and notebook-exported feature manifests.
"""

from __future__ import annotations

import re

_LAG_PATTERN = re.compile(r"(.+)_lag(\d+)(?:_\([^)]+\))?$")


def normalize_feature_name(name: str) -> str:
    """
    Normalize a raw model or process feature name into a stable lookup key.

    Feature sources use mixed punctuation, casing, and units. Normalizing them
    once gives payload, history, and model metadata a shared comparison format
    without losing the original display names.

    Args:
         - name: str - Feature name from config, payload, model metadata, or history.

    Returns:
         - return str - Lowercase underscore-normalized feature key.
    """

    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def parse_lag_feature_name(feature_name: str) -> tuple[str, int] | None:
    """
    Parse a lagged feature name into its base feature and lag step.

    Shared runtime features can encode lag as a suffix such as ``_lag4``. This
    parser lets feature builders resolve those columns from historical rows
    while leaving non-lagged features untouched.

    Args:
         - feature_name: str - Feature name that may end with a lag suffix.

    Returns:
         - return tuple[str, int] | None - Parsed base name and lag step, or None.
    """

    match = _LAG_PATTERN.match(str(feature_name))
    if not match:
        return None
    return match.group(1), int(match.group(2))
