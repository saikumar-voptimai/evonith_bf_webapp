from __future__ import annotations

import re

_LAG_PATTERN = re.compile(r"(.+)_lag(\d+)$")


def normalize_feature_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def parse_lag_feature_name(feature_name: str) -> tuple[str, int] | None:
    match = _LAG_PATTERN.match(str(feature_name))
    if not match:
        return None
    return match.group(1), int(match.group(2))
