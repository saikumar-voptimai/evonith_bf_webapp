from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def normalize_feature_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_feature_payload(
    quantities_mt: Mapping[str, float],
    ore_display_name_by_id: Mapping[str, str],
    process_context: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    payload: dict[str, float] = {}

    total_qty = float(sum(float(v) for v in quantities_mt.values()))
    sinter_qty = 0.0

    for ore_id, qty in quantities_mt.items():
        qty = float(qty)
        display = ore_display_name_by_id.get(ore_id, ore_id)
        if "sinter" in display.lower():
            sinter_qty += qty

        ore_slug = normalize_feature_name(display)
        payload[f"{ore_slug}_qty_mt"] = qty
        payload[f"{ore_slug}_share_pct"] = (qty / total_qty * 100.0) if total_qty > 0 else 0.0

    sinter_share_pct = (sinter_qty / total_qty * 100.0) if total_qty > 0 else 0.0
    payload["sinter_qty_mt"] = sinter_qty
    payload["ore_qty_mt"] = max(0.0, total_qty - sinter_qty)
    payload["sinter_share_pct"] = sinter_share_pct
    payload["ore_share_pct"] = max(0.0, 100.0 - sinter_share_pct)
    payload["total_burden_qty_mt"] = total_qty

    if process_context:
        for key, value in process_context.items():
            if value is None:
                continue
            try:
                payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
            payload[normalize_feature_name(str(key))] = payload[str(key)]

    return payload

