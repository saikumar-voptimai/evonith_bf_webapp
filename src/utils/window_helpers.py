import re
from datetime import date
from typing import Dict, Optional

from agents.memory.vector_store import QdrantVectorStore
from utils.payload_helpers import window_id_to_uuid

# Window ID builders (DERIVABLE LEVELS ONLY)


def build_shift_window_id(d: date, shift_label: str) -> str:
    """
    UUID seed used for SHIFT points in Qdrant.
    Example: shift_2025-11-17_Shift_C
    """
    return f"{d.strftime('%Y-%m-%d')}_Shift_{shift_label}"


def build_day_window_id(d: date) -> str:
    """
    UUID seed used for DAY points in Qdrant.
    Example: day_2025-11-14
    """
    return f"day_{d.strftime('%Y-%m-%d')}"


def _normalize_report_payload(
    payload: Dict,
    *,
    fallback_window_id: Optional[str] = None,
) -> Dict:
    """Return a UI-friendly payload shape for both legacy and current records."""
    normalized = dict(payload)

    if "summary_text" not in normalized and "summary" in normalized:
        normalized["summary_text"] = normalized["summary"]

    if "window_id" not in normalized and fallback_window_id:
        normalized["window_id"] = fallback_window_id

    return normalized


def _parse_shift_window_id(window_id: str) -> Optional[tuple[str, str]]:
    """Extract ``(date_str, shift_label)`` from supported shift window IDs."""
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_(?:SHIFT|Shift)_([ABC])$", window_id)
    if not match:
        return None

    return match.group(1), match.group(2)


# Qdrant fetch helper (EXACT ID LOOKUP)
def fetch_from_qdrant(
    vector_store: QdrantVectorStore,
    window_id: str,
) -> Optional[Dict]:
    """
    Fetch a report payload from Qdrant using exact UUID seed.
    """

    point_id = window_id_to_uuid(window_id)

    points = vector_store.client.retrieve(
        collection_name=vector_store.collection_name,
        ids=[point_id],
        with_payload=True,
    )

    if points:
        return _normalize_report_payload(
            points[0].payload or {},
            fallback_window_id=window_id,
        )

    shift_lookup = _parse_shift_window_id(window_id)
    if shift_lookup is None:
        return None

    date_str, shift_label = shift_lookup
    payload = vector_store.get_report_by_metadata(date_str, shift_label)
    if payload is None:
        return None

    return _normalize_report_payload(payload, fallback_window_id=window_id)