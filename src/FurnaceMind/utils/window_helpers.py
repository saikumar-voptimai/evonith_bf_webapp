# FurnaceMind/utils/window_helpers.py
# Purpose: Window ID builders and Qdrant lookup helpers
# Fixed: Uses canonical shift ID format from shift_builder

from datetime import date
from typing import Optional, Dict
from FurnaceMind.utils.payload_helpers import window_id_to_uuid
from FurnaceMind.memory.vector_store import QdrantVectorStore
from FurnaceMind.core.shift_builder import make_shift_id


# Window ID builders

def build_shift_window_id(d: date, shift_label: str) -> str:
    """
    UUID seed used for SHIFT points in Qdrant.
    Example: 2025-11-17_SHIFT_C
    """
    return make_shift_id(d.isoformat(), shift_label)


def build_day_window_id(d: date) -> str:
    """
    UUID seed used for DAY points in Qdrant.
    Example: day_2025-11-14
    """
    return f"day_{d.strftime('%Y-%m-%d')}"


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

    if not points:
        return None

    return points[0].payload