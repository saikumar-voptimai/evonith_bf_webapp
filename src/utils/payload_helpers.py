"""Payload builder utilities for shift, day, week, and bi-week summaries.

Each ``build_*_payload`` function assembles a schema-validated dictionary
suitable for storage in :class:`~memory.structured_store.StructuredStore` and
embedding in Qdrant via :class:`~memory.vector_store.QdrantVectorStore`.

All builders delegate final validation and default-merging to the shared
:func:`build_payload` helper.
"""

# utils/helpers.py
# Purpose: Centralized payload builders and helper utilities

import uuid
from datetime import datetime


# Time helpers
def utc_now_iso() -> str:
    """
    Current UTC time in ISO format.
    """
    return datetime.utcnow().isoformat()


# Generic payload builder (ALREADY YOURS)
def build_payload(base_fields: dict, schema: dict) -> dict:
    """
    Build payload using schema defaults and provided fields.

    - Applies defaults
    - Validates required fields
    """
    payload = {}

    # Apply defaults
    payload.update(schema.get("defaults", {}))

    # Apply provided fields
    payload.update(base_fields)

    # Validate required fields
    missing = [k for k in schema.get("required_fields", []) if k not in payload]
    if missing:
        raise ValueError(f"Missing required payload fields: {missing}")

    return payload


# SHIFT payload builder
def build_shift_payload(
    *,
    shift_data,
    structured_summary: dict,
    llm_text: str,
    prev_shift=None,
    schema: dict,
    operator_context: dict | None = None,
) -> dict:
    """Build a schema-compliant payload dict for a single furnace shift.

    Merges FSI (Furnace Stability Index) fields, anomaly signals, and operator
    context into a flat dictionary.  Backward-compatible: falls back to
    ``anomaly_count`` if ``num_anomalies`` is absent in *structured_summary*.

    Args:
        shift_data:         :class:`~core.shift_builder.ShiftData` instance with
                            ``shift_id``, ``shift_start``, ``shift_end``.
        structured_summary: Dict returned by :class:`~core.shift_analyzer.ShiftAnalyzer`
                            containing FSI scores, signals, and metadata.
        llm_text:           Raw LLM-generated narrative for this shift.
        prev_shift:         Optional previous :class:`~core.shift_builder.ShiftData`
                            used to populate ``previous_window_id``.
        schema:             Payload schema dict with ``defaults`` and
                            ``required_fields`` keys.
        operator_context:   Optional operator notes/feedback dict.

    Returns:
        Schema-validated payload dictionary ready for storage.
    """

    anomaly_count = structured_summary.get(
        "num_anomalies", structured_summary.get("anomaly_count", 0)
    )

    num_parameters = structured_summary.get(
        "num_parameters", len(structured_summary.get("stats", {}))
    )

    generated_at = structured_summary.get("generated_at", utc_now_iso())

    stability_index = structured_summary.get("stability_index")
    stability_status = structured_summary.get("stability_status")
    stability_penalties = structured_summary.get("stability_penalties", {})

    legacy_overall = structured_summary.get("overall_stability")

    base_fields = {
        "window_type": "shift",
        "window_id": shift_data.shift_id,
        "shift_name": getattr(shift_data, "shift_name", shift_data.shift_id),
        "start_time": shift_data.shift_start.isoformat(),
        "end_time": shift_data.shift_end.isoformat(),
        "summary_text": llm_text,
        "stability_index": stability_index,
        "stability_status": stability_status,
        "stability_penalties": stability_penalties,
        "overall_stability": legacy_overall or stability_status or "UNKNOWN",
        "num_parameters": num_parameters,
        "num_anomalies": anomaly_count,
        "anomalous_parameters": list(structured_summary.get("signals", {}).keys()),
        "anomaly_details": structured_summary.get("signals", {}),
        "generated_at": generated_at,
        "previous_window_id": prev_shift.shift_id if prev_shift else None,
        # NEW
        "operator_context": operator_context,
    }

    return build_payload(base_fields, schema)


# DAY payload builder
def build_day_payload(
    *, day_id, shift_payloads, structured_summary, llm_text, schema
) -> dict:
    """Build a schema-compliant payload dict for a daily summary.

    Args:
        day_id:             ISO date string (e.g. ``"2024-06-15"``).
        shift_payloads:     Ordered list of shift payload dicts for this day.
        structured_summary: Aggregated summary dict with FSI trend fields.
        llm_text:           Raw LLM-generated narrative for the day.
        schema:             Payload schema with ``defaults`` and
                            ``required_fields`` keys.

    Returns:
        Schema-validated daily payload dictionary.
    """
    base_fields = {
        # WINDOW METADATA
        "window_type": "day",
        "window_id": day_id,
        "start_time": shift_payloads[0]["start_time"],
        "end_time": shift_payloads[-1]["start_time"],
        # LLM SUMMARY
        "summary_text": llm_text,
        # DAY-LEVEL FSI (SCHEMA REQUIRED)
        "avg_stability_index": structured_summary.get("avg_stability_index"),
        "stability_trend": structured_summary.get("stability_trend"),
        # METADATA
        "num_records": structured_summary.get("num_shifts"),
        "generated_at": utc_now_iso(),
    }

    return build_payload(base_fields, schema)


# WEEK payload builder
def build_week_payload(
    *, week_id, day_payloads, structured_summary, llm_text, schema
) -> dict:
    """Build a schema-compliant payload dict for a weekly summary.

    Args:
        week_id:            ISO week identifier (e.g. ``"2024-W25"``).
        day_payloads:       Ordered list of daily payload dicts for the week.
        structured_summary: Aggregated summary dict with FSI trend fields.
        llm_text:           Raw LLM-generated narrative for the week.
        schema:             Payload schema with ``defaults`` and
                            ``required_fields`` keys.

    Returns:
        Schema-validated weekly payload dictionary.
    """
    base_fields = {
        # WINDOW METADATA
        "window_type": "week",
        "window_id": week_id,
        "start_time": day_payloads[0]["start_time"],
        "end_time": day_payloads[-1]["end_time"],
        # LLM SUMMARY
        "summary_text": llm_text,
        # WEEK-LEVEL FSI (SCHEMA REQUIRED)
        "avg_stability_index": structured_summary.get("avg_stability_index"),
        "stability_trend": structured_summary.get("stability_trend"),
        # METADATA
        "num_records": structured_summary.get("num_days"),
        "generated_at": utc_now_iso(),
    }

    return build_payload(base_fields, schema)


# BI-WEEK payload builder
def build_biweek_payload(
    *, biweek_id, week_payloads, structured_summary, llm_text, schema
) -> dict:
    """Build a schema-compliant payload dict for a bi-weekly summary.

    Args:
        biweek_id:          Bi-week identifier (e.g. ``"2024-BW13"``).
        week_payloads:      Ordered list of weekly payload dicts for the period.
        structured_summary: Aggregated summary dict with FSI trend fields.
        llm_text:           Raw LLM-generated narrative for the bi-week.
        schema:             Payload schema with ``defaults`` and
                            ``required_fields`` keys.

    Returns:
        Schema-validated bi-weekly payload dictionary.
    """
    base_fields = {
        # WINDOW METADATA
        "window_type": "biweek",
        "window_id": biweek_id,
        "start_time": week_payloads[0]["start_time"],
        "end_time": week_payloads[-1]["end_time"],
        # LLM SUMMARY
        "summary_text": llm_text,
        # BI-WEEK-LEVEL FSI (SCHEMA REQUIRED)
        "avg_stability_index": structured_summary.get("avg_stability_index"),
        "stability_trend": structured_summary.get("stability_trend"),
        # METADATA
        "num_records": structured_summary.get("num_weeks"),
        "generated_at": utc_now_iso(),
    }

    return build_payload(base_fields, schema)


# Qdrant-safe deterministic ID helper
def window_id_to_uuid(window_id: str) -> str:
    """Convert a human-readable window_id into a deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, window_id))
