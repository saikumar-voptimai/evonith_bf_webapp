# FurnaceMind/utils/payload_helpers.py
# Purpose: Centralized payload builders and helper utilities
# Fixed: Replaced deprecated datetime.utcnow() with timezone-aware UTC,
#        added type validation for critical fields

from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Current UTC time in ISO format (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_field_types(payload: dict) -> None:
    """Validate critical field types after payload assembly."""
    # stability_index must be numeric 0-100 or None
    si = payload.get("stability_index")
    if si is not None:
        if not isinstance(si, (int, float)):
            raise TypeError(f"stability_index must be numeric, got {type(si).__name__}: {si!r}")
        if not (0 <= si <= 100):
            logger.warning(f"stability_index={si} outside expected range [0, 100]")

    # window_id must be a non-empty string
    wid = payload.get("window_id")
    if wid is not None and (not isinstance(wid, str) or not wid.strip()):
        raise ValueError(f"window_id must be a non-empty string, got: {wid!r}")

    # summary_text must be a non-empty string
    st = payload.get("summary_text")
    if st is not None and isinstance(st, str) and not st.strip():
        logger.warning("summary_text is empty — payload may lack meaningful content")


# ---------------------------------------------------------------------------
# Generic payload builder
# ---------------------------------------------------------------------------
def build_payload(base_fields: dict, schema: dict) -> dict:
    """
    Build payload using schema defaults and provided fields.
    Applies defaults, validates required fields, checks types.
    """
    payload = {}

    # Apply defaults
    payload.update(schema.get("defaults", {}))

    # Apply provided fields
    payload.update(base_fields)

    # Validate required fields
    missing = [
        k for k in schema.get("required_fields", [])
        if k not in payload
    ]
    if missing:
        raise ValueError(f"Missing required payload fields: {missing}")

    # Type-check critical fields
    _validate_field_types(payload)

    return payload


# ---------------------------------------------------------------------------
# SHIFT payload builder
# ---------------------------------------------------------------------------
def build_shift_payload(
    *,
    shift_data,
    structured_summary: dict,
    llm_text: str,
    prev_shift=None,
    schema: dict,
    operator_context: dict | None = None,
):
    """
    Build schema-compliant payload for a single shift.
    (FSI-aware, backward-compatible)
    """
    anomaly_count = structured_summary.get(
        "num_anomalies",
        structured_summary.get("anomaly_count", 0)
    )

    num_parameters = structured_summary.get(
        "num_parameters",
        len(structured_summary.get("stats", {}))
    )

    generated_at = structured_summary.get(
        "generated_at",
        utc_now_iso()
    )

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
        "anomalous_parameters": list(
            structured_summary.get("signals", {}).keys()
        ),
        "anomaly_details": structured_summary.get("signals", {}),
        "generated_at": generated_at,

        "previous_window_id": prev_shift.shift_id if prev_shift else None,

        "operator_context": operator_context,
    }

    return build_payload(base_fields, schema)


# ---------------------------------------------------------------------------
# DAY payload builder
# ---------------------------------------------------------------------------
def build_day_payload(*, day_id, shift_payloads, structured_summary, llm_text, schema):
    base_fields = {
        "window_type": "day",
        "window_id": day_id,
        "start_time": shift_payloads[0]["start_time"],
        "end_time": shift_payloads[-1].get("end_time", shift_payloads[-1]["start_time"]),

        "summary_text": llm_text,

        "avg_stability_index": structured_summary.get("avg_stability_index"),
        "stability_trend": structured_summary.get("stability_trend"),

        "num_records": structured_summary.get("num_shifts"),
        "generated_at": utc_now_iso(),
    }

    return build_payload(base_fields, schema)


# ---------------------------------------------------------------------------
# WEEK payload builder
# ---------------------------------------------------------------------------
def build_week_payload(*, week_id, day_payloads, structured_summary, llm_text, schema):
    base_fields = {
        "window_type": "week",
        "window_id": week_id,
        "start_time": day_payloads[0]["start_time"],
        "end_time": day_payloads[-1]["end_time"],

        "summary_text": llm_text,

        "avg_stability_index": structured_summary.get("avg_stability_index"),
        "stability_trend": structured_summary.get("stability_trend"),

        "num_records": structured_summary.get("num_days"),
        "generated_at": utc_now_iso(),
    }

    return build_payload(base_fields, schema)


# ---------------------------------------------------------------------------
# BI-WEEK payload builder
# ---------------------------------------------------------------------------
def build_biweek_payload(*, biweek_id, week_payloads, structured_summary, llm_text, schema):
    base_fields = {
        "window_type": "biweek",
        "window_id": biweek_id,
        "start_time": week_payloads[0]["start_time"],
        "end_time": week_payloads[-1]["end_time"],

        "summary_text": llm_text,

        "avg_stability_index": structured_summary.get("avg_stability_index"),
        "stability_trend": structured_summary.get("stability_trend"),

        "num_records": structured_summary.get("num_weeks"),
        "generated_at": utc_now_iso(),
    }

    return build_payload(base_fields, schema)


# ---------------------------------------------------------------------------
# Qdrant-safe deterministic ID helper
# ---------------------------------------------------------------------------
def window_id_to_uuid(window_id: str) -> str:
    """Convert a human-readable window_id into a deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, window_id))