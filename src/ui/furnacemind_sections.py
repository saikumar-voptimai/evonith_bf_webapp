"""FurnaceMind UI — module-level helpers and re-exports.

Tab renderers live in dedicated modules:
  - AI Co-Operate  → FurnaceMind.ui.cooperate  (render_ai_cooperate)
  - Reports        → FurnaceMind.ui.reports     (render_reports)
  - Live Ops       → FurnaceMind.ui.live_ops    (render_live_operations)
  - Intelligence   → FurnaceMind.ui.live_ops    (render_furnace_intelligence)

This file retains:
  - Module-level constants  (MEASUREMENT_LABELS, FREQUENCY_TO_TIMEDTA, FIELD_LABELS, NAV_TABS)
  - Shared utilities        (_ensure_ist, _normalize_label, select_nav_tab, load_schemas)

These are imported by ``live_ops.py`` and by the FurnaceMind page directly.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from config.config_loader import load_config

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Navigation ───────────────────────────────────────────────────────────────

NAV_TABS = [
    "🤖 AI Co-Operate",
    "📊 Reports",
]

# ── InfluxDB / data-mapping constants (used by live_ops and cooperate) ───────

config = load_config("setting_ds_dv.yml")

MEASUREMENT_LABELS: dict[str, str] = {
    "heatload_delta_t":    "Heatload Delta T",
    "process_params":      "Process Params",
    "temperature_profile": "Temperature Profile",
}

FREQUENCY_TO_TIMEDTA: dict[str, str | None] = {
    "None":       None,
    "1 minute":   "1min",
    "5 minutes":  "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour":     "1h",
    "6 hours":    "6h",
    "8 hours":    "8h",
    "12 hours":   "12h",
    "1 day":      "1d",
}

FIELD_LABELS: dict[str, str] = {
    internal_key: human_label
    for mapping in config["data_mapping"].values()
    for human_label, internal_key in mapping.items()
}

# ── Shared timezone ───────────────────────────────────────────────────────────

IST = timezone(timedelta(hours=5, minutes=30))


# ── Shared utilities ──────────────────────────────────────────────────────────

def _ensure_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def _normalize_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def load_schemas() -> dict:
    """Load shift/day/week/biweek payload schema JSON files."""
    this_dir = Path(__file__).resolve().parent
    base = this_dir.parent / "config" / "schemas"
    return {
        "shift":  json.load(open(base / "shift_payload_schema.json")),
        "day":    json.load(open(base / "day_payload_schema.json")),
        "week":   json.load(open(base / "weekly_payload_schema.json")),
        "biweek": json.load(open(base / "biweekly_payload_schema.json")),
    }


def select_nav_tab() -> str:
    """Render a segmented-control (or radio fallback) for top-level navigation."""
    try:
        if hasattr(st, "segmented_control"):
            return st.segmented_control(
                "Navigation", NAV_TABS, default=NAV_TABS[0], key="furnacemind_nav",
            )
    except TypeError:
        pass
    return st.radio("Navigation", NAV_TABS, horizontal=True, index=0, key="furnacemind_nav")
