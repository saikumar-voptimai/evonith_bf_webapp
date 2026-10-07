"""Top-level segmented navigation for the V-Board page."""

from __future__ import annotations

import streamlit as st

VISUALISATIONS = "📈 Visualisations"
FURNACE_STATUS = "🔥 Furnace Status"
VBOARD_SECTIONS: tuple[str, str] = (VISUALISATIONS, FURNACE_STATUS)
VBOARD_NAV_KEY = "vboard_section_nav"


def select_vboard_section() -> str:
    """Render V-Board navigation and return the selected section label."""
    selected = st.segmented_control(
        "V-Board section",
        VBOARD_SECTIONS,
        default=VISUALISATIONS,
        key=VBOARD_NAV_KEY,
    )
    return selected or VISUALISATIONS
