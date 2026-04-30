"""render_live_operations and render_furnace_intelligence tabs.

Both tabs depend on classes that were in src/core/ (now deleted).
The functions are stubbed until those classes are rebuilt under agents/.
"""

from __future__ import annotations

import streamlit as st

# TODO: The following classes were in src/core/ which has been deleted.
#       They need to be rebuilt or relocated before these tabs can function.
#   - InfluenceAttribution      (was core.influence_attribution)
#   - RecurringAnomalyTracker   (was core.recurring_anomaly_tracker)
#   - ShiftAnalyzer             (was core.shift_analyzer)
#   - FurnaceStabilityIndex     (was core.stability_index)


def render_live_operations(*, structured_store, vector_store) -> None:  # noqa: ARG001
    """Render the Live Operations tab (currently under reconstruction)."""
    st.info("Live Operations tab is under reconstruction.")


def render_furnace_intelligence(*, structured_store) -> None:  # noqa: ARG001
    """Render the Furnace Intelligence tab (currently under reconstruction)."""
    st.info("Furnace Intelligence tab is under reconstruction.")
