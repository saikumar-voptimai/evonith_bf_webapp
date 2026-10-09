"""Compatibility exports for the modular Furnace Status UI.

The implementation lives under :mod:`ui.furnace_status`. The responsive module
contains the visible "Rotate your device for the best view." hint.
"""

from ui.furnace_status.components import button_label as _button_label
from ui.furnace_status.components import row_html as _row_html
from ui.furnace_status.layout import FALLBACK_PROFILE as _FALLBACK_PROFILE
from ui.furnace_status.layout import ZONES as _ZONES
from ui.furnace_status.page import render_furnace_status
from ui.furnace_status.responsive import ORIENTATION_HTML as _ORIENTATION_HTML
from ui.furnace_status.schematic import CALLOUTS as _CALLOUTS
from ui.furnace_status.schematic import build_furnace_svg
from ui.furnace_status.schematic import furnace_profile as _furnace_profile
from ui.furnace_status.schematic import schematic_html as _schematic_html
from ui.furnace_status.trends import PLOT_CONFIG as _PLOT_CONFIG
from ui.furnace_status.trends import build_trend_figure
from ui.furnace_status.trends import naive_ist as _naive_ist

__all__ = ["build_furnace_svg", "build_trend_figure", "render_furnace_status"]
