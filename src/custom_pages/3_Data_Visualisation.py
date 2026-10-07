"""V-Board entry point with conditionally rendered top-level sections."""

from __future__ import annotations

from ui.vboard_sections import (
    FURNACE_STATUS,
    VISUALISATIONS,
    select_vboard_section,
)


def main() -> None:
    """Render only the selected V-Board section."""
    section = select_vboard_section()

    if section == VISUALISATIONS:
        from ui.vboard_visualisations import render_visualisations

        render_visualisations()
        return

    if section == FURNACE_STATUS:
        from ui.furnace_status_page import render_furnace_status

        render_furnace_status()


main()
