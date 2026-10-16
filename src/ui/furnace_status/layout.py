"""Visual layout metadata and catalogue-reference validation."""

from __future__ import annotations

from collections.abc import Iterable

from data import furnace_status as fs

SECTION_SLUGS = {
    fs.SECTION_PRODUCTION: "production",
    fs.SECTION_BLAST: "blast",
    fs.SECTION_UPTAKE: "uptake",
    fs.SECTION_INJECTION: "injection",
    fs.SECTION_PERFORMANCE: "performance",
    fs.SECTION_HEAT_LOAD: "heat-load",
    fs.SECTION_HEARTH: "hearth",
}

PANEL_COLUMNS: tuple[tuple[str, ...], tuple[str, ...]] = (
    (fs.SECTION_PRODUCTION, fs.SECTION_BLAST, fs.SECTION_INJECTION),
    (fs.SECTION_PERFORMANCE, fs.SECTION_HEAT_LOAD),
)

STRIPS: dict[str, tuple[str, ...]] = {
    fs.SECTION_UPTAKE: (
        "uptake_t1",
        "uptake_t2",
        "uptake_t3",
        "uptake_t4",
        "uptake_avg",
        "uptake_std",
    ),
    fs.SECTION_HEARTH: (
        "hearth_temp_a",
        "hearth_temp_b",
        "hearth_temp_c",
        "hearth_temp_d",
    ),
}

HEAT_TOTAL_KEY = "heat_load_total"
HEAT_QUADRANT_KEYS = (
    "heat_load_q1",
    "heat_load_q2",
    "heat_load_q3",
    "heat_load_q4",
)
INSTRUCTIONS = (
    "Select a row or tile to open its trend. Hover a furnace zone, or pick one "
    "under the diagram, to highlight its related readings."
)

_COLOURS = dict(fs.UI.colours)
SURFACE = _COLOURS["surface"]
GRID = _COLOURS["grid"]
BORDER = _COLOURS["border"]
TEXT_MUTED = _COLOURS["text_muted"]
TEXT = _COLOURS["text"]
ACCENT = _COLOURS["accent"]
WARM = _COLOURS["warm"]
WARM_TEXT = _COLOURS["warm_text"]
GLOW = _COLOURS["glow"]
OUTLINE = _COLOURS["outline"]
LEADER = _COLOURS["leader"]
BODY_TOP = _COLOURS["body_top"]
BODY_UPPER = _COLOURS["body_upper"]
BODY_MID = _COLOURS["body_mid"]
BODY_LOWER = _COLOURS["body_lower"]
BODY_BOTTOM = _COLOURS["body_bottom"]
LINING = _COLOURS["lining"]
ZONE_DIVIDER = _COLOURS["zone_divider"]

FALLBACK_PROFILE: tuple[tuple[float, float], ...] = (
    (-2.8, 4.374),
    (-2.8, 6.795),
    (-3.15, 8.335),
    (-3.15, 11.29),
    (-3.65, 14.39),
    (-3.65, 15.89),
    (-2.898, 20.0),
)
ZONES = tuple((zone.name, zone.lower_m, zone.upper_m) for zone in fs.UI.zones)
ZONE_READINGS = {zone.name: zone.readings for zone in fs.UI.zones}
ZONE_WIDGET_KEY = "fs-zone"


def validate_parameter_references(extra_keys: Iterable[str] = ()) -> None:
    """Fail fast when visual metadata names a non-catalogue parameter."""
    referenced = {
        HEAT_TOTAL_KEY,
        *HEAT_QUADRANT_KEYS,
        *(key for keys in STRIPS.values() for key in keys),
        *(key for keys in ZONE_READINGS.values() for key in keys),
        *extra_keys,
    }
    unknown = sorted(referenced - fs.PARAMETERS_BY_KEY.keys())
    if unknown:
        raise ValueError(f"Furnace Status UI references unknown parameters: {unknown}")


validate_parameter_references()
