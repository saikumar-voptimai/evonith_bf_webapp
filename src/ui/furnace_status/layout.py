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

STRIPS: dict[str, tuple[tuple[str, ...], str]] = {
    fs.SECTION_UPTAKE: (
        ("uptake_t1", "uptake_t2", "uptake_t3", "uptake_t4"),
        "uptake_avg",
    ),
    fs.SECTION_HEARTH: (
        ("hearth_temp_a", "hearth_temp_b", "hearth_temp_c", "hearth_temp_d"),
        "hearth_temp_avg",
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

SURFACE = "#FFFFFF"
GRID = "#E3EAF1"
BORDER = "#DBE3EC"
TEXT_MUTED = "#64748B"
TEXT = "#0F172A"
ACCENT = "#1F6FB2"
WARM = "#EA580C"
WARM_TEXT = "#C2410C"
GLOW = "#FB923C"
OUTLINE = "#4B6584"
LEADER = "#94A3B8"

FALLBACK_PROFILE: tuple[tuple[float, float], ...] = (
    (-2.8, 4.374),
    (-2.8, 6.795),
    (-3.15, 8.335),
    (-3.15, 11.29),
    (-3.65, 14.39),
    (-3.65, 15.89),
    (-2.898, 20.0),
)
ZONES = (
    ("Stack", 15.0, 20.0),
    ("Belly", 12.9, 15.0),
    ("Bosh", 10.5, 12.9),
    ("Tuyere", 5.5, 10.5),
    ("Hearth", 0.0, 5.5),
)
ZONE_READINGS: dict[str, tuple[str, ...]] = {
    "Stack": (
        "uptake_t1",
        "uptake_t2",
        "uptake_t3",
        "uptake_t4",
        "uptake_avg",
        "top_pressure",
        "furnace_level",
        "co_utilization",
        "top_gas_h2",
    ),
    "Belly": ("permeability", HEAT_TOTAL_KEY, *HEAT_QUADRANT_KEYS),
    "Bosh": ("permeability", HEAT_TOTAL_KEY, *HEAT_QUADRANT_KEYS),
    "Tuyere": (
        "hot_blast_volume",
        "blast_pressure",
        "hbt",
        "raft",
        "tuyere_velocity",
        "pci_rate",
        "steam_injection",
        "steam_bypass_flow",
        "o2_injection",
        "oxygen_flow",
    ),
    "Hearth": (
        "hearth_temp_a",
        "hearth_temp_b",
        "hearth_temp_c",
        "hearth_temp_d",
        "hearth_temp_avg",
        "production_theoretical",
        "production_rate",
        "slag_rate",
    ),
}
ZONE_WIDGET_KEY = "fs-zone"


def validate_parameter_references(extra_keys: Iterable[str] = ()) -> None:
    """Fail fast when visual metadata names a non-catalogue parameter."""
    referenced = {
        HEAT_TOTAL_KEY,
        *HEAT_QUADRANT_KEYS,
        *(key for keys, average in STRIPS.values() for key in (*keys, average)),
        *(key for keys in ZONE_READINGS.values() for key in keys),
        *extra_keys,
    }
    unknown = sorted(referenced - fs.PARAMETERS_BY_KEY.keys())
    if unknown:
        raise ValueError(f"Furnace Status UI references unknown parameters: {unknown}")


validate_parameter_references()
