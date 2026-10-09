"""Furnace schematic/SVG generation and rendering."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import streamlit as st

from data import furnace_status as fs
from ui.furnace_status.components import h, is_linked, na_markup, zones_attr
from ui.furnace_status.layout import (
    FALLBACK_PROFILE,
    GLOW,
    HEAT_TOTAL_KEY,
    LEADER,
    OUTLINE,
    TEXT_MUTED,
    WARM,
    WARM_TEXT,
    ZONE_READINGS,
    ZONE_WIDGET_KEY,
    ZONES,
    validate_parameter_references,
)
from utils.logger import get_logger

log = get_logger(__name__)
SVG_W = 560.0
SVG_H = 532.0
PX_PER_M = 22.0
CX = SVG_W / 2
Y0 = 512.0
GUTTER = 168.0
BLAST_M = 8.0
TAP_M = 1.0
LINING_M = 0.32
SCHEMATIC_ALT = (
    "Illustrative BF2 cross-section, drawn from the configured furnace outline: "
    "stack, belly, bosh, tuyere and hearth zones, hot blast entering at the "
    "tuyeres, top gas leaving the top and hot metal and slag leaving the hearth."
)


def _sx(x_m: float) -> float:
    return CX + x_m * PX_PER_M


def _sy(y_m: float) -> float:
    return Y0 - y_m * PX_PER_M


def _pct(value: float, whole: float) -> str:
    return f"{value / whole * 100:.2f}%"


@dataclass(frozen=True)
class Callout:
    key: str
    side: Literal[-1, 1]
    y: float
    anchor: Literal["wall", "blast", "raceway"]
    elevation: float = BLAST_M


CALLOUTS: tuple[Callout, ...] = (
    Callout("top_pressure", -1, 96.0, "wall", 19.4),
    Callout("permeability", -1, 206.0, "wall", 13.6),
    Callout("hbt", -1, 284.0, "blast"),
    Callout("raft", -1, 396.0, "raceway"),
    Callout("furnace_level", 1, 96.0, "wall", 19.0),
    Callout(HEAT_TOTAL_KEY, 1, 206.0, "wall", 12.5),
    Callout("blast_pressure", 1, 284.0, "blast"),
    Callout("hearth_temp_avg", 1, 442.0, "wall", 4.4),
)
validate_parameter_references(callout.key for callout in CALLOUTS)


@lru_cache(maxsize=1)
def furnace_profile() -> tuple[tuple[float, float], ...]:
    try:
        from config.config_loader import load_config

        points = load_config("setting_ds_dv.yml")["plot"]["geometry"]["geometry_points"]
        return tuple((float(x), float(y)) for x, y in points)
    except Exception:  # noqa: BLE001 - decorative fallback
        log.warning("Furnace profile not read from config; using built-in profile.")
        return FALLBACK_PROFILE


def _half_width(profile: tuple[tuple[float, float], ...], y: float) -> float:
    points = [(abs(profile[0][0]), 0.0), *[(abs(x), yy) for x, yy in profile]]
    points = sorted((yy, xx) for xx, yy in points)
    if y <= points[0][0]:
        return points[0][1]
    for (y0, x0), (y1, x1) in zip(points, points[1:], strict=False):
        if y <= y1:
            return x0 + (x1 - x0) * ((y - y0) / (y1 - y0) if y1 > y0 else 0)
    return points[-1][1]


def _callout_anchor(
    profile: tuple[tuple[float, float], ...], callout: Callout
) -> tuple[float, float]:
    if callout.anchor == "wall":
        radius = _half_width(profile, callout.elevation)
        return _sx(callout.side * radius), _sy(callout.elevation)
    radius = _half_width(profile, BLAST_M)
    offset = 22.0 if callout.anchor == "blast" else -14.0
    return _sx(callout.side * radius) + callout.side * offset, _sy(BLAST_M)


def _arrow(
    x1: float, y1: float, x2: float, y2: float, colour: str, width: float = 2.4
) -> str:
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    bx, by = x2 - ux * 10, y2 - uy * 10
    px, py = -uy * 5, ux * 5
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
        f'stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>'
        f'<polygon points="{bx + px:.1f},{by + py:.1f} {x2:.1f},{y2:.1f} '
        f'{bx - px:.1f},{by - py:.1f}" fill="{colour}"/>'
    )


def build_furnace_svg(profile: tuple[tuple[float, float], ...]) -> str:
    left = [(profile[0][0], 0.0), *profile]
    outline = [(_sx(x), _sy(y)) for x, y in left] + [
        (_sx(-x), _sy(y)) for x, y in reversed(left)
    ]
    polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in outline)
    top = max(y for _, y in profile)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W:.0f} {SVG_H:.0f}" width="{SVG_W:.0f}" height="{SVG_H:.0f}">',
        (
            "<defs>"
            '<linearGradient id="fsBody" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#eef3f8"/>'
            '<stop offset="0.45" stop-color="#e6edf4"/>'
            '<stop offset="0.7" stop-color="#ebe6e3"/>'
            '<stop offset="0.86" stop-color="#f4d8c3"/>'
            '<stop offset="1" stop-color="#f0bf98"/></linearGradient>'
            '<radialGradient id="fsGlow">'
            f'<stop offset="0" stop-color="{GLOW}" stop-opacity="0.7"/>'
            f'<stop offset="0.45" stop-color="{WARM}" stop-opacity="0.3"/>'
            f'<stop offset="1" stop-color="{WARM}" stop-opacity="0"/></radialGradient>'
            "</defs>"
        ),
        f'<polygon points="{polygon}" fill="url(#fsBody)" stroke="{OUTLINE}" stroke-width="2" stroke-linejoin="round"/>',
    ]
    lining = [
        (max(_half_width(profile, y) - LINING_M, 0.5), y)
        for y in sorted({0.3, *(y for _, y in profile), top - 0.25})
        if 0.3 <= y <= top - 0.25
    ]
    for side in (-1, 1):
        points = " ".join(f"{_sx(side * r):.1f},{_sy(y):.1f}" for r, y in lining)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#c3d0dd" stroke-width="1" stroke-linejoin="round"/>'
        )
    for elevation in (15.9, 16.9, 17.9, 18.9):
        radius = _half_width(profile, elevation) - LINING_M - 0.1
        y = _sy(elevation)
        parts.append(
            f'<path d="M{_sx(-radius):.1f},{y:.1f} Q{CX:.1f},{y + 9:.1f} {_sx(radius):.1f},{y:.1f}" fill="none" stroke="{TEXT_MUTED}" stroke-opacity="0.22" stroke-width="1"/>'
        )
    for _, lower, _ in ZONES:
        if lower <= 0:
            continue
        radius = _half_width(profile, lower)
        parts.append(
            f'<line x1="{_sx(-radius):.1f}" y1="{_sy(lower):.1f}" x2="{_sx(radius):.1f}" y2="{_sy(lower):.1f}" stroke="#9fb3c8" stroke-width="1" stroke-dasharray="3 4"/>'
        )
    radius = _half_width(profile, BLAST_M)
    y_blast = _sy(BLAST_M)
    for side in (-1, 1):
        wall = _sx(side * radius)
        parts.append(
            f'<ellipse cx="{wall - side * 14:.1f}" cy="{y_blast:.1f}" rx="20" ry="12" fill="url(#fsGlow)"/>'
        )
        parts.append(
            _arrow(wall + side * 70, y_blast, wall + side * 1.5, y_blast, WARM)
        )
    y_tap = _sy(TAP_M)
    wall_tap = _sx(-_half_width(profile, TAP_M))
    parts.append(_arrow(wall_tap - 1.5, y_tap, wall_tap - 66, y_tap, WARM_TEXT, 2))
    parts.append(_arrow(CX, _sy(top) - 3, CX, _sy(top) - 34, TEXT_MUTED, 2))
    for callout in CALLOUTS:
        start = GUTTER + 4 if callout.side < 0 else SVG_W - GUTTER - 4
        anchor_x, anchor_y = _callout_anchor(profile, callout)
        parts.append(
            f'<line x1="{start:.1f}" y1="{callout.y:.1f}" x2="{anchor_x:.1f}" y2="{anchor_y:.1f}" stroke="{LEADER}" stroke-width="1"/>'
            f'<circle cx="{anchor_x:.1f}" cy="{anchor_y:.1f}" r="2.6" fill="{TEXT_MUTED}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


@lru_cache(maxsize=2)
def _furnace_img_src(profile: tuple[tuple[float, float], ...]) -> str:
    encoded = base64.b64encode(build_furnace_svg(profile).encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def _annotation_html(
    text: str, x: float, y: float, kind: str, align: Literal["centre", "end"]
) -> str:
    return (
        f'<span class="fs-annot fs-annot--{kind} fs-annot--{align}" aria-hidden="true" '
        f'style="left:{_pct(x, SVG_W)};top:{_pct(y, SVG_H)}">{h(text)}</span>'
    )


def _callout_html(
    reading: fs.ParameterReading, callout: Callout, *, linked: bool = False
) -> str:
    spec = reading.spec
    side = "left" if callout.side < 0 else "right"
    edge = "right" if side == "left" else "left"
    style = (
        f"{edge}:{_pct(SVG_W - GUTTER, SVG_W)};top:{_pct(callout.y, SVG_H)};"
        f"max-width:{_pct(GUTTER - 4, SVG_W)}"
    )
    unit = (
        f'<span class="fs-callout__unit">{h(spec.unit)}</span>'
        if reading.value is not None and spec.unit
        else ""
    )
    css = f"fs-callout fs-callout--{side}" + (" is-linked" if linked else "")
    return (
        f'<div class="{css}"{zones_attr(spec.key)} style="{style}">'
        f'<span class="fs-callout__label">{h(spec.label)}</span>'
        f'<span class="fs-callout__value">{na_markup(fs.format_reading(reading))}'
        f"{unit}</span></div>"
    )


def schematic_html(
    readings: dict[str, fs.ParameterReading], zone: str | None = None
) -> str:
    profile = furnace_profile()
    top = max(y for _, y in profile)
    reach = max(abs(x) for x, _ in profile) + 0.3
    parts = [
        '<div class="fs-schematic">',
        f'<img class="fs-schematic__img" src="{_furnace_img_src(profile)}" alt="{h(SCHEMATIC_ALT)}">',
    ]
    for label, lower, upper in ZONES:
        style = (
            f"left:{_pct(_sx(-reach), SVG_W)};top:{_pct(_sy(upper), SVG_H)};"
            f"width:{_pct(2 * reach * PX_PER_M, SVG_W)};"
            f"height:{_pct((upper - lower) * PX_PER_M, SVG_H)}"
        )
        selected = " is-selected" if label == zone else ""
        parts.append(
            f'<div class="fs-zone{selected}" data-zone="{label.lower()}" aria-hidden="true" style="{style}">'
            f'<span class="fs-zone__label">{h(label)}</span></div>'
        )
    blast_tail = _sx(-_half_width(profile, BLAST_M)) - 74
    tap_tip = _sx(-_half_width(profile, TAP_M)) - 72
    parts.append(_annotation_html("Top gas", CX, _sy(top) - 54, "gas", "centre"))
    parts.append(
        _annotation_html("Hot blast", blast_tail, _sy(BLAST_M), "blast", "end")
    )
    parts.append(
        _annotation_html("Hot metal & slag", tap_tip, _sy(TAP_M), "tap", "end")
    )
    for callout in CALLOUTS:
        parts.append(
            _callout_html(
                readings[callout.key],
                callout,
                linked=is_linked(callout.key, zone),
            )
        )
    parts.append("</div>")
    return "".join(parts)


def _zone_note_html(readings: dict[str, fs.ParameterReading], zone: str | None) -> str:
    if zone is None:
        linked = "Zone links group related readings; they do not mark sensor positions."
    else:
        names = ", ".join(readings[key].spec.label for key in ZONE_READINGS[zone])
        linked = f"{zone}: highlighting {names}."
    return (
        f'<div class="fs-note" role="status">{h(linked)}</div>'
        '<div class="fs-note">Illustrative schematic: shading and glow are not a '
        "measured temperature map, and callouts are related readings, not exact "
        "sensor positions.</div>"
    )


def render_furnace(
    readings: dict[str, fs.ParameterReading], zone: str | None = None
) -> None:
    with st.container(key="fs-furnace"):
        st.html(schematic_html(readings, zone))
        st.pills(
            "Highlight a zone's related readings",
            [label for label, _, _ in ZONES],
            selection_mode="single",
            key=ZONE_WIDGET_KEY,
        )
        st.html(_zone_note_html(readings, zone))
