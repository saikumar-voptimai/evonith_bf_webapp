"""Render the V-Board Furnace Status section and its parameter trends.

Value resolution, formatting and grouped fetching live in
:mod:`data.furnace_status`. This module turns those results into Streamlit
elements and HTML.

Nested views use validated, V-Board-namespaced query parameters.
Each parameter row or temperature tile is a normal Streamlit button laid
invisibly over its visible HTML, so a click updates ``st.query_params`` and
reruns without a full browser reload (which would drop the session) and the
whole row/tile is the target.

The status board has three columns: panels | uptake strip, furnace schematic,
hearth strip | panels.  Everything on it comes from one
:class:`~data.furnace_status.StatusSnapshot` per render; the strips, the
schematic callouts and the heat-load ring only re-present those readings.
"""

from __future__ import annotations

import base64
import html
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
import streamlit as st

from data import furnace_status as fs
from utils.logger import get_logger

log = get_logger(__name__)

_CSS_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "css" / "furnace_status.css"
)

_SECTION_SLUGS = {
    fs.SECTION_PRODUCTION: "production",
    fs.SECTION_BLAST: "blast",
    fs.SECTION_UPTAKE: "uptake",
    fs.SECTION_INJECTION: "injection",
    fs.SECTION_PERFORMANCE: "performance",
    fs.SECTION_HEAT_LOAD: "heat-load",
    fs.SECTION_HEARTH: "hearth",
}

# Panels in the left and right board columns, top to bottom.  The centre
# column is the uptake strip, the furnace schematic and the hearth strip.
_PANEL_COLUMNS: tuple[tuple[str, ...], tuple[str, ...]] = (
    (fs.SECTION_PRODUCTION, fs.SECTION_BLAST, fs.SECTION_INJECTION),
    (fs.SECTION_PERFORMANCE, fs.SECTION_HEAT_LOAD),
)

#: Centre-column temperature strips: four readings, then the average the data
#: layer supplies (shown as given, never recomputed from the four).
_STRIPS: dict[str, tuple[tuple[str, ...], str]] = {
    fs.SECTION_UPTAKE: (
        ("uptake_t1", "uptake_t2", "uptake_t3", "uptake_t4"),
        "uptake_avg",
    ),
    fs.SECTION_HEARTH: (
        ("hearth_temp_a", "hearth_temp_b", "hearth_temp_c", "hearth_temp_d"),
        "hearth_temp_avg",
    ),
}

_HEAT_TOTAL_KEY = "heat_load_total"
_HEAT_QUADRANT_KEYS = ("heat_load_q1", "heat_load_q2", "heat_load_q3", "heat_load_q4")

_INSTRUCTIONS = (
    "Select a row or tile to open its trend. Hover a furnace zone, or pick one "
    "under the diagram, to highlight its related readings."
)

# Page palette (the app's light theme), for the places CSS cannot reach (SVG
# image, Plotly figure).  Keep in step with the custom properties in
# furnace_status.css.
_SURFACE = "#FFFFFF"
_GRID = "#E3EAF1"
_BORDER = "#DBE3EC"
_TEXT_MUTED = "#64748B"
_TEXT = "#0F172A"
_ACCENT = "#1F6FB2"
_WARM = "#EA580C"
_WARM_TEXT = "#C2410C"
_GLOW = "#FB923C"
_OUTLINE = "#4B6584"
_LEADER = "#94A3B8"

_PLOT_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "zoomIn2d",
        "zoomOut2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}

# Fallback if the shared config cannot be read (same numbers as setting_ds_dv.yml).
_FALLBACK_PROFILE: tuple[tuple[float, float], ...] = (
    (-2.8, 4.374),
    (-2.8, 6.795),
    (-3.15, 8.335),
    (-3.15, 11.29),
    (-3.65, 14.39),
    (-3.65, 15.89),
    (-2.898, 20.0),
)
# (label, bottom m, top m) — zone table from the furnace documentation.
_ZONES = (
    ("Stack", 15.0, 20.0),
    ("Belly", 12.9, 15.0),
    ("Bosh", 10.5, 12.9),
    ("Tuyere", 5.5, 10.5),
    ("Hearth", 0.0, 5.5),
)

#: Readings highlighted with each zone.  A grouping of related readings, not a
#: statement of where a sensor sits: several of these are furnace-wide values.
_ZONE_READINGS: dict[str, tuple[str, ...]] = {
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
    "Belly": ("permeability", _HEAT_TOTAL_KEY, *_HEAT_QUADRANT_KEYS),
    "Bosh": ("permeability", _HEAT_TOTAL_KEY, *_HEAT_QUADRANT_KEYS),
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

_ZONE_WIDGET_KEY = "fs-zone"


def _reading_zones() -> dict[str, tuple[str, ...]]:
    zones: dict[str, list[str]] = {}
    for zone, keys in _ZONE_READINGS.items():
        for key in keys:
            zones.setdefault(key, []).append(zone)
    return {key: tuple(names) for key, names in zones.items()}


_READING_ZONES = _reading_zones()


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════


def _h(text: object) -> str:
    return html.escape(str(text), quote=True)


@lru_cache(maxsize=4)
def _read_css(path: str, mtime_ns: int) -> str:
    return Path(path).read_text(encoding="utf-8")


def _inject_css() -> None:
    """Inject the page stylesheet (st.markdown; st.html is sandboxed)."""
    try:
        css = _read_css(str(_CSS_PATH), _CSS_PATH.stat().st_mtime_ns)
    except OSError:
        log.warning("Furnace Status stylesheet missing: %s", _CSS_PATH)
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _embed_html(markup: str, height: int) -> None:
    """Embed an HTML document in an iframe.

    ``st.components.v1.html`` is deprecated in favour of ``st.iframe`` from
    Streamlit 1.5x (and shows a visible warning by default); use the new API when
    present and fall back to the old one on earlier versions.
    """
    iframe = getattr(st, "iframe", None)
    if callable(iframe):
        iframe(markup, height=height)
        return
    import streamlit.components.v1 as components

    components.html(markup, height=height)


def _zones_attr(key: str) -> str:
    """``data-zones`` attribute naming the zones a reading is grouped with."""
    zones = _READING_ZONES.get(key)
    if not zones:
        return ""
    return f' data-zones="{" ".join(z.lower() for z in zones)}"'


def _is_linked(key: str, zone: str | None) -> bool:
    return zone is not None and key in _ZONE_READINGS.get(zone, ())


def _selected_zone() -> str | None:
    """Zone chosen in the selector (its value is known before it is drawn)."""
    value = st.session_state.get(_ZONE_WIDGET_KEY)
    return value if isinstance(value, str) and value in _ZONE_READINGS else None


# ── Navigation callbacks ─────────────────────────────────────────────────────


def _open_trend(key: str) -> None:
    _set_view_query(fs.VIEW_TREND, key)


def _go_status() -> None:
    _set_view_query(fs.VIEW_STATUS)


def _refresh() -> None:
    fs.clear_cache()


def _set_view_query(view: str, parameter: str | None = None) -> None:
    """Update only Furnace Status query keys and preserve unrelated parameters."""
    params = st.query_params.to_dict()
    params.pop(fs.VIEW_QUERY_KEY, None)
    params.pop(fs.PARAMETER_QUERY_KEY, None)
    params[fs.VIEW_QUERY_KEY] = view
    if parameter is not None:
        params[fs.PARAMETER_QUERY_KEY] = parameter
    st.query_params.from_dict(params)


# ══════════════════════════════════════════════════════════════════════════════
# Orientation / fullscreen support
# ══════════════════════════════════════════════════════════════════════════════

# Runs inside a small iframe.  Everything is wrapped in try/catch because every
# step can be refused: orientation lock usually needs fullscreen plus a user
# gesture, and many browsers (notably iOS Safari) support neither.  It never
# touches Streamlit, so nothing reruns on rotation; the portrait hint is shown by
# CSS media queries alone.
_ORIENTATION_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;padding:0;background:transparent;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.row{display:flex;align-items:center;gap:8px;min-height:44px}
button{min-height:40px;padding:0 14px;border:1px solid #3f6a96;background:#eaf2fb;color:#17324d;
border-radius:6px;font-weight:600;font-size:14px;font-family:inherit;cursor:pointer;white-space:nowrap}
button:hover{background:#dbeaf8}
button:focus-visible{outline:2px solid #1f6fb2;outline-offset:2px}
#note{font-size:12px;color:#5b6b7c;line-height:1.25}
.short{display:none}
@media (max-width:230px){.long{display:none}.short{display:inline}button{padding:0 10px}}
</style></head><body>
<div class="row"><button id="go" type="button" aria-label="Open landscape fullscreen" title="Open landscape / fullscreen"><span aria-hidden="true">&#x26F6;</span> <span class="long" id="lbl-long"></span><span class="short" id="lbl-short"></span></button><span id="note" role="status"></span></div>
<script>
(function () {
  "use strict";
  var host = window;
  try { if (window.parent && window.parent.document) { host = window.parent; } } catch (e) { host = window; }
  var btn = document.getElementById("go");
  var note = document.getElementById("note");

  function safe(fn, fallback) { try { return fn(); } catch (e) { return fallback; } }
  function isSmallScreen() {
    return safe(function () { return host.matchMedia("(pointer: coarse), (max-width: 900px)").matches; }, false);
  }
  function orientation() {
    return safe(function () { return host.screen.orientation; }, null) ||
           safe(function () { return window.screen.orientation; }, null);
  }
  function isFullscreen() {
    return safe(function () {
      return !!(host.document.fullscreenElement || host.document.webkitFullscreenElement);
    }, false);
  }
  function lockLandscape() {
    return new Promise(function (resolve, reject) {
      try {
        var o = orientation();
        if (!o || typeof o.lock !== "function") { reject(new Error("unsupported")); return; }
        Promise.resolve(o.lock("landscape")).then(resolve, reject);
      } catch (e) { reject(e); }
    });
  }
  function enterFullscreen() {
    return new Promise(function (resolve, reject) {
      try {
        var el = host.document.documentElement;
        var req = el.requestFullscreen || el.webkitRequestFullscreen;
        if (!req) { reject(new Error("unsupported")); return; }
        Promise.resolve(req.call(el)).then(resolve, reject);
      } catch (e) { reject(e); }
    });
  }
  function exitFullscreen() {
    safe(function () {
      var d = host.document;
      (d.exitFullscreen || d.webkitExitFullscreen).call(d);
    });
    safe(function () { orientation().unlock(); });
  }
  function refreshLabel() {
    var on = isFullscreen();
    document.getElementById("lbl-long").textContent = on ? "Exit fullscreen" : "Open landscape / fullscreen";
    document.getElementById("lbl-short").textContent = on ? "Exit" : "Fullscreen";
  }

  btn.addEventListener("click", function () {
    note.textContent = "";
    if (isFullscreen()) { exitFullscreen(); return; }
    enterFullscreen()
      .catch(function () { /* still try the lock below */ })
      .then(lockLandscape)
      .catch(function () {
        note.textContent = "Your browser did not allow this. Rotate your device manually.";
      });
  });
  safe(function () { host.document.addEventListener("fullscreenchange", refreshLabel); });
  safe(function () { host.document.addEventListener("webkitfullscreenchange", refreshLabel); });
  refreshLabel();

  // Best-effort automatic attempt; browsers usually refuse outside fullscreen,
  // in which case the page's portrait hint (pure CSS) stays visible.
  if (isSmallScreen()) { lockLandscape().catch(function () {}); }
})();
</script></body></html>"""


def _render_rotate_hint() -> None:
    """Portrait-only hint; shown/hidden purely by CSS media queries."""
    with st.container(key="fs-rotate"):
        st.html(
            '<div class="fs-rotate-hint" role="status">'
            '<span aria-hidden="true">⟳</span>'
            "<span>Rotate your device for the best view.</span></div>"
        )


def _render_fullscreen_control() -> None:
    """Landscape/fullscreen button + automatic lock attempt (touch/small screens)."""
    with st.container(key="fs-fullscreen"):
        _embed_html(_ORIENTATION_HTML, height=48)


# ══════════════════════════════════════════════════════════════════════════════
# Furnace schematic
# ══════════════════════════════════════════════════════════════════════════════
#
# The drawing is an SVG <img> (st.html strips inline <svg>) holding geometry
# only.  Text — zone labels, annotations and reading callouts — is HTML laid
# over it at percentage positions of the same coordinate system, so it stays
# aligned at any width, keeps a readable minimum size and can be highlighted by
# page CSS.

_SVG_W = 560.0
_SVG_H = 532.0
_PX_PER_M = 22.0  # SVG units per metre of elevation / radius
_CX = _SVG_W / 2
_Y0 = 512.0  # SVG y of elevation 0 m
#: Width of the callout column on each side of the furnace (SVG units).
_GUTTER = 168.0
_BLAST_M = 8.0  # hot-blast arrows: illustrative tuyere level
_TAP_M = 1.0  # hot metal & slag arrow: illustrative tap hole
_LINING_M = 0.32  # inner lining contour, inset from the shell

_SCHEMATIC_ALT = (
    "Illustrative BF2 cross-section, drawn from the configured furnace outline: "
    "stack, belly, bosh, tuyere and hearth zones, hot blast entering at the "
    "tuyeres, top gas leaving the top and hot metal and slag leaving the hearth."
)


def _sx(x_m: float) -> float:
    return _CX + x_m * _PX_PER_M


def _sy(y_m: float) -> float:
    return _Y0 - y_m * _PX_PER_M


def _pct(value: float, whole: float) -> str:
    return f"{value / whole * 100:.2f}%"


@dataclass(frozen=True)
class _Callout:
    """A reading shown beside the schematic, with a leader to a related spot."""

    key: str
    side: Literal[-1, 1]  # -1 left column, 1 right column
    y: float  # vertical centre of the callout (SVG units)
    anchor: Literal["wall", "blast", "raceway"]
    elevation: float = _BLAST_M  # metres; used by "wall" anchors


#: Related readings around the furnace.  Leaders point at the part of the
#: drawing the reading describes; they are not sensor positions.
_CALLOUTS: tuple[_Callout, ...] = (
    _Callout("top_pressure", -1, 96.0, "wall", 19.4),
    _Callout("permeability", -1, 206.0, "wall", 13.6),
    _Callout("hbt", -1, 284.0, "blast"),
    _Callout("raft", -1, 396.0, "raceway"),
    _Callout("furnace_level", 1, 96.0, "wall", 19.0),
    _Callout(_HEAT_TOTAL_KEY, 1, 206.0, "wall", 12.5),
    _Callout("blast_pressure", 1, 284.0, "blast"),
    _Callout("hearth_temp_avg", 1, 442.0, "wall", 4.4),
)


@lru_cache(maxsize=1)
def _furnace_profile() -> tuple[tuple[float, float], ...]:
    try:
        from config.config_loader import load_config

        points = load_config("setting_ds_dv.yml")["plot"]["geometry"]["geometry_points"]
        return tuple((float(x), float(y)) for x, y in points)
    except Exception:  # noqa: BLE001 - decorative; fall back to the known profile
        log.warning("Furnace profile not read from config; using built-in profile.")
        return _FALLBACK_PROFILE


def _half_width(profile: tuple[tuple[float, float], ...], y: float) -> float:
    """Wall radius at elevation ``y`` by linear interpolation of the profile."""
    pts = [(abs(profile[0][0]), 0.0), *[(abs(x), yy) for x, yy in profile]]
    pts = sorted((yy, xx) for xx, yy in pts)
    if y <= pts[0][0]:
        return pts[0][1]
    for (y0, x0), (y1, x1) in zip(pts, pts[1:], strict=False):
        if y <= y1:
            return x0 + (x1 - x0) * ((y - y0) / (y1 - y0) if y1 > y0 else 0.0)
    return pts[-1][1]


def _callout_anchor(
    profile: tuple[tuple[float, float], ...], callout: _Callout
) -> tuple[float, float]:
    """SVG point a callout's leader line ends on."""
    side = callout.side
    if callout.anchor == "wall":
        r = _half_width(profile, callout.elevation)
        return _sx(side * r), _sy(callout.elevation)
    r = _half_width(profile, _BLAST_M)
    offset = 22.0 if callout.anchor == "blast" else -14.0  # on the arrow / inside
    return _sx(side * r) + side * offset, _sy(_BLAST_M)


def _arrow(
    x1: float, y1: float, x2: float, y2: float, colour: str, width: float = 2.4
) -> str:
    """Line from (x1, y1) with an arrowhead pointing at (x2, y2)."""
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    bx, by = x2 - ux * 10, y2 - uy * 10  # arrowhead base
    px, py = -uy * 5, ux * 5
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
        f'stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>'
        f'<polygon points="{bx + px:.1f},{by + py:.1f} {x2:.1f},{y2:.1f} '
        f'{bx - px:.1f},{by - py:.1f}" fill="{colour}"/>'
    )


def build_furnace_svg(profile: tuple[tuple[float, float], ...]) -> str:
    """Return the furnace cross-section (geometry only) as an SVG document.

    Drawn from the configured outline and zone table.  Not to scale for any
    process value: the cool-to-warm fill and the raceway glow are illustrative,
    not a measured temperature map.  Text is laid over it as HTML
    (:func:`_schematic_html`).
    """
    left = [(profile[0][0], 0.0), *profile]
    outline = [(_sx(x), _sy(y)) for x, y in left] + [
        (_sx(-x), _sy(y)) for x, y in reversed(left)
    ]
    polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in outline)
    top = max(y for _, y in profile)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W:.0f} {_SVG_H:.0f}" '
            f'width="{_SVG_W:.0f}" height="{_SVG_H:.0f}">'
        ),
        (
            "<defs>"
            '<linearGradient id="fsBody" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#eef3f8"/>'
            '<stop offset="0.45" stop-color="#e6edf4"/>'
            '<stop offset="0.7" stop-color="#ebe6e3"/>'
            '<stop offset="0.86" stop-color="#f4d8c3"/>'
            '<stop offset="1" stop-color="#f0bf98"/></linearGradient>'
            '<radialGradient id="fsGlow">'
            f'<stop offset="0" stop-color="{_GLOW}" stop-opacity="0.7"/>'
            f'<stop offset="0.45" stop-color="{_WARM}" stop-opacity="0.3"/>'
            f'<stop offset="1" stop-color="{_WARM}" stop-opacity="0"/></radialGradient>'
            "</defs>"
        ),
        (
            f'<polygon points="{polygon}" fill="url(#fsBody)" stroke="{_OUTLINE}" '
            'stroke-width="2" stroke-linejoin="round"/>'
        ),
    ]

    # Inner lining contour, one polyline per side.
    lining = [
        (max(_half_width(profile, y) - _LINING_M, 0.5), y)
        for y in sorted({0.3, *(y for _, y in profile), top - 0.25})
        if 0.3 <= y <= top - 0.25
    ]
    for side in (-1, 1):
        points = " ".join(f"{_sx(side * r):.1f},{_sy(y):.1f}" for r, y in lining)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#c3d0dd" '
            'stroke-width="1" stroke-linejoin="round"/>'
        )

    # Burden layers in the stack (illustrative contour lines).
    for elevation in (15.9, 16.9, 17.9, 18.9):
        r = _half_width(profile, elevation) - _LINING_M - 0.1
        y = _sy(elevation)
        parts.append(
            f'<path d="M{_sx(-r):.1f},{y:.1f} Q{_CX:.1f},{y + 9:.1f} {_sx(r):.1f},{y:.1f}" '
            f'fill="none" stroke="{_TEXT_MUTED}" stroke-opacity="0.22" stroke-width="1"/>'
        )

    # Zone boundaries, inside the shell only.
    for _, lo, _hi in _ZONES:
        if lo <= 0:
            continue
        r = _half_width(profile, lo)
        parts.append(
            f'<line x1="{_sx(-r):.1f}" y1="{_sy(lo):.1f}" x2="{_sx(r):.1f}" '
            f'y2="{_sy(lo):.1f}" stroke="#9fb3c8" stroke-width="1" '
            'stroke-dasharray="3 4"/>'
        )

    # Raceway glow and hot-blast arrows at the tuyeres.
    r_blast = _half_width(profile, _BLAST_M)
    y_blast = _sy(_BLAST_M)
    for side in (-1, 1):
        wall = _sx(side * r_blast)
        parts.append(
            f'<ellipse cx="{wall - side * 14:.1f}" cy="{y_blast:.1f}" rx="20" ry="12" '
            'fill="url(#fsGlow)"/>'
        )
        parts.append(
            _arrow(wall + side * 70, y_blast, wall + side * 1.5, y_blast, _WARM)
        )

    # Hot metal & slag out of the hearth; top gas out of the top.
    y_tap = _sy(_TAP_M)
    wall_tap = _sx(-_half_width(profile, _TAP_M))
    parts.append(_arrow(wall_tap - 1.5, y_tap, wall_tap - 66, y_tap, _WARM_TEXT, 2.0))
    parts.append(_arrow(_CX, _sy(top) - 3, _CX, _sy(top) - 34, _TEXT_MUTED, 2.0))

    # Leader lines from each callout to the related part of the drawing.
    for callout in _CALLOUTS:
        start = _GUTTER + 4 if callout.side < 0 else _SVG_W - _GUTTER - 4
        ax, ay = _callout_anchor(profile, callout)
        parts.append(
            f'<line x1="{start:.1f}" y1="{callout.y:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
            f'stroke="{_LEADER}" stroke-width="1"/>'
            f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="2.6" fill="{_TEXT_MUTED}"/>'
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
        f'style="left:{_pct(x, _SVG_W)};top:{_pct(y, _SVG_H)}">{_h(text)}</span>'
    )


def _callout_html(
    reading: fs.ParameterReading, callout: _Callout, *, linked: bool = False
) -> str:
    spec = reading.spec
    side = "left" if callout.side < 0 else "right"
    edge = "right" if side == "left" else "left"
    style = (
        f"{edge}:{_pct(_SVG_W - _GUTTER, _SVG_W)};top:{_pct(callout.y, _SVG_H)};"
        f"max-width:{_pct(_GUTTER - 4, _SVG_W)}"
    )
    unit = (
        f'<span class="fs-callout__unit">{_h(spec.unit)}</span>'
        if reading.value is not None and spec.unit
        else ""
    )
    css = f"fs-callout fs-callout--{side}" + (" is-linked" if linked else "")
    return (
        f'<div class="{css}"{_zones_attr(spec.key)} style="{style}">'
        f'<span class="fs-callout__label">{_h(spec.label)}</span>'
        f'<span class="fs-callout__value">{_na_markup(fs.format_reading(reading))}'
        f"{unit}</span></div>"
    )


def _schematic_html(
    readings: dict[str, fs.ParameterReading], zone: str | None = None
) -> str:
    """Schematic image plus its HTML overlay (zones, annotations, callouts)."""
    profile = _furnace_profile()
    top = max(y for _, y in profile)
    reach = max(abs(x) for x, _ in profile) + 0.3  # half-width of the zone bands
    parts = [
        '<div class="fs-schematic">',
        (
            f'<img class="fs-schematic__img" src="{_furnace_img_src(profile)}" '
            f'alt="{_h(_SCHEMATIC_ALT)}">'
        ),
    ]
    for label, lo, hi in _ZONES:
        style = (
            f"left:{_pct(_sx(-reach), _SVG_W)};top:{_pct(_sy(hi), _SVG_H)};"
            f"width:{_pct(2 * reach * _PX_PER_M, _SVG_W)};"
            f"height:{_pct((hi - lo) * _PX_PER_M, _SVG_H)}"
        )
        selected = " is-selected" if label == zone else ""
        parts.append(
            f'<div class="fs-zone{selected}" data-zone="{label.lower()}" '
            f'aria-hidden="true" style="{style}">'
            f'<span class="fs-zone__label">{_h(label)}</span></div>'
        )

    blast_tail = _sx(-_half_width(profile, _BLAST_M)) - 74
    tap_tip = _sx(-_half_width(profile, _TAP_M)) - 72
    parts.append(_annotation_html("Top gas", _CX, _sy(top) - 54, "gas", "centre"))
    parts.append(
        _annotation_html("Hot blast", blast_tail, _sy(_BLAST_M), "blast", "end")
    )
    parts.append(
        _annotation_html("Hot metal & slag", tap_tip, _sy(_TAP_M), "tap", "end")
    )

    for callout in _CALLOUTS:
        parts.append(
            _callout_html(
                readings[callout.key],
                callout,
                linked=_is_linked(callout.key, zone),
            )
        )
    parts.append("</div>")
    return "".join(parts)


def _zone_note_html(readings: dict[str, fs.ParameterReading], zone: str | None) -> str:
    if zone is None:
        linked = "Zone links group related readings; they do not mark sensor positions."
    else:
        names = ", ".join(readings[k].spec.label for k in _ZONE_READINGS[zone])
        linked = f"{zone}: highlighting {names}."
    return (
        f'<div class="fs-note" role="status">{_h(linked)}</div>'
        '<div class="fs-note">Illustrative schematic: shading and glow are not a '
        "measured temperature map, and callouts are related readings, not exact "
        "sensor positions.</div>"
    )


def _render_furnace(
    readings: dict[str, fs.ParameterReading], zone: str | None = None
) -> None:
    """Schematic, the accessible zone selector and the clarifying notes."""
    with st.container(key="fs-furnace"):
        st.html(_schematic_html(readings, zone))
        st.pills(
            "Highlight a zone's related readings",
            [label for label, _, _ in _ZONES],
            selection_mode="single",
            key=_ZONE_WIDGET_KEY,
        )
        st.html(_zone_note_html(readings, zone))


# ══════════════════════════════════════════════════════════════════════════════
# Temperature strips and heat-load distribution (pure helpers)
# ══════════════════════════════════════════════════════════════════════════════


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


@dataclass(frozen=True)
class StripScale:
    """Shared marker scale of one temperature strip: lowest → highest shown.

    A relative visual aid only; it carries no operating limits.
    """

    low: float
    high: float

    @property
    def flat(self) -> bool:
        return math.isclose(self.low, self.high, rel_tol=1e-9, abs_tol=1e-9)

    def position(self, value: float) -> float:
        """Percent along the track; the middle when every reading is equal."""
        if self.flat:
            return 50.0
        return (value - self.low) / (self.high - self.low) * 100.0


def strip_scale(values: Sequence[float | None]) -> StripScale | None:
    """Scale spanning the finite values, or ``None`` with fewer than two."""
    finite = [v for v in values if _finite(v)]
    if len(finite) < 2:
        return None
    return StripScale(min(finite), max(finite))


def strip_spread(values: Sequence[float | None]) -> float | None:
    """Highest minus lowest of the four readings; ``None`` unless all four exist."""
    if len(values) != 4 or not all(_finite(v) for v in values):
        return None
    return max(values) - min(values)  # type: ignore[type-var]


@dataclass(frozen=True)
class QuadrantShares:
    """Heat-load share per quadrant, from unrounded Q1–Q4 readings."""

    #: Fractions in Q1..Q4 order (sum 1), or ``None`` when undefined.
    shares: tuple[float, ...] | None
    #: 0-based indices of the largest quadrant(s); several on a tie.
    highest: tuple[int, ...] = ()
    #: Why shares are not shown ("" when they are).
    note: str = ""


def quadrant_shares(values: Sequence[float | None]) -> QuadrantShares:
    """Shares and highest quadrant(s), only from a complete, valid set.

    Undefined (with a reason) when any quadrant is missing or negative, or when
    all are zero; an incomplete subset is never normalised into a distribution.
    """
    if len(values) != 4 or not all(_finite(v) for v in values):
        return QuadrantShares(None, note="Shares need all four quadrant readings.")
    loads = [float(v) for v in values]  # type: ignore[arg-type]
    if any(v < 0 for v in loads):
        return QuadrantShares(
            None, note="Shares not shown: a quadrant reading is negative."
        )
    total = sum(loads)
    if total <= 0:
        return QuadrantShares(
            None, note="All four quadrants read zero, so shares are undefined."
        )
    top = max(loads)
    highest = tuple(
        i
        for i, v in enumerate(loads)
        if math.isclose(v, top, rel_tol=1e-9, abs_tol=1e-12)
    )
    return QuadrantShares(tuple(v / total for v in loads), highest)


def _peak_quadrants(shares: QuadrantShares) -> frozenset[int]:
    """Quadrants to accent: the highest, unless all four are equal."""
    if shares.shares is None or len(shares.highest) == len(shares.shares):
        return frozenset()
    return frozenset(shares.highest)


def _share_pct(share: float) -> str:
    return f"{share * 100:.1f}%"


def _share_summary(shares: QuadrantShares) -> str:
    if shares.shares is None:
        return shares.note
    pct = _share_pct(shares.shares[shares.highest[0]])
    names = [f"Q{i + 1}" for i in shares.highest]
    if len(names) == len(shares.shares):
        return f"All quadrants equal ({pct} each)"
    if len(names) > 1:
        return f"Highest share: {', '.join(names)} tied ({pct} each)"
    return f"Highest share: {names[0]} ({pct})"


# ══════════════════════════════════════════════════════════════════════════════
# Status view
# ══════════════════════════════════════════════════════════════════════════════


def _na_markup(text: str) -> str:
    """Escape ``text`` and mute every 'Not available' inside it."""
    return _h(text).replace(
        fs.NOT_AVAILABLE, f'<span class="fs-na">{fs.NOT_AVAILABLE}</span>'
    )


def _signed(value: float, decimals: int) -> str:
    """Formatted difference with an explicit sign (zero stays unsigned)."""
    text = fs.format_value(value, decimals)
    if text.startswith("-"):
        return "−" + text[1:]
    return text if fs.format_value(0.0, decimals) == text else f"+{text}"


def _dual_value_html(reading: fs.ParameterReading) -> str:
    """Actual first, setpoint (and actual − setpoint when both exist) beneath."""
    spec = reading.spec
    actual = _na_markup(fs.format_value(reading.value, spec.decimals))
    setpoint = _na_markup(fs.format_value(reading.setpoint, spec.decimals))
    sub = f"SP {setpoint}"
    if reading.value is None and spec.unit:
        sub += f" {_h(spec.unit)}"  # no unit column next to a missing actual
    if reading.value is not None and reading.setpoint is not None:
        delta = _signed(reading.value - reading.setpoint, spec.decimals)
        sub += f' · <span title="Actual minus setpoint">Δ {_h(delta)}</span>'
    return (
        f'<span class="fs-row__main">{actual}</span>'
        f'<span class="fs-row__sub">{sub}</span>'
    )


def _row_html(
    reading: fs.ParameterReading, *, linked: bool = False, peak: bool = False
) -> str:
    spec = reading.spec
    available = fs.has_display_value(reading)
    classes = ["fs-row"]
    if not available:
        classes.append("fs-row--na")
    if linked:
        classes.append("is-linked")
    value_class = "fs-row__value"
    if spec.setpoint_field and available:
        value = _dual_value_html(reading)
        value_class += " fs-row__value--dual"
        show_unit = reading.value is not None
    else:
        value = _na_markup(fs.format_reading(reading))
        show_unit = available
    unit = _h(spec.unit) if show_unit and spec.unit else ""
    tag = (
        '<span class="fs-row__tag" title="Highest share of the total"></span>'
        if peak
        else ""
    )
    # aria-hidden: the overlay button carries the accessible name + value.
    return (
        f'<div class="{" ".join(classes)}"{_zones_attr(spec.key)} aria-hidden="true">'
        f'<span class="fs-row__label">{_h(spec.label)}</span>{tag}'
        f'<span class="{value_class}">{value}</span>'
        f'<span class="fs-row__unit">{unit}</span>'
        '<span class="fs-row__icon"></span></div>'
    )


def _button_label(reading: fs.ParameterReading, detail: str = "") -> str:
    spec = reading.spec
    text = fs.format_reading(reading)
    unit = f" {spec.unit}" if fs.has_display_value(reading) and spec.unit else ""
    extra = f" {detail}" if detail else ""
    return f"{spec.label}: {text}{unit}.{extra} Open trend"


def _render_clickable(
    reading: fs.ParameterReading,
    markup: str,
    *,
    kind: Literal["row", "tile"] = "row",
    detail: str = "",
) -> None:
    """Visible HTML with a transparent full-size button over it."""
    key = reading.spec.key
    with st.container(key=f"fs-{kind}-{key}"):
        st.html(markup)
        st.button(
            _button_label(reading, detail),
            key=f"fs-btn-{key}",
            on_click=_open_trend,
            args=(key,),
        )


def _render_row(
    reading: fs.ParameterReading, *, zone: str | None = None, peak: bool = False
) -> None:
    linked = _is_linked(reading.spec.key, zone)
    _render_clickable(reading, _row_html(reading, linked=linked, peak=peak))


def _panel_title_html(section: str, meta: str = "") -> str:
    return (
        f'<div class="fs-panel__title"><span class="fs-panel__name">{_h(section)}</span>'
        f"{meta}</div>"
    )


def _render_panel(
    section: str,
    readings: dict[str, fs.ParameterReading],
    zone: str | None = None,
) -> None:
    with st.container(key=f"fs-panel-{_SECTION_SLUGS[section]}"):
        st.html(_panel_title_html(section))
        for spec in fs.parameters_in_section(section):
            _render_row(readings[spec.key], zone=zone)


# ── Temperature strips ───────────────────────────────────────────────────────


def _tile_track_html(
    value: float | None,
    scale: StripScale | None,
    average: float | None,
    *,
    is_average: bool,
) -> str:
    if scale is None or not _finite(value):  # nothing to place on this tile
        return '<span class="fs-tile__track fs-tile__track--off"></span>'
    marks = []
    avg_pos = scale.position(average) if _finite(average) else None
    if _finite(value) and not is_average:
        pos = scale.position(value)  # type: ignore[arg-type]
        if avg_pos is not None and not scale.flat:
            lo, hi = sorted((avg_pos, pos))
            marks.append(
                f'<span class="fs-tile__bar" style="left:{lo:.1f}%;width:{hi - lo:.1f}%"></span>'
            )
        marks.append(f'<span class="fs-tile__dot" style="left:{pos:.1f}%"></span>')
    if avg_pos is not None:
        marks.append(f'<span class="fs-tile__avg" style="left:{avg_pos:.1f}%"></span>')
    return f'<span class="fs-tile__track">{"".join(marks)}</span>'


def _tile_html(
    reading: fs.ParameterReading,
    scale: StripScale | None,
    average: float | None,
    *,
    is_average: bool = False,
    linked: bool = False,
) -> str:
    spec = reading.spec
    classes = ["fs-tile"]
    if is_average:
        classes.append("fs-tile--avg")
    if reading.value is None:
        classes.append("fs-tile--na")
    if linked:
        classes.append("is-linked")
    unit = (
        f'<span class="fs-tile__unit">{_h(spec.unit)}</span>'
        if reading.value is not None and spec.unit
        else ""
    )
    track = _tile_track_html(reading.value, scale, average, is_average=is_average)
    return (
        f'<div class="{" ".join(classes)}"{_zones_attr(spec.key)} aria-hidden="true">'
        f'<span class="fs-tile__label">{_h(spec.label)}</span>'
        f'<span class="fs-tile__value">{_na_markup(fs.format_reading(reading))}{unit}</span>'
        f"{track}</div>"
    )


def _strip_note(
    scale: StripScale | None, average: float | None, spec: fs.ParameterSpec
) -> str:
    if scale is None:
        return "Too few readings for a comparison scale."
    decimals = spec.decimals
    low = fs.format_value(scale.low, decimals)
    high = fs.format_value(scale.high, decimals)
    if scale.flat:
        return f"All shown readings are equal ({low} {spec.unit})."
    if low == high:  # distinct values that round alike
        low = fs.format_value(scale.low, decimals + 1)
        high = fs.format_value(scale.high, decimals + 1)
    note = f"Markers span {low}–{high} {spec.unit} (lowest to highest shown)"
    if not _finite(average):
        return f"{note}; average not available. Relative aid, not limits."
    return f"{note}; bars run from the average tick. Relative aid, not limits."


def _render_strip(
    section: str,
    readings: dict[str, fs.ParameterReading],
    zone: str | None = None,
) -> None:
    keys, average_key = _STRIPS[section]
    members = [readings[k] for k in keys]
    average = readings[average_key]
    scale = strip_scale([r.value for r in (*members, average)])
    spread = strip_spread([r.value for r in members])
    first, last = members[0].spec, members[-1].spec
    slug = _SECTION_SLUGS[section]

    meta = ""
    if spread is not None:
        meta = (
            f'<span class="fs-panel__meta" title="Highest minus lowest of '
            f'{_h(first.label)}–{_h(last.label)}">Spread '
            f"<b>{_h(fs.format_value(spread, first.decimals))}</b> {_h(first.unit)}</span>"
        )
    with st.container(key=f"fs-panel-{slug}"):
        st.html(_panel_title_html(section, meta))
        with st.container(key=f"fs-tilegrid-{slug}"):
            for reading in (*members, average):
                is_average = reading is average
                markup = _tile_html(
                    reading,
                    scale,
                    average.value,
                    is_average=is_average,
                    linked=_is_linked(reading.spec.key, zone),
                )
                _render_clickable(reading, markup, kind="tile")
        st.html(
            f'<div class="fs-strip__note">{_h(_strip_note(scale, average.value, first))}</div>'
        )


# ── Heat load ────────────────────────────────────────────────────────────────


def _heat_ring_html(
    total: fs.ParameterReading,
    shares: QuadrantShares,
    *,
    linked: bool = False,
) -> str:
    """Equal-segment quadrant ring with the total in the middle."""
    peaks = _peak_quadrants(shares)
    colours = []
    labels = []
    for i, corner in enumerate(("ne", "se", "sw", "nw")):
        if shares.shares is None:
            colour = "var(--fs-border)"
            share = "–"
        else:
            colour = "var(--fs-warm)" if i in peaks else "var(--fs-segment)"
            share = _share_pct(shares.shares[i])
        colours.append(f"--fs-q{i + 1}:{colour}")
        peak = " is-peak" if i in peaks else ""
        labels.append(
            f'<span class="fs-ring__q fs-ring__q--{corner}{peak}">'
            f"<b>Q{i + 1}</b><span>{_h(share)}</span></span>"
        )
    spec = total.spec
    unit = (
        f'<span class="fs-ring__unit">{_h(spec.unit)}</span>'
        if total.value is not None and spec.unit
        else ""
    )
    css = "fs-heat" + (" is-linked" if linked else "")
    return (
        f'<div class="{css}"{_zones_attr(spec.key)} aria-hidden="true">'
        f'<div class="fs-ring" style="{";".join(colours)}">'
        '<span class="fs-ring__donut"></span>'
        f"{''.join(labels)}"
        '<span class="fs-ring__centre">'
        f'<span class="fs-ring__value">{_na_markup(fs.format_reading(total))}</span>'
        f'{unit}<span class="fs-ring__caption">Total</span></span></div>'
        '<div class="fs-heat__info">'
        f'<span class="fs-heat__label">{_h(spec.label)}</span>'
        f'<span class="fs-heat__summary">{_h(_share_summary(shares))}</span>'
        '<span class="fs-row__icon"></span></div></div>'
    )


def _render_heat_load_panel(
    readings: dict[str, fs.ParameterReading], zone: str | None = None
) -> None:
    total = readings[_HEAT_TOTAL_KEY]
    quadrants = [readings[k] for k in _HEAT_QUADRANT_KEYS]
    shares = quadrant_shares([q.value for q in quadrants])
    peaks = _peak_quadrants(shares)
    section = fs.SECTION_HEAT_LOAD
    with st.container(key=f"fs-panel-{_SECTION_SLUGS[section]}"):
        st.html(
            _panel_title_html(
                section, '<span class="fs-panel__meta">Rows R6–R10</span>'
            )
        )
        _render_clickable(
            total,
            _heat_ring_html(total, shares, linked=_is_linked(total.spec.key, zone)),
            detail=f"{_share_summary(shares)}.",
        )
        for i, reading in enumerate(quadrants):
            _render_row(reading, zone=zone, peak=i in peaks)
        st.html(
            '<div class="fs-strip__note">Equal ring segments are a Q1–Q4 schematic: '
            "not proportional and not a confirmed physical orientation. Shares use "
            "unrounded quadrant readings.</div>"
        )


# ── Header and board ─────────────────────────────────────────────────────────


def _status_html(snapshot: fs.StatusSnapshot) -> str:
    status = snapshot.status
    return (
        f'<div class="fs-status fs-status--{status.level}" role="status">'
        '<span class="fs-status__dot" aria-hidden="true"></span>'
        '<div class="fs-status__text">'
        '<div class="fs-status__line">'
        f'<span class="fs-status__label">{_h(status.label)}</span>'
        '<span class="fs-status__kind" title="Live, Partial and Offline describe '
        'telemetry availability, not furnace safety or operating health.">'
        f'<div class="fs-status__time">Last updated: {_h(fs.format_ist(snapshot.last_updated))}</div>'
        "</div></div>"
    )


def _render_header(snapshot: fs.StatusSnapshot) -> None:
    with st.container(
        key="fs-header", horizontal=True, vertical_alignment="center", gap="medium"
    ):
        st.html(
            '<div class="fs-head">'
            '<div class="fs-title" role="heading" aria-level="1">'
            "BF2 blast furnace status</div>"
            f'<div class="fs-subtitle">{_h(_INSTRUCTIONS)}</div></div>'
        )
        st.html(_status_html(snapshot), width="content")
        _render_fullscreen_control()
        st.button(
            "Refresh",
            key="fs-refresh",
            icon=":material/refresh:",
            help="Reload live values",
            on_click=_refresh,
        )


def _render_status_view() -> None:
    now = datetime.now(timezone.utc)
    snapshot = fs.load_status_snapshot(now)  # the only data load of this render
    readings = {r.spec.key: r for r in snapshot.readings}
    zone = _selected_zone()

    _render_header(snapshot)

    if snapshot.failed_measurements:
        names = ", ".join(snapshot.failed_measurements)
        st.html(
            f'<div class="fs-warning" role="alert">Data source unavailable: {_h(names)}. '
            "Affected parameters show Not available.</div>"
        )

    left_sections, right_sections = _PANEL_COLUMNS
    with st.container(key="fs-dashboard"):
        with st.container(key="fs-col-left"):
            for section in left_sections:
                _render_panel(section, readings, zone)
        with st.container(key="fs-col-centre"):
            _render_strip(fs.SECTION_UPTAKE, readings, zone)
            _render_furnace(readings, zone)
            _render_strip(fs.SECTION_HEARTH, readings, zone)
        with st.container(key="fs-col-right"):
            for section in right_sections:
                if section == fs.SECTION_HEAT_LOAD:
                    _render_heat_load_panel(readings, zone)
                else:
                    _render_panel(section, readings, zone)


# ══════════════════════════════════════════════════════════════════════════════
# Trend view
# ══════════════════════════════════════════════════════════════════════════════


def _naive_ist(moment: datetime) -> datetime:
    """IST wall-clock time without tzinfo (what Plotly should display)."""
    return moment.astimezone(fs.IST).replace(tzinfo=None)


def build_trend_figure(
    trend: fs.TrendData,
    x_range: tuple[datetime, datetime],
    revision: str,
) -> go.Figure:
    """Build the responsive line chart for a successfully loaded trend."""
    spec = trend.spec
    unit = f" {spec.unit}" if spec.unit else ""
    y_title = f"{spec.title} ({spec.unit})" if spec.unit else spec.title
    lines = [
        (trend.series, "Actual" if trend.setpoint is not None else spec.title, None)
    ]
    if trend.setpoint is not None:
        lines.append((trend.setpoint, "Setpoint", "dash"))

    fig = go.Figure(
        [
            go.Scatter(
                x=series.index.tz_localize(None),
                y=series.to_numpy(),
                mode="lines",
                name=name,
                connectgaps=False,  # keep gaps; never bridge missing data
                line={
                    "color": _WARM_TEXT if dash is None else _ACCENT,
                    "width": 2,
                    "dash": dash,
                },
                hovertemplate=f"%{{y:,.{spec.decimals}f}}{_h(unit)}<extra>{_h(name)}</extra>",
            )
            for series, name, dash in lines
        ]
    )
    axis_style = {
        "gridcolor": _GRID,
        "linecolor": _BORDER,
        "tickcolor": _BORDER,
        "tickfont": {"color": _TEXT_MUTED},
        "title": {"font": {"color": _TEXT_MUTED}},
    }
    fig.update_layout(
        template="plotly_white",
        height=None,
        autosize=True,
        margin={"l": 64, "r": 16, "t": 12, "b": 48},
        showlegend=trend.setpoint is not None,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "font": {"color": _TEXT},
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": _SURFACE,
            "bordercolor": _BORDER,
            "font": {"color": _TEXT},
        },
        uirevision=revision,
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font={"color": _TEXT_MUTED},
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": _TEXT_MUTED,
            "activecolor": _ACCENT,
        },
        xaxis={
            **axis_style,
            "title": {"text": "Time (IST)", "font": {"color": _TEXT_MUTED}},
            "type": "date",
            "range": [x_range[0], x_range[1]],
            "hoverformat": "%d %b %Y, %H:%M",
            "zeroline": False,
        },
        yaxis={
            **axis_style,
            "title": {"text": y_title, "font": {"color": _TEXT_MUTED}},
            "tickformat": f",.{spec.decimals}f" if spec.decimals <= 2 else None,
            "zeroline": False,
        },
    )
    return fig


def _stat_tile(label: str, value: str, unit: str) -> str:
    suffix = f' <span class="fs-trend-current__unit">{_h(unit)}</span>' if unit else ""
    return (
        '<div class="fs-stat">'
        f'<div class="fs-stat__label">{_h(label)}</div>'
        f'<div class="fs-stat__value">{_h(value)}{suffix}</div></div>'
    )


def _source_line(spec: fs.ParameterSpec, last_data: str | None) -> str:
    if spec.components:  # summed parameter: the note says what is summed
        parts = [f"Source: {spec.measurement}"]
    elif spec.has_source:
        parts = [f"Source: {spec.measurement} · {', '.join(spec.source_fields)}"]
    else:
        parts = ["Source: not configured"]
    if spec.source_note:
        parts.append(spec.source_note)
    if last_data is not None:
        parts.append(f"Last data: {last_data}")
    return " · ".join(parts)


def _unavailable_html(why: str) -> str:
    return (
        '<div class="fs-unavailable">'
        f'<div class="fs-unavailable__title">{fs.NOT_AVAILABLE}</div>'
        f'<div class="fs-unavailable__why">{_h(why)}</div></div>'
    )


def _head_html(
    spec: fs.ParameterSpec,
    *,
    last_data: str | None = None,
    current_text: str | None = None,
) -> str:
    """Title + source caption, with the latest value on the right when known."""
    current = ""
    if current_text is not None:
        current = (
            '<div class="fs-trend-current">'
            f'<span class="fs-trend-current__value">{_h(current_text)}</span>'
            f'<span class="fs-trend-current__unit">{_h(spec.unit)}</span></div>'
        )
    return (
        '<div class="fs-trend-head"><div>'
        f'<div class="fs-trend-title" role="heading" aria-level="1">{_h(spec.title)}</div>'
        f'<div class="fs-trend-source">{_h(_source_line(spec, last_data))}</div>'
        f"</div>{current}</div>"
    )


def _render_interval_selector() -> tuple[str, tuple[datetime, datetime] | None]:
    """Render the interval control; return ``(label, (start_utc, end_utc))``.

    The range is ``None`` when a custom range is invalid (an error is shown).
    """
    now = datetime.now(timezone.utc)
    with st.container(key="fs-interval"):
        choice = st.segmented_control(
            "Interval",
            fs.INTERVAL_OPTIONS,
            default=fs.DEFAULT_INTERVAL,
            key="fs-interval-choice",
            label_visibility="collapsed",
        )
    interval = choice or fs.DEFAULT_INTERVAL

    if interval != fs.CUSTOM_INTERVAL:
        return interval, fs.resolve_fixed_range(interval, now)

    default_start, default_end = fs.default_custom_range(now)
    with st.container(key="fs-controls"):
        c1, c2, c3, c4 = st.columns(4)
        start_date = c1.date_input(
            "Start date",
            default_start.date(),
            key="fs-c-start-date",
            format="DD/MM/YYYY",
        )
        start_time = c2.time_input(
            "Start time (IST)", default_start.time(), key="fs-c-start-time", step=60
        )
        end_date = c3.date_input(
            "End date", default_end.date(), key="fs-c-end-date", format="DD/MM/YYYY"
        )
        end_time = c4.time_input(
            "End time (IST)", default_end.time(), key="fs-c-end-time", step=60
        )
    try:
        return interval, fs.resolve_custom_range(
            datetime.combine(start_date, start_time),
            datetime.combine(end_date, end_time),
            now,
        )
    except fs.RangeError as err:
        st.error(str(err))
        return interval, None


def _render_trend_view(spec: fs.ParameterSpec) -> None:
    with st.container(key="fs-toolbar"):
        back, head, fullscreen = st.columns([3, 6, 2], vertical_alignment="center")
        with back:
            st.button(
                "← Back to Furnace Status",
                key="fs-back",
                on_click=_go_status,
                width="stretch",
            )
        with fullscreen:
            _render_fullscreen_control()

    # ``head`` (title + latest value) sits in the toolbar row but is filled in
    # below, once the data is known.
    if not spec.has_source:
        with head:
            st.html(_head_html(spec))
        st.html(_unavailable_html(spec.unavailable_reason))
        return

    interval, window = _render_interval_selector()
    if window is None:  # invalid custom range; the selector already showed why
        with head:
            st.html(_head_html(spec))
        return

    start_utc, end_utc = window
    bucket = fs.choose_window(end_utc - start_utc)
    with st.spinner("Loading trend…"):
        view_data = fs.load_trend_view(
            spec,
            start_utc,
            end_utc,
            bucket,
            use_live_current=interval != fs.CUSTOM_INTERVAL,
        )
    current = view_data.current
    trend = view_data.trend
    stats = view_data.stats

    def fmt(value: float) -> str:
        return fs.format_value(value, spec.decimals)

    current_timestamp = (
        current.timestamp
        if current.timestamp is not None
        else current.setpoint_timestamp
    )
    with head:
        st.html(
            _head_html(
                spec,
                last_data=(
                    fs.format_ist(current_timestamp)
                    if current_timestamp is not None
                    else None
                ),
                current_text=fs.format_reading(current),
            )
        )

    if trend.status == "ok" and stats is not None:
        if interval == fs.CUSTOM_INTERVAL:
            revision = (
                f"{spec.key}|{interval}|{start_utc:%Y%m%d%H%M}|" f"{end_utc:%Y%m%d%H%M}"
            )
        else:
            revision = (
                f"{spec.key}|{interval}"  # fixed ranges slide each minute; keep zoom
            )
        fig = build_trend_figure(
            trend, (_naive_ist(start_utc), _naive_ist(end_utc)), revision
        )
        with st.container(key="fs-chart"):
            st.plotly_chart(
                fig,
                width="stretch",
                theme=None,
                key=f"fs-trend-{spec.key}",
                config=_PLOT_CONFIG,
            )
    else:
        st.html(_unavailable_html(trend.message or "No data was returned."))

    unit = spec.unit
    minimum = fmt(stats.minimum) if stats is not None else fs.NOT_AVAILABLE
    maximum = fmt(stats.maximum) if stats is not None else fs.NOT_AVAILABLE
    average = fmt(stats.mean) if stats is not None else fs.NOT_AVAILABLE
    current_unit = unit if fs.has_display_value(current) else ""
    range_unit = unit if stats is not None else ""
    st.html(
        '<div class="fs-stats">'
        + _stat_tile("Current", fs.format_reading(current), current_unit)
        + _stat_tile("Minimum", minimum, range_unit)
        + _stat_tile("Maximum", maximum, range_unit)
        + _stat_tile("Average", average, range_unit)
        + "</div>"
    )
    if trend.status == "ok" and stats is not None:
        st.caption(
            f"{bucket} average · {fs.format_ist(start_utc)} → {fs.format_ist(end_utc)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def render_furnace_status() -> None:
    """Render the selected nested view within V-Board's Furnace Status section."""
    _inject_css()
    state = fs.parse_view_state(st.query_params)
    if state.needs_reset:
        _set_view_query(fs.VIEW_STATUS)

    # Page root: every page-specific style hangs off this container's class.
    with st.container(key="fs-page"):
        _render_rotate_hint()
        if state.view == fs.VIEW_TREND and state.spec is not None:
            _render_trend_view(state.spec)
        else:
            _render_status_view()
