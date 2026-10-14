"""Reusable HTML and Streamlit components for status readings."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import streamlit as st

from data import furnace_status as fs
from ui.furnace_status.layout import (
    HEAT_QUADRANT_KEYS,
    HEAT_TOTAL_KEY,
    SECTION_SLUGS,
    STRIPS,
    ZONE_READINGS,
)
from utils.logger import get_logger

log = get_logger(__name__)
CSS_PATH = Path(__file__).resolve().parents[2] / "assets" / "css" / "furnace_status.css"
_CSS_COLOUR_TOKENS = {
    "surface": "surface",
    "inset": "inset",
    "hover": "hover",
    "border": "border",
    "segment": "segment",
    "text_muted": "text-2",
    "text": "text",
    "accent": "accent",
    "warm": "warm",
    "warm_text": "warm-text",
    "live": "live",
    "partial": "partial",
}


def h(text: object) -> str:
    return html.escape(str(text), quote=True)


@lru_cache(maxsize=4)
def _read_css(path: str, mtime_ns: int) -> str:
    return Path(path).read_text(encoding="utf-8")


def _palette_css() -> str:
    colours = dict(fs.UI.colours)
    tokens = ";".join(
        f"--fs-{token}:{colours[name]}" for name, token in _CSS_COLOUR_TOKENS.items()
    )
    return f".st-key-fs-page{{{tokens}}}"


def inject_css() -> None:
    try:
        css = _read_css(str(CSS_PATH), CSS_PATH.stat().st_mtime_ns)
    except OSError:
        log.warning("Furnace Status stylesheet missing: %s", CSS_PATH)
        return
    st.markdown(f"<style>{_palette_css()}{css}</style>", unsafe_allow_html=True)


def _reading_zones() -> dict[str, tuple[str, ...]]:
    zones: dict[str, list[str]] = {}
    for zone, keys in ZONE_READINGS.items():
        for key in keys:
            zones.setdefault(key, []).append(zone)
    return {key: tuple(names) for key, names in zones.items()}


READING_ZONES = _reading_zones()


def zones_attr(key: str) -> str:
    zones = READING_ZONES.get(key)
    return f' data-zones="{" ".join(z.lower() for z in zones)}"' if zones else ""


def is_linked(key: str, zone: str | None) -> bool:
    return zone is not None and key in ZONE_READINGS.get(zone, ())


def set_view_query(view: str, parameter: str | None = None) -> None:
    params = st.query_params.to_dict()
    params.pop(fs.VIEW_QUERY_KEY, None)
    params.pop(fs.PARAMETER_QUERY_KEY, None)
    params[fs.VIEW_QUERY_KEY] = view
    if parameter is not None:
        params[fs.PARAMETER_QUERY_KEY] = parameter
    st.query_params.from_dict(params)


def open_trend(key: str) -> None:
    set_view_query(fs.VIEW_TREND, key)


def go_status() -> None:
    set_view_query(fs.VIEW_STATUS)


def na_markup(text: str) -> str:
    return h(text).replace(
        fs.NOT_AVAILABLE, f'<span class="fs-na">{fs.NOT_AVAILABLE}</span>'
    )


def _signed(value: float, decimals: int) -> str:
    text = fs.format_value(value, decimals)
    if text.startswith("-"):
        return "−" + text[1:]
    return text if fs.format_value(0.0, decimals) == text else f"+{text}"


def _dual_value_html(reading: fs.ParameterReading) -> str:
    spec = reading.spec
    actual = na_markup(fs.format_value(reading.value, spec.decimals))
    setpoint = na_markup(fs.format_value(reading.setpoint, spec.decimals))
    sub = f"SP {setpoint}"
    if reading.value is None and spec.unit:
        sub += f" {h(spec.unit)}"
    if reading.value is not None and reading.setpoint is not None:
        delta = _signed(reading.value - reading.setpoint, spec.decimals)
        sub += f' · <span title="Actual minus setpoint">Δ {h(delta)}</span>'
    return (
        f'<span class="fs-row__main">{actual}</span>'
        f'<span class="fs-row__sub">{sub}</span>'
    )


def row_html(
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
        value = na_markup(fs.format_reading(reading))
        show_unit = available
    unit = h(spec.unit) if show_unit and spec.unit else ""
    tag = (
        '<span class="fs-row__tag" title="Highest share of the total"></span>'
        if peak
        else ""
    )
    return (
        f'<div class="{" ".join(classes)}"{zones_attr(spec.key)} aria-hidden="true">'
        f'<span class="fs-row__label">{h(spec.label)}</span>{tag}'
        f'<span class="{value_class}">{value}</span>'
        f'<span class="fs-row__unit">{unit}</span>'
        '<span class="fs-row__icon"></span></div>'
    )


def button_label(reading: fs.ParameterReading, detail: str = "") -> str:
    spec = reading.spec
    unit = f" {spec.unit}" if fs.has_display_value(reading) and spec.unit else ""
    extra = f" {detail}" if detail else ""
    return f"{spec.label}: {fs.format_reading(reading)}{unit}.{extra} Open trend"


def render_clickable(
    reading: fs.ParameterReading,
    markup: str,
    *,
    kind: Literal["row", "tile"] = "row",
    detail: str = "",
) -> None:
    key = reading.spec.key
    with st.container(key=f"fs-{kind}-{key}"):
        st.html(markup)
        st.button(
            button_label(reading, detail),
            key=f"fs-btn-{key}",
            on_click=open_trend,
            args=(key,),
        )


def render_row(
    reading: fs.ParameterReading, *, zone: str | None = None, peak: bool = False
) -> None:
    render_clickable(
        reading,
        row_html(reading, linked=is_linked(reading.spec.key, zone), peak=peak),
    )


def panel_title_html(section: str, meta: str = "") -> str:
    return (
        f'<div class="fs-panel__title"><span class="fs-panel__name">{h(section)}</span>'
        f"{meta}</div>"
    )


def render_panel(
    section: str,
    readings: dict[str, fs.ParameterReading],
    zone: str | None = None,
) -> None:
    with st.container(key=f"fs-panel-{SECTION_SLUGS[section]}"):
        st.html(panel_title_html(section))
        for spec in fs.parameters_in_section(section):
            render_row(readings[spec.key], zone=zone)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def strip_spread(values: Sequence[float | None]) -> float | None:
    if len(values) != 4 or not all(_finite(value) for value in values):
        return None
    return max(values) - min(values)  # type: ignore[type-var]


@dataclass(frozen=True)
class QuadrantShares:
    shares: tuple[float, ...] | None
    highest: tuple[int, ...] = ()
    note: str = ""


def quadrant_shares(values: Sequence[float | None]) -> QuadrantShares:
    if len(values) != 4 or not all(_finite(value) for value in values):
        return QuadrantShares(None, note="Shares need all four quadrant readings.")
    loads = [float(value) for value in values]  # type: ignore[arg-type]
    if any(value < 0 for value in loads):
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
        index
        for index, value in enumerate(loads)
        if math.isclose(value, top, rel_tol=1e-9, abs_tol=1e-12)
    )
    return QuadrantShares(tuple(value / total for value in loads), highest)


def _peak_quadrants(shares: QuadrantShares) -> frozenset[int]:
    if shares.shares is None or len(shares.highest) == len(shares.shares):
        return frozenset()
    return frozenset(shares.highest)


def _share_pct(share: float) -> str:
    return f"{share * 100:.1f}%"


def _share_summary(shares: QuadrantShares) -> str:
    if shares.shares is None:
        return shares.note
    percentage = _share_pct(shares.shares[shares.highest[0]])
    names = [f"Q{index + 1}" for index in shares.highest]
    if len(names) == len(shares.shares):
        return f"All quadrants equal ({percentage} each)"
    if len(names) > 1:
        return f"Highest share: {', '.join(names)} tied ({percentage} each)"
    return f"Highest share: {names[0]} ({percentage})"


def _tile_html(
    reading: fs.ParameterReading,
    *,
    linked: bool = False,
) -> str:
    spec = reading.spec
    classes = ["fs-tile"]
    if reading.value is None:
        classes.append("fs-tile--na")
    if linked:
        classes.append("is-linked")
    unit = (
        f'<span class="fs-tile__unit">{h(spec.unit)}</span>'
        if reading.value is not None and spec.unit
        else ""
    )
    return (
        f'<div class="{" ".join(classes)}"{zones_attr(spec.key)} aria-hidden="true">'
        f'<span class="fs-tile__label">{h(spec.label)}</span>'
        f'<span class="fs-tile__value">{na_markup(fs.format_reading(reading))}{unit}</span>'
        "</div>"
    )


def render_strip(
    section: str,
    readings: dict[str, fs.ParameterReading],
    zone: str | None = None,
) -> None:
    members = [readings[key] for key in STRIPS[section]]
    primary = members[:4]
    spread = strip_spread([reading.value for reading in primary])
    first, last = primary[0].spec, primary[-1].spec
    meta = ""
    if spread is not None:
        meta = (
            '<span class="fs-panel__meta" title="Highest minus lowest of '
            f'{h(first.label)}–{h(last.label)}">Spread '
            f"<b>{h(fs.format_value(spread, first.decimals))}</b> {h(first.unit)}</span>"
        )
    with st.container(key=f"fs-panel-{SECTION_SLUGS[section]}"):
        st.html(panel_title_html(section, meta))
        with st.container(key=f"fs-tilegrid-{SECTION_SLUGS[section]}"):
            for reading in members:
                render_clickable(
                    reading,
                    _tile_html(
                        reading,
                        linked=is_linked(reading.spec.key, zone),
                    ),
                    kind="tile",
                )


def _heat_ring_html(
    total: fs.ParameterReading,
    shares: QuadrantShares,
    *,
    linked: bool = False,
) -> str:
    peaks = _peak_quadrants(shares)
    colours, labels = [], []
    for index, corner in enumerate(("ne", "se", "sw", "nw")):
        if shares.shares is None:
            colour, share = "var(--fs-border)", "–"
        else:
            colour = "var(--fs-warm)" if index in peaks else "var(--fs-segment)"
            share = _share_pct(shares.shares[index])
        colours.append(f"--fs-q{index + 1}:{colour}")
        peak = " is-peak" if index in peaks else ""
        labels.append(
            f'<span class="fs-ring__q fs-ring__q--{corner}{peak}">'
            f"<b>Q{index + 1}</b><span>{h(share)}</span></span>"
        )
    spec = total.spec
    unit = (
        f'<span class="fs-ring__unit">{h(spec.unit)}</span>'
        if total.value is not None and spec.unit
        else ""
    )
    css = "fs-heat" + (" is-linked" if linked else "")
    return (
        f'<div class="{css}"{zones_attr(spec.key)} aria-hidden="true">'
        f'<div class="fs-ring" style="{";".join(colours)}">'
        '<span class="fs-ring__donut"></span>'
        f'{"".join(labels)}<span class="fs-ring__centre">'
        f'<span class="fs-ring__value">{na_markup(fs.format_reading(total))}</span>'
        f'{unit}<span class="fs-ring__caption">Total</span></span></div>'
        '<div class="fs-heat__info">'
        f'<span class="fs-heat__label">{h(spec.label)}</span>'
        f'<span class="fs-heat__summary">{h(_share_summary(shares))}</span>'
        '<span class="fs-row__icon"></span></div></div>'
    )


def render_heat_load_panel(
    readings: dict[str, fs.ParameterReading], zone: str | None = None
) -> None:
    total = readings[HEAT_TOTAL_KEY]
    quadrants = [readings[key] for key in HEAT_QUADRANT_KEYS]
    shares = quadrant_shares([reading.value for reading in quadrants])
    peaks = _peak_quadrants(shares)
    with st.container(key=f"fs-panel-{SECTION_SLUGS[fs.SECTION_HEAT_LOAD]}"):
        st.html(
            panel_title_html(
                fs.SECTION_HEAT_LOAD,
                '<span class="fs-panel__meta">Rows R6–R10</span>',
            )
        )
        render_clickable(
            total,
            _heat_ring_html(total, shares, linked=is_linked(total.spec.key, zone)),
            detail=f"{_share_summary(shares)}.",
        )
        for index, reading in enumerate(quadrants):
            render_row(reading, zone=zone, peak=index in peaks)
        st.html(
            '<div class="fs-strip__note">Equal ring segments are a Q1–Q4 schematic: '
            "not proportional and not a confirmed physical orientation. Shares use "
            "unrounded quadrant readings.</div>"
        )
