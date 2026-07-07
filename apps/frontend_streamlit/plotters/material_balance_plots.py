"""Plotly figures and Styler helpers for the Material Balance page.

All four functions are pure-Python (no Streamlit imports). The page
calls them and uses ``st.plotly_chart`` / ``st.dataframe`` to render.

Public API:
    build_sankey            — 3-column flow diagram (materials → BF2 → outputs)
    build_per_element_bars  — 4×3 grid of stacked In/Out bars
    style_closure_table     — pandas Styler with traffic-light row colours
    build_furnace_diagram   — lightweight schematic with inflow/outflow arrows
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from furnace_data.material_balance.constants import ELEMENTS

# Stable colour palette per stream so cross-tab visuals stay consistent.
INPUT_COLOURS: Dict[str, str] = {
    "Coke": "#4f4f4f",
    "Nut Coke": "#7c7c7c",
    "PCI": "#a1845b",
    "Ore": "#a23b3b",
    "Sinter": "#cc7a3a",
    "Pellet": "#c2a83a",
    "Flux": "#6db26d",
    "Hot Blast": "#3a9bd6",
    "O2 Enrichment": "#7fc4ff",
    "Steam": "#cfe3ff",
}
OUTPUT_COLOURS: Dict[str, str] = {
    "Hot Metal": "#b8281f",
    "Slag": "#7a571c",
    "Top Gas": "#6f4e8b",
    "Dust Catcher": "#888888",
    "Unaccounted": "#bbbbbb",
}

INPUT_ORDER = list(INPUT_COLOURS.keys())
OUTPUT_ORDER = list(OUTPUT_COLOURS.keys())


# ---------------------------------------------------------------------------
# 1. Sankey
# ---------------------------------------------------------------------------


def _aggregate_for_sankey(
    inputs: Dict[str, Dict[str, float]],
    outputs: Dict[str, Dict[str, float]],
    focus_element: str | None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Collapse the nested element dicts into stream→tonnes totals.

    If ``focus_element`` is given, only that element's tonnes are used
    so the link widths reflect a single-element flow.
    """
    in_totals: Dict[str, float] = {k: 0.0 for k in INPUT_ORDER}
    out_totals: Dict[str, float] = {k: 0.0 for k in OUTPUT_ORDER}

    elements = [focus_element] if focus_element else ELEMENTS
    for el in elements:
        for stream, t in inputs.get(el, {}).items():
            if stream in in_totals:
                in_totals[stream] += t
        for stream, t in outputs.get(el, {}).items():
            if stream in out_totals:
                out_totals[stream] += t

    return in_totals, out_totals


def build_sankey(
    inputs: Dict[str, Dict[str, float]],
    outputs: Dict[str, Dict[str, float]],
    focus_element: str | None = None,
) -> go.Figure:
    """3-column Sankey: input streams → BF2 hub → output streams.

    Args:
        inputs (dict): ``{element: {stream: tonnes}}`` from
            :class:`~utils.material_balance.compute.BalanceResult`.
        outputs (dict): Same structure as *inputs*.
        focus_element (str | None): When set, only that element's flow
            is shown (link widths in tonnes of that element). When
            ``None``, total mass is summed across all elements.

    Returns:
        go.Figure: Plotly Sankey figure.
    """
    in_totals, out_totals = _aggregate_for_sankey(inputs, outputs, focus_element)

    in_streams = [s for s in INPUT_ORDER if in_totals[s] > 0]
    out_streams = [s for s in OUTPUT_ORDER if out_totals[s] > 0]

    # Always render an "Unaccounted" node so the future hook is visible
    # even when v1 has 0 t there.
    if "Unaccounted" not in out_streams:
        out_streams.append("Unaccounted")

    hub = "BF2 (per day)"
    labels = in_streams + [hub] + out_streams
    label_idx = {label: i for i, label in enumerate(labels)}
    colours = (
        [INPUT_COLOURS[s] for s in in_streams]
        + ["#222222"]
        + [OUTPUT_COLOURS[s] for s in out_streams]
    )

    sources, targets, values, link_colours = [], [], [], []
    for s in in_streams:
        sources.append(label_idx[s])
        targets.append(label_idx[hub])
        values.append(max(in_totals[s], 0.0))
        link_colours.append(_translucent(INPUT_COLOURS[s], 0.55))
    for s in out_streams:
        sources.append(label_idx[hub])
        targets.append(label_idx[s])
        values.append(max(out_totals.get(s, 0.0), 0.0))
        link_colours.append(_translucent(OUTPUT_COLOURS[s], 0.55))

    title = (
        f"Material Balance — Element: {focus_element}"
        if focus_element
        else "Material Balance — Total mass"
    )

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=colours,
                pad=18,
                thickness=18,
                line=dict(color="#444", width=0.6),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colours,
                hovertemplate="%{source.label} → %{target.label}<br>%{value:.1f} t<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=title,
        font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
    )
    return fig


def _translucent(hex_colour: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to ``rgba(r,g,b,alpha)`` for Sankey link tinting."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# ---------------------------------------------------------------------------
# 2. Per-element bars
# ---------------------------------------------------------------------------


def build_per_element_bars(
    closure_table: pd.DataFrame,
    inputs: Dict[str, Dict[str, float]],
    outputs: Dict[str, Dict[str, float]],
) -> go.Figure:
    """4×3 grid of stacked In/Out bars (one subplot per element).

    Each subplot has two x categories ("In", "Out") with stream-coloured
    stacks. The legend is rendered once across the figure.

    Args:
        closure_table (pd.DataFrame): Closure table (unused directly but
            kept in the signature for future annotation overlays).
        inputs (dict): ``{element: {stream: tonnes}}`` from
            :class:`~utils.material_balance.compute.BalanceResult`.
        outputs (dict): Same structure as *inputs*.

    Returns:
        go.Figure: Plotly figure with 4×3 subplots.
    """
    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=[el for el in ELEMENTS],
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    legend_seen: set[str] = set()
    for idx, el in enumerate(ELEMENTS):
        row = idx // 3 + 1
        col = idx % 3 + 1

        in_streams = inputs.get(el, {})
        out_streams = outputs.get(el, {})

        for stream in INPUT_ORDER:
            v = in_streams.get(stream, 0.0)
            if v <= 0:
                continue
            show = stream not in legend_seen
            legend_seen.add(stream)
            fig.add_trace(
                go.Bar(
                    x=["In"],
                    y=[v],
                    name=stream,
                    legendgroup=stream,
                    showlegend=show,
                    marker_color=INPUT_COLOURS.get(stream, "#888"),
                    hovertemplate=f"{stream}<br>%{{y:.2f}} t<extra></extra>",
                ),
                row=row,
                col=col,
            )

        for stream in OUTPUT_ORDER:
            v = out_streams.get(stream, 0.0)
            if v <= 0:
                continue
            show = stream not in legend_seen
            legend_seen.add(stream)
            fig.add_trace(
                go.Bar(
                    x=["Out"],
                    y=[v],
                    name=stream,
                    legendgroup=stream,
                    showlegend=show,
                    marker_color=OUTPUT_COLOURS.get(stream, "#888"),
                    hovertemplate=f"{stream}<br>%{{y:.2f}} t<extra></extra>",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        barmode="stack",
        height=820,
        margin=dict(l=20, r=20, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Closure table styler
# ---------------------------------------------------------------------------


def style_closure_table(
    closure_table: pd.DataFrame,
    good: Sequence[float] = (95, 105),
    warning: Sequence[float] = (85, 115),
):
    """Return a pandas Styler with traffic-light row colours.

    Green for closure within *good*, yellow within *warning*, red
    otherwise. Rows with ``In_t == 0`` (NaN closure) are left white.

    Args:
        closure_table (pd.DataFrame): Output of
            :func:`~utils.material_balance.compute.build_closure_table`.
        good (Sequence[float]): ``(lo, hi)`` Closure_pct range for green.
        warning (Sequence[float]): ``(lo, hi)`` Closure_pct range for
            yellow (outside *good* but within *warning*).

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame ready for
        ``st.dataframe()``.
    """
    def _row_style(row: pd.Series) -> List[str]:
        c = row.get("Closure_pct")
        if pd.isna(c):
            return [""] * len(row)
        if good[0] <= c <= good[1]:
            colour = "background-color: #d6f0d6;"  # green
        elif warning[0] <= c <= warning[1]:
            colour = "background-color: #fff4c2;"  # yellow
        else:
            colour = "background-color: #f7c8c8;"  # red
        return [colour] * len(row)

    styler = (
        closure_table.style.apply(_row_style, axis=1)
        .format(
            {
                "In_t": "{:.1f}",
                "Out_t": "{:.1f}",
                "Closure_pct": lambda v: "—" if pd.isna(v) else f"{v:.1f} %",
                "Delta_t": "{:+.1f}",
            }
        )
    )
    return styler


# ---------------------------------------------------------------------------
# 4. Furnace schematic with annotations
# ---------------------------------------------------------------------------

# Right-half outline points (x, y) from CLAUDE.md / setting_ds_dv.yml.
# We mirror about x=0 to draw the full furnace cross-section.
_RIGHT_OUTLINE = [
    (2.8, 4.374),
    (2.8, 6.795),
    (3.15, 8.335),
    (3.15, 11.290),
    (3.65, 14.390),
    (3.65, 15.89),
    (2.898, 20.0),
]


def build_furnace_diagram(
    in_streams: Dict[str, float],
    out_streams: Dict[str, float],
) -> go.Figure:
    """Lightweight cross-section schematic with labelled inflow/outflow arrows.

    Labels are drawn at the outer (non-furnace) end of each arrow so they
    never overlap the furnace outline.  Input labels sit left of the furnace,
    output labels right.

    Args:
        in_streams (dict): ``{label: tonnes}`` to annotate inflow arrows.
            Recognised labels: ``"Burden"``, ``"Hot Blast"``, ``"PCI"``.
        out_streams (dict): ``{label: tonnes}`` to annotate outflow arrows.
            Recognised labels: ``"Top Gas"``, ``"Hot Metal"``, ``"Slag"``,
            ``"Dust Catcher"``.

    Returns:
        go.Figure: Plotly figure with the furnace cross-section and
        labelled inflow/outflow arrows.
    """
    fig = go.Figure()

    # --- Furnace outline ---
    right_x, right_y = zip(*_RIGHT_OUTLINE)
    left_x = [-x for x in right_x[::-1]]
    left_y = list(right_y[::-1])
    poly_x = list(right_x) + left_x + [right_x[0]]
    poly_y = list(right_y) + left_y + [right_y[0]]
    fig.add_trace(
        go.Scatter(
            x=poly_x,
            y=poly_y,
            mode="lines",
            line=dict(color="#222", width=2),
            fill="toself",
            fillcolor="rgba(220, 200, 170, 0.45)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --- Zone bands with centred labels inside the furnace body ---
    zones = [
        ("Hearth", 0.0, 5.5, "rgba(180, 30, 30, 0.10)"),
        ("Tuyere", 5.5, 10.5, "rgba(220, 120, 30, 0.10)"),
        ("Bosh", 10.5, 12.9, "rgba(220, 200, 60, 0.10)"),
        ("Belly", 12.9, 15.0, "rgba(120, 180, 80, 0.10)"),
        ("Stack", 15.0, 20.0, "rgba(80, 130, 200, 0.10)"),
    ]
    for z_label, y0, y1, z_colour in zones:
        fig.add_shape(
            type="rect",
            x0=-4.0, x1=4.0, y0=y0, y1=y1,
            fillcolor=z_colour,
            line=dict(width=0),
            layer="below",
        )
        fig.add_annotation(
            x=0.0,
            y=(y0 + y1) / 2,
            text=z_label,
            showarrow=False,
            font=dict(size=9, color="#777"),
            xanchor="center",
            yanchor="middle",
        )

    # --- Input streams (left side) ---
    _draw_stream_arrow(
        fig, fx=-2.9, fy=19.5, lx=-7.2, ly=19.5,
        text=f"Burden: {in_streams.get('Burden', 0):,.0f} t",
        colour="#a23b3b", label_side="left",
    )
    _draw_stream_arrow(
        fig, fx=-4.0, fy=8.0, lx=-7.2, ly=8.0,
        text=f"Blast + O2: {in_streams.get('Hot Blast', 0):,.0f} t",
        colour="#3a9bd6", label_side="left",
    )
    _draw_stream_arrow(
        fig, fx=-4.0, fy=6.5, lx=-7.2, ly=6.5,
        text=f"PCI + Steam: {in_streams.get('PCI', 0):,.0f} t",
        colour="#a1845b", label_side="left",
    )

    # --- Output streams (right side) ---
    _draw_stream_arrow(
        fig, fx=2.9, fy=19.5, lx=7.2, ly=19.5,
        text=f"Top Gas: {out_streams.get('Top Gas', 0):,.0f} t",
        colour="#6f4e8b", label_side="right",
    )
    dust_t = out_streams.get("Dust Catcher", 0.0)
    _draw_stream_arrow(
        fig, fx=3.2, fy=17.5, lx=7.2, ly=17.5,
        text=f"Dust: {dust_t:,.0f} t" if dust_t > 0 else "Dust Catcher: —",
        colour="#888888" if dust_t > 0 else "#bbbbbb",
        label_side="right",
    )
    _draw_stream_arrow(
        fig, fx=4.0, fy=5.5, lx=7.2, ly=5.5,
        text=f"Slag: {out_streams.get('Slag', 0):,.0f} t",
        colour="#7a571c", label_side="right",
    )
    _draw_stream_arrow(
        fig, fx=4.0, fy=3.8, lx=7.2, ly=3.8,
        text=f"Hot Metal: {out_streams.get('Hot Metal', 0):,.0f} t",
        colour="#b8281f", label_side="right",
    )

    fig.update_layout(
        height=680,
        margin=dict(l=18, r=18, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(range=[-11.5, 11.5], visible=False),
        yaxis=dict(range=[-0.5, 22], visible=False),
    )
    return fig


def _draw_stream_arrow(
    fig: go.Figure,
    fx: float,
    fy: float,
    lx: float,
    ly: float,
    text: str,
    colour: str,
    label_side: str,
) -> None:
    """Draw a stream arrow plus a separate text label at the outer end.

    The arrow and the label are two distinct annotations so the text is
    never rendered on top of the furnace outline.

    For inflows (``label_side="left"``) the arrowhead touches the furnace
    wall at ``(fx, fy)`` and the label is anchored at ``(lx, ly)``.
    For outflows (``label_side="right"``) the arrowhead points away from
    the furnace toward ``(lx, ly)`` and the tail sits at ``(fx, fy)``.

    Args:
        fig (go.Figure): Figure to annotate.
        fx (float): Furnace-wall x-coordinate.
        fy (float): Furnace-wall y-coordinate.
        lx (float): Label / outer-end x-coordinate.
        ly (float): Label / outer-end y-coordinate.
        text (str): Stream label text.
        colour (str): Hex colour string.
        label_side (str): ``"left"`` for inflows, ``"right"`` for outflows.
    """
    if label_side == "left":
        head_x, head_y = fx, fy
        tail_ax, tail_ay = lx, ly
    else:
        head_x, head_y = lx, ly
        tail_ax, tail_ay = fx, fy

    # Arrow (no text on annotation itself)
    fig.add_annotation(
        x=head_x, y=head_y,
        ax=tail_ax, ay=tail_ay,
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowcolor=colour,
        arrowwidth=2.5,
        text="",
    )
    # Label at the outer end
    fig.add_annotation(
        x=lx, y=ly,
        text=text,
        showarrow=False,
        font=dict(size=10, color=colour, family="Arial"),
        xanchor="right" if label_side == "left" else "left",
        yanchor="middle",
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=2,
    )
