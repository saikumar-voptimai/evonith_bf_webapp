"""Figures for the Burden & Fuel Optimiser page. Same palette and layout conventions as the Coke Energy page."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go

from ui.coke_energy.figures import AMBER, GRID, NAVY, TEAL


def frontier_figure(frontier: pd.DataFrame, best: Mapping[str, Any] | None = None, top: Mapping[str, Any] | None = None,
                    height: int = 400) -> go.Figure:
    """Ore + flux + fuel cost per tHM against production (one series), with the cheapest level and the ceiling
    labelled on the curve. The binding limit at each point is in the hover."""
    fig = go.Figure()
    if frontier.empty:
        return fig
    hover = ("%{x:,.0f} t/d · Rs %{y:,.0f}/tHM<br>coke %{customdata[0]:.1f} kg/tHM dry · fuel %{customdata[1]:.0f}"
             "<br>extra tonne Rs %{customdata[2]:,.0f}<br>binds: %{customdata[3]}<extra></extra>")
    fig.add_trace(go.Scatter(
        x=frontier.prod_t_day, y=frontier.cost_rs_thm, mode="lines+markers", name="Cheapest plan at each output",
        line=dict(color=NAVY, width=2), marker=dict(size=8, color=NAVY, line=dict(color="white", width=2)),
        customdata=frontier[["coke_dry", "fuel_rate", "marginal_rs_per_t", "binding"]].fillna("").values,
        hovertemplate=hover, showlegend=False))
    for pt, colour, label, pos in ((best, AMBER, "Cheapest", "bottom center"), (top, TEAL, "Highest output", "top left")):
        if pt is None:
            continue
        x, y = pt["P_tday"], pt["cost"]["total"]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", marker=dict(size=13, color=colour, line=dict(color="white", width=2)),
            text=[f"{label}: {x:,.0f} t/d · Rs {y:,.0f}"], textposition=pos, textfont=dict(size=12),
            hoverinfo="skip", showlegend=False, cliponaxis=False))
    fig.update_xaxes(title="Hot metal, t/d", gridcolor=GRID, zeroline=False, tickformat=",")
    fig.update_yaxes(title="Ore + flux + fuel, Rs/tHM", gridcolor=GRID, zeroline=False, tickformat=",")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

