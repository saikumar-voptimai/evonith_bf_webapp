"""Figures for the Burden & Fuel Optimiser page. Same palette and layout conventions as the Coke Energy page."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go

from ui.coke_energy.figures import AMBER, GREY, GRID, KG, NAVY, TEAL, _legend
from utils.coke_energy.accuracy import shift_timeline


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


def model_check_figure(frame: pd.DataFrame, height: int = 340) -> go.Figure:
    """Burnt coke per shift against the two balances' out-of-sample predictions (one panel).

    Burnt coke is the measurement (navy points, hollow for shifts that failed the stability filter); the energy-balance
    model, the optimiser's model, is the amber dashed line; the oxygen balance is teal dotted; charged coke is off by
    default (click it in the legend). Missing shifts are gaps on the 8-hour grid."""
    fig = go.Figure()
    if frame.empty:
        return fig
    grid = shift_timeline(frame["sdate"].min(), frame["sdate"].max())
    X = frame.reindex(grid.index)
    x = grid["t_mid"]
    stable = X["stable"].eq(True) if "stable" in X else pd.Series(True, index=X.index)
    hover = f"%{{y:,.1f}} {KG}<extra>%{{fullData.name}}</extra>"
    if "K_charged" in X:
        fig.add_trace(go.Scatter(x=x, y=X["K_charged"], mode="markers", name="Charged coke", visible="legendonly",
                                 marker=dict(color="rgba(0,0,0,0)", size=6, line=dict(color=GREY, width=1.5)),
                                 hovertemplate=hover))
    fig.add_trace(go.Scatter(x=x, y=X["K_obal"], mode="lines", name="Oxygen-balance model",
                             line=dict(color=TEAL, width=1.6, dash="dot"), connectgaps=False, hovertemplate=hover))
    fig.add_trace(go.Scatter(x=x, y=X["K_energy"], mode="lines+markers", name="Energy-balance model (optimiser)",
                             line=dict(color=AMBER, width=2.2, dash="dash"), marker=dict(color=AMBER, size=5),
                             connectgaps=False, hovertemplate=hover))
    fig.add_trace(go.Scatter(x=x, y=X["K_burnt"].where(stable), mode="markers", name="Burnt coke (realised)",
                             marker=dict(color=NAVY, size=8, line=dict(color="white", width=1)), hovertemplate=hover))
    if (~stable & X["K_burnt"].notna()).any():
        fig.add_trace(go.Scatter(x=x, y=X["K_burnt"].where(~stable), mode="markers", name="Burnt coke, non-stable shift",
                                 marker=dict(color="rgba(0,0,0,0)", size=8, line=dict(color=NAVY, width=1.5)),
                                 hovertemplate=hover))
    cols = [c for c in ("K_burnt", "K_energy", "K_obal") if c in X]
    vals = X[cols].to_numpy(dtype=float)
    lo, hi = float(pd.Series(vals.ravel()).min()), float(pd.Series(vals.ravel()).max())
    fig.update_yaxes(title=f"{KG} dry", gridcolor=GRID, zeroline=False, range=[lo - 6, hi + 6])
    fig.update_xaxes(showgrid=False, tickformat="%d %b", hoverformat="%d %b %Y, %H:%M IST")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified", legend=_legend(1.02),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
