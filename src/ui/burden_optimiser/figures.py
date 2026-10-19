"""Figures for the Burden & Fuel Optimiser page. Same palette and layout conventions as the Coke Energy page."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
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


def material_colours(ids: list[str]) -> dict[str, str]:
    """One colour per material for every donut on the page (the page-9 palette), in yard order, so a material keeps
    its colour when the plan and today's blend are compared."""

    import plotly.express as px

    pal = px.colors.qualitative.Safe
    return {i: pal[n % len(pal)] for n, i in enumerate(ids)}


def blend_donut(shares: Mapping[str, float], t_day: Mapping[str, float], names: Mapping[str, str],
                colours: Mapping[str, str], title: str, centre: str, centre_label: str, height: int = 330) -> go.Figure:
    """Blend shares as a donut, largest first, tonnage in the middle (the page-9 donut)."""

    rows = sorted(((i, float(sh)) for i, sh in shares.items() if sh > 0.05), key=lambda t: -t[1])
    fig = go.Figure()
    if not rows:
        return fig
    ids = [i for i, _ in rows]
    fig.add_trace(go.Pie(
        labels=[names.get(i, i) for i in ids], values=[sh for _, sh in rows], hole=0.62, sort=False,
        direction="clockwise", textposition="auto", texttemplate="%{percent:.1%}", insidetextorientation="horizontal",
        customdata=[[float(t_day.get(i, float("nan")))] for i in ids],
        hovertemplate="<b>%{label}</b><br>%{percent:.1%} of the ferrous charge<br>%{customdata[0]:,.0f} t/d"
                      "<extra></extra>",
        marker=dict(colors=[colours.get(i, GREY) for i in ids], line=dict(color="rgba(128,128,128,0.35)", width=1))))
    fig.add_annotation(text=f"<span style='font-size:19px'><b>{centre}</b></span><br>"
                            f"<span style='font-size:11px'>{centre_label}</span>", showarrow=False, font=dict(size=13))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), margin=dict(l=8, r=8, t=36, b=8), height=height,
                      showlegend=True, legend=dict(orientation="v", x=1.0, xanchor="left", y=0.5, yanchor="middle",
                                                   font=dict(size=11)),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def transit_window_figure(table: pd.DataFrame, split: Mapping[str, Any] | None = None, height: int = 300) -> go.Figure:
    """Burnt minus charged coke against the averaging window: the 95 % band (bias ± 1.96 SD) per window as a range,
    the bias as a point, and the floor that does not average out. One axis, kg/tHM."""

    fig = go.Figure()
    t = table.dropna(subset=["sd"])
    if t.empty:
        return fig
    x = t["window"]
    hover = ("%{x}: bias %{customdata[0]:+.1f}, 95 %% band %{customdata[1]:+.1f} to %{customdata[2]:+.1f} "
             f"{KG}<br>%{{customdata[3]}} windows · CCC %{{customdata[4]:.3f}}<extra></extra>")
    cd = t[["bias", "loa_lo", "loa_hi", "n", "ccc"]].values
    fig.add_trace(go.Bar(x=x, y=t["loa_hi"] - t["loa_lo"], base=t["loa_lo"], name="95 % band of the difference",
                         marker=dict(color="rgba(37,52,75,0.16)", line=dict(color=NAVY, width=1)), width=0.55,
                         customdata=cd, hovertemplate=hover))
    fig.add_trace(go.Scatter(x=x, y=t["bias"], mode="markers", name="Mean difference",
                             marker=dict(color=NAVY, size=9, line=dict(color="white", width=1.5)),
                             customdata=cd, hovertemplate=hover))
    floor = (split or {}).get("floor_sd")
    if floor is not None and np.isfinite(floor) and floor > 0:
        for sgn in (1, -1):
            fig.add_hline(y=sgn * 1.96 * floor, line=dict(color=AMBER, width=1.5, dash="dash"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=AMBER, width=1.5, dash="dash"),
                                 name=f"Does not average out: ±{1.96 * floor:.0f}", hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.update_yaxes(title=f"Burnt − charged, {KG}", gridcolor=GRID, zeroline=False)
    fig.update_xaxes(title="Averaging window", showgrid=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=_legend(1.02), bargap=0.4,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def transit_lag_figure(scan: pd.DataFrame, height: int = 300) -> go.Figure:
    """Correlation of the shift-to-shift movements of burnt and charged coke against the lag (burnt after charged
    is positive)."""

    fig = go.Figure()
    s = scan.dropna(subset=["r"])
    if s.empty:
        return fig
    colours = [NAVY if lag == 0 else ("rgba(37,52,75,0.55)" if lag > 0 else "rgba(138,143,152,0.6)")
               for lag in s["lag_h"]]
    fig.add_trace(go.Bar(x=s["lag_h"], y=s["r"], marker=dict(color=colours), width=5, showlegend=False,
                         text=[f"{v:.2f}" for v in s["r"]], textposition="outside", cliponaxis=False,
                         customdata=s[["n"]].values,
                         hovertemplate="burnt %{x:+.0f} h after charged: r %{y:.2f} (%{customdata[0]} shifts)"
                                       "<extra></extra>"))
    fig.update_xaxes(title="Burnt coke compared with charged coke this many hours earlier", tickvals=list(s["lag_h"]),
                     ticktext=[f"{v:+.0f} h" if v else "0" for v in s["lag_h"]], showgrid=False)
    fig.update_yaxes(title="Correlation r", gridcolor=GRID, zeroline=False, range=[0, max(0.1, float(s["r"].max())) * 1.2])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
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
