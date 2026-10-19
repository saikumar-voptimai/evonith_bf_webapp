"""Plotly figures for the Coke Energy Model page (pure builders: data in, figure out, no Streamlit).

Colours and layout follow the BMO model-accuracy charts: predictions are dashed lines, measurements are points, the
legend sits on its own row, there is no Plotly title, and missing shifts are gaps, not bridges. Each model keeps one
colour on every chart: oxygen-balance model teal, energy-balance model amber; burnt coke (the measurement) navy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.coke_energy.accuracy import accuracy
from utils.coke_energy.rist import Y_W
from utils.coke_energy.rist_diagram import RistShift

NAVY = "#25344b"        # measured: burnt coke, the shift's operating line
TEAL = "#07827f"        # oxygen-balance model (eta CO fixed)
AMBER = "#c77d2a"       # energy-balance model (omega fixed)
GREY = "#8a8f98"        # charged coke, ideal line, references
RED = "#b23a48"         # the W (wustite equilibrium) point
GRID = "rgba(120,130,145,0.18)"
KG = "kg/tHM"

MODELS = {
    # column: (name, short name, colour, band fill)
    "K_obal": ("Oxygen-balance model (ηCO fixed)", "Oxygen-balance model", TEAL, "rgba(7,130,127,0.14)"),
    "K_energy": ("Energy-balance model (ω fixed)", "Energy-balance model", AMBER, "rgba(199,125,42,0.16)"),
}
SERIES = {
    # column: (name, colour, mode, dash) - the four coke rates, used by the Rist strip
    "K_charged": ("Charged coke", GREY, "markers", None),
    "K_burnt": ("Burnt coke", NAVY, "markers", None),
    "K_obal": ("Oxygen-balance model (ηCO fixed)", TEAL, "lines+markers", "dash"),
    "K_energy": ("Energy-balance model (ω fixed)", AMBER, "lines+markers", "dash"),
}


def _legend(y: float = 1.02) -> dict:
    return dict(orientation="h", yanchor="bottom", y=y, xanchor="left", x=0, font=dict(size=12),
                bgcolor="rgba(0,0,0,0)", itemsizing="constant")


def shift_label(skey: str, fmt: str = "%d %b %Y") -> str:
    """'2026-10-14_B' -> '14 Oct 2026 · B'."""

    day, shift = skey.split("_")
    return f"{pd.Timestamp(day):{fmt}} · {shift}"


def _band_outline(x: Sequence, centre: pd.Series, band: float) -> tuple[list, list]:
    """Closed outline of centre ± band for each unbroken run, None-separated (a fill must not bridge gaps)."""

    xs: list = []
    ys: list = []
    values = np.asarray(centre, dtype=float)
    positions = list(x)
    run: list[int] = []
    for i, value in enumerate([*values, np.nan]):
        if np.isfinite(value):
            run.append(i)
            continue
        if run:
            fwd = [positions[j] for j in run]
            xs.extend([*fwd, *fwd[::-1], fwd[0], None])
            ys.extend([*(values[j] + band for j in run), *(values[j] - band for j in run[::-1]), values[run[0]] + band, None])
            run = []
    return xs, ys


def _heading(text: str, x_axis: str) -> dict:
    y_axis = x_axis.replace("x", "y")
    return dict(x=0, y=1.0, xref=f"{x_axis} domain", yref=f"{y_axis} domain", xanchor="left", yanchor="bottom",
                showarrow=False, text=text, font=dict(size=13), align="left")


# ------------------------------------------------------------------------------------------- prediction vs burnt
def prediction_figure(raw: pd.DataFrame, smooth: pd.DataFrame | None, *, show_charged: bool = False,
                      smoothing_label: str = "", highlight: pd.Timestamp | None = None, height: int = 640) -> go.Figure:
    """One panel per model against burnt coke, sharing the time axis.

    ``raw``: per-shift coke rates on the 8-hour grid (NaN = gap), indexed by shift mid-time. ``smooth``: the same columns
    smoothed for display, or None to draw the shifts themselves as the lines. The band around each model is that
    model's typical error against burnt coke at the chosen smoothing (SD of the difference on this chart).
    """

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13)
    lines = smooth if smooth is not None else raw
    faint = smooth is not None
    x_raw, x_line = raw.index, lines.index
    for row, (col, (name, short, colour, fill)) in enumerate(MODELS.items(), start=1):
        sd = accuracy(lines[col], lines["K_burnt"])["sd"]
        if np.isfinite(sd):
            bx, by = _band_outline(x_line, lines[col], sd)
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(width=0), fill="toself", fillcolor=fill,
                                     hoverinfo="skip", showlegend=False, legendgroup=col), row=row, col=1)
        # burnt coke: points per shift; with smoothing, also a solid line through the smoothed values
        fig.add_trace(go.Scatter(
            x=x_raw, y=raw["K_burnt"], mode="markers", name="Burnt coke", legendgroup="burnt", showlegend=row == 1 and not faint,
            marker=dict(color=NAVY, size=5 if faint else 6, opacity=0.35 if faint else 0.9,
                        line=dict(color="rgba(255,255,255,0.9)", width=0 if faint else 1)),
            hovertemplate=f"%{{y:,.1f}} {KG}<extra>Burnt coke, shift</extra>"), row=row, col=1)
        if faint:
            fig.add_trace(go.Scatter(
                x=x_line, y=lines["K_burnt"], mode="lines", name="Burnt coke", legendgroup="burnt", showlegend=row == 1,
                line=dict(color=NAVY, width=2), hovertemplate=f"%{{y:,.1f}} {KG}<extra>Burnt coke, {smoothing_label}</extra>"),
                row=row, col=1)
            fig.add_trace(go.Scatter(
                x=x_raw, y=raw[col], mode="markers", name=short, legendgroup=col, showlegend=False,
                marker=dict(color=colour, size=5, opacity=0.35), hovertemplate=f"%{{y:,.1f}} {KG}<extra>{short}, shift</extra>"),
                row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x_line, y=lines[col], mode="lines" if faint else "lines+markers", name=name, legendgroup=col,
            line=dict(color=colour, width=2.2 if faint else 1.8, dash="dash"), marker=dict(color=colour, size=6),
            hovertemplate=f"%{{y:,.1f}} {KG}<extra>{short}{', ' + smoothing_label if faint else ''}</extra>"), row=row, col=1)
        if show_charged and "K_charged" in raw:
            fig.add_trace(go.Scatter(
                x=x_raw, y=raw["K_charged"], mode="markers", name="Charged coke", legendgroup="charged", showlegend=row == 1,
                marker=dict(color="rgba(0,0,0,0)", size=6, line=dict(color=GREY, width=1.5)),
                hovertemplate=f"%{{y:,.1f}} {KG}<extra>Charged coke, shift</extra>"), row=row, col=1)
        basis = smoothing_label or "per shift"
        heading = (f"<b>{short}</b> vs burnt coke · typical error ±{sd:.1f} {KG} ({basis})" if np.isfinite(sd)
                   else f"<b>{short}</b> vs burnt coke")
        fig.add_annotation(**_heading(heading, "x" if row == 1 else "x2"))
    cols = ["K_burnt", *MODELS] + (["K_charged"] if show_charged else [])
    values = pd.concat([raw[cols], lines[cols]]).to_numpy(dtype=float)
    lo, hi = np.nanmin(values) - 5, np.nanmax(values) + 5
    for row in (1, 2):
        fig.update_yaxes(range=[lo, hi], title_text=f"{KG} dry", gridcolor=GRID, zeroline=False, ticksuffix=" ", row=row, col=1)
        fig.update_xaxes(showgrid=False, tickformat="%d %b<br>%Y", hoverformat="%d %b %Y, %H:%M IST", nticks=8, row=row, col=1)
        if highlight is not None:
            fig.add_vline(x=highlight, line=dict(color=GREY, width=1, dash="dot"), row=row, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=64, b=10), hovermode="x unified", legend=_legend(1.08),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def parity_figure(frame: pd.DataFrame, band: float = 10.0, height: int = 420) -> go.Figure:
    """Each model's per-shift prediction against burnt coke, with the 1:1 line and a ±band (the web model's chart)."""

    fig = go.Figure()
    vals = frame[["K_burnt", *MODELS]].to_numpy(dtype=float)
    lo = math.floor(np.nanmin(vals) / 10) * 10 - 5
    hi = math.ceil(np.nanmax(vals) / 10) * 10 + 5
    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo, lo], y=[lo + band, hi + band, hi - band, lo - band, lo + band], fill="toself",
                             mode="lines", fillcolor="rgba(138,143,152,0.14)", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GREY, width=1), hoverinfo="skip",
                             showlegend=False))
    for col, (name, short, colour, _fill) in MODELS.items():
        s = accuracy(frame[col], frame["K_burnt"])
        fig.add_trace(go.Scatter(
            x=frame["K_burnt"], y=frame[col], mode="markers", name=f"{short}: SD {s['sd']:.1f}, r {s['r']:.2f}",
            marker=dict(color=colour, size=6, opacity=0.75, line=dict(color="rgba(255,255,255,0.8)", width=0.5)),
            text=[shift_label(k) for k in frame.index],
            hovertemplate=f"%{{text}}<br>burnt %{{x:,.1f}} · model %{{y:,.1f}} {KG}<extra>{short}</extra>"))
    fig.add_annotation(x=hi, y=hi, text="1:1", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=GREY, size=11))
    fig.add_annotation(x=hi, y=hi - band, text=f"±{band:.0f} {KG}", showarrow=False, xanchor="right", yanchor="top",
                       font=dict(color=GREY, size=11))
    fig.update_xaxes(range=[lo, hi], title_text=f"Burnt coke, {KG} dry", gridcolor=GRID, zeroline=False, constrain="domain")
    fig.update_yaxes(range=[lo, hi], title_text=f"Model, {KG} dry", gridcolor=GRID, zeroline=False, scaleanchor="x",
                     scaleratio=1, constrain="domain")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=56, b=10), legend=_legend(1.02), hovermode="closest",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def residual_figure(frame: pd.DataFrame, band: float = 10.0, height: int = 420) -> go.Figure:
    """Model minus burnt coke per shift against time, with a ±band and the zero line."""

    fig = go.Figure()
    t = frame["t_mid"]
    t0, t1 = t.min(), t.max()
    fig.add_trace(go.Scatter(x=[t0, t1, t1, t0, t0], y=[band, band, -band, -band, band], fill="toself", mode="lines",
                             fillcolor="rgba(138,143,152,0.14)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    for col, (name, short, colour, _fill) in MODELS.items():
        e = frame[col] - frame["K_burnt"]
        fig.add_trace(go.Scatter(
            x=t, y=e, mode="markers", name=f"{short}: bias {e.mean():+.1f}", marker=dict(color=colour, size=6, opacity=0.75),
            text=[shift_label(k) for k in frame.index], hovertemplate=f"%{{text}}<br>%{{y:+.1f}} {KG}<extra>{short}</extra>"))
    spread = np.abs(frame[list(MODELS)].sub(frame["K_burnt"], axis=0).to_numpy(dtype=float))
    lim = max(2 * band, float(np.nanmax(spread)) + 2) if np.isfinite(spread).any() else 2 * band
    fig.update_yaxes(range=[-lim, lim], title_text=f"Model − burnt, {KG}", gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False, tickformat="%d %b<br>%Y", nticks=8)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=56, b=10), legend=_legend(1.02), hovermode="closest",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------------------------------- Rist animation
_RIST_TRACES = ("ideal", "model_line", "shift_line", "shift_pts", "model_pts", "w", "p_shift", "p_model")


def _nan_pair() -> tuple[list[float], list[float]]:
    return [math.nan, math.nan], [math.nan, math.nan]


def _rist_frame_data(rs: RistShift) -> list[go.Scatter]:
    """The Rist traces for one shift, in _RIST_TRACES order (same point counts every frame, so frames tween)."""

    a = rs.actual
    out: list[go.Scatter] = []
    if rs.ideal_mu is not None:
        x_end = (a.y_top_c - rs.ideal_y_a) / rs.ideal_mu
        out.append(go.Scatter(x=[0.0, x_end], y=[rs.ideal_y_a, rs.ideal_y_a + rs.ideal_mu * x_end]))
    else:
        x, y = _nan_pair(); out.append(go.Scatter(x=x, y=y))
    m = rs.model
    if m is not None:
        out.append(go.Scatter(x=[0.0, m.x_top], y=[m.y_a, m.y_top_c]))
    else:
        x, y = _nan_pair(); out.append(go.Scatter(x=x, y=y))
    out.append(go.Scatter(x=[0.0, a.x_top], y=[a.y_a, a.y_top_c]))
    hover = [
        f"A · oxygen brought by blast, PCI, Si/Mn/P/Ti and carbonates: {-a.y_a:.3f} O per Fe",
        f"D · ore oxygen removed directly by coke: {a.y_d:.3f} O per Fe",
        f"E · gas leaving the reserve zone: ω = {a.omega:.3f} of the way to W",
        f"T · top gas: η = {100 * (a.x_top - 1):.1f} % (carbon's share)",
    ]
    out.append(go.Scatter(x=[a.a[0], a.d[0], a.e[0], a.x_top], y=[a.a[1], a.d[1], a.e[1], a.y_top_c],
                          text=["", "", "E", ""], hovertext=hover))
    if m is not None:
        out.append(go.Scatter(x=[m.e[0]], y=[m.e[1]], hovertext=[f"E′ · model, ω fixed at {m.omega:.3f}"]))
    else:
        out.append(go.Scatter(x=[math.nan], y=[math.nan], hovertext=[""]))
    out.append(go.Scatter(x=[a.w[0]], y=[a.w[1]], text=["W"],
                          hovertext=[f"W · wüstite–iron equilibrium at {rs.T_R:.0f} °C: η = {100 * a.eta_w:.1f} %"]))
    for p, label, what in ((rs.p_actual, "P", "the shift's"), (rs.p_model, "", "the model's")):
        if p is None:
            out.append(go.Scatter(x=[math.nan], y=[math.nan], text=[""], hovertext=[""]))
        else:
            out.append(go.Scatter(x=[p[0]], y=[p[1]], text=[label],
                                  hovertext=[f"P{'' if label else '′'} · pivot of {what} lower-zone heat balance"]))
    return out


def _step_label(skey: str, info: Mapping[str, float], omega_shift: float) -> str:
    def f(v: float | None, fmt: str) -> str:
        return "–" if v is None or not np.isfinite(v) else format(v, fmt)

    return (f"{shift_label(skey, '%d %b')}   ω {f(omega_shift, '.2f')} (model {f(info.get('omega_fixed'), '.2f')})   "
            f"burnt {f(info.get('K_burnt'), '.0f')} · oxygen-balance {f(info.get('K_obal'), '.0f')} · "
            f"energy {f(info.get('K_energy'), '.0f')} {KG}")


def rist_animation_figure(frames: Sequence[RistShift], info: pd.DataFrame, strip: pd.DataFrame, *,
                          show_charged: bool = False, height: int = 760) -> go.Figure:
    """Rist diagram (full width) with the coke rates of the same shifts in a strip below, animated shift by shift.

    ``frames`` in time order; ``info`` is the shift table indexed by shift key; ``strip`` the same shifts' coke rates
    on the 8-hour grid (NaN = gap). The slider and Play run in the browser; nothing is recomputed while it plays.
    """

    fig = make_subplots(rows=2, cols=1, row_heights=[0.74, 0.26], vertical_spacing=0.10)
    last = frames[-1]
    styles = {
        "ideal": dict(mode="lines", line=dict(color=GREY, width=1.4, dash="dot"), name="Ideal line (ω = 1)", hoverinfo="skip"),
        "model_line": dict(mode="lines", line=dict(color=AMBER, width=2.2, dash="dash"), name="Energy model (ω fixed)",
                           legendgroup="energy", hoverinfo="skip"),
        "shift_line": dict(mode="lines", line=dict(color=NAVY, width=3), name="Shift line (burnt coke)", hoverinfo="skip"),
        "shift_pts": dict(mode="markers+text", marker=dict(color=NAVY, size=8, line=dict(color="white", width=1)),
                          textposition="top left", showlegend=False, textfont=dict(size=13),
                          hovertemplate="%{hovertext}<extra></extra>"),
        "model_pts": dict(mode="markers", marker=dict(color="rgba(0,0,0,0)", size=10, line=dict(color=AMBER, width=2)),
                          showlegend=False, legendgroup="energy", hovertemplate="%{hovertext}<extra></extra>"),
        "w": dict(mode="markers+text", marker=dict(color=RED, size=14, symbol="star"), textposition="bottom right",
                  showlegend=False, textfont=dict(size=13, color=RED), hovertemplate="%{hovertext}<extra></extra>"),
        "p_shift": dict(mode="markers+text", marker=dict(color=NAVY, size=11, symbol="diamond", line=dict(color="white", width=1)),
                        textposition="bottom right", showlegend=False, textfont=dict(size=13),
                        hovertemplate="%{hovertext}<extra></extra>"),
        "p_model": dict(mode="markers", marker=dict(color="rgba(0,0,0,0)", size=11, symbol="diamond", line=dict(color=AMBER, width=2)),
                        showlegend=False, legendgroup="energy", hovertemplate="%{hovertext}<extra></extra>"),
    }
    for name, trace in zip(_RIST_TRACES, _rist_frame_data(last)):
        trace.update(**styles[name])
        fig.add_trace(trace, row=1, col=1)

    strip_cols = ["K_burnt", "K_obal", "K_energy"] + (["K_charged"] if show_charged else [])
    for col in strip_cols:
        name, colour, mode, dash = SERIES[col]
        marker = (dict(color=colour, size=5, opacity=0.9) if col != "K_charged"
                  else dict(color="rgba(0,0,0,0)", size=5, line=dict(color=GREY, width=1.2)))
        fig.add_trace(go.Scatter(
            x=strip.index, y=strip[col], name=name, mode=mode, marker=marker,
            legendgroup="energy" if col == "K_energy" else None, showlegend=col != "K_energy",
            line=dict(color=colour, width=1.4, dash=dash) if "lines" in mode else None,
            hovertemplate=f"%{{y:,.1f}} {KG}<extra>{name}</extra>"), row=2, col=1)
    y_lo = float(np.nanmin(strip[strip_cols].to_numpy(dtype=float))) - 6
    y_hi = float(np.nanmax(strip[strip_cols].to_numpy(dtype=float))) + 6
    t_last = info.loc[last.skey, "t_mid"]
    fig.add_trace(go.Scatter(x=[t_last, t_last], y=[y_lo, y_hi], mode="lines", line=dict(color=GREY, width=1.2),
                             hoverinfo="skip", showlegend=False), row=2, col=1)
    ring_cols = ["K_burnt", "K_obal", "K_energy"]
    fig.add_trace(go.Scatter(x=[t_last] * 3, y=[info.loc[last.skey, c] for c in ring_cols], mode="markers",
                             marker=dict(color="rgba(0,0,0,0)", size=13, line=dict(color=[NAVY, TEAL, AMBER], width=2.2)),
                             hoverinfo="skip", showlegend=False), row=2, col=1)
    n_rist = len(_RIST_TRACES)
    cursor_idx = n_rist + len(strip_cols)
    animated = list(range(n_rist)) + [cursor_idx, cursor_idx + 1]

    plotly_frames = []
    for rs in frames:
        traces = _rist_frame_data(rs)
        t = info.loc[rs.skey, "t_mid"]
        traces.append(go.Scatter(x=[t, t], y=[y_lo, y_hi]))
        traces.append(go.Scatter(x=[t] * 3, y=[info.loc[rs.skey, c] for c in ring_cols]))
        plotly_frames.append(go.Frame(name=rs.skey, data=traces, traces=animated))
    fig.frames = plotly_frames

    y_min = min(f.actual.y_a for f in frames) - 0.08
    full = {"xaxis.range": [0, 1.55], "yaxis.range": [y_min, 1.55], "xaxis.dtick": 0.25, "yaxis.dtick": 0.5}
    zone = {"xaxis.range": [0.95, max(f.actual.x_top for f in frames) + 0.05],
            "yaxis.range": [min(f.actual.y_d for f in frames) - 0.08, max(f.actual.y_top_c for f in frames) + 0.06],
            "xaxis.dtick": 0.05, "yaxis.dtick": 0.1}
    fig.update_xaxes(range=[0, 1.55], dtick=0.25, title_text="Gas oxidation, O/C  (1 = all CO)", gridcolor=GRID,
                     zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[y_min, 1.55], title_text="O per Fe atom", gridcolor=GRID, zeroline=False, row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color=GREY, width=1), row=1, col=1)
    fig.add_vline(x=1.0, line=dict(color=GREY, width=1), row=1, col=1)
    fig.add_hline(y=Y_W, line=dict(color=GREY, width=1, dash="dot"), row=1, col=1)
    fig.add_annotation(x=0.02, y=Y_W, xref="x", yref="y", xanchor="left", yanchor="bottom", showarrow=False,
                       text="wüstite 1.056", font=dict(size=11, color=GREY))
    fig.add_annotation(x=0.02, y=0.0, xref="x", yref="y", xanchor="left", yanchor="bottom", showarrow=False,
                       text="ore oxygen ↑   blast oxygen ↓", font=dict(size=11, color=GREY))
    fig.update_xaxes(showgrid=False, tickformat="%d %b", hoverformat="%d %b %Y, %H:%M IST", nticks=8, row=2, col=1)
    fig.update_yaxes(range=[y_lo, y_hi], title_text=KG, gridcolor=GRID, zeroline=False, row=2, col=1)

    steps = [dict(method="animate", label=_step_label(rs.skey, info.loc[rs.skey], rs.actual.omega),
                  args=[[rs.skey], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for rs in frames]
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=56, b=120), legend=_legend(1.02), hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.13, xanchor="left", yanchor="top", showactive=False,
            pad=dict(t=0, r=10),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True,
                                      transition=dict(duration=250, easing="cubic-in-out"), mode="immediate")]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")]),
            ]),
            dict(type="buttons", direction="left", x=0.01, y=1.0, xanchor="left", yanchor="top", showactive=True, active=0,
                 pad=dict(t=4, r=4), bgcolor="rgba(255,255,255,0.85)", font=dict(size=12),
                 buttons=[dict(label="Whole furnace", method="relayout", args=[full]),
                          dict(label="Reserve zone", method="relayout", args=[zone])]),
        ],
        sliders=[dict(
            active=len(frames) - 1, x=0.14, len=0.86, y=-0.09, xanchor="left", yanchor="top", pad=dict(t=0, b=0),
            currentvalue=dict(prefix="", font=dict(size=13, color=GREY), xanchor="left"),
            font=dict(size=1, color="rgba(0,0,0,0)"), ticklen=2, steps=steps)],
    )
    return fig
