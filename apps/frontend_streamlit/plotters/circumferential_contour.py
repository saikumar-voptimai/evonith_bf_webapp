"""Circumferential (polar) contour plotter for V-Board quadrant data."""

import numpy as np
import plotly.colors as pc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

from .base_contour import BasePlotter


def value_to_rgba(val, vmin, vmax, colorscale):
    if not np.isfinite(val):
        frac = 0.5
    elif vmax == vmin:
        frac = 0.5
    else:
        frac = (val - vmin) / (vmax - vmin)
    if not np.isfinite(frac):
        frac = 0.5
    frac = min(max(frac, 0.0), 1.0)
    return pc.sample_colorscale(colorscale, [frac])[0]


def _axis_id(prefix: str, n: int) -> str:
    """Plotly axis ids are x/y for the first subplot, then x2/y2..."""

    return prefix if n == 1 else f"{prefix}{n}"


def _quadrant_values(values):
    array = np.asarray(values, dtype=float)
    if array.size < 4:
        array = np.r_[array, np.full(4 - array.size, np.nan)]
    array = array[:4]
    array[~np.isfinite(array)] = np.nan
    finite = np.isfinite(array)
    if not finite.any():
        return None
    if finite.sum() == 1:
        return np.full(4, array[finite][0], dtype=float)
    if not finite.all():
        x = np.arange(4)
        array = np.interp(x, x[finite], array[finite])
    return array


def _quadrant_spline(values):
    array = _quadrant_values(values)
    if array is None:
        return None
    quad_edges = np.array([0, 90, 180, 270, 360], dtype=float)
    return CubicSpline(quad_edges, np.r_[array, array[0]], bc_type="periodic")


class CircumferentialPlotter(BasePlotter):
    """Plotly-based polar contour plotter for quadrant visualisation."""

    def __init__(self, mask_file="mask_circumferential.pkl"):
        super().__init__(mask_file=mask_file)
        self.r_mesh, self.theta_mesh = self.furnace.generate_polar_mesh()
        self.mask = self.generate_mask(grid_type="polar")

    def plot_circumferential_quadrants(
        self,
        field_values_fulllist,
        titles,
        r_inner=1,
        r_outer=10,
        colorscale="YlOrRd",
        colorbar_title="Heat-load index",
        unit="",
        resolution=36,
        HORIZ_PLOTS=5,
        show_colorbar=True,
    ):
        """Render one donut chart per row or temperature level."""

        theta_edges = np.linspace(0, 360, resolution + 1)
        theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        quad_mids = np.array([45, 135, 225, 315], dtype=float)

        spline_sets = []
        all_samples = []
        for means_q, max_q, min_q in field_values_fulllist:
            cs_mean = _quadrant_spline(means_q)
            cs_max = _quadrant_spline(max_q)
            cs_min = _quadrant_spline(min_q)
            spline_sets.append((cs_mean, cs_max, cs_min))
            for spline in (cs_mean, cs_max, cs_min):
                if spline is not None:
                    all_samples.append(spline(theta_mid))

        if not all_samples:
            raise ValueError("No finite circumferential values are available.")
        all_vals = np.concatenate(all_samples)
        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size == 0:
            raise ValueError("No finite circumferential values are available.")
        vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))

        nplots = len(field_values_fulllist)
        if len(titles) != nplots:
            raise ValueError("Title count must match circumferential data count.")
        rows = int(np.ceil(nplots / HORIZ_PLOTS))
        cols = min(nplots, HORIZ_PLOTS)
        fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0, vertical_spacing=0)
        circle_theta = np.linspace(0, 2 * np.pi, 361)

        for idx, (title, splines) in enumerate(zip(titles, spline_sets)):
            row = idx // HORIZ_PLOTS + 1
            col = idx % HORIZ_PLOTS + 1
            axis_num = idx + 1
            xref = _axis_id("x", axis_num)
            yref = _axis_id("y", axis_num)
            cs_mean, cs_max, cs_min = splines

            fig.add_annotation(
                text=title,
                x=0.5,
                y=0.98,
                xref=xref,
                yref="paper",
                showarrow=False,
                font=dict(size=14, weight="bold"),
            )

            if cs_mean is not None:
                vals_mid = cs_mean(theta_mid)
                for th0, th1, val in zip(theta_edges[:-1], theta_edges[1:], vals_mid):
                    r = [r_inner, r_outer, r_outer, r_inner, r_inner]
                    t = [th0, th0, th1, th1, th0]
                    x = [r[j] * np.cos(np.deg2rad(t[j])) for j in range(5)]
                    y = [r[j] * np.sin(np.deg2rad(t[j])) for j in range(5)]
                    color = value_to_rgba(val, vmin, vmax, colorscale)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            fill="toself",
                            fillcolor=color,
                            line=dict(color=color, width=0.2),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode="markers",
                        marker=dict(opacity=0),
                        showlegend=False,
                        hovertemplate=f"{title}<br>Mean ring shown<br><extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

            for radius in (r_inner, r_outer):
                fig.add_trace(
                    go.Scatter(
                        x=radius * np.cos(circle_theta),
                        y=radius * np.sin(circle_theta),
                        mode="lines",
                        line=dict(color="black", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            if cs_mean is None:
                fig.add_annotation(
                    x=0,
                    y=0,
                    text="No data",
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                    font=dict(size=13, color="black"),
                )
                continue

            q_mean = cs_mean(quad_mids)
            q_max = cs_max(quad_mids) if cs_max is not None else q_mean
            q_min = cs_min(quad_mids) if cs_min is not None else q_mean
            label_r = r_outer * 1.22
            value_r = r_outer * 0.55

            for q in range(4):
                ang = quad_mids[q]
                ang_rad = np.deg2rad(ang)
                value = float(q_mean[q])
                plus = float(q_max[q] - q_mean[q])
                minus = float(q_min[q] - q_mean[q])
                unit_text = f" {unit}" if unit else ""

                fig.add_annotation(
                    x=label_r * np.cos(ang_rad),
                    y=label_r * np.sin(ang_rad),
                    text=f"<b>Q{q + 1}</b>",
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                    font=dict(size=14, color="black"),
                )
                fig.add_annotation(
                    x=value_r * np.cos(ang_rad),
                    y=value_r * np.sin(ang_rad),
                    text=f"<b>{value:.1f}</b>{unit_text}",
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                    font=dict(size=14, color="black"),
                )
                fig.add_annotation(
                    text=f"+{plus:.1f}{unit_text}",
                    x=value_r * np.cos(ang_rad) + 2,
                    y=value_r * np.sin(ang_rad) + 2,
                    xref=xref,
                    yref=yref,
                    font=dict(size=10, color="red", weight="bold"),
                    showarrow=False,
                )
                fig.add_annotation(
                    text=f"{minus:.1f}{unit_text}",
                    x=value_r * np.cos(ang_rad) + 2,
                    y=value_r * np.sin(ang_rad) - 2,
                    xref=xref,
                    yref=yref,
                    font=dict(size=10, color="green", weight="bold"),
                    showarrow=False,
                )

        if show_colorbar:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        colorscale=colorscale,
                        cmin=vmin,
                        cmax=vmax,
                        color=[vmin],
                        showscale=True,
                        colorbar=dict(
                            title=colorbar_title,
                            x=0.5,
                            y=-0.25,
                            len=0.85,
                            xanchor="center",
                            thickness=18,
                            orientation="h",
                        ),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        pad = 1.45
        x_range = [-r_outer * pad, r_outer * pad]
        y_range = [-r_outer * pad, r_outer * pad]
        for i in range(nplots):
            row = i // HORIZ_PLOTS + 1
            col = i % HORIZ_PLOTS + 1
            axis_num = i + 1
            fig.update_xaxes(
                visible=False,
                range=x_range,
                scaleanchor=_axis_id("y", axis_num),
                scaleratio=1,
                row=row,
                col=col,
            )
            fig.update_yaxes(visible=False, range=y_range, row=row, col=col)

        row_height = 160
        fig.update_layout(
            height=row_height * rows,
            margin=dict(t=0, b=10, l=20, r=20),
            showlegend=False,
        )
        return fig
