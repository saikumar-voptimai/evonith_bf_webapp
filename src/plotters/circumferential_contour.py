"""Circumferential (polar) contour plotter for cooling stave heat loads.

Builds interactive Plotly polar charts showing per-quadrant heat load
distributions across stave rows (R6–R10).  Uses cubic-spline interpolation
for smooth quadrant boundaries.
"""

import numpy as np
import plotly.colors as pc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

from .base_contour import BasePlotter


def value_to_rgba(val, vmin, vmax, colorscale):
    if vmax == vmin:
        frac = 0.5
    else:
        frac = (val - vmin) / (vmax - vmin)
    if not np.isfinite(frac):
        frac = 0.5
    frac = min(max(frac, 0.0), 1.0)
    return pc.sample_colorscale(colorscale, [frac])[0]


def _axis_id(prefix: str, n: int) -> str:
    """
    Plotly axis ids: x, x2, x3... (NOT x1)
    """
    return prefix if n == 1 else f"{prefix}{n}"


class CircumferentialPlotter(BasePlotter):
    """Plotly-based polar contour plotter for quadrant heat load visualisation.

    Renders one polar donut chart per stave row (R6–R10), coloured by mean
    quadrant heat load, with inner rings for min and outer rings for max.

    Attributes:
        r_mesh:    Radial meshgrid generated from the furnace geometry.
        theta_mesh: Angular meshgrid generated from the furnace geometry.
        mask:      Boolean polar mask array.
    """

    def __init__(self, mask_file="mask_circumferential.pkl"):
        """Initialise with the circumferential polar mask.

        Args:
            mask_file: Filename of the cached polar mask pickle.
        """
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
        colorbar_title="Heat Load (GJ)",
        unit="GJ",
        resolution=120,  # smoother than 36
        HORIZ_PLOTS=5,
        show_colorbar=True,
    ):
        """
        field_values_fulllist:
          [
            [means_q, max_q, min_q],   # ring 1
            [means_q, max_q, min_q],   # ring 2
            ...
          ]
        where each *_q is length 4 (Q1..Q4)
        """

        # Angles for quadrant boundaries and midpoints
        quad_edges = np.array([0, 90, 180, 270, 360], dtype=float)
        quad_mids = np.array([45, 135, 225, 315], dtype=float)

        # ---------- build global vmin/vmax from ALL rings using spline samples ----------
        all_samples = []
        theta_edges = np.linspace(0, 360, resolution + 1)
        theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])

        for means_q, max_q, min_q in field_values_fulllist:
            means_q = np.asarray(means_q, dtype=float)
            max_q = np.asarray(max_q, dtype=float)
            min_q = np.asarray(min_q, dtype=float)

            # Ensure periodic (append first value to end)
            means_p = np.r_[means_q, means_q[0]]
            max_p = np.r_[max_q, max_q[0]]
            min_p = np.r_[min_q, min_q[0]]

            cs_mean = CubicSpline(quad_edges, means_p, bc_type="periodic")
            cs_max = CubicSpline(quad_edges, max_p, bc_type="periodic")
            cs_min = CubicSpline(quad_edges, min_p, bc_type="periodic")

            all_samples.append(cs_mean(theta_mid))
            all_samples.append(cs_max(theta_mid))
            all_samples.append(cs_min(theta_mid))

        all_vals = np.concatenate(all_samples)
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

        # ---------- subplots ----------
        nplots = len(field_values_fulllist)
        rows = int(np.ceil(nplots / HORIZ_PLOTS))
        cols = min(nplots, HORIZ_PLOTS)

        fig = make_subplots(
            rows=rows, cols=cols, horizontal_spacing=0, vertical_spacing=0
        )

        circle_theta = np.linspace(0, 2 * np.pi, 361)

        for idx, (means_q, max_q, min_q) in enumerate(field_values_fulllist):
            row = idx // HORIZ_PLOTS + 1
            col = idx % HORIZ_PLOTS + 1

            axis_num = idx + 1
            xref = _axis_id("x", axis_num)
            yref = _axis_id("y", axis_num)

            means_q = np.asarray(means_q, dtype=float)
            max_q = np.asarray(max_q, dtype=float)
            min_q = np.asarray(min_q, dtype=float)

            means_p = np.r_[means_q, means_q[0]]
            max_p = np.r_[max_q, max_q[0]]
            min_p = np.r_[min_q, min_q[0]]

            cs_mean = CubicSpline(quad_edges, means_p, bc_type="periodic")
            cs_max = CubicSpline(quad_edges, max_p, bc_type="periodic")
            cs_min = CubicSpline(quad_edges, min_p, bc_type="periodic")

            # Values for coloring each angular slice (use midpoint for better look)
            vals_mid = cs_mean(theta_mid)
            fig.add_annotation(
                text=titles[idx],
                x=0.5,
                y=0.98,
                xref=xref,
                yref="paper",
                showarrow=False,
                font=dict(size=14, weight="bold"),
            )

            # Draw sectors
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

            # Optional: add a single invisible hover layer (lighter than 1 per slice)
            # (If you want hover per slice, keep your old invisible-marker approach.)
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker=dict(opacity=0),
                    showlegend=False,
                    hovertemplate=f"{titles[idx]}<br>Mean ring shown<br><extra></extra>",
                ),
                row=row,
                col=col,
            )

            # Boundary circles
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

            # -------- quadrant annotations (THIS is the key fix) --------
            q_mean = cs_mean(quad_mids)
            q_max = cs_max(quad_mids)
            q_min = cs_min(quad_mids)

            label_r = r_outer * 1.22
            value_r = r_outer * 0.55

            for q in range(4):
                ang = quad_mids[q]
                ang_rad = np.deg2rad(ang)

                v = float(q_mean[q])
                plus = float(q_max[q] - q_mean[q])
                minus = float(q_mean[q] - q_min[q])

                # Single text with sup/sub (Plotly supports <sup>/<sub>)
                txt = f"<b>{v:.1f}</b> {unit}"
                fig.add_annotation(
                    x=value_r * np.cos(ang_rad),
                    y=value_r * np.sin(ang_rad),
                    text=txt,
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                    font=dict(size=14, color="black"),
                )

                fig.add_annotation(
                    text=f"+{plus:.1f}{unit}",
                    x=value_r * np.cos(ang_rad) + 2,
                    y=value_r * np.sin(ang_rad) + 2,
                    xref=f"x{idx+1}",
                    yref=yref,
                    font=dict(size=10, color="green", weight="bold"),
                    showarrow=False,
                )
                fig.add_annotation(
                    text=f"+{minus:.1f}{unit}",
                    x=value_r * np.cos(ang_rad) + 2,
                    y=value_r * np.sin(ang_rad) - 2,
                    xref=f"x{idx+1}",
                    yref=yref,
                    font=dict(size=10, color="green", weight="bold"),
                    showarrow=False,
                )

                # Q label
                fig.add_annotation(
                    x=label_r * np.cos(ang_rad),
                    y=label_r * np.sin(ang_rad),
                    text=f"<b>Q{q+1}</b>",
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                    font=dict(size=14, color="black"),
                )

                # Value label (mean + sup/sub)
                fig.add_annotation(
                    x=value_r * np.cos(ang_rad),
                    y=value_r * np.sin(ang_rad),
                    text=txt,
                    showarrow=False,
                    xref=xref,
                    yref=yref,
                    font=dict(size=14, color="black"),
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
                            orientation="h",  # enable if your plotly supports it
                        ),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # ---------- aspect ratio / clipping ----------
        pad = 1.45
        x_range = [-r_outer * pad, r_outer * pad]
        y_range = [-r_outer * pad, r_outer * pad]

        for i in range(nplots):
            row = i // HORIZ_PLOTS + 1
            col = i % HORIZ_PLOTS + 1
            axis_num = i + 1
            y_anchor = _axis_id("y", axis_num)  # "y" for 1st, "y2" for 2nd...

            fig.update_xaxes(
                visible=False,
                range=x_range,
                scaleanchor=y_anchor,
                scaleratio=1,
                row=row,
                col=col,
            )
            fig.update_yaxes(visible=False, range=y_range, row=row, col=col)
        row_height = 160
        top_margin = 0
        bottom_margin = 10 if show_colorbar else 10
        fig.update_layout(
            height=row_height * rows,
            margin=dict(t=top_margin, b=bottom_margin, l=20, r=20),
            showlegend=False,
        )

        return fig
