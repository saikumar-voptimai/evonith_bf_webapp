import numpy as np
import pytz
from .base_contour import BasePlotter
from config.loader import load_config
from scipy.interpolate import CubicSpline, interp1d
from plotly.subplots import make_subplots
import plotly.colors
import plotly.graph_objs as go

from typing import Dict, List
import plotly.colors as pc

config = load_config()

TIMEZONE = pytz.timezone('Asia/Kolkata')  # GMT+5:30


def interpolate_fields_new(query_theta, known_theta, values):
    """Periodic interpolation for angular data."""
    interp_func = interp1d(known_theta, values, kind='linear', fill_value="extrapolate", bounds_error=False)
    return interp_func(query_theta % 360)

def value_to_rgba(val, vmin, vmax, colorscale):
    frac = (val - vmin) / (vmax - vmin + 1e-8)
    return pc.sample_colorscale(colorscale, [frac])[0]

def interpolate_fields(theta_grid, angles, field_values, query_theta=None) -> Dict[int, float]:
    """
    Interpolate temperatures along the y-axis using periodic interpolation.

    Args:
        theta_grid (ndarray): 1D array of theta coordinates.
        angles (list): Angles where values are available.
        values (list): Temperature values at the respective angles.

    Returns:
        ndarray: Interpolated field values along the theta_grid.
    """
    # Create the CubicSpline interpolator with natural boundary conditions
    cs = CubicSpline(angles, field_values, bc_type='periodic')

    # Interpolate the values at the desired theta grid points
    # interp_func = interp1d(angles, field_values, kind='linear', fill_value="extrapolate", bounds_error=False)   # Ensure theta values are within 0-360 range
    interpolated_values = cs(theta_grid * 180/ (np.pi))
    interp_dict = {int(theta*180/(np.pi)): val for theta, val in zip(theta_grid, interpolated_values)}
    return interp_dict

class CircumferentialPlotter(BasePlotter):
    """
    A plotter for circumferential temperature or heat load distributions.
    Inherits geometry and mask functionality from BasePlotter.
    """

    def __init__(self, mask_file="mask_circumferential.pkl"):
        """
        Initialize the plotter.

        Args:
            config (dict): Configuration settings.
            mask_file (str): Path to the circumferential mask file.
        """
        super().__init__(mask_file=mask_file)
        self.r_mesh, self.theta_mesh = self.furnace.generate_polar_mesh()
        self.mask = self.generate_mask(grid_type="polar")
    
    def plotter_circum_plotly(self, field_values, title="Heat Load Distribution", colorscale='Viridis', resolution=200, 
                              colorbar_title = "Heat Load (GJ)", unit="GJ"):
        """
        Plot a concentric heat ring using Plotly (polar coordinates, true ring, with colorbar and quadrant labels).
        Args:
            field_values (list): List of temperature/heatload values (theta direction only, e.g. [Q1, Q2, Q3, Q4]).
            r_inner (float): Inner radius.
            r_outer (float): Outer radius.
            title (str): Title for the plot.
            colorscale (str): Plotly colorscale name.
            resolution (int): Number of angular steps for interpolation.
        Returns:
            plotly.graph_objects.Figure
        """
        r_inner = self.furnace.r_inner
        r_outer = self.furnace.r_outer
        n_theta = len(field_values)
        theta_step = 360 / n_theta
        angles_deg = np.array([theta_step/2 + theta_step*i for i in range(n_theta)])
        # Ensure wrap-around for periodic spline
        angles_deg = np.append(angles_deg, angles_deg[0])
        values = np.append(field_values, field_values[0])
        theta_interp = np.linspace(0, 360, resolution)
        temps_interp = interpolate_fields(theta_interp, angles_deg, values)
        vmin, vmax = temps_interp.min(), temps_interp.max()
        # Get colorscale as list of tuples (fraction, color)
        colorscale_list = plotly.colors.get_colorscale(colorscale)
        fig = go.Figure()
        # Draw each angular slice as a filled sector with correct color
        for i in range(len(theta_interp) - 1):
            theta0 = theta_interp[i]
            theta1 = theta_interp[i + 1]
            theta_rad0 = np.deg2rad(theta0)
            theta_rad1 = np.deg2rad(theta1)
            # Sector polygon (r, theta)
            r = [r_inner, r_outer, r_outer, r_inner, r_inner]
            theta = [theta0, theta0, theta1, theta1, theta0]
            # Convert to cartesian for fillcolor
            x = [r[j]*np.cos(np.deg2rad(theta[j])) for j in range(5)]
            y = [r[j]*np.sin(np.deg2rad(theta[j])) for j in range(5)]
            color = value_to_rgba(temps_interp[i])
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(color=color, width=0.5),
                hoverinfo='skip',
                showlegend=False
            ))
        # Add colorbar using an invisible scatter
        colorbar_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=colorbar_title, thickness=20, x=1.05, y=.56, len=0.9),
                showscale=True
            ),
            showlegend=False
        )
        fig.add_trace(colorbar_trace)
        # Add outer and inner boundary circles
        circle_theta = np.linspace(0, 2*np.pi, 361)
        fig.add_trace(go.Scatter(
            x=r_inner*np.cos(circle_theta),
            y=r_inner*np.sin(circle_theta),
            mode='lines',
            line=dict(color='black'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=r_outer*np.cos(circle_theta),
            y=r_outer*np.sin(circle_theta),
            mode='lines',
            line=dict(color='black'),
            showlegend=False
        ))
        # Add quadrant labels and values (cartesian positions)
        label_r = r_outer * 1.12
        value_r = r_outer * 0.7
        for i, v in enumerate(field_values):
            angle = theta_step/2 + theta_step*i
            angle_rad = np.deg2rad(angle)
            # Quadrant label
            fig.add_annotation(
                x=label_r * np.cos(angle_rad),
                y=label_r * np.sin(angle_rad),
                text=f"Q{i+1}",
                showarrow=False,
                font=dict(size=16, color="black", family="Arial Black"),
                xref="x", yref="y"
            )
            # Value label
            fig.add_annotation(
                x=value_r * np.cos(angle_rad),
                y=value_r * np.sin(angle_rad),
                text=f"{v:.2f} {unit}",
                showarrow=False,
                font=dict(size=16, color="black", family="Arial Black"),
                xref="x", yref="y"
            )
        fig.update_layout(
            title=dict(text=title, y=0.05, x=0.5, font=dict(size=14)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            width=350, height=350,
            autosize=False
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    def plot_circumferential_quadrants(self, field_values_list, titles, r_inner=2, r_outer=5,
                                        colorscale='Viridis', colorbar_title="Heat Load (GJ)", unit="GJ"):
        resolution = 200
        theta_interp = np.linspace(0, 360, resolution)
        circle_theta = np.linspace(0, 2*np.pi, 361)

        # Interpolate all values and find global color scale range
        all_interp_vals = []
        field_exact_values_list = []
        unshifted_fixed_angles = [0, 90, 180, 270]
        fixed_angles = [45, 135, 225, 315]  # Fixed angles for quadrants
        for values in field_values_list:
            theta_step = 360 / len(values)
            angles = np.append([theta_step*i for i in range(len(values))], 360)
            extended_values = np.append(values, values[0])
            temps_interp_dict = interpolate_fields(circle_theta, angles, extended_values)
            temps_interp = list(temps_interp_dict.values())
            temps_query = np.array([temps_interp_dict[theta] for theta in unshifted_fixed_angles])
            all_interp_vals.append(temps_interp)
            field_exact_values_list.append(temps_query)

        all_vals = np.concatenate(all_interp_vals)
        vmin, vmax = all_vals.min(), all_vals.max()

        # Create subplots
        fig = make_subplots(rows=1, cols=len(field_values_list), horizontal_spacing=0.05)
        for idx, (field_values, temps_interp) in enumerate(zip(field_exact_values_list, all_interp_vals)):
            for i in range(len(theta_interp) - 1):
                theta0, theta1 = theta_interp[i], theta_interp[i+1]
                r = [r_inner, r_outer, r_outer, r_inner, r_inner]
                t = [theta0, theta0, theta1, theta1, theta0]
                x = [r[j]*np.cos(np.deg2rad(t[j])) for j in range(5)]
                y = [r[j]*np.sin(np.deg2rad(t[j])) for j in range(5)]
                color = value_to_rgba(temps_interp[i], vmin, vmax, colorscale)
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=0.2),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=idx+1)

            # Add boundary circles
            for radius in [r_inner, r_outer]:
                fig.add_trace(go.Scatter(
                    x=radius * np.cos(circle_theta),
                    y=radius * np.sin(circle_theta),
                    mode='lines',
                    line=dict(color='black'),
                    showlegend=False
                ), row=1, col=idx+1)

            # Add annotations
            label_r = r_outer * 1.2
            value_r = r_outer * 0.7
            theta_step = 360 / len(field_values)
            for i, v in enumerate(field_values):
                angle = theta_step/2 + theta_step*i
                angle_rad = np.deg2rad(angle)
                fig.add_annotation(
                    x=label_r * np.cos(angle_rad),
                    y=label_r * np.sin(angle_rad),
                    text=f"Q{i+1}",
                    showarrow=False,
                    font=dict(size=14, color='black', weight='bold'),
                    xref=f"x{idx+1}", yref="y"
                )
                fig.add_annotation(
                    x=value_r * np.cos(angle_rad),
                    y=value_r * np.sin(angle_rad),
                    text=f"{v:.1f} {unit}",
                    showarrow=False,
                    font=dict(size=14, color='white', weight='bold'),
                    xref=f"x{idx+1}", yref="y"
                )

            # Add subplot title
            fig.add_annotation(
                text=titles[idx],
                x=0.5,
                y=1.1,
                xref=f"x{idx+1}",
                yref="paper",
                xanchor="center",
                font=dict(size=14, weight='bold', color='black'),
                showarrow=False
            )

        # Add shared colorbar using dummy scatter
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                color=[vmin],
                colorbar=dict(
                    title=colorbar_title,
                    x=0.5,
                    y=-0.3,
                    len=0.8,
                    xanchor='center',
                    thickness=15,
                    orientation='h',
                ),
                showscale=True
            ),
            showlegend=False
        ))

        # Layout cleanup
        fig.update_layout(
            height=400,
            margin=dict(t=80, b=100, l=20, r=20),
            plot_bgcolor='white',
            showlegend=False
        )
        x_range = [-r_outer * 1, r_outer * 1]
        y_range = [-r_outer * 1, r_outer * 1]
        for i in range(len(field_values_list)):
            fig.update_xaxes(visible=False, range=x_range, scaleanchor=f"y{i+1}",  row=1, col=i+1)
            fig.update_yaxes(visible=False, range=y_range, row=1, col=i+1)

        return fig

    
    @staticmethod
    def _get_angles(field_values):
      theta_step = 360 / len(field_values)
      theta_values = [theta_step / 2 + theta_step * i for i in range(len(field_values))]
      theta_values = np.concatenate(([0], theta_values, [360]))
      return theta_values

    @staticmethod
    def _get_ext_heatloads(field_values):
        return np.concatenate(([0.5 * (field_values[0] + field_values[-1])], 
                                field_values, 
                                [0.5 * (field_values[0] + field_values[-1])]))

