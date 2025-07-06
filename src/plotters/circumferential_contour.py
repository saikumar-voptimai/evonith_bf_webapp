import numpy as np
import pytz
from .base_contour import BasePlotter
from src.config.config_loader import load_config
from scipy.interpolate import CubicSpline, interp1d
from plotly.subplots import make_subplots
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
    if vmax == vmin:
        frac = 0.5
    else:
        frac = (val - vmin) / (vmax - vmin)
    # Clamp to [0, 1] and handle NaN/inf
    if not np.isfinite(frac):
        frac = 0.5
    frac = min(max(frac, 0.0), 1.0)
    return pc.sample_colorscale(colorscale, [frac])[0]

def interpolate_fields(theta_grid, angles, field_values) -> Dict[int, float]:
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
    interpolated_values = cs(theta_grid)
    interp_dict = {int(theta): val for theta, val in zip(theta_grid, interpolated_values)}
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
    
    def plot_circumferential_quadrants(self, field_values_fulllist, titles, r_inner=1, r_outer=5,
                                        colorscale='YlOrRd', colorbar_title="Heat Load (GJ)", unit="GJ"):
        """
        Plot circumferential quadrants with interpolated values.
        Args:
            field_values_fulllist (list): List of lists containing - field values, max, and min for each quadrant.
            For each row:
            [[field_values_Q1, field_values_Q2, field_values_Q3, field_values_Q4],
             [field_values_max_Q1, field_values_max_Q2, field_values_max_Q3, field_values_max_Q4],
             [field_values_min_Q1, field_values_min_Q2, field_values_min_Q3, field_values_min_Q4]],

            titles (list): Titles for each subplot.
            r_inner (float): Inner radius of the plot.
            r_outer (float): Outer radius of the plot.
            colorscale (str): Colorscale for the plot.
            colorbar_title (str): Title for the colorbar.
            unit (str): Unit of measurement for the field values.
        Returns:
            fig (plotly.graph_objs.Figure): Plotly figure object with the circumferential quadrants.
        """
        resolution = 360
        theta_interp = np.linspace(0, 360, resolution)
        circle_theta = np.linspace(0, 2*np.pi, 361)

        # Interpolate all values and find global color scale range
        all_interp_vals = []
        field_exact_values_list = []
        unshifted_fixed_angles = [0, 90, 180, 270]
        fixed_angles = [45, 135, 225, 315]  # Fixed angles for quadrants
        field_values_list = [item[0] for item in field_values_fulllist] # Extract field values for each row
        field_values_maxlist = [item[1] for item in field_values_fulllist] # Extract max field values for each row
        field_values_minlist = [item[2] for item in field_values_fulllist] # Extract min field values for each row
        
        newfield_list = []
        for field in [field_values_list, 
                       field_values_maxlist, 
                       field_values_minlist]:
            newval = []
            for values in field:
                values.append(values[0])
                newval.append([0.5*(values[i] + values[i + 1]) for i in range(len(values)-1)])
            newfield_list.append(newval)

        all_vals = np.concatenate([np.array(newval) for newval in newfield_list])
        vmin, vmax = all_vals.min(), all_vals.max()

        # Create subplots
        fig = make_subplots(rows=1, cols=len(field_values_fulllist), horizontal_spacing=0.05)
        for idx, (field_values, 
                  fieldmax_values, 
                  fieldmin_values) in enumerate(zip(field_values_list, field_values_maxlist, field_values_minlist)):
            unshifted_fixed_angles = [0, 90, 180, 270, 360]
            all_temps_interp = list(interpolate_fields(theta_interp, unshifted_fixed_angles, field_values).values())
            temps_interp_dict = interpolate_fields(theta_interp, unshifted_fixed_angles, fieldmax_values)
            temps_interp = [v for k, v in temps_interp_dict.items() if k in fixed_angles]
            tempsmax_interp_dict = interpolate_fields(theta_interp, unshifted_fixed_angles, fieldmax_values)
            tempsmax_interp = [v for k, v in tempsmax_interp_dict.items() if k in fixed_angles]
            tempsmin_interp_dict = interpolate_fields(theta_interp, unshifted_fixed_angles, fieldmin_values)
            tempsmin_interp = [v for k, v in tempsmin_interp_dict.items() if k in fixed_angles]
            for i in range(len(theta_interp) - 1):
                theta0, theta1 = theta_interp[i], theta_interp[i+1]
                r = [r_inner, r_outer, r_outer, r_inner, r_inner]
                t = [theta0, theta0, theta1, theta1, theta0]
                x = [r[j]*np.cos(np.deg2rad(t[j])) for j in range(5)]
                y = [r[j]*np.sin(np.deg2rad(t[j])) for j in range(5)]
                color = value_to_rgba(all_temps_interp[i], vmin, vmax, colorscale)
                # Main filled polygon (no hover)
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=0.2),
                    showlegend=False,
                    hoverinfo='skip',
                ), row=1, col=idx+1)
                # Invisible marker for hover
                fig.add_trace(go.Scatter(
                    x=[np.mean(x)], y=[np.mean(y)],
                    mode='markers',
                    marker=dict(opacity=0),
                    showlegend=False,
                    hovertemplate=f"θ: {theta0:.1f}°<br>{colorbar_title[:-3]}: {all_temps_interp[i]:.2f} {unit}",
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
            theta_step = 360 / (len(field_values)-1)
            for i, (v, v_max, v_min) in enumerate(zip(temps_interp, 
                                                      tempsmax_interp, 
                                                      tempsmin_interp)):
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
                    font=dict(size=14, color='black', weight='bold'),
                    xref=f"x{idx+1}", yref="y"
                )
                fig.add_annotation(
                    x=value_r * np.cos(angle_rad) + 0.2,
                    y=value_r * np.sin(angle_rad) + 0.8,
                    text=f"+{int(12*(v_max-v))}",
                    showarrow=False,
                    font=dict(size=10, color='white', weight='bold'),
                    xref=f"x{idx+1}", yref="y"
                )
                fig.add_annotation(
                    x=value_r * np.cos(angle_rad) + 0.1,
                    y=value_r * np.sin(angle_rad) - 0.7,
                    text=f"{int(10*(v_min-v))}",
                    showarrow=False,
                    font=dict(size=12, color='green', weight='bold'),
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
            # plot_bgcolor='white',
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

