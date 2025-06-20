import os
import pickle
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import CubicSpline, interp1d
from pathlib import Path
import plotly.colors

# Load configuration settings
with open("src/config/setting.yaml", "r") as f:
    config = yaml.safe_load(f)

R_INNER = float(config["plot"]["circular"]["r_inner"])
R_OUTER = float(config["plot"]["circular"]["r_outer"])

RADIAL_STEPS = int(config["plot"]["circular"]["radial_steps"])
THETA_STEPS = int(config["plot"]["circular"]["theta_steps"])

MASK_PATH = config["paths"]["geometry"]

# Generate radius and theta grid points
radius_grid = np.linspace(0, R_OUTER, RADIAL_STEPS)
theta_grid = np.radians(np.arange(0, THETA_STEPS+1, 1))

r_mesh, theta_mesh = np.meshgrid(radius_grid, theta_grid)


def generate_mask(r_inner, r_outer, mask_file="mask_circular.pkl"):
    """
    Generate or load a mask for the furnace geometry.

    The mask determines if a point in the meshgrid is inside the defined geometry.
    If the mask file exists, it is loaded instead of generating a new mask.

    Args:
        r_inner (float): Inner radius.
        r_outer (ndarray): Outer radius.
        mask_file (str): Path to save or load the mask file.

    Returns:
        ndarray: Boolean mask where True indicates points inside the geometry.
    """
    fullpath = Path(__file__).resolve().parents[2] / "geometries" / mask_file

    if os.path.exists(fullpath):
        with open(fullpath, "rb") as f:
            return pickle.load(f)
    mask = np.zeros_like(r_mesh, dtype=bool)
    # Generate meshgrid and create mask using the geometry
    for i, r_val in enumerate(radius_grid):
        if (r_val <= r_outer) and (r_val >= r_inner):
            mask[:, i] = True
        else:
            mask[:, i] = False
    with open(fullpath, "wb") as f:
        pickle.dump(mask, f)
    return mask


def interpolate_fields(theta_grid, angles, field_values):
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
    interpolated_values = cs(theta_grid % 360)  # Ensure theta values are within 0-360 range
    return interpolated_values


def plotter_circum(field_values, fig, ax, r_inner=R_INNER, r_outer=R_OUTER, title="Heat Load Distribution"):
    """
    Plot contour maps for multiple furnaces along a longitudinal layout.

    Args:
        field_values (list): List of temperature/heatload values.
        angles (list): Angles where values are available,
        title (str): Title for the plot.

    Returns:
        plt.figure
    """
    ext_angles = _get_angles(field_values)
    ext_field_values = _get_ext_heatloads(field_values)
    mask = generate_mask(r_inner, r_outer)
    field_interpolated = np.interp(theta_grid, np.radians(ext_angles), ext_field_values, period=90)

    # Create a 2D array of field values and apply the mask
    Z = np.zeros((len(theta_grid), len(radius_grid)))
    for j in range(len(radius_grid)):
        Z[:, j] = field_interpolated
    Z[~mask] = np.nan

    # Add contour plot for the furnace
    contour = ax.contourf(theta_mesh, r_mesh, Z, 100)
    ax.set_rticks([])
    ax.set_xticks([])

    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, [R_INNER] * len(theta_circle), linestyle='-', color='black', linewidth=0.5)

    # Regularize colorbar ticks to even intervals
    cbar = fig.colorbar(contour, ax=ax, anchor=(0, 0.5), shrink=0.7)
    cbar_ticks = np.linspace(np.nanmin(Z), np.nanmax(Z), 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])
    cbar.set_label("Heat Load (GJ)")

    # Add quadrant labels
    quadrant_labels = ["Q" + str(i+1) for i in range(len(field_values)) ]
    quadrant_angles = ext_angles[1:-1] # Corresponding angles in degrees
    for label, angle in zip(quadrant_labels, quadrant_angles):
        ax.text(np.radians(angle), r_outer * (1 + 0.15), label, color="black", fontsize=8, fontweight="bold", ha='center', va='center')

    # Add heatload values
    for value, angle in zip(field_values, quadrant_angles):
        ax.text(np.radians(angle), r_outer * (1 - 0.25), f"{value:.2f} GJ", color="black", fontsize=8, fontweight="bold", ha='center', va='center')

    fig.suptitle(title, y=0.05, fontsize=6)
    fig.set_size_inches(3, 3)
    return fig


def plotter_circum_plotly(field_values, r_inner=R_INNER, r_outer=R_OUTER, title="Heat Load Distribution", colorscale='Viridis', resolution=200):
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
    n_theta = len(field_values)
    theta_step = 360 / n_theta
    angles_deg = np.array([theta_step/2 + theta_step*i for i in range(n_theta)])
    # Ensure wrap-around for periodic spline
    angles_deg = np.append(angles_deg, 360)
    values = np.append(field_values, field_values[0])
    theta_interp = np.linspace(0, 360, resolution)
    temps_interp = interpolate_fields(theta_interp, angles_deg, values)
    vmin, vmax = temps_interp.min(), temps_interp.max()
    # Get colorscale as list of tuples (fraction, color)
    colorscale_list = plotly.colors.get_colorscale(colorscale)
    def value_to_rgba(val):
        # Normalize
        frac = (val - vmin) / (vmax - vmin + 1e-8)
        # Interpolate color
        return plotly.colors.sample_colorscale(colorscale, [frac])[0]
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
            colorbar=dict(title="Heat Load (GJ)", thickness=20, x=1.05, y=.56, len=0.9),
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
            text=f"{v:.2f} GJ",
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


def _get_angles(field_values):
    theta_step = 360 / len(field_values)
    theta_values = [theta_step / 2 + theta_step * i for i in range(len(field_values))]
    theta_values = np.concatenate(([0], theta_values, [360]))
    return theta_values


def _get_ext_heatloads(field_values):
    return np.concatenate(([0.5 * (field_values[0] + field_values[-1])], 
                            field_values, 
                            [0.5 * (field_values[0] + field_values[-1])]))
