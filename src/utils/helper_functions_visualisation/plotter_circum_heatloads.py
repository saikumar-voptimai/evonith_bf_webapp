import os
import pickle
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import CubicSpline
from pathlib import Path

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
    cbar.set_label("Heat Load (kW)")

    # Add quadrant labels
    quadrant_labels = ["Q" + str(i+1) for i in range(len(field_values)) ]
    quadrant_angles = ext_angles[1:-1] # Corresponding angles in degrees
    for label, angle in zip(quadrant_labels, quadrant_angles):
        ax.text(np.radians(angle), r_outer * (1 + 0.15), label, color="black", fontsize=8, fontweight="bold", ha='center', va='center')

    # Add heatload values
    for value, angle in zip(field_values, quadrant_angles):
        ax.text(np.radians(angle), r_outer * (1 - 0.25), f"{int(value)} kW", color="black", fontsize=8, fontweight="bold", ha='center', va='center')

    fig.suptitle(title, y=0.05, fontsize=6)
    fig.set_size_inches(3, 3)
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
