import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from matplotlib.path import Path as plotter_path
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

# Load configuration settings
with open("src/config/setting.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract geometry and grid configuration from the YAML file
geometry_points = config["plot"]["geometry"]["geometry_points"]
geometry_points = [tuple(point) for point in geometry_points]
heights = config["plot"]["geometry"]["heights"][0]

X_GRID_LOWLIM = float(config["plot"]["contour"]["furnace_grid_X_low"])
X_GRID_HIGHLIM = float(config["plot"]["contour"]["furnace_grid_X_high"])
Y_GRID_LOWLIM = float(config["plot"]["contour"]["furnace_grid_Y_low"])
Y_GRID_HIGHLIM = float(config["plot"]["contour"]["furnace_grid_Y_high"])
GRID_SPACING = int(config["plot"]["contour"]["grid_divisions"])

MASK_PATH = config["paths"]["geometry"]

# Generate x and y grid points
x_grid = np.linspace(X_GRID_LOWLIM, X_GRID_HIGHLIM, GRID_SPACING)
y_grid = np.linspace(Y_GRID_LOWLIM, Y_GRID_HIGHLIM, GRID_SPACING)

X, Y = np.meshgrid(x_grid, y_grid)


def generate_mask(x_grid, y_grid, geometry_points, mask_file="mask_longitudinal.pkl"):
    """
    Generate or load a mask for the furnace geometry.

    The mask determines if a point in the meshgrid is inside the defined geometry.
    If the mask file exists, it is loaded instead of generating a new mask.

    Args:
        x_grid (ndarray): 1D array of x-coordinates.
        y_grid (ndarray): 1D array of y-coordinates.
        geometry_points (list): Points defining the geometry boundary.
        mask_file (str): Path to save or load the mask file.

    Returns:
        ndarray: Boolean mask where True indicates points inside the geometry.
    """
    fullpath = Path(__file__).resolve().parents[2] / "geometries" / mask_file
    if os.path.exists(fullpath):
        with open(fullpath, "rb") as f:
            return pickle.load(f)

    # Generate meshgrid and create mask using the geometry
    points = geometry_points + [(X_GRID_HIGHLIM, Y_GRID_HIGHLIM), (X_GRID_HIGHLIM, Y_GRID_LOWLIM), (X_GRID_LOWLIM, Y_GRID_LOWLIM)]  # Close the polygon
    path = plotter_path(points)
    mask = path.contains_points(np.column_stack((X.flatten(), Y.flatten()))).reshape(X.shape)

    # Save mask to file
    with open(fullpath, "wb") as f:
        pickle.dump(mask, f)

    return mask


def plotter_longitudinal_temp(temperatures_list):
    """
    Plot contour maps for multiple furnaces along a longitudinal layout.

    Args:
        temperatures_list (list): List of temperature profiles for each furnace.

    Returns:
        list: List of Matplotlib figures.
    """
    mask = generate_mask(x_grid, y_grid, geometry_points)
    num_plots = len(temperatures_list)

    # Define regions and their approximate y-locations
    regions = [
        ("Hearth", 4.5),
        ("Tuyere", 6.5),
        ("Bosh", 9.0),
        ("Belly", 12.0),
        ("Stack", 17.0)
    ]

    # Compute global colorbar limits
    all_temperatures = np.concatenate(temperatures_list)
    colorbar_min, colorbar_max = np.min(all_temperatures), np.max(all_temperatures)

    fig, ax = plt.subplots(1, 4, sharey=True, width_ratios=[0.2, 0.2, 0.2, 0.25])
    fig.set_figheight(3)
    for i, temperatures in enumerate(temperatures_list):
        temp_interpolated = np.interp(y_grid, heights, temperatures)

        Z = np.zeros((len(y_grid), len(x_grid)))
        for j in range(len(y_grid)):
            Z[j, :] = temp_interpolated[j]
        Z[~mask] = np.nan
        contour = ax[i].contourf(X, Y, Z, levels=100, cmap='viridis', vmin=colorbar_min, vmax=colorbar_max)

        boundary_x = [-4] + [p[0] for p in geometry_points] + [0, 0, -4]
        boundary_y = [0] + [p[1] for p in geometry_points] + [20, 0, 0]
        ax[i].plot(boundary_x, boundary_y, color='black', linewidth=2)
        ax[i].set_ylim(4, 20)
        ax[i].axis('off')
        ax[i].set_title(f"Q{str(i + 1)}", fontsize=6, fontweight='bold')

        for region_name, region_y in regions:
            temp_at_region = np.interp(region_y, y_grid, temp_interpolated)
            if i == 0:
                ax[i].text(-5.5, region_y, region_name, color='black', fontsize=10, fontweight='bold', va='center')
            ax[i].text(-2.7, region_y, f"{temp_at_region:.1f}°C", color='white', fontsize=8, fontweight='bold', va='center')

        # Add colorbar only for the last plot

    cbar = fig.colorbar(contour, ax=ax[num_plots-1], shrink=0.8, pad=0.04)
    cbar.set_label("Temperature (°C)")
    cbar_ticks = np.linspace(np.nanmin(colorbar_min), np.nanmax(colorbar_max), 5)
    cbar.set_ticks(cbar_ticks)
    return fig

def plotly_longitudinal_temp_plotly(temperatures_list, x_grid, y_grid, heights, geometry_points, generate_mask):
    """
    Plot contour maps for multiple furnaces along a longitudinal layout using Plotly.

    Args:
        temperatures_list (list): List of temperature profiles for each furnace.
        x_grid, y_grid (1D arrays): Grid points.
        heights (1D array): Original height positions corresponding to each temperature reading.
        geometry_points (list of tuples): Points defining the furnace boundary.
        generate_mask (function): Function to return a 2D mask for the furnace interior.

    Returns:
        Plotly Figure object.
    """
    # Generate geometry mask
    mask = generate_mask(x_grid, y_grid, geometry_points)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Define furnace regions for annotation
    regions = [
        ("Hearth", 4.5),
        ("Tuyere", 6.5),
        ("Bosh", 9.0),
        ("Belly", 12.0),
        ("Stack", 17.0)
    ]

    num_plots = len(temperatures_list)
    fig = make_subplots(rows=1, cols=num_plots, shared_yaxes=True, horizontal_spacing=0.02)

    all_temperatures = np.concatenate(temperatures_list)
    vmin, vmax = np.nanmin(all_temperatures), np.nanmax(all_temperatures)

    for i, temperatures in enumerate(temperatures_list):
        # Interpolate along y_grid
        temp_interpolated = np.interp(y_grid, heights, temperatures)

        # Form 2D Z array
        Z = np.tile(temp_interpolated[:, np.newaxis], (1, len(x_grid)))
        Z[~mask] = np.nan

        # Add contour plot as heatmap
        heatmap = go.Heatmap(
            z=Z,
            x=x_grid,
            y=y_grid,
            colorscale='Viridis',
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Temperature (°C)") if i == num_plots - 1 else None,
            showscale=(i == num_plots - 1)
        )

        fig.add_trace(heatmap, row=1, col=i+1)

        # Plot furnace boundary
        boundary_x = [-4] + [p[0] for p in geometry_points] + [0, 0, -4]
        boundary_y = [0] + [p[1] for p in geometry_points] + [20, 0, 0]

        fig.add_trace(
            go.Scatter(
                x=boundary_x, y=boundary_y,
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ),
            row=1, col=i+1
        )

        # Add region labels and interpolated values
        annotations = []
        for region_name, region_y in regions:
            temp_val = float(np.interp(region_y, y_grid, temp_interpolated))
            if i == 0:
                annotations.append(dict(
                    x=-5.5, y=region_y,
                    text=region_name,
                    font=dict(size=10, color='black'),
                    showarrow=False,
                    xref=f"x{i+1}", yref=f"y"
                ))
            annotations.append(dict(
                x=-2.7, y=region_y,
                text=f"{temp_val:.1f}°C",
                font=dict(size=8, color='white'),
                showarrow=False,
                xref=f"x{i+1}", yref=f"y"
            ))

        fig.update_layout(annotations=fig.layout.annotations + tuple(annotations))

        # Add quadrant title
        fig.add_annotation(
            text=f"Q{i+1}",
            xref=f"x{i+1} domain",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=10, color='black')
        )

    # Layout settings
    fig.update_layout(
        height=400,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor='white',
    )

    # Hide axes
    for i in range(num_plots):
        fig.update_xaxes(visible=False, row=1, col=i+1)
        fig.update_yaxes(visible=(i==0), row=1, col=i+1, range=[4, 20])

    return fig
