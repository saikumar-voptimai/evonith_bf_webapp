import numpy as np
import pytz
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
from geometries.furnace_gen import Furnace
from .base_contour import BasePlotter
from config.loader import load_config

config = load_config()

TIMEZONE = pytz.timezone('Asia/Kolkata')  # GMT+5:30

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

    def plot(self, field_values_list, titles=None, data_type="Value"):
        """
        Generate circumferential contour plots for multiple datasets using periodic cubic spline interpolation.

        Args:
            field_values_list (list): A list of datasets (each a list of values).
            titles (list): Titles for the plots.
            data_type (str): Type of data (e.g., "Temperature" or "Heat Load").

        Returns:
            plt.Figure: A matplotlib figure with the plots.
        """
        from scipy.interpolate import CubicSpline
        num_plots = len(field_values_list)
        width_ratios = [1 for _ in range(num_plots-1)]
        width_ratios.append(1.2)
        fig, axes = plt.subplots(1, int(num_plots), subplot_kw={'projection': 'polar'}, dpi=1200, sharey=True, sharex=True)

        if num_plots == 1:
            axes = [axes]
        step = 100
        global_min = int(min(min(values) for values in field_values_list)/step) * step
        global_max = int(max(max(values) for values in field_values_list)/step) * step + step
        contour = []
        for i, (field_values, ax) in enumerate(zip(field_values_list, axes.flat)):
            n_theta = len(field_values)
            theta_step = 360 / n_theta
            angles_deg = np.array([theta_step/2 + theta_step*j for j in range(n_theta)])
            # Periodic cubic spline interpolation
            angles_deg = np.append(angles_deg, 360)
            values = np.append(field_values, field_values[0])
            theta_interp = np.linspace(0, 360, self.theta_mesh.shape[0])
            cs = CubicSpline(angles_deg, values, bc_type='periodic')
            interpolated_values = cs(np.degrees(self.theta_mesh[:,0]) % 360)

            Z = np.zeros_like(self.r_mesh)
            for j in range(self.r_mesh.shape[1]):
                Z[:, j] = interpolated_values
            Z[~self.mask] = np.nan

            cont = ax.contourf(self.theta_mesh, self.r_mesh, Z, levels=100, cmap='viridis', vmin=global_min, vmax=global_max)
            contour.append(cont)
            theta_circle = np.linspace(0, 2 * np.pi, 100)
            ax.plot(theta_circle, [self.furnace.r_inner] * len(theta_circle), linestyle='-', color='black', linewidth=0.5)

            ax.set_rticks([])
            ax.set_xticks([])
            ax.set_title(titles[i] if titles else f"Plot {i+1}", va='bottom', fontsize=8, fontweight="bold")

            # Add data values and quadrant labels
            for idx, value in enumerate(field_values):
                angle = theta_step/2 + theta_step*idx
                ax.text(np.radians(angle), self.furnace.r_outer * 0.65, f"{value:.1f}", color="white", fontsize=8, fontweight="bold", ha='center', va='center')
                if i == 0:
                    ax.text(np.radians(angle), self.furnace.r_outer * 1.2, f"Q{idx+1}", color="black", fontsize=8, fontweight="bold", ha="center", va="center")

        # Create a dummy plot for the colorbar
        hidden_ax = fig.add_axes([0, 0, 0.01, 0.01])  # Completely hidden
        Z_hid = np.random.rand(361, 100) + global_max
        Z_hid[Z_hid < global_max + .5] = global_min
        vmin, vmax = global_min, global_max
        hidden_contour = hidden_ax.contourf(self.theta_mesh, self.r_mesh, Z_hid, levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
        hidden_ax.remove()              
        cbar_ax = fig.add_axes([0.14, 0.3, 0.75, 0.02]) # start x, start y, height, width
        cbar = fig.colorbar(hidden_contour, cax=cbar_ax, orientation='horizontal')
        cbar_ticks = np.arange(global_min, global_max+step, step)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])
        cbar.set_label("Temperature (°C)")
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

