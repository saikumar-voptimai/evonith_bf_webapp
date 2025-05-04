import numpy as np
import matplotlib.pyplot as plt
from .base_contour import BasePlotter

class LongitudinalTemperaturePlotter(BasePlotter):
    def plot(self, temperatures_list):
        """
        Plot longitudinal temperature distribution.

        Args:
            temperatures_list (list): List of temperature profiles.
            geometry_points (list): Geometry of the furnace.
            grid (tuple): (X, Y) meshgrid.

        Returns:
            Matplotlib figure.
        """
        X, Y = self.furnace.X, self.furnace.Y
        heights = self.furnace.get_heights()
        regions = self.furnace.get_regions()

        # Compute global colorbar limits
        step = 100
        global_min = int(min(min(values) for values in temperatures_list)/step) * step
        global_max = int(max(max(values) for values in temperatures_list)/step) * step + step

        mask = self.generate_mask("cartesian")
        width_ratios = [1 for _ in range(len(temperatures_list)-1)]
        width_ratios.append(1)
        fig, axes = plt.subplots(1, len(temperatures_list), figsize=(12, 4), gridspec_kw={'width_ratios': width_ratios})
        
        for idx, (temperatures, ax) in enumerate(zip(temperatures_list, axes)):
            temp_interpolated = np.interp(Y[:, 0], heights, temperatures)

            Z = np.zeros_like(X)
            for j in range(len(temp_interpolated)):
                Z[j, :] = temp_interpolated[j]
            Z[~mask] = np.nan

            contour = ax.contourf(X, Y, Z, levels=100, cmap="viridis", vmin=global_min, vmax=global_max)
            ax.set_title(f"Q{idx + 1}", fontsize=10, fontweight="bold")
            ax.axis("off")
            
            # Add furnace boundaries
            boundary_x = [-4] + [p[0] for p in self.furnace.geometry_points] + [0, 0, -4]
            boundary_y = [0] + [p[1] for p in self.furnace.geometry_points] + [20, 0, 0]
            ax.plot(boundary_x, boundary_y, color="black", linewidth=2)

            ax.set_ylim(4, 20)  # Limit Y-axis range
            ax.axis("off")
            ax.set_title(f"Q{idx + 1}", fontsize=10, fontweight="bold")

                        # Add region labels for the first plot
            
            for region_name, region_y in regions:
                temp_at_region = np.interp(region_y, Y[:, 0], temp_interpolated)
                if idx == 0:
                    ax.text(-5.5, region_y, region_name, color="black", fontsize=10, fontweight="bold", va="center")
            
                ax.text(-2.7, region_y, f"{temp_at_region:.1f}°C", color="white", fontsize=10, fontweight="bold", va="center")
        
         # Create a dummy plot for the colorbar
        hidden_ax = fig.add_axes([0, 0, 0.01, 0.01])  # Completely hidden
        Z_hid = np.random.rand(200, 200) + global_max
        Z_hid[Z_hid < global_max + .5] = global_min
        vmin, vmax = global_min, global_max
        hidden_contour = hidden_ax.contourf(X, Y, Z_hid, levels=100, cmap='viridis', vmin=vmin, vmax=vmax)
        hidden_ax.remove()              

        fig.subplots_adjust(right=0.75)
        cbar_ax = fig.add_axes([0.8, 0.1, 0.025, 0.8]) # start x, start y, weight, height
        cbar = fig.colorbar(hidden_contour, cax=cbar_ax)
        cbar_ticks = np.arange(global_min, global_max+step, step)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])
        cbar.set_label("Temperature (°C)")
        return fig
