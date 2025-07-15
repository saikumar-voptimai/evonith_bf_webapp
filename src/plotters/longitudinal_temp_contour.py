import numpy as np
import matplotlib.pyplot as plt
from .base_contour import BasePlotter

import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
    
    def plot_plotly(self, temperatures_list, temperatures_max_list, temperatures_min_list):
        """ Plot longitudinal temperature distribution using Plotly.
        Args:
            temperatures_list (list): List of temperature profiles for each quadrant.
            temperatures_max_list (list): List of maximum temperatures for each quadrant.
            temperatures_min_list (list): List of minimum temperatures for each quadrant.
        Returns:
            plotly.graph_objs.Figure: Plotly figure object.
        """
        X, Y = self.furnace.X, self.furnace.Y
        heights = self.furnace.get_heights()
        regions = self.furnace.get_regions()
        mask = self.generate_mask("cartesian")

        x_grid = X[0, :]
        y_grid = Y[:, 0]

        step = 100
        all_temps = np.concatenate(temperatures_list)
        min_val = min(all_temps) // step
        max_val = max(all_temps) // step
        
        if np.isnan(min_val) or np.isnan(max_val):
            # Probable data missing error
            raise ValueError('Missing data in DB - Try different time range')
        vmin = int(min_val) * step
        vmax = int(max_val) * step

        fig = make_subplots(rows=1, cols=len(temperatures_list), shared_yaxes=True, 
                            horizontal_spacing=0.02, column_widths=[0.3, 0.22, 0.22, 0.22])
        # Iterate over each quadrant's temperatures
        for idx, (temperatures, temperatures_max, temperatures_min) in enumerate(zip(temperatures_list, 
                                                                   temperatures_max_list,
                                                                   temperatures_min_list)):
            temp_interp = np.interp(y_grid, heights, temperatures)
            tempmax_interp = np.interp(y_grid, heights, temperatures_max)
            tempmin_interp = np.interp(y_grid, heights, temperatures_min)

            Z = np.tile(temp_interp[:, np.newaxis], (1, len(x_grid)))
            Z[~mask] = np.nan

            fig.add_trace(go.Heatmap(
                z=Z,
                x=x_grid,
                y=y_grid,
                zmin=vmin,
                zmax=vmax,
                colorscale='Viridis',
                colorbar=dict(
                    title='Temperature (°C)',
                    ticks='outside',
                    tickvals=np.arange(vmin, vmax+step, step),
                ) if idx == len(temperatures_list) - 1 else None,
                showscale=(idx == len(temperatures_list) - 1)
            ), row=1, col=idx + 1)

            # Add furnace boundary
            boundary_x = [-4] + [p[0] for p in self.furnace.geometry_points] + [0, 0, -4]
            boundary_y = [0] + [p[1] for p in self.furnace.geometry_points] + [20, 0, 0]
            fig.add_trace(go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ), row=1, col=idx + 1)

            # Iterate over regions to add labels and temperature values
            for i, (region_name, region_y) in enumerate(regions):
                temp_val = float(np.interp(region_y, y_grid, temp_interp))
                tempmax_val = float(np.interp(region_y, y_grid, tempmax_interp))
                tempmin_val = float(np.interp(region_y, y_grid, tempmin_interp))
                if idx == 0:
                    fig.add_annotation(
                        text=region_name,
                        x=-5.5, y=region_y,
                        xref=f'x{idx+1}', yref='y',
                        font=dict(size=15, color='black', weight='bold'),
                        showarrow=False
                    )
                fig.add_annotation(
                    text=f"{temp_val:.1f}°C",
                    x=-1.8, y=region_y,
                    xref=f'x{idx+1}', yref='y',
                    font=dict(size=15, color='white', weight='bold'),
                    showarrow=False
                )
                fig.add_annotation(
                    text=f"+{(tempmax_val - temp_val):.1f}°C",
                    x=-1.2, y=region_y+0.5,
                    xref=f'x{idx+1}', yref='y',
                    font=dict(size=12, color='red', weight='bold'),
                    showarrow=False
                )
                fig.add_annotation(
                    text=f"{(tempmin_val - temp_val):.1f}°C",
                    x=-1.2, y=region_y-0.5,
                    xref=f'x{idx+1}', yref='y',
                    font=dict(size=12, color='green', weight='bold'),
                    showarrow=False
                )

            fig.add_annotation(
                text=f"Q{idx + 1}",
                xref=f'x{idx+1}', yref='paper',
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=20, color='black', weight='bold')
            )

        # Hide axes and finalize layout
        for idx in range(len(temperatures_list)):
            fig.update_xaxes(visible=False, row=1, col=idx + 1)
            fig.update_yaxes(visible=False, row=1, col=idx + 1, range=[4, 20])

        fig.update_layout(
            height=400,
            margin=dict(t=40, b=20, l=20, r=20),
            plot_bgcolor='white'
        )
        return fig