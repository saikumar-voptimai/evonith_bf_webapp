"""Polar Matplotlib heat-load contour plotter.

Provides a simple polar contour plot of per-quadrant heat load values.
For the Plotly-based multi-row circumferential view, use
:class:`~plotters.circumferential_contour.CircumferentialPlotter`.
"""

import matplotlib.pyplot as plt
import numpy as np

from .base_contour import BasePlotter


class HeatLoadPlotter(BasePlotter):
    """Matplotlib polar plotter for heat load per quadrant.

    Inherits mask and colorbar helpers from :class:`~plotters.base_contour.BasePlotter`.
    """

    def plot(self, heatload_values, grid):
        """
        Plot heat load distribution.

        Args:
            heatload_values (list): Heatload values for each quadrant.
            grid (tuple): (R, Θ) meshgrid.

        Returns:
            Matplotlib figure.
        """
        R, THETA = grid
        THETA_grid = np.linspace(0, 2 * np.pi, len(heatload_values))
        heat_interpolated = np.interp(THETA[0, :], THETA_grid, heatload_values)
        Z = np.tile(heat_interpolated, (len(R), 1)).T

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        contour = ax.contourf(Θ, R, Z, levels=100, cmap="hot")
        ax.set_rticks([])

        self.add_colorbar(fig, contour, ax, label="Heat Load (GJ)")
        return fig
