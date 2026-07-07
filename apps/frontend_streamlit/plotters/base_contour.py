"""Abstract base class for all furnace contour plotters.

Provides shared helpers (mask generation, figure initialisation, colorbar)
that are inherited by :class:`~plotters.circumferential_contour.CircumferentialPlotter`,
:class:`~plotters.heatload_contour.HeatLoadPlotter`, and
:class:`~plotters.longitudinal_temp_contour.LongitudinalTemperaturePlotter`.
"""

import matplotlib.pyplot as plt

from apps.frontend_streamlit.geometries.furnace_gen import Furnace


class BasePlotter:
    """Base plotter that owns the :class:`~geometries.furnace_gen.Furnace` instance.

    Attributes:
        mask_file: Filename (relative to the geometry path) for the cached mask pickle.
        furnace:   :class:`~geometries.furnace_gen.Furnace` geometry instance.
    """

    def __init__(self, mask_file):
        """Initialise the plotter with a specific mask file.

        Args:
            mask_file: Filename of the geometry mask pickle (e.g.
                       ``"mask_circumferential.pkl"``).
        """
        self.mask_file = mask_file
        self.furnace = Furnace()

    def generate_mask(self, grid_type="cartesian"):
        """
        Delegate mask generation to the Furnace object.
        """
        return self.furnace.load_or_generate_mask(
            grid_type=grid_type, mask_file=self.mask_file
        )

    def initialize_figure(self, n_cols, height_ratios=None):
        """Create a 1-row Matplotlib figure with *n_cols* axes.

        Args:
            n_cols:        Number of subplot columns.
            height_ratios: Optional list of relative column widths, passed to
                           ``gridspec_kw['width_ratios']``.

        Returns:
            Tuple ``(fig, ax)`` — the :class:`~matplotlib.figure.Figure` and
            array of :class:`~matplotlib.axes.Axes`.
        """
        fig, ax = plt.subplots(
            1,
            n_cols,
            figsize=(4 * n_cols, 4),
            gridspec_kw={"width_ratios": height_ratios},
        )
        return fig, ax

    def add_colorbar(self, fig, contour, ax, label="Value", shrink=0.8, pad=0.05):
        """Attach a colorbar to *ax* using an existing contour set.

        Args:
            fig:     Parent :class:`~matplotlib.figure.Figure`.
            contour: Filled contour object returned by ``ax.contourf()``.
            ax:      Target :class:`~matplotlib.axes.Axes` for the colorbar.
            label:   Colorbar axis label text.
            shrink:  Fraction of the parent axes height to use (0–1).
            pad:     Spacing between colorbar and parent axes.

        Returns:
            The :class:`~matplotlib.colorbar.Colorbar` instance.
        """
        cbar = fig.colorbar(contour, ax=ax, shrink=shrink, pad=pad)
        cbar.set_label(label)
        return cbar
