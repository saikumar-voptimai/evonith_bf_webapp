import matplotlib.pyplot as plt
from geometries.furnace_gen import Furnace

class BasePlotter:
    def __init__(self, mask_file):
        self.mask_file = mask_file
        self.furnace = Furnace()

    def generate_mask(self, grid_type="cartesian"):
        """
        Delegate mask generation to the Furnace object.
        """
        return self.furnace.load_or_generate_mask(grid_type=grid_type, mask_file=self.mask_file)
    
    def initialize_figure(self, n_cols, height_ratios=None):
        fig, ax = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), gridspec_kw={'width_ratios': height_ratios})
        return fig, ax

    def add_colorbar(self, fig, contour, ax, label="Value", shrink=0.8, pad=0.05):
        cbar = fig.colorbar(contour, ax=ax, shrink=shrink, pad=pad)
        cbar.set_label(label)
        return cbar
