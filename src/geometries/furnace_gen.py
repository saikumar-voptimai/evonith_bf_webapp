"""Furnace geometry: grid generation and boundary masking.

The :class:`Furnace` class reads geometry parameters from
``config/setting_ds_dv.yml`` and builds Cartesian and polar meshgrids
that are used by all contour plotters.  Masks that exclude points outside
the furnace profile are cached as pickle files to avoid recomputation.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import yaml

from config.config_loader import load_config

config = load_config()


class Furnace:
    """Furnace geometry container with Cartesian and polar mesh support.

    Reads geometry points, height arrays, and region boundaries from the
    application config file and exposes meshgrids plus mask generation
    helpers to the plotter classes.

    Attributes:
        x_grid_lowlim:   Left (negative) boundary of the Cartesian X grid (m).
        x_grid_highlim:  Right boundary of the Cartesian X grid (m).
        y_grid_lowlim:   Bottom boundary of the Cartesian Y grid (m).
        y_grid_highlim:  Top boundary of the Cartesian Y grid (m).
        grid_spacing:    Number of grid divisions along each Cartesian axis.
        r_inner:         Inner radius of the polar ring (m).
        r_outer:         Outer radius of the polar ring (m).
        radial_steps:    Number of radial divisions in the polar grid.
        theta_steps:     Number of angular divisions in the polar grid.
        geometry_points: List of (x, y) tuples defining the furnace profile outline.
        heights:         List of sensor elevation arrays for longitudinal plots.
        regions:         List of ``(region_name, y_elevation)`` region boundary pairs.
        X:               Cartesian X meshgrid (shape: grid_spacing × grid_spacing).
        Y:               Cartesian Y meshgrid (shape: grid_spacing × grid_spacing).
        r_mesh:          Polar radial meshgrid.
        theta_mesh:      Polar angular meshgrid.
        mask_path:       Directory path where mask pickle files are saved.
    """

    def __init__(self):
        """Load geometry configuration and build initial meshgrids."""
        # Cartesian grid configuration
        self.x_grid_lowlim = float(config["plot"]["contour"]["furnace_grid_X_low"])
        self.x_grid_highlim = float(config["plot"]["contour"]["furnace_grid_X_high"])
        self.y_grid_lowlim = float(config["plot"]["contour"]["furnace_grid_Y_low"])
        self.y_grid_highlim = float(config["plot"]["contour"]["furnace_grid_Y_high"])
        self.grid_spacing = int(config["plot"]["contour"]["grid_divisions"])

        # Polar grid configuration
        self.r_inner = float(config["plot"]["circular"]["r_inner"])
        self.r_outer = float(config["plot"]["circular"]["r_outer"])
        self.radial_steps = int(config["plot"]["circular"]["radial_steps"])
        self.theta_steps = int(config["plot"]["circular"]["theta_steps"])

        # Geometry points
        self.geometry_points = [
            tuple(point) for point in config["plot"]["geometry"]["geometry_points"]
        ]
        self.heights = config["plot"]["geometry"]["heights"][0]
        self.regions = config["plot"]["geometry"]["regions"]  # Region names and heights

        # Mesh placeholders
        x_grid = np.linspace(self.x_grid_lowlim, self.x_grid_highlim, self.grid_spacing)
        y_grid = np.linspace(self.y_grid_lowlim, self.y_grid_highlim, self.grid_spacing)

        self.X, self.Y = np.meshgrid(x_grid, y_grid)
        # Generate radius and theta grid points
        self.radius_grid = np.linspace(0, self.r_outer, self.radial_steps)
        self.theta_grid = np.radians(np.linspace(0, 360, self.theta_steps))

        self.r_mesh, self.theta_mesh = np.meshgrid(
            self.radius_grid, self.theta_grid, indexing="ij"
        )

        # Mask path
        self.mask_path = config["paths"]["geometry"]

    def get_heights(self):
        """
        Retrieve the heights for longitudinal sensor arrays.
        """
        return self.heights

    def get_regions(self):
        """Return the list of ``(region_name, y_elevation)`` region boundary pairs."""
        return self.regions

    def generate_cartesian_mesh(self):
        """
        Generate Cartesian meshgrid (X, Y).
        """
        x_grid = np.linspace(self.x_grid_lowlim, self.x_grid_highlim, self.grid_spacing)
        y_grid = np.linspace(self.y_grid_lowlim, self.y_grid_highlim, self.grid_spacing)
        self.X, self.Y = np.meshgrid(x_grid, y_grid)
        return self.X, self.Y

    def generate_polar_mesh(self):
        """
        Generate polar meshgrid (R, Θ).
        """
        radius_grid = np.linspace(0, self.r_outer, self.radial_steps)
        theta_grid = np.radians(np.arange(0, self.theta_steps + 1, 1))
        self.r_mesh, self.theta_mesh = np.meshgrid(radius_grid, theta_grid)
        return self.r_mesh, self.theta_mesh

    def load_or_generate_mask(self, grid_type="cartesian", mask_file="mask.pkl"):
        """
        Load or generate a mask for the furnace geometry.

        Args:
            grid_type (str): "cartesian" or "polar".
            mask_file (str): Path to the mask file.

        Returns:
            ndarray: Boolean mask.
        """
        fullpath = Path(self.mask_path) / mask_file
        if fullpath.exists():
            with open(fullpath, "rb") as f:
                return pickle.load(f)

        if grid_type == "cartesian":
            if self.X is None or self.Y is None:
                self.generate_cartesian_mesh()
            points = self.geometry_points + [(self.x_grid_highlim, self.y_grid_highlim)]
            from matplotlib.path import Path as plotter_path

            path = plotter_path(points)
            mask = path.contains_points(
                np.column_stack((self.X.flatten(), self.Y.flatten()))
            ).reshape(self.X.shape)
        elif grid_type == "polar":
            if self.r_mesh is None or self.theta_mesh is None:
                self.generate_polar_mesh()
            mask = (self.r_mesh >= self.r_inner) & (self.r_mesh <= self.r_outer)
        else:
            raise ValueError("Invalid grid_type. Must be 'cartesian' or 'polar'.")

        # Save mask to file for reuse
        with open(fullpath, "wb") as f:
            pickle.dump(mask, f)

        return mask
