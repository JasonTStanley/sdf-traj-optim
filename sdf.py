from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field
from matplotlib.axes import Axes

from obstacles import *


@define
class TrajPoint:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray


@define
class Trajectory:
    points: list[TrajPoint]


@define
class Boundary:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def mins(self):
        return self.x_min, self.y_min

    def maxs(self):
        return self.x_max, self.y_max
    # def generate_grid(self, grid_size: tuple[int, int]) -> np.ndarray:
    #     """
    #     Generate a grid of points within the specified bounds.
    #     """
    #     # ensure grid_size is valid
    #     assert (
    #         grid_size[0] > 0 and grid_size[1] > 0
    #     ), "Grid size must be positive integers."
    #     xMin, xMax = self.bounds.x_min, self.bounds.x_max
    #     yMin, yMax = self.bounds.y_min, self.bounds.y_max
    #     x = np.linspace(xMin, xMax, grid_size[1])
    #     y = np.linspace(yMin, yMax, grid_size[0])
    #     grid = np.array(np.meshgrid(x, y, indexing="ij")).T.reshape(-1, 2)
    #     return grid
    #
    # def compute_sdf(self, grid_size: tuple[int, int]) -> np.ndarray:
    #     """
    #     Compute the SDF value for each point in the grid.
    #     """
    #     grid = self.generate_grid(grid_size)
    #     self.sdf_grid = np.array([self.sdf.query(point) for point in grid])
    #     self.sdf_grid = self.sdf_grid.reshape(grid_size)
    #     return self.sdf_grid


        # if view_sdf_grid:
        #     assert self.sdf_grid is not None, "SDF grid has not been computed yet."
        #     pos_mask = self.sdf_grid >= 0
        #     neg_mask = self.sdf_grid < 0
        #
        #     # Create masked arrays
        #     pos_grid = np.ma.masked_where(
        #         ~pos_mask, self.sdf_grid
        #     )  # only positive values
        #     neg_grid = np.ma.masked_where(
        #         ~neg_mask, self.sdf_grid
        #     )  # only negative values
        #
        #     # Plot positive SDF
        #     im_pos = ax.imshow(
        #         pos_grid,
        #         extent=self.bounds.as_tuple(),
        #         origin="lower",
        #         cmap="viridis",
        #         alpha=1.0,
        #     )
        #
        #     # Plot negative SDF
        #     im_neg = ax.imshow(
        #         neg_grid,
        #         extent=self.bounds.as_tuple(),
        #         origin="lower",
        #         cmap="coolwarm",
        #         alpha=1.0,
        #     )
        #
        #     ax.set_title("Environment Visualization with SDF")
        #
        #     # Create two colorbars
        #     cbar_pos = plt.colorbar(im_pos, ax=ax, fraction=0.046, pad=0.184)
        #     cbar_pos.set_label("Positive SDF")
        #
        #     cbar_neg = plt.colorbar(im_neg, ax=ax, fraction=0.046, pad=0.08)
        #     cbar_neg.set_label("Negative SDF")


@define
class Environment:
    bounds: Boundary
    obstacles: []

    def plot_obstacles(self, ax: Axes):
        for obs in self.obstacles:
            obs.plot(ax)

    def plot(self, ax: Axes):
        ax.set_title("Environment Visualization")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_xlim(self.bounds.x_min, self.bounds.x_max)
        ax.set_ylim(self.bounds.y_min, self.bounds.y_max)
        self.plot_obstacles(ax)

@define
class SDF:
    env: Environment
    bounds: Boundary
    def query(self, pos: np.ndarray) -> float:
        val = np.inf
        for obs in self.env.obstacles:
            value = obs.check_sdf(pos)
            if value < val:
                val = value
        return val

    def query_with_deriv(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        val = np.inf
        closest_obs = None
        for obs in self.env.obstacles:
            value = obs.check_sdf(pos)
            if value < val:
                val = value
                closest_obs = obs

        return val, closest_obs.sdf_deriv(pos)

    def get_bounds(self) -> Boundary:
        return self.bounds

    def plot_sample(self, ax: Axes, sample: np.ndarray):
        sdf_val, deriv = self.query_with_deriv(sample)
        circ = patches.Circle(sample, sdf_val, color="k", fill=False, linewidth=1)
        ax.add_patch(circ)
        vec = -sdf_val * deriv
        ax.quiver(
            sample[0],
            sample[1],
            vec[0],
            vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="k",
            width=0.005,
        )

    def plot_samples(self, ax: Axes, samples: np.ndarray):
        for sample in samples:
            self.plot_sample(ax, sample)
