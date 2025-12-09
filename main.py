import matplotlib
import numpy as np
from attrs import define, field
from matplotlib.axes import Axes

from obstacles import *

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


@define
class TrajPoint:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray


@define
class Trajectory:
    points: list[TrajPoint]


@define
class SDF:
    obstacles: list[Obstacle]

    def query(self, pos: np.ndarray):
        val = np.inf
        for obs in self.obstacles:
            value = obs.check_sdf(pos)
            if value < val:
                val = value
        return val

    def plot(self, ax: Axes):
        for obs in self.obstacles:
            obs.plot(ax)


@define
class Environment:
    grid_size: tuple[int, int]  # (rows, cols)
    bounds: tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    sdf: SDF
    sdf_grid: np.ndarray = field(init=False, default=None)

    def generate_grid(self) -> np.ndarray:
        """
        Generate a grid of points within the specified bounds.
        """
        x = np.linspace(self.bounds[0], self.bounds[1], self.grid_size[1])
        y = np.linspace(self.bounds[2], self.bounds[3], self.grid_size[0])
        grid = np.array(np.meshgrid(x, y, indexing="ij")).T.reshape(-1, 2)
        return grid

    def compute_sdf(self) -> np.ndarray:
        """
        Compute the SDF value for each point in the grid.
        """
        grid = self.generate_grid()
        self.sdf_grid = np.array([self.sdf.query(point) for point in grid])
        self.sdf_grid = self.sdf_grid.reshape(self.grid_size)
        return self.sdf_grid

    def plot(self, ax: Axes):
        if self.sdf_grid is None:
            self.compute_sdf()

        pos_mask = self.sdf_grid >= 0
        neg_mask = self.sdf_grid < 0

        # Create masked arrays
        pos_grid = np.ma.masked_where(~pos_mask, self.sdf_grid)  # only positive values
        neg_grid = np.ma.masked_where(~neg_mask, self.sdf_grid)  # only negative values
        extent = env.bounds

        # Plot positive SDF
        im_pos = ax.imshow(
            pos_grid, extent=extent, origin="lower", cmap="viridis", alpha=1.0
        )

        # Plot negative SDF
        im_neg = ax.imshow(
            neg_grid, extent=extent, origin="lower", cmap="coolwarm", alpha=1.0
        )

        ax.set_title("Environment Visualization with SDF")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        # Create two colorbars
        cbar_pos = fig.colorbar(im_pos, ax=ax, fraction=0.046, pad=0.184)
        cbar_pos.set_label("Positive SDF")

        cbar_neg = fig.colorbar(im_neg, ax=ax, fraction=0.046, pad=0.08)
        cbar_neg.set_label("Negative SDF")


if __name__ == "__main__":
    # Define obstacles
    circle = Circle(center=np.array([0.0, 0.0]), radius=1.0)
    rectangle = Rectangle(
        center=np.array([0.0, 2.0]), half_extents=np.array([1.0, 0.5])
    )

    # Create SDF with obstacles
    sdf = SDF(obstacles=[circle, rectangle])

    # Create environment
    env = Environment(grid_size=(100, 100), bounds=(-3, 3, -3, 3), sdf=sdf)

    # Plot environment

    fig, ax = plt.subplots(figsize=(10, 10))
    env.plot(ax)
    plt.show()
