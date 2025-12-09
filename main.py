import numpy as np
from attrs import define, field
from obstacles import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
@define
class TrajPoint:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray

@define
class Trajectory:
    points: [TrajPoint]


@define
class SDF:
    obstacles: field(type=list[Obstacle])

    def query(self, pos:np.ndarray):
        val = np.inf
        for obs in self.obstacles:
            value = obs.check_sdf(pos)
            if value < val:
                val = value
        return val

    def plot(self, ax: plt.Axes):
        for obs in self.obstacles:
            obs.plot(ax)
@define
class Environment:
    grid_size: tuple[int, int]  # (rows, cols)
    bounds: tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    sdf: SDF

    def generate_grid(self) -> np.ndarray:
        """
        Generate a grid of points within the specified bounds.
        """
        x = np.linspace(self.bounds[0], self.bounds[1], self.grid_size[1])
        y = np.linspace(self.bounds[2], self.bounds[3], self.grid_size[0])
        grid = np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2)
        return grid

    def compute_sdf(self) -> np.ndarray:
        """
        Compute the SDF value for each point in the grid.
        """
        grid = self.generate_grid()
        sdf_values = np.array([self.sdf.query(point) for point in grid])
        return sdf_values.reshape(self.grid_size)

if __name__ == "__main__":
    # Define obstacles
    circle = Circle(center=np.array([0.0, 0.0]), radius=1.0)
    rectangle = Rectangle(center=np.array([0.0, 2.0]), half_extents=np.array([1.0, 0.5]))

    # Create SDF with obstacles
    sdf = SDF(obstacles=[circle, rectangle])

    # Create environment
    env = Environment(grid_size=(100, 100), bounds=(-3, 3, -3, 3), sdf=sdf)

    # Compute SDF values for the grid
    sdf_grid = env.compute_sdf()
    plt.imshow(sdf_grid, extent=env.bounds, origin='lower', cmap='viridis')
    sdf.plot(plt.gca())
    plt.colorbar(label='SDF Value')
    plt.title('Environment Visualization with SDF')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()