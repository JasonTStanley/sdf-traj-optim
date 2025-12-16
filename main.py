import matplotlib
import numpy as np
from attrs import define, field
from matplotlib.axes import Axes

from obstacles import *
from sdf_env import SDF, Boundary, Environment

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Define obstacles
    circle = Circle(center=np.array([0.0, 0.0]), radius=1.0)
    rectangle = Rectangle(
        center=np.array([0.0, 2.0]), half_extents=np.array([1.0, 0.5])
    )
    boundary = Boundary(x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0)
    # Create SDF with obstacles
    sdf = SDF(obstacles=[circle, rectangle], bounds=boundary)

    # Create environment
    env = Environment(bounds=boundary, sdf=sdf)

    # Plot environment

    fig, ax = plt.subplots(figsize=(10, 10))
    # env.compute_sdf(grid_size=(100, 100))
    env.plot(ax)  # , view_sdf_grid=True)

    plt.show()
