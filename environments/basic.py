import numpy as np
from attrs import define, field
from matplotlib.axes import Axes

from obstacles import *
from sdf import SDF, Boundary, Environment

@define
class BasicEnv:
    env: Environment = field(init=False)
    def __attrs_post_init__(self):
        circle1 = Circle(center=np.array([0.0, 0.0]), radius=1.0)
        rectangle1 = Rectangle(
            center=np.array([0.0, 2.0]), half_extents=np.array([1.0, 0.5])
        )
        circle2 = Circle(center=np.array([-2, -2]), radius=0.75)
        boundary = Boundary(x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0)
        # Create environment
        self.env = Environment(obstacles=[circle1, rectangle1, circle2], bounds=boundary)

    def plot(self, ax: Axes):
        self.env.plot(ax)