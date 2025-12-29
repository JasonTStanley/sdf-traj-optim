from obstacles import *
from sdf import Boundary, Environment


@define
class BasicEnv:
    env: Environment = field(init=False)

    def __attrs_post_init__(self):
        circle1 = Circle(center=np.array([-1.0, -1.0]), radius=1.5)
        rectangle1 = Rectangle(
            center=np.array([0.0, 2.0]), half_extents=np.array([1.0, 0.5])
        )
        circle2 = Circle(center=np.array([-2, -2]), radius=0.75, observed=False)
        rectangle2 = Rectangle(
            center=np.array([1, -1.5]), half_extents=np.array([0.25, 0.25])
        )
        boundary = Boundary(x_min=-3.0, x_max=3.0, y_min=-3.0, y_max=3.0)
        unobserved_rect = Rectangle(
            center=np.array([-2, -2]), half_extents=np.array([0.75, 0.75])
        )
        # Create environment
        self.env = Environment(
            bounds=boundary,
            obstacles=[circle1, rectangle1, circle2, rectangle2],
            unknown_regions=[unobserved_rect],
        )

    def plot(self, ax: Axes):
        self.env.plot(ax)
