from abc import ABC, abstractmethod
import numpy as np
from attrs import define
import matplotlib.pyplot as plt
class Obstacle(ABC):
    @abstractmethod
    def check_sdf(self, pos: np.ndarray) -> float:
        """
        Calculate the signed distance function (SDF) value for a given position.
        """
        pass

    @abstractmethod
    def plot(self, ax: plt.Axes):
        """Visualize to a matplotlib plot"""
        pass

@define
class Circle(Obstacle):
    center: np.ndarray
    radius: float

    def check_sdf(self, pos: np.ndarray) -> float:
        return np.linalg.norm(pos - self.center) - self.radius

    def plot(self, ax: plt.Axes):
        circle = plt.Circle(self.center, self.radius, color='red', fill=False, linewidth=2)
        ax.add_artist(circle)


@define
class Rectangle(Obstacle):
    center: np.ndarray
    half_extents: np.ndarray

    def check_sdf(self, pos: np.ndarray) -> float:
        d = np.abs(pos - self.center) - self.half_extents
        return np.linalg.norm(np.maximum(d, 0)) + min(max(d[0], d[1]), 0)

    def plot(self, ax: plt.Axes):
        lower_left = self.center - self.half_extents
        width, height = 2 * self.half_extents
        rect = plt.Rectangle(lower_left, width, height, color='red', fill=False, linewidth=2)
        ax.add_artist(rect)
@define
class Oval(Obstacle):
    center: np.ndarray
    radii: np.ndarray

    def check_sdf(self, pos: np.ndarray) -> float:
        q = (pos - self.center) / self.radii
        return np.linalg.norm(q) - 1

    def plot(self, ax: plt.Axes):
        ellipse = plt.Ellipse(self.center, 2 * self.radii[0], 2 * self.radii[1], color='green', fill=False, linewidth=2)
        ax.add_artist(ellipse)