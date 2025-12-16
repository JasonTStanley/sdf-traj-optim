from abc import ABC, abstractmethod

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field
from matplotlib.axes import Axes


@define
class Obstacle(ABC):
    @abstractmethod
    def check_sdf(self, pos: np.ndarray) -> float:
        """
        Calculate the signed distance function (SDF) value for a given position.
        """
        pass

    @abstractmethod
    def sdf_deriv(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the SDF value wrt. a given position.
        """
        pass

    @abstractmethod
    def plot(self, ax: Axes):
        """Visualize to a matplotlib plot"""
        pass
    @abstractmethod
    def set_observed(self):
        """Set if the SDF should see this obstacle"""
        pass
    @abstractmethod
    def is_observed(self):
        """True if the SDF should see this obstacle"""
        pass


@define
class Circle(Obstacle):
    center: np.ndarray
    radius: float
    observed: bool = field(default=True)
    def check_sdf(self, pos: np.ndarray) -> float:
        if not self.observed:
            return np.inf
        return float(np.linalg.norm(pos - self.center) - self.radius)

    def sdf_deriv(self, pos: np.ndarray) -> np.ndarray:
        direction = pos - self.center
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros_like(direction)
        return direction / norm

    def plot(self, ax: Axes):
        circle = patches.Circle(
            self.center, self.radius, color="red", fill=False, linewidth=2
        )
        ax.add_artist(circle)

    def set_observed(self, obs: bool):
        self.observed = obs
    def is_observed(self):
        return self.observed

@define
class Rectangle(Obstacle):
    center: np.ndarray
    half_extents: np.ndarray
    observed: bool = field(default=True)

    def check_sdf(self, pos: np.ndarray) -> float:
        if not self.observed:
            return np.inf
        d = np.abs(pos - self.center) - self.half_extents
        return np.linalg.norm(np.maximum(d, 0)) + min(max(d[0], d[1]), 0)

    def sdf_deriv(self, pos: np.ndarray) -> np.ndarray:
        d = pos - self.center
        clamped = np.maximum(np.abs(d) - self.half_extents, 0)
        norm = np.linalg.norm(clamped)
        if norm == 0:
            return np.zeros_like(d)
        sign = np.sign(d)
        deriv = clamped / norm
        deriv *= sign
        return deriv

    def plot(self, ax: Axes):
        lower_left = self.center - self.half_extents
        width, height = 2 * self.half_extents
        rect = patches.Rectangle(
            lower_left, width, height, color="red", fill=False, linewidth=2
        )
        ax.add_artist(rect)

    def set_observed(self, obs: bool):
        self.observed = obs
    def is_observed(self):
        return self.is_observed


@define
class Oval(Obstacle):
    center: np.ndarray
    radii: np.ndarray
    observed: bool = field(default=True)

    def check_sdf(self, pos: np.ndarray) -> float:
        if not self.observed:
            return np.inf
        q = (pos - self.center) / self.radii
        return float(np.linalg.norm(q) - 1)

    def sdf_deriv(self, pos: np.ndarray) -> np.ndarray:
        return (
            2
            * (pos - self.center)
            / (self.radii**2)
            / np.linalg.norm((pos - self.center) / self.radii)
        )

    def plot(self, ax: Axes):
        ellipse = patches.Ellipse(
            self.center,
            2 * self.radii[0],
            2 * self.radii[1],
            color="green",
            fill=False,
            linewidth=2,
        )
        ax.add_artist(ellipse)

    def set_observed(self, obs: bool):
        self.observed = obs
    def is_observed(self):
        return self.observed
