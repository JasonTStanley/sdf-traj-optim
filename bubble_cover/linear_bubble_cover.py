from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class Bubble:
    center: np.ndarray
    radius: float


class BubbleCoverInterface(ABC):
    pass


class LinearBubbleCover(BubbleCoverInterface):
    _centers: np.ndarray  # N_bubbles x N_dim array
    _radii: np.ndarray  # N_bubbles array

    def __init__(self, start_point, radius):
        self._centers = np.atleast_2d(start_point)
        self._radii = np.atleast_1d(radius)

    def distance_to_point(self, point):
        dists = cdist(np.atleast_2d(point), self._centers)  # N_points X N_bubbles
        dist_to_boundary = dists - self._radii
        return dist_to_boundary

    def append(self, bubble: Bubble):
        return self.add(bubble.center, bubble.radius)

    def add(self, center, radii):
        self._centers = np.append(self._centers, np.atleast_2d(center), axis=0)
        self._radii = np.append(self._radii, radii, axis=0)
        return None

    @property
    def ndim(self):
        return self._centers.shape[-1]

    @property
    def n_bubbles(self):
        return self._centers.shape[0]

    def __len__(self):
        return self.n_bubbles
