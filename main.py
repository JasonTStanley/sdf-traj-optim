import matplotlib
import numpy as np
from attrs import define, field
from matplotlib.axes import Axes

from obstacles import *
from sdf import SDF, Boundary, Environment

from environments.basic import BasicEnv
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from bubble_cover.rrt import get_rapidly_exploring
if __name__ == "__main__":
    # Define obstacles
    env = BasicEnv()
    # Plot environment
    sdf = SDF(env.env, env.env.bounds)
    fig, ax = plt.subplots(figsize=(10, 10))
    # env.compute_sdf(grid_size=(100, 100))
    env.plot(ax)  # , view_sdf_grid=True)

    # Parameters
    num_test_positions = 100
    epsilon = 0.05  # ALL: clearance distance
    minimum_radius = 0.05  # ALL: minimum radius
    terminate_early = True  # EBT/RBT: early termination if end_point
    overlap_factor = 0.5  # EBT: overlap factor
    max_retry = 100  # RBT: max retry for rejection samping
    max_retry_epsilon = 1000  # RBT: max retry for min radius check
    inflate_factor = 1.0  # RBT: inflate factor for bounds

    start_position = np.array([-2.0, 2.0])
    end_position = np.array([2.1, -2.1])

    rng = np.random.default_rng(seed=3)
    jumpstart = None
    overlaps_graph, max_circles, _ = get_rapidly_exploring(
        sdf,
        epsilon,
        minimum_radius,
        num_test_positions,
        sdf.bounds.mins(),
        sdf.bounds.maxs(),
        start_position,
        end_point=np.array(end_position) if terminate_early else None,
        max_retry=max_retry,
        max_retry_epsilon=max_retry_epsilon,
        inflate_factor=inflate_factor,
        rng=rng,
        bubble_jumpstart=jumpstart
    )


    plt.show()
