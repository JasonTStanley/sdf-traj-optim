import matplotlib
import numpy as np
from attrs import define, field
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from environments.basic import BasicEnv
from obstacles import *
from sdf import SDF, Boundary, Environment

matplotlib.use("TkAgg")
import sys

import matplotlib.pyplot as plt

from bubble_cover.circles import Circle
from bubble_cover.rrt import get_rapidly_exploring
from erl_gcopter import pyminco

if __name__ == "__main__":
    # Define obstacles
    env = BasicEnv()
    sdf = SDF(env.env, env.env.bounds)
    fig, ax = plt.subplots(figsize=(10, 10))
    # env.compute_sdf(grid_size=(100, 100))
    env.plot(ax)  # , view_sdf_grid=True)

    # Parameters
    num_test_positions = 100
    epsilon = 0.1  # ALL: clearance distance
    minimum_radius = 0.1  # ALL: minimum radius
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
        sdf.multi_query,
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
        bubble_jumpstart=jumpstart,
    )

    for circle in max_circles:
        sdf.plot_sample(ax, circle.centre)
    ax.plot(start_position[0], start_position[1], "gv")
    ax.plot(end_position[0], end_position[1], "g^")

    # add this block after plotting the circles/start/end (before plt.show())
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=8,
            label="Optimistic",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=8,
            label="Pessimistic",
        ),
        patches.Patch(color="red", label="Obstacle"),
        Line2D(
            [0, 1],
            [0, 0],
            color="black",
            lw=2,
            marker=">",
            markevery=[1],
            markersize=8,
            label="neg. Gradient",
        ),
    ]

    ax.legend(handles=legend_handles, loc="upper right")

    plt.show()
