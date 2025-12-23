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
from bubble_cover.discrete import epath_to_vpath, get_shortest_path
from bubble_cover.overlap import position_to_max_circle_idx
from bubble_cover.rrt import get_rapidly_exploring
from bubble_cover.tracing import trace_toward_graph_all
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
    terminate_early = False  # EBT/RBT: early termination if end_point
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

    start_idx = position_to_max_circle_idx(overlaps_graph, start_position)

    if start_idx < 0:
        print("repairing graph for start")
        overlaps_graph, start_idx = trace_toward_graph_all(
            overlaps_graph, sdf, epsilon, minimum_radius, start_position
        )

    end_idx = position_to_max_circle_idx(overlaps_graph, end_position)
    if end_idx < 0:
        print("repairing graph for end")
        overlaps_graph, end_idx = trace_toward_graph_all(
            overlaps_graph, sdf, epsilon, minimum_radius, end_position
        )

    overlaps_graph.to_directed()
    # plan shortest path on the graph
    epath_centre_distance = get_shortest_path(
        lambda from_circle, to_circle: from_circle.hausdorff_distance_to(to_circle),
        overlaps_graph,
        start_idx,
        end_idx,
        cost_name="hausdorff_distance",
        return_epath=True,
    )

    # for circle in max_circles:
    #    sdf.plot_sample(ax, circle.centre)

    if epath_centre_distance[0]:
        vpath_centre_distance = epath_to_vpath(overlaps_graph, epath_centre_distance[0])
    else:
        vpath_centre_distance = []

    chosen_path_circles = []
    for idx in vpath_centre_distance:
        chosen_path_circles.append(max_circles[idx])

    # convert circles to pyminco circles:
    minco_bubbles = []
    for circle in chosen_path_circles:
        minco_bubbles.append(pyminco.Bubble2D(circle.centre, circle.radius))

    print(f"num_bubbles in path: { len(minco_bubbles)}")

    cfg = pyminco.Config()
    cfg.boundary_lambda = 1e-2

    # print("calculating short path:")
    # short_path = pyminco.shortestPath2D(
    #    start_position, end_position, minco_bubbles, cfg
    # )
    # ax.scatter(short_path[0, :], short_path[1, :], c="red", zorder=5)

    trajectory: pyminco.Trajectory | None = pyminco.solveTrajectory2D(
        start_position, end_position, minco_bubbles, cfg
    )
    if trajectory is None:
        print("No trajectory found")
        sys.exit(1)

    print(trajectory)

    traj_duration = trajectory.getTotalDuration()
    traj_positions = np.array(
        [trajectory.getPos(t) for t in np.linspace(0, traj_duration, num=1000)]
    )
    print(traj_positions.shape)
    print(traj_positions)
    waypoints = trajectory.getPositions()
    ax.scatter(waypoints[0, :], waypoints[1, :], c="orange", zorder=4)
    ax.plot(traj_positions[:, 0], traj_positions[:, 1], "k-", linewidth=2, zorder=3)

    for circle in chosen_path_circles:
        ax.add_artist(
            patches.Circle(
                circle.centre,
                circle.radius,
                color="cyan",
                zorder=2,
                fill=True,
                alpha=0.5,
                linewidth=2,
            )
        )

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
