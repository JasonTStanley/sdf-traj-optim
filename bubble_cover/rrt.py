import matplotlib.pyplot as plt
import numpy as np

from circles import Circle
from linear_bubble_cover import LinearBubbleCover
from overlap import get_overlaps_graph
from points_sampler import get_uniform_random_points


class SDF:
    def __init__(self, sdf_function):
        self.sdf_function = sdf_function

    def __call__(self, points):
        return self.sdf_function(points)


def get_rapidly_exploring(
    dist_function,
    epsilon,
    minimum_radius,
    num_samples,  # maximum number of samples?
    mins,
    maxs,
    start_point,
    max_retry=500,
    max_retry_epsilon=100,
    max_num_iterations=np.inf,
    inflate_factor=1.0,
    end_point=None,
    rng=None,
    bubble_jumpstart=None,
):
    if rng is None:
        rng = np.random.default_rng()

    bubble_cover = LinearBubbleCover(
        np.array(start_point), dist_function(np.array([start_point])) - epsilon
    )
    # if bubble len != none
    # query radii
    # use bubble_cover.add to jumpstart
    # bubble jumpstart is a list of centers
    # use dist function to get radii
    inflated_mins, inflated_maxs = inflate_min_and_max(mins, maxs, inflate_factor)

    sampler = lambda: get_uniform_random_points(
        1, inflated_mins, inflated_maxs, rng=rng
    )

    num_retry_epsilon = 0

    num_iterations = 0
    while len(bubble_cover) < num_samples and num_iterations < max_num_iterations:
        print(f"{len(bubble_cover)} at {num_iterations}")
        new_circle, _, _ = get_rapidly_exploring_loop(
            bubble_cover,
            dist_function,
            sampler,
            epsilon,
            minimum_radius,
            max_retry=max_retry,
        )
        num_iterations += 1

        if new_circle is None:
            # print("failed epsilon check")
            num_retry_epsilon += 1
            # print(f"with {len(circles)} circles, retried {num_retry_epsilon}")
            if num_retry_epsilon > max_retry_epsilon:
                print("reached max retry epsilon")
                break
            continue  # failed the epsilon check
        else:
            num_retry_epsilon = 0
            bubble_cover.add(new_circle.centre, new_circle.radius)

        # NOTE: assuming starting circle does not contain end_point
        if end_point is not None and new_circle.contains_point(end_point):
            print(
                f"breaking because end_point is in new circle at iter {len(bubble_cover)}"
            )
            break

    # if len(circles) >= num_samples:
    #     print("reached max samples")
    if num_iterations >= max_num_iterations:
        print("reached max iterations")
    print("num_samples: ", len(bubble_cover))
    print("num_iterations: ", num_iterations)
    circles = [
        Circle(center, radius)
        for center, radius in zip(bubble_cover._centers, bubble_cover._radii)
    ]
    if bubble_jumpstart is not None:
        for idx in range(bubble_jumpstart.shape[0]):
            centers = bubble_jumpstart[idx, :]
            radii = dist_function(centers) - epsilon
            circ = Circle(centers, radii)
            circles.append(circ)
    return get_overlaps_graph(circles), circles, num_iterations


def pick_circle(sampler, bubble_cover, max_retry=500):
    near_idx = -1
    random_position = None
    for _ in range(max_retry):
        random_position = sampler()
        # dists = np.fromiter(list( map(lambda circle: circle.distance_to_point(random_position)[0], circles)), np.float32)
        dists = bubble_cover.distance_to_point(random_position)
        near_idx = np.argmin(dists, axis=-1)
        if np.all(dists[:, near_idx] > 0):
            break

    # Give up and return any random position
    return near_idx, random_position


def inflate_min_and_max(mins, maxs, factor):
    diff = maxs - mins
    centre = (maxs + mins) / 2
    return centre - factor * diff / 2, centre + factor * diff / 2


def get_valid_circle(
    bubble_cover, dist_function, new_position, near_idx, epsilon, minimum_radius
):
    diff_vector = new_position.flatten() - bubble_cover._centers[near_idx, :]
    new_centre = bubble_cover._centers[near_idx, :] + bubble_cover._radii[
        near_idx
    ] * diff_vector / np.linalg.norm(diff_vector)
    new_radius = dist_function(new_centre)
    new_circle = (
        Circle(new_centre.flatten(), new_radius - epsilon)
        if new_radius - epsilon > minimum_radius
        else None
    )
    return new_circle


def get_rapidly_exploring_loop(
    bubble_cover, dist_function, sampler, epsilon, minimum_radius, max_retry=500
):
    near_idx, random_position = pick_circle(sampler, bubble_cover, max_retry=max_retry)

    new_circle = get_valid_circle(
        bubble_cover, dist_function, random_position, near_idx, epsilon, minimum_radius
    )

    return new_circle, random_position, near_idx
