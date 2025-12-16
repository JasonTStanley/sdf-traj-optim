import numpy as np


def get_uniform_random_points(num_samples, mins, maxs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return np.stack(
        [
            rng.uniform(low=min, high=max, size=[num_samples])
            for min, max in zip(mins, maxs)
        ],
        axis=-1,
    )


# TODO
def get_goal_biased_random_points(
    num_samples, goals, goal_prob, default_sampler, rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    pass
