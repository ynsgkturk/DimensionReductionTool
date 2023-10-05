import numpy as np


def problem_terminate(dimension):
    """
    Problem specific parameter settings.
    :param dimension: Dimension of search space.
    :return: N: Particle number,
    max_fe: Max fitness evaluation number,
    lb: Lower bound array,
    ub: Upper bound array
    """
    n = 25
    max_fe = 300
    lb = np.zeros((1, dimension))
    ub = np.ones((1, dimension))

    return n, max_fe, lb, ub
