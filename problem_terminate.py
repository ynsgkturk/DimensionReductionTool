import numpy as np


def problem_terminate(dimension):
    n = 25
    max_FE = 300
    lb = np.zeros((1, dimension))
    ub = np.ones((1, dimension))

    return n, max_FE, lb, ub