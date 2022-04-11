import numpy as np


def like_lpdf(x, state):
    return np.log(1 / np.sqrt(2 * np.pi * state[1] * state[1])) - 1 / 2 * (
            (x - state[0]) * (x - state[0]) / state[1] / state[1])


def marg_lpdf(x, params):
    sig_n = np.sqrt(params[3] * params[1] + 1) / (params[2] * params[1])
