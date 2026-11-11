"""Reference solutions for chapter 09. Peek only after trying!"""

import numpy as np


def variance(v):
    return np.mean((v - np.mean(v)) ** 2)


def batches(n, size, rng):
    order = rng.permutation(n)
    result = []
    for start in range(0, n, size):
        result.append(order[start:start + size])
    return result


def momentum_step(w, velocity, grad, rate):
    velocity = 0.9 * velocity + grad
    return w - rate * velocity, velocity


def adam_step(w, m, v, grad, t, rate):
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad * grad
    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)
    return w - rate * m_hat / (np.sqrt(v_hat) + 1e-8), m, v
