"""Reference solutions for chapter 24. Peek only after trying!"""

import numpy as np


def fit_line(xs, ys):
    dx = xs - np.mean(xs)
    slope = np.sum(dx * (ys - np.mean(ys))) / np.sum(dx * dx)
    return slope, np.mean(ys) - slope * np.mean(xs)


def fit_power_law(xs, ys):
    k, log_a = fit_line(np.log(xs), np.log(ys))
    return np.exp(log_a), k


def tokens_for(budget, knobs):
    return budget / (6 * knobs)


def joint_law(knobs, tokens, floor, a, alpha, b, beta):
    return floor + a * knobs ** -alpha + b * tokens ** -beta


def best_size(budget, sizes, law):
    losses = [law(n, tokens_for(budget, n)) for n in sizes]
    return sizes[np.argmin(losses)]
