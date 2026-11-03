"""Reference solutions for chapter 12. Peek only after trying!"""

import numpy as np


def bigram_counts(ids, size):
    C = np.zeros((size, size))
    for a, b in zip(ids, ids[1:]):
        C[a, b] = C[a, b] + 1
    return C


def bigram_probs(C, k):
    smoothed = C + k
    return smoothed / np.sum(smoothed, axis=1, keepdims=True)


def sample_next(p, rng):
    return rng.choice(len(p), p=p)


def perplexity(log_probs):
    return np.exp(-np.mean(log_probs))


def trigram_chance(table, a, b, c):
    if (a, b) not in table:
        return 0.0
    followers = table[(a, b)]
    return followers.get(c, 0) / sum(followers.values())
