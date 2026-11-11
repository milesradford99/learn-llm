"""Reference solutions for chapter 06. Peek only after trying!"""

import numpy as np


def guess(S):
    return np.argmax(S, axis=1)


def accuracy(S, labels):
    return np.mean(guess(S) == labels)


def template(X, labels, c):
    return np.mean(X[labels == c], axis=0)
