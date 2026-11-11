"""Reference solutions for chapter 07. Peek only after trying!"""

import numpy as np


def softmax(S):
    S = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=1, keepdims=True)


def cross_entropy(P, Y):
    return -np.mean(np.sum(Y * np.log(P), axis=1))


def gradient(X, Y, W, b):
    P = softmax(X @ W + b)
    G = (P - Y) / len(X)
    return X.T @ G, np.sum(G, axis=0)
