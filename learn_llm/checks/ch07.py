import numpy as np

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain

# Small inputs whose answers can be worked out by hand.
X_TINY = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
Y_TINY = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
W_TINY = [[0.5, -1.0, 0.0], [0.0, 1.0, -0.5]]
B_TINY = [0.0, 0.5, 0.0]


def _softmax(S):
    E = np.exp(S - np.max(S, axis=1, keepdims=True))
    return E / np.sum(E, axis=1, keepdims=True)


def _gradient(X, Y, W, b):
    G = (_softmax(X @ W + b) - Y) / len(X)
    return (X.T @ G).tolist(), np.sum(G, axis=0).tolist()


def _a(rows):
    return np.array(rows, dtype=float)


def _call(f, *args):
    return _plain(f(*[_a(a) for a in args]))


# Two rows of three: a missing keepdims can't hide behind a square matrix.
ROWS = [[1, 2, 3], [10, 20, 30]]


EXERCISES = {
    "softmax": [
        Case("ch07.softmax.three", lambda f: _call(f, [[2, 1, 0.1]]), _softmax(_a([[2, 1, 0.1]])).tolist(), tolerance=1e-6, hint="ch07.softmax.hint"),
        Case("ch07.softmax.rows", lambda f: _call(f, ROWS), _softmax(_a(ROWS)).tolist(), tolerance=1e-6, hint="ch07.softmax.rows_hint"),
        Case("ch07.softmax.huge", lambda f: _call(f, [[1000, 1000]]), [[0.5, 0.5]], tolerance=1e-9, hint="ch07.softmax.huge_hint"),
    ],
    "cross_entropy": [
        Case("ch07.cross_entropy.half", lambda f: _call(f, [[0.5, 0.5]], [[1, 0]]), float(np.log(2)), tolerance=1e-9, hint="ch07.cross_entropy.hint"),
        Case("ch07.cross_entropy.ten", lambda f: _call(f, [[0.1] * 10], [[0] * 9 + [1]]), float(np.log(10)), tolerance=1e-9, hint="ch07.cross_entropy.hint"),
        Case("ch07.cross_entropy.mean", lambda f: _call(f, [[0.5, 0.5], [0.9, 0.1]], [[1, 0], [1, 0]]), float((np.log(2) - np.log(0.9)) / 2), tolerance=1e-9, hint="ch07.cross_entropy.hint"),
    ],
    "gradient": [
        Case("ch07.gradient.tiny", lambda f: _call(f, X_TINY, Y_TINY, W_TINY, B_TINY), tuple(_gradient(_a(X_TINY), _a(Y_TINY), _a(W_TINY), _a(B_TINY))), tolerance=1e-9, hint="ch07.gradient.hint"),
        Case("ch07.gradient.zero", lambda f: _call(f, [[1.0, 2.0]], [[0, 1]], [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0]), ([[0.5, -0.5], [1.0, -1.0]], [0.5, -0.5]), tolerance=1e-9, hint="ch07.gradient.hint"),
    ],
}
