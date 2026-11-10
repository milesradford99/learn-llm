import numpy as np

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain

# A tiny network: 2 pictures, 3 pixels, 2 hidden units, 2 digits. One hidden value is
# negative before the bend, so a wrong ReLU slope shows.
X = np.array([[1.0, 0.0, 2.0], [0.5, 1.0, -1.0]])
Y = np.array([[1.0, 0.0], [0.0, 1.0]])
W1 = np.array([[0.5, -1.0], [1.0, 0.5], [-0.25, 0.5]])
B1 = np.array([0.1, -0.2])
W2 = np.array([[1.0, -0.5], [0.5, 2.0]])
B2 = np.array([0.0, 0.3])


def _softmax(S):
    E = np.exp(S - np.max(S, axis=1, keepdims=True))
    return E / np.sum(E, axis=1, keepdims=True)


def _forward():
    Z = X @ W1 + B1
    H = np.maximum(Z, 0)
    return Z, H, H @ W2 + B2


def _backward():
    Z, H, S = _forward()
    G = (_softmax(S) - Y) / len(X)
    dZ = (G @ W2.T) * (Z > 0)
    return X.T @ dZ, np.sum(dZ, axis=0), H.T @ G, np.sum(G, axis=0)


def _list(values):
    return tuple(np.asarray(v).tolist() for v in values)


def _count(f):
    """How many things forward gave back (None counts as 0, a single array as 1)."""
    out = f(X, W1, B1, W2, B2)
    return 0 if out is None else len(out) if isinstance(out, tuple) else 1


def _part(i):
    """Item i of forward's (Z, H, S); anything else is shown as it is, for the hint to explain."""
    def get(f):
        out = f(X, W1, B1, W2, B2)
        if not isinstance(out, tuple) or len(out) != 3:
            return _plain(out)
        return _plain(out[i])
    return get


def _grads(part):
    """backward's slopes for one layer: (dW2, db2) or (dW1, db1); anything else shown as it is."""
    def get(f):
        out = f(X, Y, *_forward(), W2)
        if not isinstance(out, tuple) or len(out) != 4:
            return _plain(out)
        return _plain(out[part])
    return get


EXERCISES = {
    "relu": [
        Case("ch08.relu.mixed", lambda f: _plain(f(np.array([[-2.0, 0.0, 3.0], [1.5, -0.5, 0.0]]))), [[0.0, 0.0, 3.0], [1.5, 0.0, 0.0]], tolerance=1e-12, hint="ch08.relu.hint"),
        Case("ch08.relu.shape", lambda f: list(np.shape(f(np.zeros((4, 7))))), [4, 7], hint="ch08.relu.hint"),
    ],
    "forward": [
        Case("ch08.forward.count", _count, 3, hint="ch08.forward.hint"),
        Case("ch08.forward.z", _part(0), _forward()[0].tolist(), tolerance=1e-9, hint="ch08.forward.hint"),
        Case("ch08.forward.h", _part(1), _forward()[1].tolist(), tolerance=1e-9, hint="ch08.forward.hint"),
        Case("ch08.forward.s", _part(2), _forward()[2].tolist(), tolerance=1e-9, hint="ch08.forward.hint"),
    ],
    "backward": [
        Case("ch08.backward.second", _grads(slice(2, 4)), _list(_backward()[2:]), tolerance=1e-9, hint="ch08.backward.second_hint"),
        Case("ch08.backward.first", _grads(slice(0, 2)), _list(_backward()[:2]), tolerance=1e-9, hint="ch08.backward.first_hint"),
    ],
}
