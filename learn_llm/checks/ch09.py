import numpy as np

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain


def _batches(f, n, size):
    """What the learner's batches gave: sizes, and whether together they cover 0..n-1 once."""
    out = f(n, size, np.random.default_rng(3))
    if not isinstance(out, list):
        return _plain(out)
    sizes = [len(b) for b in out]
    joined = sorted(int(i) for b in out for i in b)
    return sizes, joined == list(range(n))


def _shuffled(f):
    """The first batch differs from 0, 1, 2, ... (with this seed, a shuffle can't give that)."""
    out = f(100, 10, np.random.default_rng(3))
    return isinstance(out, list) and len(out) > 0 and [int(i) for i in out[0]] != list(range(10))


def _adam(f, steps):
    w, m, v = np.array([1.0, -2.0]), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        out = f(w, m, v, np.array([0.5, -3.0]) * t, t, 0.1)
        if not isinstance(out, tuple) or len(out) != 3:
            return _plain(out)
        w, m, v = out
    return _plain(w)


def _adam_count(f):
    """How many things adam_step gave back (None counts as 0, a single array as 1)."""
    out = f(np.array([1.0, -2.0]), np.zeros(2), np.zeros(2), np.array([0.5, -3.0]), 1, 0.1)
    return 0 if out is None else len(out) if isinstance(out, tuple) else 1


def _adam_reference(steps):
    w, m, v = np.array([1.0, -2.0]), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        g = np.array([0.5, -3.0]) * t
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w.tolist()


def _momentum(f, steps):
    w, velocity = np.array([1.0, -2.0]), np.zeros(2)
    for _ in range(steps):
        out = f(w, velocity, np.array([0.5, -3.0]), 0.1)
        if not isinstance(out, tuple) or len(out) != 2:
            return _plain(out)
        w, velocity = out
    return _plain((w, velocity))


EXERCISES = {
    "variance": [
        Case("ch09.variance.small", lambda f: _plain(f(np.array([1.0, 3.0]))), 1.0, tolerance=1e-12, hint="ch09.variance.hint"),
        Case("ch09.variance.five", lambda f: _plain(f(np.array([2.0, 4.0, 4.0, 4.0, 6.0]))), 1.6, tolerance=1e-12, hint="ch09.variance.hint"),
        Case("ch09.variance.same", lambda f: _plain(f(np.array([7.0, 7.0, 7.0]))), 0.0, tolerance=1e-12, hint="ch09.variance.hint"),
    ],
    "batches": [
        Case("ch09.batches.even", lambda f: _batches(f, 10, 5), ([5, 5], True), hint="ch09.batches.hint"),
        Case("ch09.batches.uneven", lambda f: _batches(f, 10, 4), ([4, 4, 2], True), hint="ch09.batches.hint"),
        Case("ch09.batches.shuffled", _shuffled, True, hint="ch09.batches.shuffle_hint"),
    ],
    "momentum_step": [
        Case("ch09.momentum_step.one", lambda f: _momentum(f, 1), ([0.95, -1.7], [0.5, -3.0]), tolerance=1e-12, hint="ch09.momentum_step.hint"),
        Case("ch09.momentum_step.two", lambda f: _momentum(f, 2), ([0.855, -1.13], [0.95, -5.7]), tolerance=1e-12, hint="ch09.momentum_step.hint"),
    ],
    "adam_step": [
        Case("ch09.adam_step.count", _adam_count, 3, hint="ch09.adam_step.hint"),
        Case("ch09.adam_step.one", lambda f: _adam(f, 1), _adam_reference(1), tolerance=1e-6, hint="ch09.adam_step.hint"),
        Case("ch09.adam_step.five", lambda f: _adam(f, 5), _adam_reference(5), tolerance=1e-6, hint="ch09.adam_step.hint"),
    ],
}
