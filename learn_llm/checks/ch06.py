import numpy as np

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain

# Small inputs whose answers can be worked out by hand.
SCORES = [[1, 3, 2], [0, 0, 5], [4, 1, 1]]
TABLE = [[1, 3, 2], [0, 0, 5], [4, 1, 1], [2, 6, 3]]
ROWS = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def _a(rows):
    return np.array(rows, dtype=float)


def _call(f, *args):
    return _plain(f(*[_a(a) for a in args]))


def _template(f, labels, c):
    return _plain(f(_a(ROWS), np.array(labels), c))


EXERCISES = {
    "guess": [
        Case("ch06.guess.table", lambda f: _call(f, TABLE), [1, 2, 0, 1], hint="ch06.guess.hint"),
        Case("ch06.guess.tie", lambda f: _call(f, [[0, 0, 0], [0, 7, 7]]), [0, 1], hint="ch06.guess.hint"),
    ],
    "accuracy": [
        Case("ch06.accuracy.two_of_three", lambda f: _call(f, SCORES, [1, 2, 1]), 2 / 3, tolerance=1e-9, hint="ch06.accuracy.hint"),
        Case("ch06.accuracy.all", lambda f: _call(f, SCORES, [1, 2, 0]), 1.0, tolerance=1e-9, hint="ch06.accuracy.hint"),
        Case("ch06.accuracy.none", lambda f: _call(f, SCORES, [0, 0, 1]), 0.0, tolerance=1e-9, hint="ch06.accuracy.hint"),
    ],
    "template": [
        Case("ch06.template.two", lambda f: _template(f, [7, 9, 7], 7), [3.0, 4.0], tolerance=1e-9, hint="ch06.template.hint"),
        Case("ch06.template.one", lambda f: _template(f, [7, 9, 7], 9), [3.0, 4.0], tolerance=1e-9, hint="ch06.template.hint"),
        Case("ch06.template.other", lambda f: _template(f, [0, 0, 1], 0), [2.0, 3.0], tolerance=1e-9, hint="ch06.template.hint"),
    ],
}
