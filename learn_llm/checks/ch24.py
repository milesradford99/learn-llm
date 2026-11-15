import numpy as np

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain


def _fit(f, xs, ys):
    out = f(np.array(xs, dtype=float), np.array(ys, dtype=float))
    return tuple(_plain(out)) if isinstance(out, (tuple, list)) else _plain(out)


def _toy_law(knobs, tokens):
    """The page's hand exercise: 1 + 100 / N + 400 / D."""
    return 1 + 100 / knobs + 400 / tokens


def _best(f, budget, sizes, law):
    return _plain(f(budget, sizes, law))


EXERCISES = {
    "fit_line": [
        Case("ch24.fit_line.exact", lambda f: _fit(f, [0, 1, 2], [1, 3, 5]), (2.0, 1.0), tolerance=1e-9, hint="ch24.fit_line.hint"),
        Case("ch24.fit_line.hand", lambda f: _fit(f, [0, 2, 4], [6, 5, 4]), (-0.5, 6.0), tolerance=1e-9, hint="ch24.fit_line.hint"),
        Case("ch24.fit_line.scattered", lambda f: _fit(f, [1, 2, 3, 4], [2, 4, 5, 9]), (2.2, -0.5), tolerance=1e-9, hint="ch24.fit_line.hint"),
    ],
    "fit_power_law": [
        Case("ch24.fit_power_law.hand", lambda f: _fit(f, [1, 4, 16], [64, 32, 16]), (64.0, -0.5), tolerance=1e-9, hint="ch24.fit_power_law.hint"),
        Case("ch24.fit_power_law.growing", lambda f: _fit(f, [1, 2, 4, 8], [3, 12, 48, 192]), (3.0, 2.0), tolerance=1e-9, hint="ch24.fit_power_law.hint"),
    ],
    "tokens_for": [
        Case("ch24.tokens_for.hand", lambda f: _plain(f(6e12, 1e6)), 1e6, tolerance=1e-3, hint="ch24.tokens_for.hint"),
        Case("ch24.tokens_for.bigger", lambda f: _plain(f(6e12, 4e6)), 250000.0, tolerance=1e-3, hint="ch24.tokens_for.hint"),
    ],
    "joint_law": [
        Case("ch24.joint_law.round", lambda f: _plain(f(100.0, 10000.0, 2, 10, 0.5, 300, 0.5)), 6.0, tolerance=1e-9, hint="ch24.joint_law.hint"),
        Case("ch24.joint_law.toy", lambda f: _plain(f(50.0, 200.0, 1, 100, 1, 400, 1)), 5.0, tolerance=1e-9, hint="ch24.joint_law.hint"),
        Case("ch24.joint_law.arrays", lambda f: _plain(f(np.array([100.0, 400.0]), np.array([10000.0, 2500.0]), 2, 10, 0.5, 300, 0.5)), [6.0, 8.5], tolerance=1e-9, hint="ch24.joint_law.hint"),
    ],
    "best_size": [
        Case("ch24.best_size.hand", lambda f: _best(f, 60000, [10, 50, 100, 200], _toy_law), 50, hint="ch24.best_size.hint"),
        Case("ch24.best_size.rich", lambda f: _best(f, 6000000, [10, 50, 100, 200], _toy_law), 200, hint="ch24.best_size.hint"),
        Case("ch24.best_size.poor", lambda f: _best(f, 600, [10, 50, 100, 200], _toy_law), 10, hint="ch24.best_size.hint"),
    ],
}
