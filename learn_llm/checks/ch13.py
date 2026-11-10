import math

import numpy as np

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain

TINY = [0, 1, 2, 0, 1, 3, 0, 4, 2]
TABLE = {(0, 1): {2: 3, 3: 1}, (1, 2): {0: 5}}


def _sums(f):
    P = f(np.array([[1.0, 3.0], [0.0, 0.0]]), 1)
    return _plain(np.sum(P, axis=1)) if isinstance(P, np.ndarray) else _plain(P)


def _draws(f, chance, draws):
    """How many of `draws` samples landed on id 0, when id 0 has this chance."""
    p = np.array([chance, 1 - chance])
    rng = np.random.default_rng(5)
    return sum(int(f(p, rng)) == 0 for _ in range(draws))


EXERCISES = {
    "bigram_counts": [
        Case("ch13.bigram_counts.pair", lambda f: _plain(f(TINY, 5))[0], [0.0, 2.0, 0.0, 0.0, 1.0], hint="ch13.bigram_counts.hint"),
        Case("ch13.bigram_counts.total", lambda f: _plain(np.sum(f(TINY, 5))), 8.0, hint="ch13.bigram_counts.hint"),
    ],
    "bigram_probs": [
        Case("ch13.bigram_probs.none", lambda f: _plain(f(np.array([[1.0, 3.0], [0.0, 0.0]]), 0.5)), [[0.3, 0.7], [0.5, 0.5]], tolerance=1e-9, hint="ch13.bigram_probs.hint"),
        Case("ch13.bigram_probs.sums", _sums, [1.0, 1.0], tolerance=1e-9, hint="ch13.bigram_probs.hint"),
    ],
    "sample_next": [
        Case("ch13.sample_next.sure", lambda f: _draws(f, 1.0, 20), 20, hint="ch13.sample_next.hint"),
        Case("ch13.sample_next.never", lambda f: _draws(f, 0.0, 20), 0, hint="ch13.sample_next.hint"),
        Case("ch13.sample_next.fair", lambda f: 300 < _draws(f, 0.4, 1000) < 500, True, hint="ch13.sample_next.hint"),
    ],
    "perplexity": [
        Case("ch13.perplexity.uniform", lambda f: _plain(f(np.full(5, math.log(1 / 50)))), 50.0, tolerance=1e-6, hint="ch13.perplexity.hint"),
        Case("ch13.perplexity.perfect", lambda f: _plain(f(np.zeros(3))), 1.0, tolerance=1e-9, hint="ch13.perplexity.hint"),
        Case("ch13.perplexity.mixed", lambda f: _plain(f(np.log(np.array([0.5, 0.125])))), 4.0, tolerance=1e-6, hint="ch13.perplexity.hint"),
    ],
    "trigram_chance": [
        Case("ch13.trigram_chance.share", lambda f: f(TABLE, 0, 1, 2), 0.75, tolerance=1e-9, hint="ch13.trigram_chance.hint"),
        Case("ch13.trigram_chance.unseen_next", lambda f: f(TABLE, 0, 1, 4), 0.0, tolerance=1e-9, hint="ch13.trigram_chance.hint"),
        Case("ch13.trigram_chance.unseen_pair", lambda f: f(TABLE, 3, 3, 0), 0.0, tolerance=1e-9, hint="ch13.trigram_chance.hint"),
    ],
}
