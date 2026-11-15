import numpy as np
import torch

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain
from learn_llm.checks.ch10 import _plain as _plain_t


def _batch():
    u = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    v = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    n = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0], [2.0, 0.0]]])
    return u, v, n


def _reference_loss():
    def surprise(s):
        return float(np.log1p(np.exp(-s)))

    first = surprise(1) + surprise(-1) + surprise(1) + surprise(0)
    second = surprise(2) + surprise(-2) + surprise(0) + surprise(0)
    return (first + second) / 2


def _no_noise(f):
    u = torch.tensor([[0.0, 0.0]])
    return _plain_t(f(u, torch.tensor([[1.0, 2.0]]), torch.tensor([[[3.0, 4.0]]])))


def _hand(f):
    """The lesson's hand exercise: scores 2 and -1, so surprises 0.127 and 0.313."""
    return _plain_t(f(torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 0.0]]), torch.tensor([[[1.0, -1.0]]])))


def _lengths(f):
    out = f([1, 2, 3, 4, 5, 6], 2)
    return [len(part) for part in out] if isinstance(out, tuple) else _plain(out)


EXERCISES = {
    "count_words": [
        Case("ch12.count_words.small", lambda f: f(["the", "dog", "the"]), {"the": 2, "dog": 1}, hint="ch12.count_words.hint"),
        Case("ch12.count_words.empty", lambda f: f([]), {}, hint="ch12.count_words.hint"),
    ],
    "similarity": [
        Case("ch12.similarity.swapped", lambda f: _plain(f(np.array([3.0, 4.0]), np.array([4.0, 3.0]))), 0.96, tolerance=1e-9, hint="ch12.similarity.hint"),
        Case("ch12.similarity.same", lambda f: _plain(f(np.array([3.0, 4.0]), np.array([6.0, 8.0]))), 1.0, tolerance=1e-9, hint="ch12.similarity.hint"),
        Case("ch12.similarity.right", lambda f: _plain(f(np.array([1.0, 0.0]), np.array([0.0, 5.0]))), 0.0, tolerance=1e-9, hint="ch12.similarity.hint"),
        Case("ch12.similarity.opposite", lambda f: _plain(f(np.array([1.0, 2.0]), np.array([-2.0, -4.0]))), -1.0, tolerance=1e-9, hint="ch12.similarity.hint"),
        Case("ch12.similarity.mixed", lambda f: _plain(f(np.array([1.0, 0.0]), np.array([3.0, 4.0]))), 0.6, tolerance=1e-9, hint="ch12.similarity.hint"),
    ],
    "skipgram_pairs": [
        Case("ch12.skipgram_pairs.page", lambda f: f([0, 1, 2, 3], 1), ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]), hint="ch12.skipgram_pairs.hint"),
        Case("ch12.skipgram_pairs.wide", _lengths, [18, 18], hint="ch12.skipgram_pairs.hint"),
        Case("ch12.skipgram_pairs.one", lambda f: f([7], 2), ([], []), hint="ch12.skipgram_pairs.hint"),
    ],
    "pair_loss": [
        Case("ch12.pair_loss.hand", _hand, float(np.log1p(np.exp(-2.0)) + np.log1p(np.exp(-1.0))), tolerance=1e-4, hint="ch12.pair_loss.hint"),
        Case("ch12.pair_loss.batch", lambda f: _plain_t(f(*_batch())), _reference_loss(), tolerance=1e-4, hint="ch12.pair_loss.hint"),
        Case("ch12.pair_loss.zero", _no_noise, 2 * float(np.log(2)), tolerance=1e-4, hint="ch12.pair_loss.hint"),
    ],
}
