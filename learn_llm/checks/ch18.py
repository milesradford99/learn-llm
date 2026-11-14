import numpy as np
import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain

GUESSES = np.array([1, 1, 1, 0, 0, 2])
LABELS = np.array([1, 1, 0, 1, 0, 2])


HAND_GUESSES = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
HAND_LABELS = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0])


def _classified(f):
    """Two stories of 3 words with 2 numbers each, and a head with known knobs."""
    head = nn.Linear(2, 4)
    with torch.no_grad():
        head.weight[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        head.bias[:] = torch.tensor([0.0, 0.0, 0.0, 0.5])
        features = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 3.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 6.0]]])
        out = f(features, head)
    return out.tolist() if isinstance(out, torch.Tensor) else out


def _frozen(f):
    model = nn.Linear(2, 2)
    f(model)
    return [p.requires_grad for p in model.parameters()]


EXERCISES = {
    "blank_out": [
        Case("ch18.blank_out.some", lambda f: f([5, 7, 5, 9], [5], 0), [0, 7, 0, 9], hint="ch18.blank_out.hint"),
        Case("ch18.blank_out.none", lambda f: f([1, 2], [], 0), [1, 2], hint="ch18.blank_out.hint"),
    ],
    "classify": [
        Case("ch18.classify.values", _classified, [[2.0, 1.0, 3.0, 0.5], [0.0, 2.0, 2.0, 0.5]], tolerance=1e-5, hint="ch18.classify.hint"),
    ],
    "majority_share": [
        Case("ch18.majority_share.page", lambda f: _plain(f(np.array([1, 1, 1, 0, 2, 2]))), 0.5, tolerance=1e-9, hint="ch18.majority_share.hint"),
        Case("ch18.majority_share.all", lambda f: _plain(f(np.array([3, 3, 3]))), 1.0, tolerance=1e-9, hint="ch18.majority_share.hint"),
    ],
    "precision_recall": [
        Case("ch18.precision_recall.hand", lambda f: _plain(f(HAND_GUESSES, HAND_LABELS, 1)), (0.8, 4 / 6), tolerance=1e-9, hint="ch18.precision_recall.hint"),
        Case("ch18.precision_recall.one", lambda f: _plain(f(GUESSES, LABELS, 1)), (2 / 3, 2 / 3), tolerance=1e-9, hint="ch18.precision_recall.hint"),
        Case("ch18.precision_recall.order", lambda f: _plain(f(np.array([1, 1, 1, 0]), np.array([1, 0, 0, 0]), 1)), (1 / 3, 1.0), tolerance=1e-9, hint="ch18.precision_recall.hint"),
        Case("ch18.precision_recall.perfect", lambda f: _plain(f(GUESSES, LABELS, 2)), (1.0, 1.0), tolerance=1e-9, hint="ch18.precision_recall.hint"),
        Case("ch18.precision_recall.never", lambda f: _plain(f(GUESSES, LABELS, 3)), (0.0, 0.0), tolerance=1e-9, hint="ch18.precision_recall.hint"),
    ],
    "margin": [
        Case("ch18.margin.hand", lambda f: _plain(f(0.5, 100)), 0.1, tolerance=1e-9, hint="ch18.margin.hint"),
        Case("ch18.margin.page", lambda f: _plain(f(0.6, 265)), 2 * float(np.sqrt(0.24 / 265)), tolerance=1e-9, hint="ch18.margin.hint"),
        Case("ch18.margin.sure", lambda f: _plain(f(1.0, 100)), 0.0, tolerance=1e-9, hint="ch18.margin.hint"),
    ],
    "clean_split": [
        Case("ch18.clean_split.page", lambda f: f(["a", "b", "a", "c", "b"], 2), (["a", "a"], ["c", "b"]), hint="ch18.clean_split.hint"),
        Case("ch18.clean_split.clean", lambda f: f(["a", "b", "c", "d"], 1), (["a", "b", "c"], ["d"]), hint="ch18.clean_split.hint"),
        Case("ch18.clean_split.all", lambda f: f(["x", "x", "x"], 1), ([], ["x"]), hint="ch18.clean_split.hint"),
    ],
    "freeze": [
        Case("ch18.freeze.all", _frozen, [False, False], hint="ch18.freeze.hint"),
    ],
}
