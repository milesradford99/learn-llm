import numpy as np
import torch

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain

Q = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
K = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]])
V = torch.tensor([[[1.0], [2.0], [3.0]]])
OPEN = torch.zeros(3, 3, dtype=torch.bool)
TRI = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)


def _reference(mask):
    scores = (Q @ K.transpose(-2, -1) / np.sqrt(2)).masked_fill(mask, -torch.inf)
    A = torch.softmax(scores, dim=-1)
    return (A @ V).tolist(), A.tolist()


def _four_way(f):
    """Works with a heads dimension too, so masks and transposes must use the last dimensions."""
    x = torch.rand(2, 4, 5, 8)
    return [list(part.shape) for part in f(x, x, x, torch.zeros(5, 5, dtype=torch.bool))]


def _blocked(f):
    return _plain(f(Q, K, V, TRI)[1])[0][0][1:]


def _split(f):
    x = torch.arange(12.0).reshape(1, 2, 6)
    return _plain(f(x, 3))


def _split_shape(f):
    out = f(torch.zeros(5, 7, 128), 4)
    return list(out.shape) if isinstance(out, torch.Tensor) else _plain(out)


def _ref_split(x, heads):
    batch, words, d = x.shape
    return x.reshape(batch, words, heads, d // heads).transpose(1, 2)


def _join_shape(f):
    out = f(torch.zeros(5, 4, 7, 32))
    return list(out.shape) if isinstance(out, torch.Tensor) else _plain(out)


def _round_trip(f):
    x = torch.arange(12.0).reshape(1, 2, 6)
    return _plain(f(_ref_split(x, 3)))


X = torch.tensor([[[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 1.0, 3.0], [2.0, 2.0, 0.0, 1.0]]])
SAME = lambda t: t
DOUBLE = lambda t: 2 * t


def _ref_step(to_query, to_key, to_value, mix, heads):
    Q, K, V = _ref_split(to_query(X), heads), _ref_split(to_key(X), heads), _ref_split(to_value(X), heads)
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    A = torch.softmax((Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])).masked_fill(mask, -torch.inf), dim=-1)
    blended = (A @ V).transpose(1, 2).reshape(1, 3, 4)
    return mix(blended).tolist(), A.tolist()


def _step_shapes(f):
    out, A = f(torch.zeros(2, 5, 8), SAME, SAME, SAME, SAME, 4)
    return list(out.shape), list(A.shape)


def _step_future(f):
    return _plain(f(X, SAME, SAME, SAME, SAME, 2)[1])[0][0][0][1:]


EXERCISES = {
    "future_mask": [
        Case("ch16.future_mask.three", lambda f: _plain(f(3)), [[False, True, True], [False, False, True], [False, False, False]], hint="ch16.future_mask.hint"),
        Case("ch16.future_mask.one", lambda f: _plain(f(1)), [[False]], hint="ch16.future_mask.hint"),
    ],
    "attend": [
        Case("ch16.attend.open", lambda f: _plain(f(Q, K, V, OPEN)), _reference(OPEN), tolerance=1e-5, hint="ch16.attend.hint"),
        Case("ch16.attend.masked", lambda f: _plain(f(Q, K, V, TRI)), _reference(TRI), tolerance=1e-5, hint="ch16.attend.hint"),
        Case("ch16.attend.blocked", _blocked, [0.0, 0.0], tolerance=1e-9, hint="ch16.attend.hint"),
        Case("ch16.attend.heads", _four_way, [[2, 4, 5, 8], [2, 4, 5, 5]], hint="ch16.attend.hint"),
    ],
    "split_heads": [
        Case("ch16.split_heads.shape", _split_shape, [5, 4, 7, 32], hint="ch16.split_heads.hint"),
        Case("ch16.split_heads.values", _split, [[[[0.0, 1.0], [6.0, 7.0]], [[2.0, 3.0], [8.0, 9.0]], [[4.0, 5.0], [10.0, 11.0]]]], hint="ch16.split_heads.hint"),
    ],
    "join_heads": [
        Case("ch16.join_heads.shape", _join_shape, [5, 7, 128], hint="ch16.join_heads.hint"),
        Case("ch16.join_heads.undo", _round_trip, torch.arange(12.0).reshape(1, 2, 6).tolist(), hint="ch16.join_heads.hint"),
    ],
    "self_attention_step": [
        Case("ch16.self_attention_step.shapes", _step_shapes, ([2, 5, 8], [2, 4, 5, 5]), hint="ch16.self_attention_step.hint"),
        Case("ch16.self_attention_step.future", _step_future, [0.0, 0.0], tolerance=1e-9, hint="ch16.self_attention_step.hint"),
        Case("ch16.self_attention_step.values", lambda f: _plain(f(X, SAME, SAME, SAME, SAME, 2)), _ref_step(SAME, SAME, SAME, SAME, 2), tolerance=1e-5, hint="ch16.self_attention_step.hint"),
        Case("ch16.self_attention_step.layers", lambda f: _plain(f(X, DOUBLE, SAME, lambda t: t + 1, DOUBLE, 2)), _ref_step(DOUBLE, SAME, lambda t: t + 1, DOUBLE, 2), tolerance=1e-5, hint="ch16.self_attention_step.hint"),
    ],
}
