import numpy as np
import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch09 import _plain

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


def _norm(f):
    torch.manual_seed(2)
    out = f(torch.rand(3, 4, 8) * 10)
    return round(torch.mean(torch.abs(torch.mean(out, dim=-1))).item(), 3), round(torch.mean(torch.var(out, dim=-1, unbiased=False)).item(), 2)


def _matches_pytorch(f):
    torch.manual_seed(3)
    x = torch.rand(2, 3, 8)
    mine = f(x)
    theirs = nn.LayerNorm(8, eps=1e-5, elementwise_affine=False)(x)
    return bool(torch.max(torch.abs(mine - theirs)) < 1e-4)


def _block(f):
    x = torch.ones(1, 2, 3)
    double = lambda t: 2 * t
    add_one = lambda t: t + 1
    # 1 + 2 × (2 × 1) = 5, then 5 + 2 × (5 + 1) = 17
    out = f(x, double, double, add_one, double)
    return None if out is None else _plain(out)[0][0][0]


def _order(f):
    calls = []

    def note(name):
        def layer(t):
            calls.append(name)
            return t
        return layer

    f(torch.ones(1, 1, 1), note("norm1"), note("attention"), note("norm2"), note("think"))
    return calls


EXERCISES = {
    "future_mask": [
        Case("ch15.future_mask.three", lambda f: _plain(f(3)), [[False, True, True], [False, False, True], [False, False, False]], hint="ch15.future_mask.hint"),
        Case("ch15.future_mask.one", lambda f: _plain(f(1)), [[False]], hint="ch15.future_mask.hint"),
    ],
    "attend": [
        Case("ch15.attend.open", lambda f: _plain(f(Q, K, V, OPEN)), _reference(OPEN), tolerance=1e-5, hint="ch15.attend.hint"),
        Case("ch15.attend.masked", lambda f: _plain(f(Q, K, V, TRI)), _reference(TRI), tolerance=1e-5, hint="ch15.attend.hint"),
        Case("ch15.attend.blocked", _blocked, [0.0, 0.0], tolerance=1e-9, hint="ch15.attend.hint"),
        Case("ch15.attend.heads", _four_way, [[2, 4, 5, 8], [2, 4, 5, 5]], hint="ch15.attend.hint"),
    ],
    "split_heads": [
        Case("ch15.split_heads.shape", _split_shape, [5, 4, 7, 32], hint="ch15.split_heads.hint"),
        Case("ch15.split_heads.values", _split, [[[[0.0, 1.0], [6.0, 7.0]], [[2.0, 3.0], [8.0, 9.0]], [[4.0, 5.0], [10.0, 11.0]]]], hint="ch15.split_heads.hint"),
    ],
    "layer_norm": [
        Case("ch15.layer_norm.scale", _norm, (0.0, 1.0), tolerance=0.01, hint="ch15.layer_norm.hint"),
        Case("ch15.layer_norm.pytorch", _matches_pytorch, True, hint="ch15.layer_norm.hint"),
    ],
    "block_step": [
        Case("ch15.block_step.value", _block, 17.0, tolerance=1e-6, hint="ch15.block_step.hint"),
        Case("ch15.block_step.order", _order, ["norm1", "attention", "norm2", "think"], hint="ch15.block_step.hint"),
    ],
}
