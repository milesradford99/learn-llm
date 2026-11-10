import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain


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


def _tens(ids):
    return ids.unsqueeze(-1) * 10.0


def _ones(ids):
    return ids.unsqueeze(-1) * 1.0


def _embed(f):
    return _plain(f(torch.tensor([[3, 5, 2], [1, 1, 1]]), _tens, _ones))


def _embed_real(f):
    torch.manual_seed(4)
    out = f(torch.tensor([[1, 0, 1, 2]]), nn.Embedding(3, 5), nn.Embedding(6, 5))
    return list(out.shape) if isinstance(out, torch.Tensor) else _plain(out)


def _forward(f):
    out = f(torch.tensor([[3, 5]]), _tens, _ones, lambda x: x * 2, lambda x: x + 1, lambda x: x * 100)
    return None if out is None else _plain(out)


def _forward_order(f):
    calls = []

    def note(name):
        def layer(t):
            calls.append(name)
            return t
        return layer

    f(torch.tensor([[3, 5]]), _tens, _ones, note("blocks"), note("norm"), note("out"))
    return calls


def _counting_model(lengths):
    """Always predicts the id after the last one, modulo 5, and notes how many ids it was given."""
    def model(ids):
        lengths.append(ids.shape[1])
        S = torch.full((1, ids.shape[1], 5), -1000.0)
        S[0, -1, (ids[0, -1].item() + 1) % 5] = 1000.0
        return S
    return model


def _write(f):
    return _plain(f(_counting_model([]), 3, 6, 1.0, 4))


def _longest(f):
    lengths = []
    f(_counting_model(lengths), 0, 9, 1.0, 4)
    return max(lengths), lengths[:3]


EXERCISES = {
    "embed": [
        Case("ch17.embed.values", _embed, [[[30.0], [51.0], [22.0]], [[10.0], [11.0], [12.0]]], tolerance=1e-6, hint="ch17.embed.hint"),
        Case("ch17.embed.shape", _embed_real, [1, 4, 5], hint="ch17.embed.hint"),
    ],
    "layer_norm": [
        Case("ch17.layer_norm.scale", _norm, (0.0, 1.0), tolerance=0.01, hint="ch17.layer_norm.hint"),
        Case("ch17.layer_norm.pytorch", _matches_pytorch, True, hint="ch17.layer_norm.hint"),
    ],
    "block_step": [
        Case("ch17.block_step.value", _block, 17.0, tolerance=1e-6, hint="ch17.block_step.hint"),
        Case("ch17.block_step.order", _order, ["norm1", "attention", "norm2", "think"], hint="ch17.block_step.hint"),
    ],
    "gpt_forward": [
        Case("ch17.gpt_forward.value", _forward, [[[6100.0], [10300.0]]], tolerance=1e-6, hint="ch17.gpt_forward.hint"),
        Case("ch17.gpt_forward.order", _forward_order, ["blocks", "norm", "out"], hint="ch17.gpt_forward.hint"),
    ],
    "write_ids": [
        Case("ch17.write_ids.counts", _write, [3, 4, 0, 1, 2, 3, 4], hint="ch17.write_ids.hint"),
        Case("ch17.write_ids.longest", _longest, (4, [1, 2, 3]), hint="ch17.write_ids.hint"),
    ],
}
