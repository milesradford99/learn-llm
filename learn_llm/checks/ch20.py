import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch09 import _plain


def _average(f):
    slopes = [[torch.tensor([1.0, 2.0]), torch.tensor([[0.0]])], [torch.tensor([3.0, 6.0]), torch.tensor([[4.0]])]]
    return _plain(f(slopes))


def _accumulate(f, pieces):
    torch.manual_seed(6)
    model = nn.Linear(4, 3)
    X, Y = torch.rand(12, 4), torch.randint(0, 3, (12,))
    model.zero_grad()
    nn.functional.cross_entropy(model(X), Y).backward()
    whole = [p.grad.clone() for p in model.parameters()]
    out = f(model, X, Y, pieces)
    if not isinstance(out, list):
        return _plain(out)
    return max(torch.max(torch.abs(a - b)).item() for a, b in zip(whole, out)) < 1e-6


EXERCISES = {
    "average_slopes": [
        Case("ch20.average_slopes.two", _average, [[2.0, 4.0], [[2.0]]], tolerance=1e-9, hint="ch20.average_slopes.hint"),
    ],
    "accumulated_slopes": [
        Case("ch20.accumulated_slopes.four", lambda f: _accumulate(f, 4), True, hint="ch20.accumulated_slopes.hint"),
        Case("ch20.accumulated_slopes.three", lambda f: _accumulate(f, 3), True, hint="ch20.accumulated_slopes.hint"),
    ],
    "ring_bytes": [
        Case("ch20.ring_bytes.four", lambda f: f(1000, 4), 1500.0, tolerance=1e-9, hint="ch20.ring_bytes.hint"),
        Case("ch20.ring_bytes.one", lambda f: f(1000, 1), 0.0, tolerance=1e-9, hint="ch20.ring_bytes.hint"),
    ],
    "shard": [
        Case("ch20.shard.even", lambda f: [f(12, 4, r) for r in range(4)], [(0, 3), (3, 6), (6, 9), (9, 12)], hint="ch20.shard.hint"),
        Case("ch20.shard.uneven", lambda f: [f(10, 4, r) for r in range(4)], [(0, 3), (3, 6), (6, 8), (8, 10)], hint="ch20.shard.hint"),
    ],
    "bytes_per_device": [
        Case("ch20.bytes_per_device.stages", lambda f: [f(1000, 4, s) for s in range(4)], [16000, 10000, 7000, 4000], tolerance=1e-9, hint="ch20.bytes_per_device.hint"),
    ],
}
