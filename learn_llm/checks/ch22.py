import numpy as np
import torch

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain


def _calls(f):
    calls = []
    f(lambda: calls.append(1), 4)
    return len(calls)


_CLOCKS = ("perf_counter", "time", "monotonic")


def _seconds(f):
    """The first call is slow (0.1 s), later ones take 0.01 s: a mean of 0.01 means the clock
    started after the warm-up and the total was divided by the repeats.

    The clock is a fake one that only the task moves, so the answer is exact on any machine
    (real sleeps overran on busy CI runners)."""
    import time

    calls = []
    now = [1000.0]

    def task():
        now[0] += 0.1 if not calls else 0.01
        calls.append(1)

    real = {name: getattr(time, name) for name in _CLOCKS}
    for name in _CLOCKS:
        setattr(time, name, lambda: now[0])
    try:
        out = f(task, 5)
    finally:
        for name, clock in real.items():
            setattr(time, name, clock)
    return isinstance(out, float) and abs(out - 0.01) < 1e-6


def _tiled(f, tile, scale=4):
    torch.manual_seed(5)
    q, K, V = torch.rand(8) * scale, torch.rand(10, 8) * scale, torch.rand(10, 3)
    reference = torch.softmax(K @ q / np.sqrt(8), dim=0) @ V
    out = f(q, K, V, tile)
    return bool(torch.max(torch.abs(out - reference)) < 1e-5) if isinstance(out, torch.Tensor) else _plain(out)


EXERCISES = {
    "time_it": [
        Case("ch22.time_it.calls", _calls, 5, hint="ch22.time_it.hint"),
        Case("ch22.time_it.mean", _seconds, True, hint="ch22.time_it.hint"),
    ],
    "matmul_flops": [
        Case("ch22.matmul_flops.small", lambda f: f(2, 3, 4), 48, hint="ch22.matmul_flops.hint"),
        Case("ch22.matmul_flops.square", lambda f: f(1024, 1024, 1024), 2 * 1024 ** 3, hint="ch22.matmul_flops.hint"),
    ],
    "training_flops": [
        Case("ch22.training_flops.page", lambda f: f(7e9, 1e12), 4.2e22, tolerance=1e10, hint="ch22.training_flops.hint"),
    ],
    "adam_bytes": [
        Case("ch22.adam_bytes.page", lambda f: f(7e9), 1.12e11, tolerance=1e-3, hint="ch22.adam_bytes.hint"),
    ],
    "tiled_attention": [
        Case("ch22.tiled_attention.one", lambda f: _tiled(f, 10), True, hint="ch22.tiled_attention.hint"),
        Case("ch22.tiled_attention.tiles", lambda f: _tiled(f, 3), True, hint="ch22.tiled_attention.hint"),
        Case("ch22.tiled_attention.single", lambda f: _tiled(f, 1), True, hint="ch22.tiled_attention.hint"),
        Case("ch22.tiled_attention.big", lambda f: _tiled(f, 3, 60), True, hint="ch22.tiled_attention.hint"),
    ],
}
