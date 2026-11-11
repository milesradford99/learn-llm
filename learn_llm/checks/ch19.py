import numpy as np
import torch

from learn_llm.checking import Case
from learn_llm.checks.ch09 import _plain


def _calls(f):
    calls = []
    f(lambda: calls.append(1), 4)
    return len(calls)


def _seconds(f):
    """The first call is slow (0.1 s), later ones take 0.01 s: a mean near 0.01 means the clock
    started after the warm-up and the total was divided by the repeats."""
    import time

    calls = []

    def task():
        time.sleep(0.1 if not calls else 0.01)
        calls.append(1)

    out = f(task, 5)
    return isinstance(out, float) and 0.0095 < out < 0.022


def _tiled(f, tile, scale=4):
    torch.manual_seed(5)
    q, K, V = torch.rand(8) * scale, torch.rand(10, 8) * scale, torch.rand(10, 3)
    reference = torch.softmax(K @ q / np.sqrt(8), dim=0) @ V
    out = f(q, K, V, tile)
    return bool(torch.max(torch.abs(out - reference)) < 1e-5) if isinstance(out, torch.Tensor) else _plain(out)


EXERCISES = {
    "time_it": [
        Case("ch19.time_it.calls", _calls, 5, hint="ch19.time_it.hint"),
        Case("ch19.time_it.mean", _seconds, True, hint="ch19.time_it.hint"),
    ],
    "matmul_flops": [
        Case("ch19.matmul_flops.small", lambda f: f(2, 3, 4), 48, hint="ch19.matmul_flops.hint"),
        Case("ch19.matmul_flops.square", lambda f: f(1024, 1024, 1024), 2 * 1024 ** 3, hint="ch19.matmul_flops.hint"),
    ],
    "training_flops": [
        Case("ch19.training_flops.page", lambda f: f(7e9, 1e12), 4.2e22, tolerance=1e10, hint="ch19.training_flops.hint"),
    ],
    "adam_bytes": [
        Case("ch19.adam_bytes.page", lambda f: f(7e9), 1.12e11, tolerance=1e-3, hint="ch19.adam_bytes.hint"),
    ],
    "tiled_attention": [
        Case("ch19.tiled_attention.one", lambda f: _tiled(f, 10), True, hint="ch19.tiled_attention.hint"),
        Case("ch19.tiled_attention.tiles", lambda f: _tiled(f, 3), True, hint="ch19.tiled_attention.hint"),
        Case("ch19.tiled_attention.single", lambda f: _tiled(f, 1), True, hint="ch19.tiled_attention.hint"),
        Case("ch19.tiled_attention.big", lambda f: _tiled(f, 3, 60), True, hint="ch19.tiled_attention.hint"),
    ],
}
