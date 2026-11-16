import torch

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain


def _read(f, table, length):
    """A pool of 4 blocks of 2 tokens, 1 head, 1 number: block b holds 10 b and 10 b + 1."""
    pool = torch.tensor([[[[10.0 * b], [10.0 * b + 1]]] for b in range(4)])
    out = f(pool, table, length)
    if not isinstance(out, torch.Tensor):
        return _plain(out)
    return list(out.shape), out.reshape(-1).tolist()


def _names(requests):
    return [r["name"] for r in requests]


def _admit(f, prompts, running, free_blocks, max_running):
    waiting = [{"name": f"w{i}", "ids": [0] * n} for i, n in enumerate(prompts)]
    out = f(waiting, [{"name": f"r{i}", "ids": [0]} for i in range(running)], free_blocks, max_running, 16)
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        return _plain(out)
    return _names(out[0]), _names(out[1])


EXERCISES = {
    "blocks_for": [
        Case("ch29.blocks_for.hand", lambda f: (f(300, 16), f(32, 16)), (19, 2), hint="ch29.blocks_for.hint"),
        Case("ch29.blocks_for.edges", lambda f: (f(0, 16), f(1, 16), f(17, 16)), (0, 1, 2), hint="ch29.blocks_for.hint"),
    ],
    "slot": [
        Case("ch29.slot.hand", lambda f: _plain(f([7, 2, 9], 20, 16)), (2, 4), hint="ch29.slot.hint"),
        Case("ch29.slot.more", lambda f: (_plain(f([7, 2, 9], 0, 16)), _plain(f([7, 2, 9], 47, 16))), ((7, 0), (9, 15)), hint="ch29.slot.hint"),
    ],
    "read_cache": [
        Case("ch29.read_cache.example", lambda f: _read(f, [3, 0], 3), ([1, 1, 3, 1], [30.0, 31.0, 0.0]), hint="ch29.read_cache.hint"),
        Case("ch29.read_cache.full", lambda f: _read(f, [1, 2], 4), ([1, 1, 4, 1], [10.0, 11.0, 20.0, 21.0]), hint="ch29.read_cache.hint"),
    ],
    "admit": [
        Case("ch29.admit.example", lambda f: _admit(f, [20, 10, 5], 0, 3, 4), (["w2"], ["w0", "w1"]), hint="ch29.admit.hint"),
        Case("ch29.admit.full", lambda f: _admit(f, [5, 5], 3, 10, 4), (["w1"], ["r0", "r1", "r2", "w0"]), hint="ch29.admit.hint"),
        Case("ch29.admit.order", lambda f: _admit(f, [100, 5], 0, 3, 4), (["w0", "w1"], []), hint="ch29.admit.hint"),
    ],
    "chunk_sizes": [
        Case("ch29.chunk_sizes.hand", lambda f: f([1, 1, 1, 540], 64), [1, 1, 1, 61], hint="ch29.chunk_sizes.hint"),
        Case("ch29.chunk_sizes.out", lambda f: f([1, 100, 30], 64), [1, 63, 0], hint="ch29.chunk_sizes.hint"),
        Case("ch29.chunk_sizes.room", lambda f: f([1, 5, 2], 64), [1, 5, 2], hint="ch29.chunk_sizes.hint"),
    ],
}
