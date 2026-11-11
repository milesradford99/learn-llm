"""Reference solutions for chapter 19. Peek only after trying!"""

import time

import numpy as np
import torch


def time_it(f, repeats):
    f()
    started = time.perf_counter()
    for i in range(repeats):
        f()
    return (time.perf_counter() - started) / repeats


def matmul_flops(n, k, m):
    return 2 * n * k * m


def training_flops(knobs, tokens):
    return 6 * knobs * tokens


def adam_bytes(knobs):
    return 16 * knobs


def tiled_attention(q, K, V, tile):
    best = -np.inf
    total = 0.0
    acc = torch.zeros(V.shape[1])
    for start in range(0, len(K), tile):
        scores = K[start:start + tile] @ q / np.sqrt(len(q))
        new_best = max(best, torch.max(scores).item())
        fix = np.exp(best - new_best)
        weights = torch.exp(scores - new_best)
        total = total * fix + torch.sum(weights).item()
        acc = acc * fix + weights @ V[start:start + tile]
        best = new_best
    return acc / total
