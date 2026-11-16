"""Reference solutions for chapter 28. Peek only after trying!"""

import numpy as np
import torch


def attend_new(Q, K, V):
    new, total = Q.shape[-2], K.shape[-2]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones(new, total, dtype=torch.bool), diagonal=total - new + 1)
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


def remember(cache, k, v):
    return torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)


def generate_cached(step, caches, ids, limit, stop):
    new = ids
    for turn in range(limit):
        scores, caches = step(new, caches)
        token = torch.argmax(scores).item()
        if token == stop:
            return ids
        ids = ids + [token]
        new = [token]
    return ids


def quantize(W, bits):
    levels = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1) / levels
    return torch.round(W / scale[:, None]).to(torch.int8), scale


def quantized_matmul(x, q, scale):
    return (x @ q.float().T) * scale
