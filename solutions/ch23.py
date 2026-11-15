"""Reference solutions for chapter 23. Peek only after trying!"""

import torch
import torch.nn.functional as F


def average_slopes(slopes):
    averaged = []
    for k in range(len(slopes[0])):
        averaged.append(torch.mean(torch.stack([device[k] for device in slopes], dim=0), dim=0))
    return averaged


def accumulated_slopes(model, X, Y, pieces):
    model.zero_grad()
    size = len(X) // pieces
    for i in range(pieces):
        part = slice(i * size, (i + 1) * size)
        (F.cross_entropy(model(X[part]), Y[part]) / pieces).backward()
    return [p.grad.clone() for p in model.parameters()]


def ring_bytes(size, devices):
    return 2 * (devices - 1) / devices * size


def shard(count, devices, rank):
    size = count // devices
    extra = count % devices
    start = rank * size + min(rank, extra)
    stop = start + size + (1 if rank < extra else 0)
    return start, stop


def bytes_per_device(knobs, devices, stage):
    weights = 4 * knobs
    slopes = 4 * knobs
    adam = 8 * knobs
    if stage >= 1:
        adam = adam / devices
    if stage >= 2:
        slopes = slopes / devices
    if stage >= 3:
        weights = weights / devices
    return weights + slopes + adam


def split_product(X, W, devices):
    pieces = []
    for rank in range(devices):
        start, stop = shard(W.shape[1], devices, rank)
        pieces.append(X @ W[:, start:stop])
    return torch.cat(pieces, dim=1)
