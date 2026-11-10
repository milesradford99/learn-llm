"""Reference solutions for chapter 16. Peek only after trying!"""

import numpy as np
import torch


def future_mask(n):
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)


def attend(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    scores = scores.masked_fill(mask, -torch.inf)
    A = torch.softmax(scores, dim=-1)
    return A @ V, A


def split_heads(x, heads):
    batch, words, d = x.shape
    return x.reshape(batch, words, heads, d // heads).transpose(1, 2)


def layer_norm(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(variance + 1e-5)


def block_step(x, norm1, attention, norm2, think):
    x = x + attention(norm1(x))
    x = x + think(norm2(x))
    return x
