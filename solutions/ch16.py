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


def join_heads(x):
    batch, heads, words, piece = x.shape
    return x.transpose(1, 2).reshape(batch, words, heads * piece)


def self_attention_step(x, to_query, to_key, to_value, mix, heads):
    Q = split_heads(to_query(x), heads)
    K = split_heads(to_key(x), heads)
    V = split_heads(to_value(x), heads)
    blended, A = attend(Q, K, V, future_mask(x.shape[1]))
    return mix(join_heads(blended)), A
