"""Small, fast training runs for chapter 24: one model size, one amount of text, one loss.

The model is chapter 20's (RMSNorm, RoPE, SwiGLU) and the training is chapter 21's (AdamW,
warmup, cosine decay), written compactly here so the lesson can run a dozen of them and spend
its page on what the results say.
"""

import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from learn_llm.data import stories
from learn_llm.tokenizer import Tokenizer, story_words

WINDOW, BATCH, TEST_WINDOWS = 32, 16, 600


def story_tokens():
    """(train tokens, test tokens, vocabulary size): the text chapters 20 and 21 trained on."""
    train_stories, test_stories = stories()
    counts: dict[str, int] = {}
    for story in train_stories:
        for word in re.findall(r"[a-z']+", story.lower()):
            counts[word] = counts.get(word, 0) + 1
    known = set(sorted(counts, key=counts.get, reverse=True)[:2000])
    tokenizer = Tokenizer.load()
    train = torch.tensor(tokenizer.encode(story_words(train_stories, known)))
    test = torch.tensor(tokenizer.encode(story_words(test_stories, known)))
    return train, test, tokenizer.size


def _rope(x):
    d, words = x.shape[-1], x.shape[-2]
    angles = torch.arange(words)[:, None] / 10000 ** (torch.arange(0, d, 2) / d)
    a, b = x[..., 0::2], x[..., 1::2]
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).reshape(x.shape)


def _norm(x, gain):
    return gain * x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)


class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.gain1, self.gain2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.mix = nn.Linear(d, d, bias=False)
        hidden = 8 * d // 3
        self.W1, self.W3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.W2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        batch, words, d = x.shape
        q, k, v = self.qkv(_norm(x, self.gain1)).reshape(batch, words, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        blended = F.scaled_dot_product_attention(_rope(q), _rope(k), v, is_causal=True)
        x = x + self.mix(blended.transpose(1, 2).reshape(batch, words, d))
        y = _norm(x, self.gain2)
        return x + self.W2(F.silu(self.W1(y)) * self.W3(y))


class SmallGPT(nn.Module):
    def __init__(self, vocab_size, d, layers=2):
        super().__init__()
        self.words = nn.Embedding(vocab_size, d)
        self.blocks = nn.Sequential(*[_Block(d, max(1, d // 16)) for _ in range(layers)])
        self.gain = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, vocab_size, bias=False)

    def forward(self, ids):
        return self.out(_norm(self.blocks(self.words(ids)), self.gain))


def knobs_of(d: int, vocab_size: int) -> int:
    """How many knobs a `SmallGPT` of width `d` has."""
    return sum(p.numel() for p in SmallGPT(vocab_size, d).parameters())


def small_run(d: int, tokens: int, train: torch.Tensor, test: torch.Tensor, vocab_size: int, seed: int = 0) -> float:
    """Train a `SmallGPT` of width `d` on `tokens` tokens of `train`, read once, and give back
    its mean surprise per token on the first test windows."""
    torch.manual_seed(seed)
    model = SmallGPT(vocab_size, d)
    steps = int(tokens) // (WINDOW * BATCH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1)
    starts = torch.randperm((len(train) - 1) // WINDOW)[: steps * BATCH] * WINDOW
    offsets = torch.arange(WINDOW)
    warmup = max(1, steps // 10)
    for step in range(steps):
        progress = (step - warmup) / max(1, steps - warmup)
        rate = 0.003 * (step + 1) / warmup if step < warmup else 0.003 * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        at = starts[step * BATCH:(step + 1) * BATCH, None] + offsets
        optimizer.zero_grad()
        S = model(train[at])
        F.cross_entropy(S.reshape(-1, vocab_size), train[at + 1].reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    at = torch.arange(TEST_WINDOWS)[:, None] * WINDOW + offsets
    with torch.no_grad():
        S = model(test[at])
        return F.cross_entropy(S.reshape(-1, vocab_size), test[at + 1].reshape(-1)).item()
