"""Reference solutions for chapter 20. Peek only after trying!"""

import numpy as np
import torch
import torch.nn.functional as F


def rms_norm(x, gain):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    return gain * x / rms


def rotate(a, b, angle):
    return a * np.cos(angle) - b * np.sin(angle), a * np.sin(angle) + b * np.cos(angle)


def rope_one(x, position, speeds):
    half = len(x) // 2
    a, b = rotate(x[:half], x[half:], position * speeds)
    return np.concatenate([a, b])


def swiglu(x, W1, W2, W3):
    return W2(F.silu(W1(x)) * W3(x))


def modern_block_step(x, gain1, look, gain2, W1, W2, W3):
    x = x + look(rms_norm(x, gain1))
    return x + swiglu(rms_norm(x, gain2), W1, W2, W3)
