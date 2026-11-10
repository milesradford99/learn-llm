"""Reference solutions for chapter 11. Peek only after trying!"""

import numpy as np
import torch
from torch import nn


def convolve(image, kernel):
    h, w = image.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1))
    for r in range(h - k + 1):
        for c in range(w - k + 1):
            patch = image[r:r + k, c:c + k]
            out[r, c] = np.sum(patch * kernel)
    return out


def max_pool(M):
    h, w = M.shape
    out = np.zeros((h // 2, w // 2))
    for r in range(h // 2):
        for c in range(w // 2):
            out[r, c] = np.max(M[2 * r:2 * r + 2, 2 * c:2 * c + 2])
    return out


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, 10),
    )


def residual_block(x, conv1, conv2):
    h = torch.relu(conv1(x))
    h = conv2(h)
    return torch.relu(x + h)
