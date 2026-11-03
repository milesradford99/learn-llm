"""Reference solutions for chapter 09. Peek only after trying!"""

import torch
import torch.nn.functional as F
from torch import nn


def torch_gradient(w, b, xs, ys):
    w = torch.tensor(w, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    e = w * x + b - y
    L = torch.mean(e * e)
    L.backward()
    return w.grad.item(), b.grad.item()


def scores(X, W1, b1, W2, b2):
    return torch.relu(X @ W1 + b1) @ W2 + b2


def make_model(hidden):
    return nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def train_step(model, optimizer, X, labels):
    optimizer.zero_grad()
    S = model(X)
    loss = F.cross_entropy(S, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
